"""
train_cnn.py

Train the CNN autoencoder for flow-level intrusion detection.

Run from the FLAIR root directory:
    python CNN/train_cnn.py

The model is trained ONLY on normal windows (target=0).
At inference, it detects intrusions from reconstruction error alone —
target labels are NEVER fed to the model.

Reads:
    data/processed/preprocessed.npz   (built by scripts/preprocess_data.py)
    CNN/config.yaml                    (CNN hyperparameters)

Writes:
    CNN/experiments/results/cnn_minimal.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from both FLAIR root (src.data.dataset) and CNN package
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import FLAIRDataset, DatasetConfig
from CNN.models.cnn_autoencoder import CNNAutoencoder, CNNConfig


@dataclass
class TrainConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs: int = 32
    seed: int = 42
    device: str = "auto"
    checkpoint_path: str = "CNN/experiments/results/cnn_minimal.pt"
    val_split: float = 0.1    # fraction of normal windows used for validation
    test_split: float = 0.1   # fraction of normal windows held out for test (never trained on)
    patience: Optional[int] = 10
    num_workers: int = 0       # set >0 on Linux; keep 0 on Windows to avoid spawn issues
    amp: bool = True


def _resolve_device(device_str: str) -> torch.device:
    if device_str in ("auto", "cuda"):
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            print(f"[train] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            dev = torch.device("cpu")
            print("[train] CUDA not available, using CPU")
    else:
        dev = torch.device(device_str)
    return dev


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_train_val(
    Xn: np.ndarray, Xc: np.ndarray, val_split: float, test_split: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal (chronological) split — NO shuffling.

    Windows in the npz are already in StartTime order from preprocessing.
    Splitting by position preserves this: the model only trains on past
    flows and is validated/tested on later ones.

    Layout:  |<-- train (1-val_split-test_split) -->|<-- val -->|<-- test -->|
    """
    n = len(Xn)
    test_n = max(1, int(n * test_split))
    val_n  = max(1, int(n * val_split))
    train_n = n - val_n - test_n

    tr_idx  = np.arange(0, train_n)
    val_idx = np.arange(train_n, train_n + val_n)
    # test slice is kept but not returned here — used only in evaluate_cnn.py

    return Xn[tr_idx], Xc[tr_idx], Xn[val_idx], Xc[val_idx]


def train_one_epoch(
    model: CNNAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    use_amp = scaler is not None

    for (x_num, x_cat), y_num in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_num = y_num.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x_num, x_cat)
            loss = model.reconstruction_loss(y_num, out["x_hat_num"], x_cat, out)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        batches += 1

    return total / max(batches, 1)


@torch.no_grad()
def eval_one_epoch(
    model: CNNAutoencoder, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    batches = 0

    for (x_num, x_cat), y_num in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_num = y_num.to(device)

        out = model(x_num, x_cat)
        loss = model.reconstruction_loss(y_num, out["x_hat_num"], x_cat, out)

        total += float(loss.item())
        batches += 1

    return total / max(batches, 1)


def train_from_preprocessed(
    npz_path: str = "data/processed/preprocessed.npz",
    train_cfg: Optional[TrainConfig] = None,
    config_path: Optional[str] = None,
) -> Dict[str, object]:
    if train_cfg is None:
        train_cfg = TrainConfig()

    set_seed(train_cfg.seed)
    device = _resolve_device(train_cfg.device)

    # Load preprocessed data (built from the original CSV by scripts/preprocess_data.py)
    bundle = np.load(npz_path, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)   # (N, T, D_num)
    X_cat = bundle["X_cat"].astype(np.int64)     # (N, T, 3)
    y_seq = bundle["y_seq"].astype(np.int64)     # (N,)  — 0=normal, 1=attack

    sport_vocab = bundle["sport_vocab"][0]
    dport_vocab = bundle["dport_vocab"][0]
    proto_vocab = bundle["proto_vocab"][0]

    print(f"[train] Loaded: {npz_path}")
    print(f"[train] X_num: {X_num.shape}  X_cat: {X_cat.shape}  y_seq: {y_seq.shape}")
    print(f"[train] attack windows: {int(y_seq.sum())}/{len(y_seq)}")

    # Use ONLY normal windows for training — target=1 (attack) windows are withheld.
    # This ensures the model never sees intrusion patterns during learning,
    # so high reconstruction error at inference reliably signals an anomaly.
    normal_mask = (y_seq == 0)
    Xn = X_num[normal_mask]
    Xc = X_cat[normal_mask]
    print(f"[train] Normal windows used for training: {len(Xn)}")

    if len(Xn) < 10:
        raise ValueError("Not enough normal windows to train.")

    Xn_tr, Xc_tr, Xn_val, Xc_val = split_train_val(
        Xn, Xc, train_cfg.val_split, train_cfg.test_split
    )
    n_test = max(1, int(len(Xn) * train_cfg.test_split))
    print(f"[train] Temporal split — Train: {len(Xn_tr)}  Val: {len(Xn_val)}  Test (held out): {n_test}")

    # Build model config
    cfg_model: Dict = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_model = yaml.safe_load(f).get("model", {})

    model_cfg = CNNConfig(
        numeric_dim=int(X_num.shape[-1]),
        sport_vocab_size=len(sport_vocab) + 1,
        dport_vocab_size=len(dport_vocab) + 1,
        proto_vocab_size=len(proto_vocab) + 1,
        embed_dim=int(cfg_model.get("embed_dim", 8)),
        encoder_channels=list(cfg_model.get("encoder_channels", [64, 128, 128])),
        latent_dim=int(cfg_model.get("latent_dim", 128)),
        kernel_size=int(cfg_model.get("kernel_size", 3)),
        seq_len=int(X_num.shape[1]),
        cat_loss_weight=float(cfg_model.get("cat_loss_weight", 0.1)),
    )

    train_ds = FLAIRDataset(Xn_tr, Xc_tr, config=DatasetConfig(return_targets=True))
    val_ds = FLAIRDataset(Xn_val, Xc_val, config=DatasetConfig(return_targets=True))

    pin = device.type == "cuda"
    persistent = train_cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    model = CNNAutoencoder(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    use_amp = train_cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[train] AMP: {'enabled' if use_amp else 'disabled'}")

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_left = train_cfg.patience
    train_losses = []
    val_losses = []

    for epoch in range(1, train_cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, scaler)
        va = eval_one_epoch(model, val_loader, device)

        train_losses.append(tr)
        val_losses.append(va)
        print(f"Epoch {epoch}/{train_cfg.epochs} - train: {tr:.6f}  val: {va:.6f}")

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if patience_left is not None:
                patience_left = train_cfg.patience
        else:
            if patience_left is not None:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train] Early stopping at epoch {epoch} (best: epoch {best_epoch}, val {best_val:.6f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = Path(train_cfg.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    print(f"\n[train] Checkpoint saved: {ckpt_path}")
    print(f"[train] Best val loss: {best_val:.6f} at epoch {best_epoch}")

    return {
        "checkpoint_path": str(ckpt_path),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    _config_path = "CNN/config.yaml"
    with open(_config_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    _t = _cfg.get("training", {})
    _p = _cfg.get("paths", {})

    cfg = TrainConfig(
        batch_size=int(_t.get("batch_size", 512)),
        learning_rate=float(_t.get("learning_rate", 1e-3)),
        epochs=int(_t.get("epochs", 32)),
        seed=int(_t.get("seed", 42)),
        device=str(_t.get("device", "auto")),
        checkpoint_path=str(_t.get("checkpoint_path", "CNN/experiments/results/cnn_minimal.pt")),
        val_split=float(_t.get("val_split", 0.1)),
        test_split=float(_t.get("test_split", 0.1)),
        patience=_t.get("patience", 10),
        num_workers=int(_t.get("num_workers", 0)),
        amp=bool(_t.get("amp", True)),
    )
    npz_path = str(_p.get("processed_npz", "data/processed/preprocessed.npz"))

    train_from_preprocessed(npz_path, train_cfg=cfg, config_path=_config_path)
