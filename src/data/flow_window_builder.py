"""
flow_window_builder.py

This module groups individual network flows into fixed-length
sequences using a sliding window approach.

Each sequence represents a short snapshot of network behavior
over time and serves as a single input sample for the FLAIR model.

No machine learning occurs here. This module only prepares data
in a structured and testable format.
"""
from typing import List
import pandas as pd
import numpy as np

def sort_flows_by_time(flows_df: pd.DataFrame,
                       time_column: str = "StartTime") -> pd.DataFrame:
    """
    Sorts flows in chronological order.

    Parameters:
        flows_df: DataFrame where each row is a flow
        time_column: Name of the column representing flow start time

    Returns:
        Time-ordered DataFrame of flows
    """
    return flows_df.sort_values(by=time_column).reset_index(drop=True)

def build_sliding_windows(flows_df: pd.DataFrame,
                          window_size: int) -> List[pd.DataFrame]:
    """
    Builds fixed-length sliding windows of flows.

    Parameters:
        flows_df: Time-ordered DataFrame of flows
        window_size: Number of flows per sequence

    Returns:
        A list of DataFrames, each representing one flow sequence
    """
    sequences = []

    total_flows = len(flows_df)
    if total_flows < window_size:
        return sequences

    for start_idx in range(0, total_flows - window_size + 1):
        window = flows_df.iloc[start_idx:start_idx + window_size]
        sequences.append(window)

    return sequences

def sequences_to_numpy(sequences: List[pd.DataFrame],
                       feature_columns: List[str]) -> np.ndarray:
    """
    Converts flow sequences into a NumPy array.

    Output shape:
        (num_sequences, window_size, num_features)
    """
    array_sequences = []

    for seq in sequences:
        array_sequences.append(seq[feature_columns].values)

    return np.array(array_sequences)

def build_flow_sequences(flows_df: pd.DataFrame,
                         feature_columns: List[str],
                         window_size: int,
                         time_column: str = "StartTime") -> np.ndarray:
    """
    High-level helper that builds flow sequences from raw flow data.
    """
    sorted_flows = sort_flows_by_time(flows_df, time_column)
    windows = build_sliding_windows(sorted_flows, window_size)
    return sequences_to_numpy(windows, feature_columns)

