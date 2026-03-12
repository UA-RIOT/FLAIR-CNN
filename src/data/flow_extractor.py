"""
flow_extractor.py

This module converts raw network traffic into flow-level records
that can be used by the FLAIR model.

A network flow represents a short conversation between two devices
(e.g., PLC ↔ HMI) and summarizes communication using statistical
features such as packet counts, byte counts, rates, and duration.

This module does NOT perform machine learning.
It only prepares data in a form that FLAIR can later analyze.
"""
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import numpy as np

from .feature_definitions import FLOW_FEATURES

def get_flow_key(row: pd.Series) -> tuple:
    """
    Creates a unique identifier for a network flow.

    A flow is defined using the 5-tuple:
    (source IP, destination IP, source port, destination port, protocol)
    """
    return (
        row["SrcIP"],
        row["DstIP"],
        row["Sport"],
        row["Dport"],
        row["Proto"]
    )

def initialize_flow_stats() -> Dict:
    """
    Initializes all counters and accumulators for a single flow.
    """
    return {
        "SrcPkts": 0,
        "DstPkts": 0,
        "Tpkts": 0,

        "SrcBytes": 0,
        "DstBytes": 0,
        "TBytes": 0,

        "start_time": None,
        "end_time": None
    }

def process_packets(df: pd.DataFrame) -> Dict:
    """
    Processes packet-level data and groups packets into flows.

    Parameters:
        df: A pandas DataFrame where each row represents a packet.

    Returns:
        A dictionary mapping flow keys to accumulated statistics.
    """
    flows = defaultdict(initialize_flow_stats)

    for _, row in df.iterrows():
        key = get_flow_key(row)
        flow = flows[key]

        # Packet counts
        flow["Tpkts"] += 1

        if row["Direction"] == "src_to_dst":
            flow["SrcPkts"] += 1
            flow["SrcBytes"] += row["Bytes"]
        else:
            flow["DstPkts"] += 1
            flow["DstBytes"] += row["Bytes"]

        flow["TBytes"] = flow["SrcBytes"] + flow["DstBytes"]

        # Time tracking
        timestamp = row["Timestamp"]
        if flow["start_time"] is None:
            flow["start_time"] = timestamp
        flow["end_time"] = timestamp

    return flows

def compute_flow_features(flow_stats: Dict) -> Dict:
    """
    Computes final flow-level features from accumulated statistics.
    """
    duration = flow_stats["end_time"] - flow_stats["start_time"]
    duration = max(duration, 1e-6)  # avoid divide-by-zero

    features = {
        "SrcPkts": flow_stats["SrcPkts"],
        "DstPkts": flow_stats["DstPkts"],
        "Tpkts": flow_stats["Tpkts"],

        "SrcBytes": flow_stats["SrcBytes"],
        "DstBytes": flow_stats["DstBytes"],
        "TBytes": flow_stats["TBytes"],

        "Srate": flow_stats["SrcPkts"] / duration,
        "Drate": flow_stats["DstPkts"] / duration,
        "Trate": flow_stats["Tpkts"] / duration,

        "Duration": duration
    }

    return features

def extract_flows(packet_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts packet-level data into a flow-level DataFrame.

    Parameters:
        packet_df: DataFrame containing packet-level data

    Returns:
        DataFrame where each row represents a network flow
    """
    flows = process_packets(packet_df)

    records = []
    for key, stats in flows.items():
        features = compute_flow_features(stats)

        record = {
            "SrcIP": key[0],
            "DstIP": key[1],
            "Sport": key[2],
            "Dport": key[3],
            "Proto": key[4],
            **features
        }

        records.append(record)

    return pd.DataFrame(records)
