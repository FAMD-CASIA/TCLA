import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleSessionDataset(Dataset):
    """Dataset wrapper for one session pickle."""

    def __init__(
        self,
        spikes: np.ndarray,
        behavior: np.ndarray,
        labels: np.ndarray,
        session_id: str,
    ):
        self.spikes = torch.from_numpy(spikes).float()
        self.behavior = torch.from_numpy(behavior).float()
        self.labels = torch.from_numpy(labels).long()
        self.session_id = session_id

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, idx):
        return {
            "signal": self.spikes[idx].permute(1, 0),
            "behavior": self.behavior[idx].permute(1, 0),
            "label": self.labels[idx],
            "session_id": self.session_id,
        }


def get_session_data_and_configs(session_list, datapath_root):
    """Load session pickle files and infer each session's neuron count."""
    all_session_data = {}
    session_configs = {}

    print(f"Loading data for sessions: {session_list}...")
    for session_id in session_list:
        cache_file = os.path.join(datapath_root, f"{session_id}.pickle")
        with open(cache_file, "rb") as f:
            data_dict = pickle.load(f)

        spikes = data_dict["spike"]
        behavior = data_dict["behavior"]
        labels = data_dict["label"]
        all_session_data[session_id] = {
            "spikes": spikes,
            "behavior": behavior,
            "label": labels,
        }
        session_configs[session_id] = {"neuron_count": spikes.shape[2]}
        print(
            f"-> Loaded session '{session_id}' with "
            f"{spikes.shape[0]} trials, {spikes.shape[2]} neurons, and labels."
        )

    return all_session_data, session_configs
