import os
from typing import Tuple, Dict, List, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm.auto import tqdm
from einops import rearrange
import pdb

# --------------------------------------------------------------------------------
# 1. 单个Session的数据集类
# --------------------------------------------------------------------------------

class SingleSessionDataset(Dataset):
    """
    Dataset for a single session of monkey data, initialized with pre-split data.
    It also carries the session_id for multi-session model routing.
    """
    def __init__(
        self,
        spikes: np.ndarray,
        behavior: np.ndarray,
        labels: np.ndarray,
        session_id: str,
        time_last: bool = True,
    ):
        """
        Args:
            spikes (np.ndarray): Spike data for this split [Trials, Time, Neurons].
            behavior (np.ndarray): Behavior data for this split [Trials, Time, Dims].
            lanels (np.ndarray): Label for this split [Trials,].
            session_id (str): The identifier for this session.
            time_last (bool): If true, return tensors with time as the last dimension.
        """
        super().__init__()
        self.spikes = torch.from_numpy(spikes).float()
        self.behavior = torch.from_numpy(behavior).float()
        self.labels = torch.from_numpy(labels).long()
        self.session_id = session_id
        self.time_last = time_last
        self.neuron_count = self.spikes.shape[2]  # Get neuron count for model init

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, idx: int):
        """
        Returns a dictionary containing the signal, behavior, and session_id.
        The data is transposed to [C, L] format as required by Conv1d layers.
        """
        signal = self.spikes[idx].permute(1, 0)
        behavior = self.behavior[idx].permute(1, 0)
        label = self.labels[idx]
        
        return {
            "signal": signal,
            "behavior": behavior,
            "label": label, 
            "session_id": self.session_id,
        }

# --------------------------------------------------------------------------------
# 2. 主数据加载和划分函数 
# --------------------------------------------------------------------------------

def get_session_data_and_configs(session_list, datapath_root):
    """
    Loads all data for the given sessions and collects configurations.

    Args:
        session_list (List[str]): List of session names to load (e.g., ['session_1', 'session_2']).
        datapath_root (str): The root directory containing session data folders.

    Returns:
        Tuple containing:
        - A dictionary with all session data: {'session_1': {'spikes': ..., 'behavior': ...}, ...}
        - A dictionary with session configs: {'session_1': {'neuron_count': 70}, ...}
    """
    all_session_data = {}
    session_configs = {}

    print(f"Loading data for sessions: {session_list}...")
    for session_id in session_list:
        # Each session has a pre-processed .pickle file
        # The file is named after the session, e.g., /path/to/data/session_1.pickle
        cache_file = os.path.join(datapath_root, f"{session_id}.pickle")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Data file not found for session '{session_id}' at {cache_file}")
        
        with open(cache_file, "rb") as f:
            # The pickle file contains a dictionary {'spike': ..., 'behavior': ...}
            data_dict = pickle.load(f)
            spikes = data_dict['spike']      # Shape: [Trials, Time, Neurons]
            behavior = data_dict['behavior']  # Shape: [Trials, Time, Dims]
            labels = data_dict['label'] # Shape: [Trials, ]

        all_session_data[session_id] = {'spikes': spikes, 'behavior': behavior, 'label': labels}
        session_configs[session_id] = {'neuron_count': spikes.shape[2]}
        print(f"-> Loaded session '{session_id}' with {spikes.shape[0]} trials, {spikes.shape[2]} neurons, and labels.")
        
    return all_session_data, session_configs

# def get_left_out_session_dataloader(
#     left_out_session_id: str,
#     all_session_data: Dict[str, Dict[str, np.ndarray]],
#     num_folds: int,
#     fold_k: int,
#     batch_size: int,
#     num_workers: int = 4,
#     random_seed: int = 42
# ):
#     """
#     For the single left-out session, creates train (finetune) and test dataloaders
#     based on a specific fold of StratifiedKFold.

#     Args:
#         left_out_session_id (str): The ID of the session to process.
#         all_session_data (Dict): The dictionary containing all loaded session data.
#         num_folds (int): Total number of folds for StratifiedKFold.
#         fold_k (int): The k-th fold to use (0-indexed). Used 1:4 split.
#         batch_size (int): Batch size for the dataloaders.
#         num_workers (int): Number of workers for dataloaders.
#         random_seed (int): Seed for KFold shuffling.

#     Returns:
#         A tuple of (finetune_dataloader, test_dataloader).
#     """
#     session_data = all_session_data[left_out_session_id]
#     spikes = session_data['spikes']
#     behavior = session_data['behavior']
#     labels = session_data['label']

#     # StratifiedKFold split for train:test.
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
#     all_splits = list(skf.split(spikes, labels))

#     # In K-Fold, the k-th split usually means fold_k is the test set.
#     # train_indices are the 20% for finetuning, test_indices are the 80%.

#     test_indices, finetune_indices = all_splits[fold_k]

#     # Create datasets for this fold
#     finetune_dataset = SingleSessionDataset(
#         spikes=spikes[finetune_indices],
#         behavior=behavior[finetune_indices],
#         labels=labels[finetune_indices],
#         session_id=left_out_session_id,
#     )
    
#     test_dataset = SingleSessionDataset(
#         spikes=spikes[test_indices],
#         behavior=behavior[test_indices],
#         labels=labels[test_indices], 
#         session_id=left_out_session_id,
#     )
    
#     # Create dataloaders
#     finetune_loader = DataLoader(
#         finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
#     )
    
#     print(f"--- Session '{left_out_session_id}' (Fold {fold_k+1}/{num_folds}) ---")
#     print(f"Finetune set size: {len(finetune_dataset)}, Test set size: {len(test_dataset)}")
    
#     return finetune_loader, test_loader
#     # return finetune_loader, test_loader, finetune_indices, test_indices


def get_left_out_session_dataloader(
    session_id: str,
    all_session_data: Dict[str, Dict[str, np.ndarray]],
    num_folds: int, 
    fold_k: int, 
    val_test_ratio: float = 0.25, # 80%中的1/4作为验证集
    batch_size: int = 64,
    num_workers: int = 4,
    random_seed: int = 42
):
    """
    使用双重分层划分，为指定的 fold_k 创建独立的训练、验证和测试集。
    
    Args:
        session_id (str): 要处理的 session ID。
        all_session_data (Dict): 包含 spikes, behavior, label 的数据字典。
        num_folds (int): 第一次划分的总折数 (例如 5 对应 20% 训练集)。
        fold_k (int): 当前是第 k 个 fold (0-indexed)。
        val_test_ratio (float): 从 第一次划分的测试集 中分出多少比例作为验证集。
                                0.25 对应 80% * 0.25 = 20% 总数据。
    """
    session_data = all_session_data[session_id]
    spikes = session_data['spikes']
    behavior = session_data['behavior']
    labels = session_data['label']

    skf_first = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    all_splits_first = list(skf_first.split(spikes, labels))

    # test_indices 是大份 (80%), train_indices 是小份 (20%)
    temp_test_indices, train_indices = all_splits_first[fold_k]

    # 将临时的80%再次划分为验证集和测试集 ---
    temp_test_spikes = spikes[temp_test_indices]
    temp_test_labels = labels[temp_test_indices]
    n_splits_second = int(1 / val_test_ratio)
    skf_second = StratifiedKFold(n_splits=n_splits_second, shuffle=True, random_state=random_seed)
    
    # 在80%的数据上只进行一次划分即可，我们固定取第0个划分方案
    # 每次 fold_k 变化时，temp_test_indices 会变化，所以最终的 val/test 集也会不同
    second_split_generator = skf_second.split(temp_test_spikes, temp_test_labels)
    test_indices_relative, val_indices_relative = next(second_split_generator)

    # 将相对索引转换回原始数据集的绝对索引
    val_indices = temp_test_indices[val_indices_relative]
    test_indices = temp_test_indices[test_indices_relative]

    print(f"--- Data Split for '{session_id}' (Fold {fold_k+1}/{num_folds}, Stratified) ---")
    print(f"Total trials: {len(spikes)}")
    print(f"  - Train set size: {len(train_indices)}")
    print(f"  - Validation set size: {len(val_indices)}")
    print(f"  - Test set size: {len(test_indices)}")
    
    # --- 创建 Datasets 和 DataLoaders ---
    train_dataset = SingleSessionDataset(
        spikes=spikes[train_indices], behavior=behavior[train_indices],
        labels=labels[train_indices], session_id=session_id
    )
    val_dataset = SingleSessionDataset(
        spikes=spikes[val_indices], behavior=behavior[val_indices],
        labels=labels[val_indices], session_id=session_id
    )
    test_dataset = SingleSessionDataset(
        spikes=spikes[test_indices], behavior=behavior[test_indices],
        labels=labels[test_indices], session_id=session_id
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader



def custom_collate_for_multisession(batch):
    """
    Collates data for the multi-session dataloader.
    - 'signal': returned as a list of tensors, since their shapes are different.
    - 'behavior': stacked into a single tensor, as their shapes are consistent.
    - 'session_id': returned as a list of strings.
    """
    signals = [item['signal'] for item in batch]
    behaviors = torch.stack([item['behavior'] for item in batch], 0)
    labels = torch.stack([item['label'] for item in batch], 0)
    session_ids = [item['session_id'] for item in batch]
    
    return {
        'signal': signals,
        'behavior': behaviors,
        'label': labels,
        'session_id': session_ids
    }


def get_main_train_dataloader(
    train_session_ids: List[str],
    all_session_data: Dict[str, Dict[str, np.ndarray]],
    batch_size: int,
    num_workers: int = 4,
):
    """
    Creates a single combined DataLoader for training the main model on n-1 sessions.
    """
    datasets_to_combine = []
    for session_id in train_session_ids:
        session_data = all_session_data[session_id]
        dataset = SingleSessionDataset(
            spikes=session_data['spikes'],
            behavior=session_data['behavior'],
            labels=session_data['label'],
            session_id=session_id
        )
        datasets_to_combine.append(dataset)
    
    combined_dataset = ConcatDataset(datasets_to_combine)
    
    train_loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_for_multisession
    )
    
    print(f"Created main training dataloader with {len(combined_dataset)} total trials from {len(train_session_ids)} sessions.")
    return train_loader

# --------------------------------------------------------------------------------
# 3. Latent Dataset Class
# --------------------------------------------------------------------------------

class LatentMonkeyDatasetMulti(Dataset):
    """
    Dataset class for latent representations of multi-session monkey neural data.
    It encodes data from a specific session using a multi-session autoencoder.
    """
    def __init__(
        self, 
        dataloader: DataLoader, 
        ae_model: torch.nn.Module, 
        session_id: str,
        clip: bool = True, 
        latent_means: torch.Tensor = None, 
        latent_stds: torch.Tensor = None
    ):
        """
        Args:
            dataloader (DataLoader): DataLoader containing data for ONE session.
            ae_model (torch.nn.Module): The trained MultiSessionAutoEncoder model.
            session_id (str): The session ID for this dataset, to use the correct read-in layer.
            clip (bool): Whether to clip latent values to [-5, 5].
            latent_means (torch.Tensor, optional): Precomputed means for normalization.
            latent_stds (torch.Tensor, optional): Precomputed stds for normalization.
        """
        self.full_dataloader = dataloader
        self.ae_model = ae_model
        self.session_id = session_id

        # Encode the entire dataset to get latents and other data
        self.latents, self.spikes, self.behavior, self.labels = self._create_latents()
        
        # Normalize latents to N(0,1)
        if latent_means is None or latent_stds is None:
            # Calculate stats from the data if not provided
            self.latent_means = self.latents.mean(dim=(0, 2), keepdim=True)
            self.latent_stds = self.latents.std(dim=(0, 2), keepdim=True)
        else:
            self.latent_means = latent_means
            self.latent_stds = latent_stds
            
        self.latents = (self.latents - self.latent_means) / self.latent_stds
        
        # Optionally clip extreme values
        if clip:
            self.latents = torch.clamp(self.latents, -5, 5)

        # Verify data alignment
        assert len(self.latents) == len(self.behavior) == len(self.spikes) == len(self.labels), \
            f"Data length mismatch: {len(self.latents)}, {len(self.behavior)}, {len(self.spikes)}, {len(self.labels)}"

    def _create_latents(self):
        """
        Encodes all data from the dataloader for this session.
        This now correctly uses the session_id.
        """
        latent_list, spike_list, behavior_list, label_list = [], [], [], []
        self.ae_model.eval()
        device = next(self.ae_model.parameters()).device
        
        with torch.no_grad():
            for batch in tqdm(self.full_dataloader, desc=f"Encoding session '{self.session_id}'"):
                signal = batch["signal"].to(device)
                
                # Use the model's encode method with the session_id
                z = self.ae_model.encode(signal, session_id=self.session_id)
                
                latent_list.append(z.cpu())
                spike_list.append(batch["signal"])
                behavior_list.append(batch["behavior"])
                label_list.append(batch["label"])
                
        return torch.cat(latent_list), torch.cat(spike_list), torch.cat(behavior_list), torch.cat(label_list)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        """
        Returns a sample from the latent dataset.
        """
        return {
            "signal": self.spikes[idx],
            "latent": self.latents[idx],
            "behavior": self.behavior[idx],
            "label": self.labels[idx], 
        }