import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
# ------------------ dataset classes ------------------


class ReconDataset(torch.utils.data.Dataset):
    """Dataset class for reconstruction data containing latents, samples and rates.

    Args:
        latents: Latent vectors
        samples: Sample data
        rates: Rate data
    """

    def __init__(self, latents, samples, rates):
        self.latents = latents
        self.samples = samples
        self.rates = rates

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.samples[idx], self.rates[idx]


# ------------------ rate statistics ------------------


def average_rates(data, mode="neur", fps=None):
    """
    Computes the average of spike counts or rates with the option to do neuronwise averaging
    or treating each neuron and time series individually.

    Args:
        data: 3D numpy array containing spike counts with shape (n_samples, n_seqlen, n_neurons)
        mode: The mode of averaging - 'neur', 'neurtime', or 'neursample'
        fps: Frames per second for converting to Hz

    Returns:
        np.ndarray: Averaged rates
    """
    if mode == "neur":
        # Average across the first two dimensions (samples and time series)
        averaged = np.nanmean(data, axis=(0, 1)).flatten()
    elif mode == "neurtime":
        # Average across the samples dimension only
        averaged = np.nanmean(data, axis=0).flatten()
    elif mode == "neursample":
        # Average across the time dimension only
        averaged = np.nanmean(data, axis=1).flatten()
    else:
        raise ValueError("Invalid mode. Choose either 'neuronwise' or 'neurontimewise' or 'neuronsamplewise'.")

    if fps is not None:
        # Convert to Hz by multiplying with the frame rate
        averaged = averaged * fps
    return averaged


def std_rates(data, mode="neur", fps=None):
    """
    Computes the standard deviation of spike counts or rates.

    Args:
        data: 3D numpy array containing spike counts
        mode: The mode of averaging - 'neur', 'neurtime', or 'neursample'
        fps: Frames per second for converting to Hz

    Returns:
        np.ndarray: Standard deviation of rates
    """
    if mode == "neur":
        # Average across the first two dimensions (samples and time series)
        stdev = np.nanstd(data, axis=(0, 1)).flatten()
    elif mode == "neurtime":
        # Average across the samples dimension only
        stdev = np.nanstd(data, axis=0).flatten()
    elif mode == "neursample":
        # Average across the time dimension only
        stdev = np.nanstd(data, axis=1).flatten()
    else:
        raise ValueError("Invalid mode. Choose either 'neuronwise' or 'neurontimewise' or 'neuronsamplewise'.")

    if fps is not None:
        # Convert to Hz by multiplying with the frame rate
        stdev = stdev * fps
    return stdev


# ------------------ correlation analysis ------------------


def correlation_matrix(data, sample=None, mode="concatenate"):
    """
    Compute averaged correlation matrix across samples.

    Args:
        data: Input data array
        sample: Sample index to compute correlation for
        mode: 'average' or 'concatenate' mode for computing correlations

    Returns:
        np.ndarray: Correlation matrix
    """
    if len(data.shape) > 2:
        if sample is not None:
            return np.corrcoef(data[sample, :, :].T)
        elif mode == "average":
            n_samples = data.shape[0]
            # Compute correlation matrix for each sample
            corrs = np.array([np.corrcoef(data[i, :, :].T) for i in range(n_samples)])
            return np.nanmean(corrs, axis=0)
        elif mode == "concatenate":
            # Concatenate all samples and n_seqlen
            return np.corrcoef(data.reshape(-1, data.shape[-1]).T)
    else:
        return np.corrcoef(data.T)


def get_corr_mat(data, mod="trial", name=""):
    """
    Compute correlation matrix averaged across trials or all data.

    Args:
        data: Input data array
        mod: Mode for computation ('trial' or other)
        name: Name prefix for output

    Returns:
        tuple: (correlation matrix, name string)
    """
    name = "Corr Matrix " + name
    _, no_samples, xdim = data.shape
    if mod == "trial":
        C_corr = np.zeros((xdim, xdim))

        for s in range(no_samples):
            C_corr += np.corrcoef(data[:, s, :].T)
        C_corr = C_corr / no_samples
        name = name + "av. across trials"
    else:
        data_reshaped = np.reshape(np.transpose(data, (2, 1, 0)), (xdim, -1))
        C_corr = np.corrcoef(data_reshaped)
    return C_corr, name


def group_neurons_temp_corr(data_x, num_groups=4):
    """
    Group neurons based on total correlation.

    Args:
        data_x: Time-series data [seq_len, num_trials, x_dim]
        num_groups: Number of groups to divide neurons into

    Returns:
        list: Groups of neuron indices
    """
    assert isinstance(data_x, np.ndarray), "Input data has to be numpy array!"
    corr_mat, _ = get_corr_mat(data_x, mod="all")
    np.fill_diagonal(corr_mat, 0)
    c_val = np.sum(corr_mat**2, axis=0)
    sorted_neurons = np.argsort(-c_val)
    g_size = data_x.shape[-1] // num_groups
    neuron_groups = []
    neuron_groups = [sorted_neurons[i : i + g_size] for i in range(0, len(sorted_neurons), g_size)]
    if len(neuron_groups) > num_groups:
        neuron_groups[-2] = np.concatenate((neuron_groups[-2], neuron_groups[-1]), axis=0)
        del neuron_groups[-1]
    return neuron_groups


# ------------------ temporal correlation ------------------


def get_temp_corr(x1, x2, nlags=10, mode="biased"):
    """
    Compute temporal correlation between two time series.

    Args:
        x1, x2: Input time series
        nlags: Number of lags
        mode: 'biased' or 'unbiased' normalization

    Returns:
        np.ndarray: Temporal correlation values
    """
    T = x1.shape[0]
    assert x2.shape[0] == T, "Must be same length!"

    norm_factor = np.ones((nlags * 2 + 1,))
    if mode == "biased":
        norm_factor = norm_factor * T
    elif mode == "unbiased":
        norm_factor = T - abs(np.arange(-nlags, nlags + 1))

    full_corr = np.correlate(x1, x2, "full")
    zero_lag_ind = int(((2 * T - 1) + 1) / 2 - 1)

    corr_result = full_corr[zero_lag_ind - nlags : zero_lag_ind + nlags + 1]
    return np.divide(corr_result, norm_factor)


def get_temp_corr_trial_av(data_x, nlags=10, mode="biased"):
    """
    Compute trial-averaged temporal correlations.

    Args:
        data_x: Input data array
        nlags: Number of lags
        mode: Correlation mode

    Returns:
        tuple: (cross correlations, auto correlations)
    """
    cross_corr = []
    auto_corr = []

    for i_trial in range(data_x.shape[1]):
        for ii in range(data_x.shape[-1]):
            for jj in range(ii, data_x.shape[-1]):
                xc = get_temp_corr(data_x[:, i_trial, ii], data_x[:, i_trial, jj], nlags, mode=mode)
                if ii == jj:
                    xc[nlags] = 0
                    auto_corr.append(xc)
                else:
                    cross_corr.append(xc)
    return np.array(cross_corr), np.array(auto_corr)


def get_temp_corr_summary(data_x, groups, nlags=10, binWidth=100, mode="biased", batch_first=False):
    """
    Compute temporal correlation summary for neuron groups.

    Args:
        data_x: Input data
        groups: Neuron groups
        nlags: Number of lags
        binWidth: Width of time bins
        mode: Correlation mode
        batch_first: If True, expects batch dimension first

    Returns:
        tuple: (cross correlation groups, auto correlation groups)
    """
    if batch_first:
        data_x = np.transpose(data_x, (1, 0, 2))
    seq_len, no_samples, xdim = data_x.shape

    mean_tensor = np.mean(data_x, axis=0)
    data_x = data_x - np.tile(mean_tensor, (seq_len, 1, 1))

    num_groups = len(groups)
    cross_corr_groups = []
    auto_corr_groups = []

    for k in range(num_groups):
        data_group = data_x[:, :, groups[k]]
        cross_corr_g, auto_corr_g = get_temp_corr_trial_av(data_group, nlags, mode)
        cross_corr_groups.append(np.mean(cross_corr_g, axis=0))
        auto_corr_groups.append(np.mean(auto_corr_g, axis=0))

    return cross_corr_groups, auto_corr_groups


# ------------------ spike train analysis ------------------


def counts_to_spike_trains_ragged(bin_counts, fps):
    """
    Generate spike trains from bin counts. Especially for ragged data.

    Parameters:
    - bin_counts: List of 2D numpy arrays (n_seqlen_i, n_neurons) with spike counts.
    - fps: Frames per second, defining the duration of each bin.

    Returns:
    - A dictionary with keys as (sample_index, neuron_index) and values as arrays of spike times.
    """
    n_samples = len(bin_counts)
    n_neurons = bin_counts[0].shape[1]
    bin_duration = 1.0 / fps
    spike_trains = {}

    for sample_idx in range(n_samples):
        for neuron_idx in range(n_neurons):
            spike_times = []
            n_seqlen = bin_counts[sample_idx].shape[0]
            for bin_idx in range(n_seqlen):
                count = int(bin_counts[sample_idx][bin_idx, neuron_idx])
                if count > 0:
                    start_time = bin_idx * bin_duration
                    spikes = np.linspace(start_time, start_time + bin_duration, count + 2)[1:-1]
                    spike_times.extend(spikes)
            spike_trains[(sample_idx, neuron_idx)] = np.array(spike_times)

    return spike_trains


def counts_to_spike_trains(bin_counts, fps):
    """
    Generate spike trains from bin counts.

    Args:
        bin_counts: 3D numpy array (n_samples, n_seqlen, n_neurons) with spike counts
        fps: Frames per second, defining the duration of each bin

    Returns:
        dict: Keys as (sample_index, neuron_index) and values as arrays of spike times
    """
    n_samples, n_seqlen, n_neurons = bin_counts.shape
    bin_duration = 1.0 / fps
    spike_trains = {}

    for sample_idx in range(n_samples):
        for neuron_idx in range(n_neurons):
            spike_times = []
            for bin_idx in range(n_seqlen):
                count = int(bin_counts[sample_idx, bin_idx, neuron_idx])
                if count > 0:
                    start_time = bin_idx * bin_duration
                    spikes = np.linspace(start_time, start_time + bin_duration, count + 2)[1:-1]
                    spike_times.extend(spikes)
            spike_trains[(sample_idx, neuron_idx)] = np.array(spike_times)

    return spike_trains


def compute_spike_stats_per_neuron(spike_trains, n_samples, n_neurons, mean_output=True):
    """
    Compute statistics for spike trains.

    Args:
        spike_trains: Dictionary of spike trains with keys as (sample_index, neuron_idx)
        n_samples: Total number of samples
        n_neurons: Total number of neurons
        mean_output: Whether to output mean values

    Returns:
        dict: mean and std ISIs per neuron
    """
    isi = {n: [] for n in range(n_neurons)}
    for (sample_idx, neuron_idx), spikes in spike_trains.items():
        if len(spikes) > 1:
            isis = np.diff(spikes)
            isi[neuron_idx].extend(isis)

    mean_isi = np.full(n_neurons, np.nan)
    std_isi = np.full(n_neurons, np.nan)

    for n in range(n_neurons):
        mean_isi[n] = np.mean(isi[n]) if len(isi[n]) > 0 else np.nan
        std_isi[n] = np.std(isi[n]) if len(isi[n]) > 0 else np.nan

    return {
        "mean_isi": mean_isi,
        "std_isi": std_isi,
    }


# ------------------ model evaluation metrics ------------------


def reconstruct_spikes(model, full_dataloader):
    """
    Reconstruct spikes using trained model.

    Args:
        model: Trained neural network model
        full_dataloader: DataLoader containing input data

    Returns:
        dict: Contains latents, spikes and reconstructed spikes
    """
    model.eval()
    latents = []
    spikes = []
    rec_spikes = []
    for batch in full_dataloader:
        signal = batch["signal"]
        with torch.no_grad():
            output_rates, z = model(signal)
            z = z.cpu()
        latents.append(z)
        spikes.append(signal.cpu())
        rec_spikes.append(torch.poisson(output_rates.cpu()))

    return {"latents": torch.cat(latents, 0), "spikes": torch.cat(spikes, 0), "rec_spikes": torch.cat(rec_spikes, 0)}


def rmse_nan(y_pred, y):
    """
    Compute RMSE ignoring NaN values.

    Args:
        y_pred: Predicted values
        y: True values

    Returns:
        float: RMSE value
    """
    return np.sqrt(np.nanmean((y_pred - y) ** 2))


def kl_div(p, q):
    """
    Compute KL divergence between distributions.

    Args:
        p, q: Input distributions

    Returns:
        float: KL divergence value
    """
    return np.nansum(p * np.log(p / q))




def get_spikes_with_history(neural_data,bins_before,bins_after=0,bins_current=1):
    num_examples = neural_data.size(0)  # Number of total time bins we have neural data for
    num_neurons = neural_data.size(1)  # Number of neurons
    surrounding_bins = bins_before + bins_after + bins_current  # Number of surrounding time bins used for prediction
    
    # Initialize an empty tensor with the desired shape
    X = torch.zeros(num_examples, surrounding_bins, num_neurons, dtype=torch.float32, device=neural_data.device)
    for i in range(num_examples - bins_before - bins_after):  # The first bins_before and last bins_after bins don't get filled in
        end_idx = i + surrounding_bins  # The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i + bins_before, :, :] = neural_data[i:end_idx, :]  # Put neural data from surrounding bins in X, starting at row "bins_before"
    return X


def downsample_trial_data(neural, behavior, factor: int = 10): # 5ms->50ms
    """
    将神经数据在时间维上做 `factor` 倍降采样（求和），
    将行为数据在时间维上做 `factor` 倍降采样（求平均）。

    返回:
        neural_ds  [A, B, C//factor]，神经数据
        behavior_ds  [A, B2, C//factor]，行为数据
    """

    # ------- 重塑并降采样 -------
    # 对神经数据：求和
    A, B, C = neural.shape
    neural_rs = neural.reshape(A, B, C // factor, factor)
    neural_ds = neural_rs.sum(-1)          # [A, B, C//factor]

    # 对行为学数据：求平均
    A, B2, C = behavior.shape
    behavior_rs = behavior.reshape(A, B2, C // factor, factor)
    behavior_ds = behavior_rs.mean(-1)     # [A, B2, C//factor]

    return neural_ds, behavior_ds



class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, num_layers=2):
        super(LSTMRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_size]

        # Only use the last time step output
        out = out[:, -1, :]  # [batch, hidden_size]

        # Fully connected layer
        out = self.fc(out)  # [batch, output_size]
        return out



def train_lstm_regression(train_rates, train_behavior, model, criterion, optimizer, batch_size=32, num_epochs=10, device=None):
    dataset = TensorDataset(train_rates, train_behavior)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device) #[batch, seq_len, input_size]
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X) 
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0) 

        epoch_loss /= len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss:{epoch_loss:.4f}')
    # torch.save(model.state_dict(), f"/data0/user/cyzhao/LD-NAE/exp/eye_movement_FEF_pos_5ms/7/lstm_model.pt")
    return epoch_loss


def lstm_regression(model, val_rates, batch_size=32, device=None):
    dataset = TensorDataset(val_rates)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    predicted_behavior = []

    with torch.no_grad():
        for batch_X in dataloader:
            batch_X = batch_X[0].to(device)
            
            outputs = model(batch_X)
            predicted_behavior.append(outputs.cpu())

    predicted_behavior = torch.cat(predicted_behavior, dim=0)

    return predicted_behavior


def get_R2(y_test, y_test_pred):
    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]):
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2)
    R2_array=np.array(R2_list)
    return R2_array #Return an array of R2s

def get_corrcoef(y_test, y_pred):
    corr_all = torch.zeros(y_test.shape[1])
    for j in range(y_test.shape[1]):
        y_true_j = y_test[:, j]
        y_pred_j = y_pred[:, j]
        corr = torch.corrcoef(torch.stack([y_pred_j, y_true_j]))[0, 1].item()
        corr_all[j] = corr
    return corr_all

