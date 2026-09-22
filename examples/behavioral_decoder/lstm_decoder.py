import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "behavioral_decoder" / "lstm"
DEFAULT_DECODER_CONFIG = REPO_ROOT / "config" / "behavioral_decoders.yaml"


def load_lstm_config(path):
    return OmegaConf.load(path).lstm


class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class LSTMRegression:
    def __init__(
        self,
        input_size,
        output_size,
        units,
        dropout,
        num_epochs,
        batch_size,
        learning_rate,
        num_workers,
        device,
        use_amp=True,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.model = LSTMRegressionModel(input_size, units, output_size, dropout).to(self.device)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.train_loss_history = []

    def fit(self, x_train, y_train):
        dataset = TensorDataset(x_train.float(), y_train.float())
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(dataset, **loader_kwargs)

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    pred = self.model(batch_x)
                    loss = self.criterion(pred, batch_y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(dataset)
            self.train_loss_history.append(float(epoch_loss))
            print(f"LSTM epoch {epoch + 1}/{self.num_epochs} | loss={epoch_loss:.5f}")
        return self

    @torch.no_grad()
    def predict(self, x):
        dataset = TensorDataset(x.float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        for (batch_x,) in loader:
            preds.append(self.model(batch_x.to(self.device)).cpu())
        return torch.cat(preds, dim=0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_stage2_npz(args):
    pair_tag = f"{args.source_session}_to_{args.target_session}"
    if args.stage2_example == "chewie_to_mihili":
        stage2_dir = REPO_ROOT / "outputs" / "Chewie_CO_2016" / "to_Mihili_CO_2014"
    else:
        stage2_dir = REPO_ROOT / "outputs" / "Chewie_CO_2016"

    return (
        stage2_dir
        / "Stage2_MMD"
        / "results_use_pnba_read_adapter_without_reset"
        / pair_tag
        / "eval"
        / "target_test_behavior_prediction_conditional_mmd.npz"
    )


def load_decoder_splits(npz_path, args):
    data = np.load(npz_path)
    latents = data["z_latent"].astype(np.float32)
    behavior = data["y_true"].astype(np.float32)
    indices = np.arange(latents.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )
    return latents[train_idx], behavior[train_idx], latents[test_idx], behavior[test_idx]


def flatten_time_major(x):
    x = torch.from_numpy(x)
    return x.permute(0, 2, 1).reshape(-1, x.shape[1])


def get_spikes_with_history(neural_data, bins_before, bins_after=0, bins_current=1):
    num_examples = neural_data.size(0)
    num_neurons = neural_data.size(1)
    surrounding_bins = bins_before + bins_after + bins_current
    x = torch.zeros(num_examples, surrounding_bins, num_neurons, dtype=torch.float32)
    for i in range(num_examples - bins_before - bins_after):
        end_idx = i + surrounding_bins
        x[i + bins_before, :, :] = neural_data[i:end_idx, :]
    return x


def build_history_decoder_matrices(latents, behavior, bins_before, bins_after, bins_current):
    flat_latents = flatten_time_major(latents).float()
    flat_behavior = flatten_time_major(behavior).float()
    x = get_spikes_with_history(
        flat_latents,
        bins_before=bins_before,
        bins_after=bins_after,
        bins_current=bins_current,
    )
    end = x.size(0) - bins_after if bins_after > 0 else x.size(0)
    valid = slice(bins_before, end)
    return x[valid].contiguous(), flat_behavior[valid].contiguous()


def normalize_decoder_inputs(x_train, x_test):
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    return (x_train - mean) / (std + 1e-8), (x_test - mean) / (std + 1e-8), mean, std


def center_decoder_targets(y_train, y_test):
    mean = y_train.mean(dim=0, keepdim=True)
    return y_train - mean, y_test - mean, mean


def raw_and_mean_r2(y_true, y_pred):
    raw = r2_score(y_true, y_pred, multioutput="raw_values")
    return float(raw[0]), float(raw[1]), float(np.mean(raw))


def save_outputs(output_dir, result, y_true, y_pred, train_loss_history):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "lstm_decoder_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        train_loss_history=np.asarray(train_loss_history, dtype=np.float32),
        **result,
    )
    with open(output_dir / "lstm_decoder_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--decoder-config", type=Path, default=DEFAULT_DECODER_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    lstm_config = load_lstm_config(pre_args.decoder_config)

    parser = argparse.ArgumentParser(
        description="LSTM decoder for Stage2 TCLA latents.",
        parents=[pre_parser],
    )
    parser.add_argument("--stage2-npz", type=Path, default=None)
    parser.add_argument("--stage2-example", choices=["chewie", "chewie_to_mihili"], default="chewie")
    parser.add_argument("--source-session", default="session_0")
    parser.add_argument("--target-session", default="session_0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=lstm_config.random_seed)
    parser.add_argument("--bins-before", type=int, default=lstm_config.bins_before)
    parser.add_argument("--bins-after", type=int, default=lstm_config.bins_after)
    parser.add_argument("--bins-current", type=int, default=lstm_config.bins_current)
    parser.add_argument("--lstm-units", type=int, default=lstm_config.units)
    parser.add_argument("--dropout", type=float, default=lstm_config.dropout)
    parser.add_argument("--lstm-epochs", type=int, default=lstm_config.epochs)
    parser.add_argument("--lstm-batch-size", type=int, default=lstm_config.batch_size)
    parser.add_argument("--learning-rate", type=float, default=lstm_config.learning_rate)
    parser.add_argument("--lstm-num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.random_seed)
    device = torch.device(args.device)
    npz_path = args.stage2_npz or default_stage2_npz(args)
    train_latents, train_behavior, test_latents, test_behavior = load_decoder_splits(npz_path, args)

    x_train, y_train = build_history_decoder_matrices(
        train_latents,
        train_behavior,
        args.bins_before,
        args.bins_after,
        args.bins_current,
    )
    x_test, y_test = build_history_decoder_matrices(
        test_latents,
        test_behavior,
        args.bins_before,
        args.bins_after,
        args.bins_current,
    )

    x_train_norm, x_test_norm, _, _ = normalize_decoder_inputs(x_train, x_test)
    y_train_norm, _, y_mean = center_decoder_targets(y_train, y_train)

    model = LSTMRegression(
        input_size=x_train_norm.shape[-1],
        output_size=y_train_norm.shape[-1],
        units=args.lstm_units,
        dropout=args.dropout,
        num_epochs=args.lstm_epochs,
        batch_size=args.lstm_batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.lstm_num_workers,
        device=device,
        use_amp=not args.no_amp,
    )
    model.fit(x_train_norm, y_train_norm)
    y_pred = model.predict(x_test_norm) + y_mean
    r2_x, r2_y, mean_r2 = raw_and_mean_r2(y_test.numpy(), y_pred.numpy())

    result = {
        "decoder": "lstm",
        "stage2_npz": str(npz_path),
        "bins_before": int(args.bins_before),
        "bins_after": int(args.bins_after),
        "bins_current": int(args.bins_current),
        "lstm_units": int(args.lstm_units),
        "dropout": float(args.dropout),
        "lstm_epochs": int(args.lstm_epochs),
        "learning_rate": float(args.learning_rate),
        "source_train_loss_final": model.train_loss_history[-1] if model.train_loss_history else np.nan,
        "r2_x": r2_x,
        "r2_y": r2_y,
        "mean_r2": mean_r2,
        "num_train_trials": int(train_latents.shape[0]),
        "num_test_trials": int(test_latents.shape[0]),
    }
    save_outputs(args.output_dir, result, y_test.numpy(), y_pred.numpy(), model.train_loss_history)
    print(result)
    print(f"Saved LSTM decoder outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
