import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "behavioral_decoder" / "ridge"
DEFAULT_DECODER_CONFIG = REPO_ROOT / "config" / "behavioral_decoders.yaml"


def load_ridge_config(path):
    return OmegaConf.load(path).ridge


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
    return np.transpose(x, (0, 2, 1)).reshape(-1, x.shape[1])


def generate_lagged_matrix(input_matrix, lag):
    if input_matrix.shape[0] <= lag:
        raise ValueError(f"Not enough time points ({input_matrix.shape[0]}) for lag={lag}")
    lagged = np.zeros(
        (input_matrix.shape[0] - lag, input_matrix.shape[1] * (lag + 1)),
        dtype=input_matrix.dtype,
    )
    for i in range(lag + 1):
        start = i * input_matrix.shape[1]
        end = (i + 1) * input_matrix.shape[1]
        lagged[:, start:end] = input_matrix[lag - i : (-i if i != 0 else None)]
    return lagged


def build_lagged_decoder_matrices(latents, behavior, lag):
    x = flatten_time_major(latents)
    y = flatten_time_major(behavior)
    return generate_lagged_matrix(x, lag), y[lag:, :]


def predict_lagged(model, latents, lag):
    x = flatten_time_major(latents)
    return model.predict(generate_lagged_matrix(x, lag))


def raw_and_mean_r2(y_true, y_pred):
    raw = r2_score(y_true, y_pred, multioutput="raw_values")
    return float(raw[0]), float(raw[1]), float(np.mean(raw))


def fit_ridge(x_train, y_train, args):
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.num_alphas)
    search = GridSearchCV(
        Ridge(),
        {"alpha": alphas},
        cv=args.cv,
        scoring="r2",
        n_jobs=args.n_jobs,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, float(search.best_params_["alpha"]), float(search.best_score_)


def save_outputs(output_dir, result, y_true, y_pred):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "ridge_decoder_predictions.npz", y_true=y_true, y_pred=y_pred, **result)
    with open(output_dir / "ridge_decoder_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--decoder-config", type=Path, default=DEFAULT_DECODER_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    ridge_config = load_ridge_config(pre_args.decoder_config)

    parser = argparse.ArgumentParser(
        description="Lagged Ridge decoder for Stage2 TCLA latents.",
        parents=[pre_parser],
    )
    parser.add_argument("--stage2-npz", type=Path, default=None)
    parser.add_argument("--stage2-example", choices=["chewie", "chewie_to_mihili"], default="chewie")
    parser.add_argument("--source-session", default="session_0")
    parser.add_argument("--target-session", default="session_1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lag", type=int, default=ridge_config.lag)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha-min-exp", type=float, default=ridge_config.alpha_min_exp)
    parser.add_argument("--alpha-max-exp", type=float, default=ridge_config.alpha_max_exp)
    parser.add_argument("--num-alphas", type=int, default=ridge_config.num_alphas)
    parser.add_argument("--cv", type=int, default=ridge_config.cv)
    parser.add_argument("--n-jobs", type=int, default=ridge_config.grid_jobs)
    args = parser.parse_args()

    npz_path = args.stage2_npz or default_stage2_npz(args)
    train_latents, train_behavior, test_latents, test_behavior = load_decoder_splits(npz_path, args)

    x_train, y_train = build_lagged_decoder_matrices(train_latents, train_behavior, args.lag)
    model, best_alpha, source_cv_r2 = fit_ridge(x_train, y_train, args)

    y_true = flatten_time_major(test_behavior)[args.lag :, :]
    y_pred = predict_lagged(model, test_latents, args.lag)
    r2_x, r2_y, mean_r2 = raw_and_mean_r2(y_true, y_pred)

    result = {
        "decoder": "lagged_ridge",
        "stage2_npz": str(npz_path),
        "lag": int(args.lag),
        "best_alpha": best_alpha,
        "source_cv_r2": source_cv_r2,
        "r2_x": r2_x,
        "r2_y": r2_y,
        "mean_r2": mean_r2,
        "num_train_trials": int(train_latents.shape[0]),
        "num_test_trials": int(test_latents.shape[0]),
    }
    save_outputs(args.output_dir, result, y_true, y_pred)
    print(result)
    print(f"Saved ridge decoder outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
