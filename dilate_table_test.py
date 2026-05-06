"""Reproduce and extend the DILATE results-table experiments.

Run from the DILATE repo root, for example:

    python dilate_table_test.py \
        --dataset synthetic \
        --model seq2seq \
        --losses mse soft_dtw dilate \
        --epochs 500 \
        --n-runs 10 \
        --output results/synthetic_seq2seq_dilate_repro.csv

The reported metrics are held-out evaluation metrics:
    MSE, hard DTW, TDI, hard MSM

The raw training losses are written for debugging only and should not be
compared directly across objectives.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tslearn.metrics import dtw_path

try:
    from aeon.distances import msm_distance
except ImportError:  # pragma: no cover - compatibility with older aeon layouts
    try:
        from aeon.distances.elastic import msm_distance
    except ImportError:  # pragma: no cover
        msm_distance = None

try:
    from aeon.datasets import load_classification
except ImportError:  # pragma: no cover
    load_classification = None

from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from loss.dilate_loss import dilate_loss
from loss.soft_msm_dilate_loss import soft_msm_loss, soft_msm_dilate_loss
from load_ecg import load_ecg5000_dilate_format
warnings.simplefilter("ignore")


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Datasets
# -------------------------------------------------------------------------


class ForecastDataset(Dataset):
    """Simple forecasting dataset returning (input, target, metadata).

    X and Y are stored as float32 arrays with shapes (N, T_in, 1) and
    (N, T_out, 1).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, metadata: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        if X.ndim == 2:
            X = X[:, :, None]
        if Y.ndim == 2:
            Y = Y[:, :, None]
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError("X and Y must have shape (N, T) or (N, T, C).")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of cases.")

        self.X = X
        self.Y = Y
        self.metadata = (
            np.zeros(X.shape[0], dtype=np.float32)
            if metadata is None
            else np.asarray(metadata)
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int):
        return self.X[index], self.Y[index], self.metadata[index]


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    val: Dataset | None
    n_input: int
    n_output: int


def _subset(dataset: Dataset, indices: Iterable[int]) -> Subset:
    return Subset(dataset, list(indices))


def split_train_val(
    train_dataset: Dataset,
    val_frac: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    """Split a training dataset into train/validation subsets."""
    if val_frac <= 0:
        return train_dataset, None
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in [0, 1).")

    n = len(train_dataset)
    n_val = int(round(n * val_frac))
    if n_val <= 0:
        return train_dataset, None
    if n_val >= n:
        raise ValueError("val_frac leaves no training cases.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return _subset(train_dataset, train_idx), _subset(train_dataset, val_idx)


def make_synthetic_bundle(args) -> DatasetBundle:
    """Create the DILATE synthetic forecasting dataset.

    This uses the DILATE repository generator already available in this repo.
    The generator returns train and test splits. A validation split is optionally
    carved out of the training split by --val-frac.
    """
    data = create_synthetic_dataset(
        args.n_series,
        args.n_input,
        args.n_output,
        args.sigma,
    )
    x_train_input, x_train_target, x_test_input, x_test_target, train_bkp, test_bkp = data

    train_dataset = SyntheticDataset(x_train_input, x_train_target, train_bkp)
    test_dataset = SyntheticDataset(x_test_input, x_test_target, test_bkp)
    train_dataset, val_dataset = split_train_val(train_dataset, args.val_frac, args.seed)

    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        n_input=args.n_input,
        n_output=args.n_output,
    )


def _read_ucr_tsv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a UCR-style TSV file, returning labels and series."""
    data = np.loadtxt(path, delimiter="\t")
    y = data[:, 0]
    X = data[:, 1:]
    return y, X


def _load_ecg5000_from_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ECG5000 from a UCR directory or a specific train/test TSV path."""
    path = Path(path)
    if path.is_dir():
        train_path = path / "ECG5000_TRAIN.tsv"
        test_path = path / "ECG5000_TEST.tsv"
    else:
        raise ValueError(
            "--ecg-path should be a directory containing ECG5000_TRAIN.tsv "
            "and ECG5000_TEST.tsv."
        )

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find {train_path} and {test_path}."
        )

    _, X_train = _read_ucr_tsv(train_path)
    _, X_test = _read_ucr_tsv(test_path)
    return X_train.astype(np.float32), X_test.astype(np.float32)


def _load_ecg5000_from_aeon() -> tuple[np.ndarray, np.ndarray]:
    """Load ECG5000 through aeon."""
    if load_classification is None:
        raise ImportError(
            "aeon is required to load ECG5000 automatically. Install aeon or "
            "provide --ecg-path pointing at UCR TSV files."
        )

    try:
        X_train, _ = load_classification("ECG5000", split="train")
        X_test, _ = load_classification("ECG5000", split="test")
    except TypeError:
        X_train, _ = load_classification(name="ECG5000", split="train")
        X_test, _ = load_classification(name="ECG5000", split="test")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # aeon usually returns (N, C, T). Convert univariate to (N, T).
    if X_train.ndim == 3:
        X_train = X_train[:, 0, :]
    if X_test.ndim == 3:
        X_test = X_test[:, 0, :]

    return X_train, X_test

def make_ecg5000_bundle(args) -> DatasetBundle:
    """Create the ECG5000 forecasting dataset from .ts files.

    Following the DILATE setup, the first 84 points are used as input and the
    remaining 56 points as target by default.
    """
    if args.ecg_path is None:
        ecg_dir = Path("data/ECG5000")
    else:
        ecg_dir = Path(args.ecg_path)

    if ecg_dir.is_dir():
        train_path = ecg_dir / "ECG5000_TRAIN.ts"
        test_path = ecg_dir / "ECG5000_TEST.ts"
    else:
        train_path = ecg_dir
        test_path = train_path.with_name(
            train_path.name.replace("TRAIN", "TEST").replace("train", "test")
        )

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find ECG5000 .ts files:\n"
            f"  train: {train_path}\n"
            f"  test:  {test_path}"
        )

    print(f"Loading ECG5000 train file: {train_path}")
    print(f"Loading ECG5000 test file:  {test_path}")

    (
        x_train_input,
        x_train_target,
        x_test_input,
        x_test_target,
        y_train,
        y_test,
    ) = load_ecg5000_dilate_format(
        train_path,
        test_path,
        n_input=args.n_input,
        n_output=args.n_output,
        channel=args.ecg_channel,
    )

    train_dataset = ForecastDataset(x_train_input, x_train_target, y_train)
    test_dataset = ForecastDataset(x_test_input, x_test_target, y_test)
    train_dataset, val_dataset = split_train_val(train_dataset, args.val_frac, args.seed)

    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        n_input=args.n_input,
        n_output=args.n_output,
    )

def _load_univariate_series(path: Path, column: int | None = None) -> np.ndarray:
    """Load a univariate series from .npy, .csv or .txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        # Try comma first, then whitespace.
        try:
            arr = np.loadtxt(path, delimiter=",")
        except ValueError:
            arr = np.loadtxt(path)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if column is None:
            # Use the first numeric column by default.
            column = 0
        return arr[:, column]

    raise ValueError("Traffic data must be a 1D array or a 2D table.")


def _make_sliding_windows(
    series: np.ndarray,
    n_input: int,
    n_output: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one long series into forecasting windows."""
    total = n_input + n_output
    if len(series) < total:
        raise ValueError("Series is shorter than n_input+n_output.")

    X = []
    Y = []
    for start in range(0, len(series) - total + 1, stride):
        window = series[start : start + total]
        X.append(window[:n_input])
        Y.append(window[n_input:])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def make_traffic_bundle(args) -> DatasetBundle:
    """Create Traffic forecasting dataset from one univariate series.

    The DILATE traffic setup predicts 24 future points from 168 previous points.
    We use chronological 60/20/20 splitting by default.
    """
    if args.traffic_path is None:
        raise ValueError("--traffic-path is required for --dataset traffic.")

    series = _load_univariate_series(Path(args.traffic_path), args.traffic_column)
    X, Y = _make_sliding_windows(
        series,
        args.n_input,
        args.n_output,
        stride=args.traffic_stride,
    )

    n = X.shape[0]
    n_train = int(np.floor(args.traffic_train_frac * n))
    n_val = int(np.floor(args.traffic_val_frac * n))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Traffic split leaves no training or test cases.")

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train : n_train + n_val]
    Y_val = Y[n_train : n_train + n_val]

    X_test = X[n_train + n_val :]
    Y_test = Y[n_train + n_val :]

    train_dataset = ForecastDataset(X_train, Y_train)
    val_dataset = ForecastDataset(X_val, Y_val) if n_val > 0 else None
    test_dataset = ForecastDataset(X_test, Y_test)

    return DatasetBundle(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        n_input=args.n_input,
        n_output=args.n_output,
    )


def make_dataset_bundle(args) -> DatasetBundle:
    """Create train/validation/test datasets for the selected experiment."""
    if args.dataset == "synthetic":
        return make_synthetic_bundle(args)
    if args.dataset == "ecg5000":
        return make_ecg5000_bundle(args)
    if args.dataset == "traffic":
        return make_traffic_bundle(args)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def make_loaders(args, bundle: DatasetBundle) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Create train, validation and test loaders."""
    trainloader = DataLoader(
        bundle.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    valloader = None
    if bundle.val is not None and len(bundle.val) > 0:
        valloader = DataLoader(
            bundle.val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    testloader = DataLoader(
        bundle.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return trainloader, valloader, testloader


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------


class MLPForecast(nn.Module):
    """One-hidden-layer MLP forecasting model."""

    def __init__(self, n_input: int, n_output: int, hidden_size: int = 128):
        super().__init__()
        self.n_output = n_output
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, 1)
        x_flat = x[:, :, 0]
        y = self.net(x_flat)
        return y.unsqueeze(-1)


class Seq2SeqGRUForecast(nn.Module):
    """GRU encoder-decoder forecasting model.

    This avoids the fixed-batch-size assumption in the original DILATE model
    wrappers, which makes ECG5000 and traffic evaluation easier.
    """

    def __init__(self, n_output: int, hidden_size: int = 128, fc_units: int = 16):
        super().__init__()
        self.n_output = n_output
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x)

        decoder_input = x[:, -1:, :]
        outputs = []
        for _ in range(self.n_output):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            prediction = self.out(torch.relu(self.fc(decoder_output)))
            outputs.append(prediction)
            decoder_input = prediction

        return torch.cat(outputs, dim=1)


def make_model(args, bundle: DatasetBundle, device: torch.device) -> nn.Module:
    """Create the requested forecasting model."""
    if args.model == "mlp":
        return MLPForecast(
            n_input=bundle.n_input,
            n_output=bundle.n_output,
            hidden_size=args.hidden_size,
        ).to(device)

    if args.model == "seq2seq":
        return Seq2SeqGRUForecast(
            n_output=bundle.n_output,
            hidden_size=args.hidden_size,
            fc_units=args.fc_units,
        ).to(device)

    raise ValueError(f"Unknown model: {args.model}")


# -------------------------------------------------------------------------
# Losses and evaluation
# -------------------------------------------------------------------------


def compute_loss(
    loss_type: str,
    target: torch.Tensor,
    output: torch.Tensor,
    alpha: float,
    gamma: float,
    c: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one of the differentiable training objectives.

    Returns exactly:
        loss, loss_shape, loss_temporal
    """
    mse = nn.MSELoss()

    if loss_type == "mse":
        loss = mse(target, output)
        zero = torch.tensor(0.0, device=device, dtype=output.dtype)
        return loss, loss, zero

    if loss_type == "dilate":
        result = dilate_loss(target, output, alpha, gamma, device)
        if not isinstance(result, tuple):
            raise ValueError("dilate_loss should return a tuple.")
        return result[0], result[1], result[2]

    if loss_type == "soft_dtw":
        _, loss_shape, _ = dilate_loss(target, output, alpha, gamma, device)
        zero = torch.tensor(0.0, device=device, dtype=output.dtype)
        return loss_shape, loss_shape, zero

    if loss_type == "soft_msm":
        return soft_msm_loss(target, output, gamma, device, c=c)

    if loss_type == "soft_msm_dilate":
        return soft_msm_dilate_loss(target, output, alpha, gamma, device, c=c)

    raise ValueError(f"Unknown loss_type: {loss_type}")


@torch.no_grad()
def evaluate_model(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    msm_c: float,
) -> dict[str, float]:
    """Evaluate MSE, hard DTW, TDI and hard MSM on a held-out loader."""
    if msm_distance is None:
        raise ImportError(
            "aeon is required for the hard MSM evaluation row. Install aeon, "
            "or remove the MSM metric from evaluate_model."
        )

    criterion = nn.MSELoss(reduction="mean")

    total_cases = 0
    mse_total = 0.0
    dtw_total = 0.0
    tdi_total = 0.0
    msm_total = 0.0

    net.eval()
    for data in loader:
        inputs, target, _ = data
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        target = torch.as_tensor(target, dtype=torch.float32, device=device)

        output = net(inputs)
        batch_size, n_output = target.shape[0:2]

        mse_total += criterion(target, output).item() * batch_size

        for k in range(batch_size):
            target_k = target[k, :, 0].detach().cpu().numpy()
            output_k = output[k, :, 0].detach().cpu().numpy()

            path, sim = dtw_path(target_k, output_k)
            dtw_total += sim

            dist = 0.0
            for i, j in path:
                dist += (i - j) * (i - j)
            tdi_total += dist / (n_output * n_output)

            msm_total += msm_distance(target_k, output_k, c=msm_c)

        total_cases += batch_size

    net.train()
    return {
        "mse": float(mse_total / total_cases),
        "dtw": float(dtw_total / total_cases),
        "tdi": float(tdi_total / total_cases),
        "msm": float(msm_total / total_cases),
    }


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------


def train_one_model(
    args,
    loss_type: str,
    seed: int,
    bundle: DatasetBundle,
    trainloader: DataLoader,
    valloader: DataLoader | None,
    testloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int | str]:
    """Train one model and return final held-out metrics."""
    set_seed(seed)
    net = make_model(args, bundle, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    final_train_loss = None
    final_shape_loss = None
    final_time_loss = None
    best_epoch = None
    best_val_metric = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        net.train()
        for data in trainloader:
            inputs, target, _ = data
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            target = torch.as_tensor(target, dtype=torch.float32, device=device)

            output = net(inputs)
            loss, loss_shape, loss_temporal = compute_loss(
                loss_type,
                target,
                output,
                args.alpha,
                args.gamma,
                args.c,
                device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_train_loss = float(loss.detach().cpu())
            final_shape_loss = float(loss_shape.detach().cpu())
            final_time_loss = float(loss_temporal.detach().cpu())

        val_metrics = None
        if valloader is not None and args.early_stopping:
            val_metrics = evaluate_model(net, valloader, device, args.msm_c_eval)
            current = val_metrics[args.early_stop_metric]

            if best_val_metric is None or current < best_val_metric - args.min_delta:
                best_val_metric = current
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                if args.verbose:
                    print(
                        f"seed={seed} loss={loss_type} early_stop_epoch={epoch:04d} "
                        f"best_epoch={best_epoch:04d} "
                        f"best_val_{args.early_stop_metric}={best_val_metric:.6f}"
                    )
                break

        if args.verbose and (epoch % args.print_every == 0 or epoch == args.epochs - 1):
            metrics = evaluate_model(net, testloader, device, args.msm_c_eval)
            val_text = ""
            if val_metrics is not None:
                val_text = f" val_{args.early_stop_metric}={val_metrics[args.early_stop_metric]:.6f}"
            print(
                f"dataset={args.dataset} model={args.model} seed={seed} "
                f"loss={loss_type} epoch={epoch:04d} "
                f"train_loss={final_train_loss:.6f} "
                f"shape={final_shape_loss:.6f} "
                f"temporal={final_time_loss:.6f}" 
                f"{val_text} "
                f"eval_mse={metrics['mse']:.6f} "
                f"eval_dtw={metrics['dtw']:.6f} "
                f"eval_tdi={metrics['tdi']:.6f} "
                f"eval_msm={metrics['msm']:.6f}"
            )

    if best_state is not None:
        net.load_state_dict(best_state)

    metrics = evaluate_model(net, testloader, device, args.msm_c_eval)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": args.model,
            "seed": seed,
            "loss_type": loss_type,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "c": args.c,
            "msm_c_eval": args.msm_c_eval,
            "epochs": args.epochs,
            "best_epoch": best_epoch if best_epoch is not None else -1,
            "best_val_metric": best_val_metric if best_val_metric is not None else np.nan,
            "train_loss": final_train_loss,
            "train_shape_loss": final_shape_loss,
            "train_temporal_loss": final_time_loss,
        }
    )
    return metrics


# -------------------------------------------------------------------------
# Output and summaries
# -------------------------------------------------------------------------


def write_results(results: list[dict], output_path: str | Path) -> None:
    """Write results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "seed",
        "loss_type",
        "alpha",
        "gamma",
        "c",
        "msm_c_eval",
        "epochs",
        "best_epoch",
        "best_val_metric",
        "mse",
        "dtw",
        "tdi",
        "msm",
        "train_loss",
        "train_shape_loss",
        "train_temporal_loss",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def summarise(results: list[dict]) -> None:
    """Print raw and DILATE-scaled summaries by loss type."""
    print("\nSummary, raw metrics")
    for loss_type in sorted({r["loss_type"] for r in results}):
        subset = [r for r in results if r["loss_type"] == loss_type]
        print(f"\n{loss_type}")
        for metric in ["mse", "dtw", "tdi", "msm"]:
            values = np.array([r[metric] for r in subset], dtype=float)
            if len(values) > 1:
                print(f"  {metric}: {values.mean():.6f} ± {values.std(ddof=1):.6f}")
            else:
                print(f"  {metric}: {values.mean():.6f}")

    print("\nSummary, DILATE table scaling")
    print("  MSE x100, DTW x100, TDI x10, MSM raw")
    for loss_type in sorted({r["loss_type"] for r in results}):
        subset = [r for r in results if r["loss_type"] == loss_type]
        print(f"\n{loss_type}")
        scaled = {
            "mse_x100": 100.0 * np.array([r["mse"] for r in subset], dtype=float),
            "dtw_x100": 100.0 * np.array([r["dtw"] for r in subset], dtype=float),
            "tdi_x10": 10.0 * np.array([r["tdi"] for r in subset], dtype=float),
            "msm": np.array([r["msm"] for r in subset], dtype=float),
        }
        for metric, values in scaled.items():
            if len(values) > 1:
                print(f"  {metric}: {values.mean():.3f} ± {values.std(ddof=1):.3f}")
            else:
                print(f"  {metric}: {values.mean():.3f}")


# -------------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------------


def _apply_dataset_defaults(args) -> None:
    """Set dataset-specific defaults when not explicitly provided."""
    if args.dataset == "synthetic":
        if args.n_input is None:
            args.n_input = 20
        if args.n_output is None:
            args.n_output = 20
        if args.alpha is None:
            args.alpha = 0.5

    elif args.dataset == "ecg5000":
        if args.n_input is None:
            args.n_input = 84
        if args.n_output is None:
            args.n_output = 56
        if args.alpha is None:
            args.alpha = 0.5

    elif args.dataset == "traffic":
        if args.n_input is None:
            args.n_input = 168
        if args.n_output is None:
            args.n_output = 24
        if args.alpha is None:
            args.alpha = 0.8

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["synthetic", "ecg5000", "traffic"],
        default="synthetic",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "seq2seq"],
        default="seq2seq",
    )

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-series", type=int, default=500)
    parser.add_argument("--n-input", type=int, default=None)
    parser.add_argument("--n-output", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.01)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--fc-units", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--msm-c",
        type=float,
        default=None,
        help="c used for hard MSM evaluation. Defaults to --c.",
    )

    parser.add_argument("--val-frac", type=float, default=0.0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=["mse", "dtw", "tdi", "msm"],
        default="mse",
    )

    parser.add_argument("--ecg-path", default=None)
    parser.add_argument("--traffic-path", default=None)
    parser.add_argument("--traffic-column", type=int, default=None)
    parser.add_argument("--traffic-stride", type=int, default=1)
    parser.add_argument("--traffic-train-frac", type=float, default=0.6)
    parser.add_argument("--traffic-val-frac", type=float, default=0.2)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument(
        "--losses",
        nargs="+",
        default=["mse", "soft_dtw", "dilate"],
        choices=["mse", "soft_dtw", "dilate", "soft_msm", "soft_msm_dilate"],
    )

    parser.add_argument("--output", default="results/dilate_table_repro.csv")

    args = parser.parse_args()
    _apply_dataset_defaults(args)
    args.msm_c_eval = args.c if args.msm_c is None else args.msm_c
    return args


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def quick_test_ecg_loader():
    """Minimal check that ECG5000 loads and msm_distance works."""
    train_path = "data/ECG5000/ECG5000_TRAIN.ts"
    test_path = "data/ECG5000/ECG5000_TEST.ts"

    (
        x_train_input,
        x_train_target,
        x_test_input,
        x_test_target,
        y_train,
        y_test,
    ) = load_ecg5000_dilate_format(
        train_path,
        test_path,
        n_input=84,
        n_output=56,
        channel=0,
    )

    print("x_train_input:", x_train_input.shape)
    print("x_train_target:", x_train_target.shape)
    print("x_test_input:", x_test_input.shape)
    print("x_test_target:", x_test_target.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    if msm_distance is None:
        print("msm_distance is not available")
        return

    d = msm_distance(
        x_train_input[0, :, 0],
        x_train_input[1, :, 0],
        c=1.0,
    )

    print("MSM distance between first two train inputs:", d)
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if torch.cuda.is_available():
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    print(
        f"dataset={args.dataset} model={args.model} "
        f"n_input={args.n_input} n_output={args.n_output} "
        f"alpha={args.alpha} gamma={args.gamma} "
        f"c={args.c} msm_c_eval={args.msm_c_eval}"
    )
    if args.gamma > 0:
        print(f"c/gamma={args.c / args.gamma:.6f}")

    set_seed(args.seed)
    bundle = make_dataset_bundle(args)
    trainloader, valloader, testloader = make_loaders(args, bundle)

    print(
        f"cases: train={len(bundle.train)} "
        f"val={0 if bundle.val is None else len(bundle.val)} "
        f"test={len(bundle.test)}"
    )

    results = []
    for run in range(args.n_runs):
        seed = args.seed + run
        for loss_type in args.losses:
            result = train_one_model(
                args,
                loss_type,
                seed,
                bundle,
                trainloader,
                valloader,
                testloader,
                device,
            )
            results.append(result)
            write_results(results, args.output)

    summarise(results)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
