"""Reproduce DILATE synthetic forecasting experiments.

Run from the DILATE repo root:

    python run_synthetic_repro.py

This script compares Seq2Seq models trained with MSE and DILATE on the
synthetic step-change forecasting problem.
"""

import argparse
import csv
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tslearn.metrics import dtw_path

from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from loss.dilate_loss import dilate_loss
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.soft_msm_dilate_loss import soft_msm_loss, soft_msm_dilate_loss

warnings.simplefilter("ignore")


def set_seed(seed):
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    """Create synthetic train and test loaders."""
    data = create_synthetic_dataset(
        args.n_series,
        args.n_input,
        args.n_output,
        args.sigma,
    )
    x_train_input, x_train_target, x_test_input, x_test_target, train_bkp, test_bkp = data

    dataset_train = SyntheticDataset(x_train_input, x_train_target, train_bkp)
    dataset_test = SyntheticDataset(x_test_input, x_test_target, test_bkp)

    trainloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    testloader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return trainloader, testloader


def make_model(args, device):
    """Create the original GRU encoder-decoder model."""
    encoder = EncoderRNN(
        input_size=1,
        hidden_size=args.hidden_size,
        num_grulstm_layers=1,
        batch_size=args.batch_size,
    ).to(device)

    decoder = DecoderRNN(
        input_size=1,
        hidden_size=args.hidden_size,
        num_grulstm_layers=1,
        fc_units=args.fc_units,
        output_size=1,
    ).to(device)

    return Net_GRU(encoder, decoder, args.n_output, device).to(device)


def compute_loss(loss_type, target, output, alpha, gamma, device):
    """Compute one of the training losses.

    Always returns exactly:
        loss, loss_shape, loss_temporal
    """
    mse = torch.nn.MSELoss()

    if loss_type == "mse":
        loss = mse(target, output)
        zero = torch.tensor(0.0, device=device)
        return loss, loss, zero

    if loss_type == "dilate":
        result = dilate_loss(target, output, alpha, gamma, device)

        if not isinstance(result, tuple):
            raise ValueError("dilate_loss should return a tuple.")

        return result[0], result[1], result[2]

    if loss_type == "soft_msm":
        return soft_msm_loss(target, output, gamma, device)

    if loss_type == "soft_msm_dilate":
        return soft_msm_dilate_loss(target, output, alpha, gamma, device)

    if loss_type == "soft_dtw":
        _, loss_shape, _ = dilate_loss(target, output, alpha, gamma, device)
        zero = torch.tensor(0.0, device=device)
        return loss_shape, loss_shape, zero
    raise ValueError(f"Unknown loss_type: {loss_type}")


def evaluate_model(net, loader, device):
    """Evaluate MSE, DTW and TDI on a loader."""
    criterion = torch.nn.MSELoss()

    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, target, _ = data
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            target = torch.as_tensor(target, dtype=torch.float32, device=device)

            output = net(inputs)
            batch_size, n_output = target.shape[0:2]

            loss_mse = criterion(target, output).item()
            loss_dtw = 0.0
            loss_tdi = 0.0

            for k in range(batch_size):
                target_k = target[k, :, 0:1].view(-1).detach().cpu().numpy()
                output_k = output[k, :, 0:1].view(-1).detach().cpu().numpy()

                path, sim = dtw_path(target_k, output_k)
                loss_dtw += sim

                dist = 0.0
                for i, j in path:
                    dist += (i - j) * (i - j)
                loss_tdi += dist / (n_output * n_output)

            losses_mse.append(loss_mse)
            losses_dtw.append(loss_dtw / batch_size)
            losses_tdi.append(loss_tdi / batch_size)

    net.train()
    return {
        "mse": float(np.mean(losses_mse)),
        "dtw": float(np.mean(losses_dtw)),
        "tdi": float(np.mean(losses_tdi)),
    }


def train_one_model(args, loss_type, seed, trainloader, testloader, device):
    """Train one model and return final metrics."""
    set_seed(seed)
    net = make_model(args, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    final_train_loss = None
    final_shape_loss = None
    final_time_loss = None

    for epoch in range(args.epochs):
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
                device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_train_loss = float(loss.detach().cpu())
            final_shape_loss = float(loss_shape.detach().cpu())
            final_time_loss = float(loss_temporal.detach().cpu())

        if args.verbose and (epoch % args.print_every == 0 or epoch == args.epochs - 1):
            metrics = evaluate_model(net, testloader, device)
            print(
                f"seed={seed} loss={loss_type} epoch={epoch:04d} "
                f"train_loss={final_train_loss:.6f} "
                f"shape={final_shape_loss:.6f} temporal={final_time_loss:.6f} "
                f"eval_mse={metrics['mse']:.6f} "
                f"eval_dtw={metrics['dtw']:.6f} "
                f"eval_tdi={metrics['tdi']:.6f}"
            )

    metrics = evaluate_model(net, testloader, device)
    metrics.update(
        {
            "seed": seed,
            "loss_type": loss_type,
            "train_loss": final_train_loss,
            "train_shape_loss": final_shape_loss,
            "train_temporal_loss": final_time_loss,
        }
    )
    return metrics


def write_results(results, output_path):
    """Write results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "loss_type",
        "mse",
        "dtw",
        "tdi",
        "train_loss",
        "train_shape_loss",
        "train_temporal_loss",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def summarise(results):
    """Print mean and standard deviation by loss type."""
    print("\nSummary")
    for loss_type in sorted({r["loss_type"] for r in results}):
        subset = [r for r in results if r["loss_type"] == loss_type]
        print(f"\n{loss_type}")
        for metric in ["mse", "dtw", "tdi"]:
            values = np.array([r[metric] for r in subset], dtype=float)
            print(f"  {metric}: {values.mean():.6f} ± {values.std(ddof=1):.6f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-series", type=int, default=500)
    parser.add_argument("--n-input", type=int, default=20)
    parser.add_argument("--n-output", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.01)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--fc-units", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument(
        "--losses",
        nargs="+",
        default=["mse", "dilate"],
    )

    parser.add_argument("--output", default="results/synthetic_repro.csv")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainloader, testloader = make_loaders(args)

    results = []
    for run in range(args.n_runs):
        seed = args.seed + run

        for loss_type in args.losses:
            result = train_one_model(
                args,
                loss_type,
                seed,
                trainloader,
                testloader,
                device,
            )
            results.append(result)
            write_results(results, args.output)

    summarise(results)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()