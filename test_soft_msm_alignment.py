import torch
import numpy as np

from loss.soft_msm_torch.soft_msm_torch import soft_msm_alignment_matrix


def main():
    torch.manual_seed(0)

    # Small synthetic batch
    B = 2
    C = 1
    T = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Case 1: identical sequences → diagonal alignment
    x = torch.randn(B, C, T, device=device)
    y = x.clone()

    print("\n=== CASE 1: identical series ===")
    test_alignment(x, y)

    # Case 2: shifted sequences → off-diagonal mass
    y_shift = torch.roll(x, shifts=2, dims=-1)

    print("\n=== CASE 2: shifted series ===")
    test_alignment(x, y_shift)

    # Case 3: completely different
    y_rand = torch.randn(B, C, T, device=device)

    print("\n=== CASE 3: random series ===")
    test_alignment(x, y_rand)


def test_alignment(x, y):
    E, s = soft_msm_alignment_matrix(x, y, c=1.0, gamma=1.0)

    print("E shape:", E.shape)
    print("cost shape:", s.shape)

    total_mass = E.sum().item()
    diag_mass = E.diagonal(dim1=1, dim2=2).sum().item()
    offdiag_mass = total_mass - diag_mass

    print("total mass:", total_mass)
    print("diag mass:", diag_mass)
    print("offdiag mass:", offdiag_mass)

    # Temporal distortion check
    T = x.shape[-1]
    idx = torch.arange(T, dtype=E.dtype, device=E.device)
    omega = (idx[:, None] - idx[None, :]) ** 2 / (T * T)

    temporal = torch.sum(E * omega[None, :, :], dim=(1, 2)).mean().item()

    print("temporal distortion:", temporal)

    # Sanity assertions
    if total_mass == 0:
        print("ERROR: alignment matrix is zero")

    if temporal == 0:
        print("WARNING: temporal loss is zero (likely bug)")

    print("-" * 40)


if __name__ == "__main__":
    main()
