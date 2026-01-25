import argparse
import glob
import os
import random
from typing import List

import matplotlib.pyplot as plt
import torch

from eeg_dataset import EEGWindowDataset
from eeg_mae import EEGMaskedAutoencoder


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EEG MAE reconstruction.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .npz files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAE checkpoint.")
    parser.add_argument("--global-index", type=int, default=-1, help="Global window index.")
    parser.add_argument("--output", type=str, default="reconstruction.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def list_npz_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return paths


def load_model(checkpoint_path: str, device: torch.device) -> EEGMaskedAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EEGMaskedAutoencoder()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    npz_paths = list_npz_files(args.data_dir)
    dataset = EEGWindowDataset(npz_paths, show_progress=True, progress_desc="Indexing EEG")

    if args.global_index < 0:
        index = random.randint(0, len(dataset) - 1)
    else:
        index = args.global_index

    x_window, valid_mask, subject_id = dataset[index]
    x_window = x_window.unsqueeze(0).to(device)

    model = load_model(args.checkpoint, device)
    with torch.no_grad():
        reconstruction, _, _ = model(x_window, valid_mask=valid_mask.unsqueeze(0).to(device))

    original = x_window.squeeze(0).cpu().numpy()
    recon = reconstruction.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(len(BANDS), 2, figsize=(10, 12), constrained_layout=True)
    fig.suptitle(f"Subject: {subject_id} | Index: {index} | Valid: {float(valid_mask):.0f}")

    for band_idx, band_name in enumerate(BANDS):
        ax_orig = axes[band_idx, 0]
        ax_recon = axes[band_idx, 1]
        ax_orig.imshow(original[band_idx], aspect="auto", origin="lower")
        ax_recon.imshow(recon[band_idx], aspect="auto", origin="lower")
        ax_orig.set_title(f"{band_name} - original")
        ax_recon.set_title(f"{band_name} - recon")
        ax_orig.set_xlabel("time")
        ax_recon.set_xlabel("time")
        ax_orig.set_ylabel("channel")
        ax_recon.set_ylabel("channel")

    fig.savefig(args.output, dpi=150)
    print(f"Saved reconstruction figure to {args.output}")


if __name__ == "__main__":
    main()
