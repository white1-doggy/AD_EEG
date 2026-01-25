import argparse
import glob
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from eeg_dataset import EEGWindowDataset
from eeg_mae import EEGMaskedAutoencoder


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EEG MAE reconstruction on validation set.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .npz files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MAE checkpoint.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-size", type=int, default=2)
    return parser.parse_args()


def list_npz_files(data_dir: str) -> List[str]:
    pattern = os.path.join(data_dir, "*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    return paths


def split_dataset(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


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
    dataset = EEGWindowDataset(
        npz_paths,
        max_cache_size=args.cache_size,
        show_progress=True,
        progress_desc="Loading EEG subjects",
    )
    _, val_set = split_dataset(dataset, args.val_split, args.seed)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(args.checkpoint, device)

    band_sse = torch.zeros(len(BANDS), device=device)
    band_count = 0.0

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Evaluating", leave=False)
        for x, valid_mask, _ in progress:
            x = x.to(device)
            valid_mask = valid_mask.to(device)
            recon, _, _ = model(x, valid_mask=valid_mask)

            diff = (recon - x) ** 2
            per_band_mse = diff.mean(dim=(2, 3))
            valid_mask = valid_mask.view(-1, 1)
            per_band_mse = per_band_mse * valid_mask

            batch_valid = valid_mask.sum().item()
            if batch_valid > 0:
                band_sse += per_band_mse.sum(dim=0)
                band_count += batch_valid

    if band_count == 0:
        print("No valid windows found in validation set.")
        return

    band_mse = (band_sse / band_count).cpu().tolist()
    print("Per-band MSE on validation set:")
    for name, mse in zip(BANDS, band_mse):
        print(f"  {name}: {mse:.6f}")


if __name__ == "__main__":
    main()
