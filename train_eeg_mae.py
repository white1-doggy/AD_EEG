import argparse
import glob
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from eeg_dataset import EEGWindowDataset
from eeg_mae import EEGMaskedAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG MAE for anomaly detection.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with .npz files.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--band-index", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume.")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def run_epoch(
    model: EEGMaskedAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    epoch: int,
    num_epochs: int,
) -> float:
    if train:
        model.train()
        desc = f"Epoch {epoch}/{num_epochs} [train]"
    else:
        model.eval()
        desc = f"Epoch {epoch}/{num_epochs} [val]"

    total_loss = 0.0
    total_count = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for x, valid_mask, _ in progress:
            x = x.to(device)
            valid_mask = valid_mask.to(device)

            if train:
                optimizer.zero_grad()
            _, loss, per_sample_loss = model(x, valid_mask=valid_mask)

            if train:
                loss.backward()
                optimizer.step()

            batch_count = valid_mask.sum().item()
            if batch_count > 0:
                total_loss += per_sample_loss.sum().item()
                total_count += batch_count
            progress.set_postfix(loss=loss.item())

    if total_count == 0:
        return 0.0
    return total_loss / total_count


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    npz_paths = list_npz_files(args.data_dir)
    dataset = EEGWindowDataset(
        npz_paths,
        max_cache_size=args.cache_size,
        show_progress=True,
        progress_desc="Loading EEG subjects",
    )
    train_set, val_set = split_dataset(dataset, args.val_split, args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGMaskedAutoencoder(
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
        time_steps=args.time_steps,
        band_index=args.band_index,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            epoch=epoch,
            num_epochs=args.epochs,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
            epoch=epoch,
            num_epochs=args.epochs,
        )
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.save_dir, f"mae_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
