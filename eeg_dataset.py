import bisect
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SubjectIndex:
    subject_id: str
    path: str
    num_windows: int


class EEGWindowDataset(Dataset):
    """Dataset that exposes each EEG window as an individual sample."""

    def __init__(self, npz_paths: List[str], max_cache_size: int = 0) -> None:
        if not npz_paths:
            raise ValueError("npz_paths must contain at least one file.")
        self._subjects: List[SubjectIndex] = []
        self._cumulative_sizes: List[int] = []
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._max_cache_size = max_cache_size

        total = 0
        for path in npz_paths:
            subject_id = os.path.splitext(os.path.basename(path))[0]
            with np.load(path, mmap_mode="r") as data:
                num_windows = int(data["x"].shape[0])
            total += num_windows
            self._subjects.append(SubjectIndex(subject_id, path, num_windows))
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")
        subject_idx = bisect.bisect_right(self._cumulative_sizes, index)
        subject = self._subjects[subject_idx]
        subject_offset = self._cumulative_sizes[subject_idx - 1] if subject_idx > 0 else 0
        window_idx = index - subject_offset

        data = self._load_subject(subject)
        x_window = torch.as_tensor(data["x"][window_idx], dtype=torch.float32)
        valid_mask = torch.as_tensor(data["valid_mask"][window_idx], dtype=torch.float32)
        return x_window, valid_mask, subject.subject_id

    def _load_subject(self, subject: SubjectIndex) -> Dict[str, np.ndarray]:
        if subject.subject_id in self._cache:
            self._cache.move_to_end(subject.subject_id)
            return self._cache[subject.subject_id]

        with np.load(subject.path) as data:
            x = data["x"].astype(np.float32)
            valid_mask = data["valid_mask"].astype(np.float32)
        self._cache[subject.subject_id] = {"x": x, "valid_mask": valid_mask}
        self._enforce_cache_limit()
        return self._cache[subject.subject_id]

    def _enforce_cache_limit(self) -> None:
        if self._max_cache_size and len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
