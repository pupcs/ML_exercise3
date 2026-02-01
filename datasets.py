import h5py
import numpy as np
from torch.utils.data import Dataset


class _H5Base(Dataset):
    """
    Safe HDF5 usage with PyTorch DataLoader:
    - Do NOT open/close per sample
    - Open lazily once per worker process
    - Avoid pickling an open file handle
    """
    def __init__(self, h5_file: str):
        super().__init__()
        self.h5_file = h5_file

        # Cache length once (fast + avoids repeated file opens in __len__)
        with h5py.File(self.h5_file, "r") as f:
            self._length = len(f["lr"])

        # Per-worker handles (initialized lazily in the worker process)
        self._h5 = None
        self._lr = None
        self._hr = None

    def _ensure_open(self):
        if self._h5 is None:
            # Each worker opens its own handle the first time it needs data
            self._h5 = h5py.File(self.h5_file, "r")
            self._lr = self._h5["lr"]
            self._hr = self._h5["hr"]

    def __len__(self):
        return self._length

    def __getstate__(self):
        # When DataLoader forks/spawns, the dataset gets pickled.
        # Never pickle open HDF5 handles; reopen in each worker instead.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_lr"] = None
        state["_hr"] = None
        return state

    def __del__(self):
        # Best-effort cleanup
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass


class TrainDataset(_H5Base):
    def __init__(self, h5_file):
        super().__init__(h5_file)

    def __getitem__(self, idx):
        self._ensure_open()
        lr = self._lr[idx] / 255.0
        hr = self._hr[idx] / 255.0
        return np.expand_dims(lr, 0), np.expand_dims(hr, 0)


class EvalDataset(_H5Base):
    def __init__(self, h5_file):
        super().__init__(h5_file)

    def __getitem__(self, idx):
        self._ensure_open()
        # eval file stores samples under string keys
        k = str(idx)
        lr = self._lr[k][:, :] / 255.0
        hr = self._hr[k][:, :] / 255.0
        return np.expand_dims(lr, 0), np.expand_dims(hr, 0)
