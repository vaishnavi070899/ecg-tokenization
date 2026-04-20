import ast

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from config import PTB_XL_PATH

def load_sample_ecg(record_index=0, lead=0, sampling_rate=100):
    """Load a single ECG lead from the PTB-XL dataset.

    Args:
        record_index: Row index into ptbxl_database.csv (0-based).
        lead: Which of the 12 leads to return (0 = lead I).
        sampling_rate: 100 or 500 Hz. 100 Hz → 1000 samples, 500 Hz → 5000 samples.

    Returns:
        1-D numpy array of shape (n_samples,).
    """
    df = pd.read_csv(PTB_XL_PATH + "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    row = df.iloc[record_index]
    filename = row.filename_lr if sampling_rate == 100 else row.filename_hr
    signal, _ = wfdb.rdsamp(PTB_XL_PATH + filename)  # shape: (n_samples, 12)
    return signal[:, lead]


class PTBXLDataset(Dataset):
    """Lazy-loading PyTorch Dataset for PTB-XL ECG records.

    Args:
        path:          Root directory of the PTB-XL dataset.
        sampling_rate: 100 or 500 Hz.
        folds:         List of strat_fold values to include (1–10).
                       Folds 1–9 = train, fold 10 = val (PTB-XL convention).
                       None = all folds.
        n_records:     Cap on number of records after fold filtering (for quick runs).
        lead:          Which of the 12 leads to use (0 = lead I).
    """

    def __init__(self, path=PTB_XL_PATH, sampling_rate=100, folds=None, n_records=None, lead=0):
        df = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
        if folds is not None:
            df = df[df.strat_fold.isin(folds)]
        if n_records is not None:
            df = df.iloc[:n_records]
        col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.filenames = [path + f for f in df[col]]
        self.lead = lead

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        signal, _ = wfdb.rdsamp(self.filenames[idx])                    # (1000, 12)
        x = torch.tensor(signal[:, self.lead], dtype=torch.float32)     # (1000,)
        # Per-sample z-score normalization: zero mean, unit variance
        std = x.std()
        if std > 0:
            x = (x - x.mean()) / std
        return x
