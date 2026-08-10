import csv
import os
import re

import numpy as np
import pandas as pd


def _natural_sort_key(name):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(name))]


def scan_location_files(root_dir):
    """掃描主資料夾 → {位置名: {檔名: 完整路徑}}。"""
    result = {}
    if not root_dir or not os.path.isdir(root_dir):
        return result
    try:
        entries = os.listdir(root_dir)
    except OSError:
        return result

    for entry in entries:
        loc_path = os.path.join(root_dir, entry)
        if not os.path.isdir(loc_path):
            continue

        files = {}
        try:
            for fname in os.listdir(loc_path):
                if fname.startswith("~$"):
                    continue
                lower = fname.lower()
                if lower.endswith((".xlsx", ".xls", ".csv", ".npy")):
                    files[fname] = os.path.join(loc_path, fname)
        except OSError:
            continue

        if files:
            result[entry] = files

    return result


def load_numeric_matrix(path):
    """載入 CSV / Excel / NPY，回傳數值矩陣。"""
    if not path:
        raise ValueError("No file path provided")

    lower = str(path).lower()
    if lower.endswith(".csv"):
        return _read_csv_matrix(path)
    if lower.endswith(".npy"):
        return _read_npy_matrix(path)
    return _read_excel_matrix(path)


def _read_excel_matrix(path):
    df = pd.read_excel(path, header=None, skiprows=4)
    return df.dropna(how="all").astype(float).to_numpy()


def _read_npy_matrix(path):
    arr = np.load(path)
    if isinstance(arr, np.ndarray):
        return np.asarray(arr, dtype=float)
    raise ValueError(f"NPY content is not a numeric array: {path}")


def _read_csv_matrix(path):
    def is_number(token):
        token = str(token).strip()
        if not token:
            return False
        try:
            float(token)
            return True
        except ValueError:
            return False

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            rows.append([cell.strip() for cell in row])

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    start_index = None
    for idx, row in enumerate(rows):
        if not row:
            continue
        numeric_cells = sum(1 for cell in row if is_number(cell))
        if numeric_cells >= max(2, (len(row) + 1) // 2):
            start_index = idx
            break

    if start_index is None:
        raise ValueError(f"No parseable numeric block found in {path}")

    block_rows = rows[start_index:]
    numeric_rows = []
    for row in block_rows:
        parsed_row = []
        for cell in row:
            try:
                parsed_row.append(float(cell))
            except ValueError:
                parsed_row.append(np.nan)
        numeric_rows.append(parsed_row)

    parsed = pd.DataFrame(numeric_rows)
    parsed = parsed.dropna(how="all")
    if parsed.empty:
        raise ValueError(f"Matrix is empty after cleanup: {path}")

    if parsed.shape[1] > 1 and _looks_like_index_column(parsed.iloc[:, 0]):
        parsed = parsed.iloc[:, 1:]

    # Avoid removing a real data row simply because it looks like a numbered sequence.
    # The numeric block itself is already selected from the first parseable rows.

    # Preserve the full numeric block; if a column is all NaN, drop only that column.
    parsed = parsed.loc[:, parsed.notna().any()].astype(float)
    if parsed.shape[1] == 0:
        raise ValueError(f"Matrix is empty after cleanup: {path}")

    return parsed.to_numpy()


def _looks_like_index_column(col):
    values = [x for x in col.dropna().tolist() if x is not None]
    if len(values) < 3:
        return False
    if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
        return False

    ints = [int(v) for v in values]
    if len(ints) == 0:
        return False

    seq_asc = ints == list(range(ints[0], ints[0] + len(ints)))
    seq_desc = ints == list(range(ints[0], ints[0] - len(ints), -1))
    return seq_asc or seq_desc
