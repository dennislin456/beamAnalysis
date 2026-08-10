import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))

from batch_data_loader import load_numeric_matrix, scan_location_files


def test_scan_location_files_accepts_csv_and_excel(tmp_path):
    loc_dir = tmp_path / "siteA"
    loc_dir.mkdir()
    (loc_dir / "sample.csv").write_text("meta\n1,2\n3,4\n", encoding="utf-8")
    (loc_dir / "sample.xlsx").write_bytes(b"fake-xlsx")

    result = scan_location_files(str(tmp_path))

    assert set(result["siteA"].keys()) == {"sample.csv", "sample.xlsx"}


def test_load_numeric_matrix_reads_csv_block(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["meta", "info"])
        writer.writerow(["header", "header"])
        writer.writerow(["1", "2"])
        writer.writerow(["3", "4"])
        writer.writerow(["5", "6"])

    matrix = load_numeric_matrix(str(csv_path))

    assert matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))


def test_scan_and_load_npy_matrix(tmp_path):
    loc_dir = tmp_path / "siteB"
    loc_dir.mkdir()
    npy_path = loc_dir / "sample.npy"
    np.save(npy_path, np.array([[1.0, 2.0], [3.0, 4.0]]))

    result = scan_location_files(str(tmp_path))
    assert "sample.npy" in result["siteB"]

    matrix = load_numeric_matrix(str(npy_path))
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))
