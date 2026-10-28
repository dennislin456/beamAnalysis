"""批次 HDR 融合並量雙光斑距離，不開視窗。

資料夾裡每個輪次是一個 L01、L02… 子資料夾。影像是膝關檔名，例如
Basler_..._L01_C01_X17p00_Y99p60_..._S2_5000us_001_YYYYMMDD_HHMMSS.npy。
點位欄名固定含 X、Y、R1、R2。同一點位、同一輪次裡，
依曝光張數對齊：張數較少的曝光用到最後一張後就固定重用，和 hdr_test.py
的批次融合相同。每一筆融合後用條帶波峰量雙光斑距離。

預設各寫一份 CSV：

  count,loop,cycle,X16.00_Y123.80_R1_359.310_R2_341.320,X17.00_Y99.60_R1_359.180_R2_329.260
  1,L01,C01,...
  2,L01,C01,...
  1,L02,C01,...

count 是左側欄，依該輪次的長曝光序號由上往下排。點位是欄名。
距離是像素，不乘相機像素尺寸。找不到下光斑的格子留白。

未指定 --pair 時會各跑一組：5000+100000、500+100000。

用法（在任何目錄）：
  python hdr_batch_distance.py "D:/data/for dennis"
  python hdr_batch_distance.py "D:/data/for dennis" --pair 5000 100000 --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def _install_import_stubs() -> None:
    """Load spot finding and blur without the Qt windows in those packages."""
    hdr_root = Path(__file__).resolve().parents[1]
    root = str(hdr_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    def stub(name: str, folder: Path) -> None:
        if name in sys.modules:
            return
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]  # type: ignore[attr-defined]
        module.__file__ = str(folder / "__init__.py")
        sys.modules[name] = module

    stub("analysis", hdr_root / "analysis")
    stub("common", hdr_root / "common")


_install_import_stubs()

import numpy as np

from analysis.spots import find_upper_spot, locate_dual_spots
from common.blur import gaussian_blur
from common.images import is_merged_name
from hdr_algorithms import run_hdr


IMAGE_EXTS = {".npy", ".png", ".bmp", ".tif", ".tiff"}
META_COLUMNS = ("count", "loop", "cycle")
NAME_RE = re.compile(
    r"(?P<session>[A-Za-z]+_\d{8}_\d{6})_"
    r"(?P<file_loop>L\d+)_"
    r"(?P<cycle>C\d+)_"
    r"X(?P<x>\d+p\d+)_"
    r"Y(?P<y>\d+p\d+)_"
    r"(?P<z>Z[^_]+)_"
    r"R1_(?P<r1>\d+p\d+)_"
    r"R2_(?P<r2>\d+p\d+)_"
    r"S\d+_"
    r"(?P<exposure>\d+(?:\.\d+)?)us_"
    r"(?P<frame>\d+)_"
    r"(?P<stamp>\d{8}_\d{6})",
    re.IGNORECASE,
)
LOOP_RE = re.compile(r"L\d+", re.IGNORECASE)


@dataclass(frozen=True)
class Shot:
    path: str
    session: str
    loop: str
    cycle: str
    x_token: str
    y_token: str
    z_token: str
    r1_token: str
    r2_token: str
    exposure: float
    frame: int
    stamp: str

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.x_token, self.y_token, self.z_token, self.r1_token, self.r2_token)


def _stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def natural_key(text: str) -> list[int | str]:
    parts = re.split(r"(\d+)", text)
    key: list[int | str] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def token_text(token: str) -> str:
    """Turn ``17p00`` into ``17.00`` and leave ``Zoff`` unchanged."""
    if "p" not in token.lower():
        return token
    whole, frac = re.split(r"p", token, maxsplit=1, flags=re.IGNORECASE)
    if not whole.isdigit() or not frac.isdigit():
        return token
    return f"{int(whole)}.{frac}"


def same_us(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-3


def us_token(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def discover_loops(root: Path) -> list[tuple[str, Path]]:
    """Loop folders named L01, L02, … Direct images are one loop."""
    if not root.is_dir():
        raise SystemExit(f"找不到資料夾：{root}")
    direct = [path for path in root.iterdir() if path.is_dir() and LOOP_RE.fullmatch(path.name)]
    if direct:
        ordered = sorted(direct, key=lambda path: natural_key(path.name))
        return [(path.name, path) for path in ordered]
    nested: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.lower() == "hdr_out":
            continue
        for path in child.iterdir():
            if path.is_dir() and LOOP_RE.fullmatch(path.name):
                nested.append((path.name, path))
    if nested:
        return sorted(nested, key=lambda item: (natural_key(item[1].parent.name), natural_key(item[0])))
    has_image = any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTS and not is_merged_name(path.name)
        for path in root.iterdir()
    )
    if has_image:
        loop = root.name if LOOP_RE.fullmatch(root.name) else "L01"
        return [(loop, root)]
    raise SystemExit(f"{root} 裡沒有 L01/L02 資料夾，也沒有影像檔。")


def iter_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if is_merged_name(path.name):
            continue
        if any(part.lower() == "hdr_out" for part in path.parts):
            continue
        files.append(path)
    return files


def parse_shot(path: Path, loop_name: str) -> Shot | None:
    match = NAME_RE.search(path.name)
    if match is None:
        return None
    loop = loop_name if LOOP_RE.fullmatch(loop_name) else match.group("file_loop")
    return Shot(
        path=str(path),
        session=match.group("session"),
        loop=loop,
        cycle=match.group("cycle"),
        x_token=match.group("x"),
        y_token=match.group("y"),
        z_token=match.group("z"),
        r1_token=match.group("r1"),
        r2_token=match.group("r2"),
        exposure=float(match.group("exposure")),
        frame=int(match.group("frame")),
        stamp=match.group("stamp"),
    )


def point_header(identity: tuple[str, str, str, str, str]) -> str:
    """Full knee point: X, Y, R1, R2 from the filename."""
    x_token, y_token, _z_token, r1_token, r2_token = identity
    return (
        f"X{token_text(x_token)}_Y{token_text(y_token)}"
        f"_R1_{token_text(r1_token)}_R2_{token_text(r2_token)}"
    )


def assign_headers(identities: list[tuple[str, str, str, str, str]]) -> dict[tuple, str]:
    by_header: dict[str, list[tuple]] = defaultdict(list)
    for identity in identities:
        by_header[point_header(identity)].append(identity)
    headers: dict[tuple, str] = {}
    for identity in identities:
        base = point_header(identity)
        if len(by_header[base]) == 1:
            headers[identity] = base
            continue
        headers[identity] = f"{base}_{identity[2]}"
    return headers


def load_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        array = np.load(path, allow_pickle=False)
    elif ext == ".csv":
        array = np.loadtxt(path, delimiter=",", dtype=np.float64)
    elif ext in {".png", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        array = np.array(Image.open(path))
    else:
        raise ValueError(f"不支援的檔案格式：{path.suffix}")
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 3):
        return array
    raise ValueError(f"{path.name} 不是二維或 RGB 影像，形狀為 {array.shape}")


def preprocess(image: np.ndarray, cfg: dict) -> np.ndarray:
    data = np.asarray(image, dtype=np.float64)
    if not cfg["blur"]:
        return data
    width_x = int(cfg["blur_x"])
    width_y = int(cfg["blur_y"])
    sigma = float(cfg["sigma"])
    if data.ndim == 2:
        return gaussian_blur(data, width_x, width_y, sigma)
    channels = [
        gaussian_blur(data[:, :, index], width_x, width_y, sigma)
        for index in range(data.shape[2])
    ]
    return np.stack(channels, axis=2)


def fuse_plane(images: list[np.ndarray], times: list[float], cfg: dict) -> np.ndarray:
    settings = {
        "samples": 70,
        "lambda": 10.0,
        "random": False,
        "max_iter": 30,
        "threshold": 0.01,
        "contrast": float(cfg["contrast"]),
        "saturation": float(cfg["saturation"]),
        "exposure": float(cfg["exposure_weight"]),
        "sigma": float(cfg["local_sigma"]),
    }
    raw, _response = run_hdr(
        images,
        times,
        str(cfg["algorithm"]),
        float(cfg["full_scale"]),
        settings,
    )
    plane = np.asarray(raw, dtype=np.float64)
    if plane.ndim == 3:
        plane = np.mean(plane, axis=2)
    if str(cfg["algorithm"]) not in {"mertens", "mertens_local"}:
        plane = plane * 255.0
    return plane


def dual_distance_px(image: np.ndarray, cfg: dict) -> float | None:
    data = np.asarray(image, dtype=np.float64)
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    if data.ndim != 2 or data.size == 0:
        return None
    height, width = data.shape
    thresh = float(cfg["thresh"])
    seed, _mask = find_upper_spot(data, thresh)
    if seed is None:
        return None
    roi_w = int(np.clip(int(cfg["strip_width"]), 1, width))
    roi_x = int(round(float(seed[0]) - roi_w / 2.0))
    roi_x = int(np.clip(roi_x, 0, max(width - roi_w, 0)))
    spots = locate_dual_spots(
        data,
        roi_x,
        0,
        roi_w,
        int(height),
        min_peak_distance=int(cfg["peak_distance"]),
        min_peak_ratio=float(cfg["peak_ratio"]) / 100.0,
        use_log=False,
        thresh_percent=thresh,
        lower_thresh_percent=float(cfg["lower_thresh"]),
        center_mode=str(cfg["center_mode"]),
    )
    upper = spots.get("upper")
    lower = spots.get("lower")
    if upper is None or lower is None:
        return None
    return float(np.hypot(float(lower[0]) - float(upper[0]), float(lower[1]) - float(upper[1])))


def format_distance(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def measure_point(task: dict) -> dict:
    """Fuse every frame of one point and measure dual-spot distance."""
    cfg = task["cfg"]
    cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    warnings: list[str] = []

    def prepared(path: str) -> np.ndarray:
        image = cache.get(path)
        if image is None:
            image = preprocess(load_image(Path(path)), cfg)
            cache[path] = image
        return image

    for frame in task["frames"]:
        try:
            images = [prepared(path) for path in frame["paths"]]
            plane = fuse_plane(images, list(frame["times"]), cfg)
            distance_px = dual_distance_px(plane, cfg)
        except Exception as exc:
            distance_px = None
            warnings.append(
                f"{task['header']} {frame['loop']} {frame['cycle']} count {frame['count']}：{exc}"
            )
        if distance_px is None:
            text = ""
            if not warnings or not warnings[-1].startswith(f"{task['header']} {frame['loop']}"):
                warnings.append(
                    f"{task['header']} {frame['loop']} {frame['cycle']} count {frame['count']}：找不到雙光斑"
                )
        else:
            text = format_distance(distance_px)
        rows.append(
            {
                "count": int(frame["count"]),
                "loop": frame["loop"],
                "cycle": frame["cycle"],
                "session": frame["session"],
                "distance": text,
            }
        )
    finite = [row["distance"] for row in rows if row["distance"]]
    return {
        "header": task["header"],
        "rows": rows,
        "warnings": warnings,
        "n": len(rows),
        "n_ok": len(finite),
    }


def shots_for_exposure(shots: list[Shot], exposure: float) -> list[Shot]:
    chosen = [shot for shot in shots if same_us(shot.exposure, exposure)]
    chosen.sort(key=lambda shot: (shot.frame, shot.session, shot.stamp, shot.path))
    return chosen


def build_frames(group: list[Shot], pair: tuple[float, float]) -> tuple[list[dict], str | None]:
    left = shots_for_exposure(group, pair[0])
    right = shots_for_exposure(group, pair[1])
    if not left or not right:
        missing = []
        if not left:
            missing.append(f"{us_token(pair[0])}us")
        if not right:
            missing.append(f"{us_token(pair[1])}us")
        return [], "缺少 " + "、".join(missing)
    if len(left) > len(right) or (len(left) == len(right) and pair[0] >= pair[1]):
        driving = left
    else:
        driving = right
    frames: list[dict] = []
    for index, drive in enumerate(driving):
        left_shot = left[min(index, len(left) - 1)]
        right_shot = right[min(index, len(right) - 1)]
        frames.append(
            {
                "count": int(drive.frame),
                "loop": drive.loop,
                "cycle": drive.cycle,
                "session": drive.session,
                "paths": [left_shot.path, right_shot.path],
                "times": [left_shot.exposure, right_shot.exposure],
            }
        )
    return frames, None


def collect_shots(root: Path) -> tuple[list[Shot], list[str]]:
    shots: list[Shot] = []
    unmatched: list[str] = []
    for loop_name, folder in discover_loops(root):
        for path in iter_images(folder):
            shot = parse_shot(path, loop_name)
            if shot is None:
                unmatched.append(path.name)
                continue
            shots.append(shot)
    return shots, unmatched


def point_tasks(shots: list[Shot], pair: tuple[float, float], cfg: dict) -> tuple[list[dict], list[str]]:
    grouped: dict[tuple, list[Shot]] = defaultdict(list)
    for shot in shots:
        grouped[shot.identity].append(shot)
    headers = assign_headers(list(grouped))
    tasks: list[dict] = []
    notes: list[str] = []
    for identity, group in grouped.items():
        header = headers[identity]
        frames, error = build_frames(group, pair)
        if error:
            notes.append(f"{header}：{error}，略過")
            continue
        tasks.append({"header": header, "frames": frames, "cfg": cfg})
    tasks.sort(key=lambda task: natural_key(task["header"]))
    return tasks, notes


def needs_session(results: list[dict]) -> bool:
    seen: dict[tuple, set[str]] = defaultdict(set)
    for result in results:
        for row in result["rows"]:
            key = (result["header"], row["loop"], row["cycle"], row["count"])
            seen[key].add(row["session"])
            if len(seen[key]) > 1:
                return True
    return False


def write_table(path: Path, results: list[dict], with_session: bool) -> None:
    headers = [result["header"] for result in results]
    columns = list(META_COLUMNS)
    if with_session:
        columns.append("session")
    columns.extend(headers)
    cells: dict[tuple, dict[str, str]] = {}
    for result in results:
        for row in result["rows"]:
            key = (int(row["count"]), row["loop"], row["cycle"], row["session"] if with_session else "")
            bucket = cells.setdefault(
                key,
                {
                    "count": str(int(row["count"])),
                    "loop": row["loop"],
                    "cycle": row["cycle"],
                    "session": row["session"],
                },
            )
            previous = bucket.get(result["header"])
            if previous not in (None, ""):
                print(
                    f"  重複列 {row['loop']} {row['cycle']} count {row['count']} {result['header']}，保留後一筆",
                    flush=True,
                )
            bucket[result["header"]] = row["distance"]
    ordered = sorted(cells, key=lambda key: (natural_key(key[2]), natural_key(key[1]), key[0], key[3]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        for key in ordered:
            writer.writerow(cells[key])


def write_params(path: Path, cfg: dict, pair: tuple[float, float], source: Path, headers: list[str]) -> None:
    payload = {
        "source": str(source),
        "pair_us": [pair[0], pair[1]],
        "distance_unit": "px",
        "algorithm": cfg["algorithm"],
        "mertens": {
            "contrast": cfg["contrast"],
            "saturation": cfg["saturation"],
            "exposure": cfg["exposure_weight"],
        },
        "gaussian": {
            "enabled": cfg["blur"],
            "width_x": cfg["blur_x"],
            "width_y": cfg["blur_y"],
            "sigma": cfg["sigma"],
        },
        "strip": {
            "width": cfg["strip_width"],
            "min_peak_distance": cfg["peak_distance"],
            "peak_ratio_percent": cfg["peak_ratio"],
            "threshold_percent": cfg["thresh"],
            "lower_threshold_percent": cfg["lower_thresh"],
            "center_mode": cfg["center_mode"],
        },
        "full_scale": cfg["full_scale"],
        "points": headers,
        "alignment": "張數較少的曝光在超過最後一張後固定重用，count 用張數較多那一側的檔名序號。",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pair(tasks: list[dict], jobs: int) -> list[dict]:
    if jobs <= 1:
        results = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            print(f"  [{index}/{total}] {task['header']}  {len(task['frames'])} 筆", flush=True)
            results.append(measure_point(task))
        return results
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: list[dict | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        future_map = {pool.submit(measure_point, task): index for index, task in enumerate(tasks)}
        done = 0
        for future in as_completed(future_map):
            done += 1
            index = future_map[future]
            result = future.result()
            results[index] = result
            print(f"  [{done}/{len(tasks)}] {result['header']}  完成 {result['n_ok']}/{result['n']}", flush=True)
    return [result for result in results if result is not None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依膝關檔名分點位，做 HDR 融合並輸出各輪次、各 count 的雙光斑距離 CSV。"
    )
    parser.add_argument("folder", type=Path, help="含 L01、L02… 的資料夾，或單一輪次資料夾")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="輸出資料夾。未指定時寫到輸入資料夾下的 hdr_out",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        type=float,
        action="append",
        metavar=("US_A", "US_B"),
        help="一組要融合的曝光（us），可重複。未指定時輸出 5000+100000 與 500+100000",
    )
    parser.add_argument("--algorithm", default="mertens", choices=("mertens", "mertens_local", "debevec", "robertson"))
    parser.add_argument("--contrast", type=float, default=1.0, help="Mertens 對比。預設 1")
    parser.add_argument("--saturation", type=float, default=0.0, help="Mertens 飽和。預設 0")
    parser.add_argument("--exposure-weight", type=float, default=1.0, help="Mertens 曝光權重。預設 1")
    parser.add_argument("--local-sigma", type=float, default=0.2, help="只給 Mertens 自寫用。預設 0.2")
    parser.add_argument("--full-scale", type=float, default=4095.0, help="非 8-bit 的滿量。預設 4095")
    parser.add_argument("--strip-width", type=int, default=250, help="條帶寬。預設 250")
    parser.add_argument("--peak-distance", type=int, default=110, help="上下峰最小距離（像素）。預設 110")
    parser.add_argument("--peak-ratio", type=float, default=10.0, help="下峰比 %%。預設 10")
    parser.add_argument("--thresh", type=float, default=50.0, help="上門檻 %%。預設 50")
    parser.add_argument("--lower-thresh", type=float, default=50.0, help="下門檻 %%。條帶波峰不使用。預設 50")
    parser.add_argument(
        "--center-mode",
        default="peak",
        choices=("peak", "shape", "intensity", "line", "slant_iter"),
        help="雙光斑演算法。預設 peak（條帶波峰）",
    )
    parser.add_argument("--blur-x", type=int, default=80, help="高斯模糊水平核寬。預設 80")
    parser.add_argument("--blur-y", type=int, default=80, help="高斯模糊垂直核寬。預設 80")
    parser.add_argument("--sigma", type=float, default=2.0, help="高斯模糊 sigma。預設 2")
    parser.add_argument("--no-blur", action="store_true", help="融合前不做高斯模糊")
    parser.add_argument("--jobs", type=int, default=1, help="平行點位數。預設 1")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs 至少為 1")
    if args.strip_width < 1 or args.peak_distance < 1:
        parser.error("條帶寬和距離至少為 1")
    if args.blur_x < 1 or args.blur_y < 1 or args.sigma <= 0:
        parser.error("高斯核寬至少為 1，sigma 必須大於 0")
    if args.full_scale <= 0:
        parser.error("滿量必須大於 0")
    if args.pair:
        for left, right in args.pair:
            if left <= 0 or right <= 0:
                parser.error("曝光時間必須大於 0")
            if same_us(left, right):
                parser.error("同一組的兩個曝光時間不能相同")
    return args


def main(argv: list[str] | None = None) -> None:
    _stdio_utf8()
    args = parse_args(argv)
    root = args.folder.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else root / "hdr_out"
    pairs = [tuple(pair) for pair in args.pair] if args.pair else [(5000.0, 100000.0), (500.0, 100000.0)]
    shots, unmatched = collect_shots(root)
    if not shots:
        raise SystemExit(f"{root} 沒有可辨認的膝關影像檔名。")
    cfg = {
        "algorithm": args.algorithm,
        "contrast": args.contrast,
        "saturation": args.saturation,
        "exposure_weight": args.exposure_weight,
        "local_sigma": args.local_sigma,
        "full_scale": args.full_scale,
        "strip_width": args.strip_width,
        "peak_distance": args.peak_distance,
        "peak_ratio": args.peak_ratio,
        "thresh": args.thresh,
        "lower_thresh": args.lower_thresh,
        "center_mode": args.center_mode,
        "blur": not args.no_blur,
        "blur_x": args.blur_x,
        "blur_y": args.blur_y,
        "sigma": args.sigma,
    }
    print(f"資料夾 {root}", flush=True)
    print(f"點位識別 {len({shot.identity for shot in shots})} 個，影像 {len(shots)} 張", flush=True)
    if unmatched:
        preview = "、".join(unmatched[:8])
        extra = "" if len(unmatched) <= 8 else f" …共 {len(unmatched)} 個"
        print(f"略過無法辨認的檔名：{preview}{extra}", flush=True)
    for pair in pairs:
        label = f"{us_token(pair[0])}us+{us_token(pair[1])}us"
        print(f"\n融合 {label}", flush=True)
        tasks, notes = point_tasks(shots, pair, cfg)
        for note in notes:
            print(f"  {note}", flush=True)
        if not tasks:
            print("  這一組沒有可融合的點位。", flush=True)
            continue
        results = run_pair(tasks, args.jobs)
        with_session = needs_session(results)
        csv_path = output / f"distance_{us_token(pair[0])}us_{us_token(pair[1])}us.csv"
        write_table(csv_path, results, with_session)
        write_params(csv_path.with_suffix(".params.json"), cfg, pair, root, [item["header"] for item in results])
        warning_lines = [line for result in results for line in result["warnings"]]
        if warning_lines:
            warning_path = csv_path.with_suffix(".warnings.txt")
            warning_path.write_text("\n".join(warning_lines) + "\n", encoding="utf-8")
        ok = sum(result["n_ok"] for result in results)
        total = sum(result["n"] for result in results)
        print(f"  寫入 {csv_path}", flush=True)
        print(f"  距離 {ok}/{total} 筆（單位 px）", flush=True)
        if warning_lines:
            print(f"  警告 {len(warning_lines)} 則，見 {csv_path.with_suffix('.warnings.txt')}", flush=True)


if __name__ == "__main__":
    main()
