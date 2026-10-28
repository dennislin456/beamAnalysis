"""從距離 CSV 裡，為每個點位找出不重複 index 的分組。

每一組從「同一個點位」取 group-size 筆。組數等於目標檔的列數（常見是十組）。
每一組有自己的預期平均與樣本標準差（ddof=1，和 HDR 分析視窗相同）。
同一個點位裡，一個 index 只會進一個組。index 是距離 CSV 的資料列號，
從 1 開始，不含標題列。

目標檔 CSV：

  group,mean,std
  1,120.5,0.40
  2,121.0,0.35
  ...

若有 point 欄，則每個點位用自己的十組目標；沒有 point 欄時，同一組目標套到每個點位。

搜尋：每組 1 筆時用匈牙利演算法，答案是最佳解。只有一組且組合數可數完時逐一列舉。
其餘情況用多起點的交換搜尋，組合很大時不保證是唯一的全域最佳。

用法：
  python hdr_group_match.py distance_5000us_100000us.csv --group-size 5 --targets targets.csv
  python hdr_group_match.py distance.csv --group-size 1 --target 120.5,0.4 --target 121.0,0.35
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


META_COLUMNS = {"count", "loop", "cycle", "session", "index"}
MEAN_NAMES = {"mean", "target_mean", "平均", "預期平均", "期望平均"}
STD_NAMES = {"std", "stdev", "target_std", "標準差", "預期標準差", "期望標準差"}
POINT_NAMES = {"point", "點位", "header"}
GROUP_NAMES = {"group", "組", "組別"}


def _stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"{path} 沒有標題列。")
        return list(reader)


def _header_map(fieldnames: list[str]) -> dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames if name}


def _pick(mapping: dict[str, str], names: set[str]) -> str | None:
    for name in names:
        if name.lower() in mapping:
            return mapping[name.lower()]
    return None


def load_distance(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = read_dicts(path)
    if not rows:
        raise SystemExit(f"{path} 沒有資料列。")
    points = [name for name in rows[0] if name and name.strip().lower() not in META_COLUMNS]
    if not points:
        raise SystemExit(f"{path} 沒有點位欄。")
    return points, rows


def load_targets(path: Path | None, inline: list[str] | None) -> dict[str | None, list[tuple[float, float]]]:
    """Map point name → [(mean, std), ...]. Key None means the same targets for every point."""
    if path is None and not inline:
        raise SystemExit("請用 --targets 或重複的 --target mean,std 指定每一組的預期平均與標準差。")
    if path is not None and inline:
        raise SystemExit("--targets 和 --target 請只留一個。")
    if inline:
        pairs = [_parse_target(text, index) for index, text in enumerate(inline, start=1)]
        return {None: pairs}
    assert path is not None
    rows = read_dicts(path)
    mapping = _header_map(list(rows[0]))
    mean_key = _pick(mapping, MEAN_NAMES)
    std_key = _pick(mapping, STD_NAMES)
    if mean_key is None or std_key is None:
        raise SystemExit(f"{path} 需要 mean 與 std 欄（或 平均 / 標準差）。")
    point_key = _pick(mapping, POINT_NAMES)
    group_key = _pick(mapping, GROUP_NAMES)
    grouped: dict[str | None, list[tuple[int, float, float]]] = {}
    for index, row in enumerate(rows, start=1):
        point = row[point_key].strip() if point_key and row.get(point_key) else None
        if point == "":
            point = None
        try:
            mean = float(str(row[mean_key]).strip())
            std = float(str(row[std_key]).strip())
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{path} 第 {index} 列的平均或標準差不是數字。") from exc
        if not np.isfinite(mean) or not np.isfinite(std):
            raise SystemExit(f"{path} 第 {index} 列的平均或標準差不是有限數字。")
        if std < 0:
            raise SystemExit(f"{path} 第 {index} 列的標準差不能小於 0。")
        order = index
        if group_key and str(row.get(group_key, "")).strip():
            try:
                order = int(float(row[group_key]))
            except ValueError:
                order = index
        grouped.setdefault(point, []).append((order, mean, std))
    if None in grouped and len(grouped) > 1:
        raise SystemExit("目標檔若有點位欄，每一列都要填點位。")
    return {key: [(mean, std) for _order, mean, std in sorted(items)] for key, items in grouped.items()}


def _parse_target(text: str, index: int) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--target 第 {index} 組要寫成 mean,std，例如 120.5,0.4")
    try:
        mean = float(parts[0])
        std = float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"--target 第 {index} 組不是數字。") from exc
    if std < 0 or not np.isfinite(mean) or not np.isfinite(std):
        raise SystemExit(f"--target 第 {index} 組的標準差不能小於 0，平均和標準差都要是有限數字。")
    return mean, std


def sample_mean_std(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, 0.0
    return mean, float(np.std(data, ddof=1))


def group_cost(
    values: np.ndarray,
    target: tuple[float, float],
    mean_weight: float,
    std_weight: float,
) -> float:
    mean, std = sample_mean_std(values)
    target_mean, target_std = target
    return mean_weight * (mean - target_mean) ** 2 + std_weight * (std - target_std) ** 2


def _fast_cost(mean: np.ndarray, std: np.ndarray, target: tuple[float, float], mean_weight: float, std_weight: float) -> np.ndarray:
    target_mean, target_std = target
    return mean_weight * (mean - target_mean) ** 2 + std_weight * (std - target_std) ** 2


def _sums(values: np.ndarray) -> tuple[float, float]:
    return float(np.sum(values)), float(np.dot(values, values))


def _mean_std_from_sums(total: np.ndarray, squares: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    mean = total / size
    if size < 2:
        return mean, np.zeros_like(mean)
    var = (squares - total * total / size) / (size - 1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def combination_count(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    count = 1
    for index in range(k):
        count = count * (n - index) // (index + 1)
    return count


def _hungarian(values: np.ndarray, targets: list[tuple[float, float]], mean_weight: float, std_weight: float):
    from scipy.optimize import linear_sum_assignment

    cost = np.empty((values.size, len(targets)), dtype=np.float64)
    for index, target in enumerate(targets):
        target_mean, target_std = target
        cost[:, index] = mean_weight * (values - target_mean) ** 2 + std_weight * (target_std ** 2)
    rows, cols = linear_sum_assignment(cost)
    groups: list[list[int]] = [[] for _ in targets]
    total = 0.0
    for row, col in zip(rows, cols):
        groups[int(col)] = [int(row)]
        total += float(cost[row, col])
    return groups, total


def _exact_one_group(
    values: np.ndarray,
    target: tuple[float, float],
    group_size: int,
    mean_weight: float,
    std_weight: float,
):
    import itertools

    best_cost = None
    best_group: tuple[int, ...] = ()
    for combo in itertools.combinations(range(values.size), group_size):
        cost = group_cost(values[list(combo)], target, mean_weight, std_weight)
        if best_cost is None or cost < best_cost - 1e-15 or (
            abs(cost - best_cost) <= 1e-15 and combo < best_group
        ):
            best_cost = cost
            best_group = combo
    return [list(best_group)], float(best_cost or 0.0)


def _greedy_init(values: np.ndarray, targets: list[tuple[float, float]], group_size: int, order: list[int]) -> list[list[int]]:
    used: set[int] = set()
    groups: list[list[int]] = [[] for _ in targets]
    n = int(values.size)
    for target_index in order:
        target_mean = targets[target_index][0]
        candidates = [index for index in range(n) if index not in used]
        candidates.sort(key=lambda index: (abs(float(values[index]) - target_mean), index))
        chosen = candidates[:group_size]
        groups[target_index] = chosen
        used.update(chosen)
    return groups


def _random_init(n: int, n_groups: int, group_size: int, rng: np.random.Generator) -> list[list[int]]:
    picked = rng.permutation(n)[: n_groups * group_size]
    return [picked[index * group_size : (index + 1) * group_size].astype(int).tolist() for index in range(n_groups)]


def _replacement_move(
    groups: list[list[int]],
    values: np.ndarray,
    targets: list[tuple[float, float]],
    mean_weight: float,
    std_weight: float,
) -> tuple[float, tuple | None]:
    used = {index for group in groups for index in group}
    unused = np.array([index for index in range(values.size) if index not in used], dtype=int)
    if unused.size == 0:
        return 0.0, None
    best_delta = 0.0
    best: tuple | None = None
    cand = values[unused]
    for group_index, group in enumerate(groups):
        current = values[np.asarray(group, dtype=int)]
        current_cost = group_cost(current, targets[group_index], mean_weight, std_weight)
        total, squares = _sums(current)
        size = current.size
        for pos, old_index in enumerate(group):
            old = float(values[old_index])
            new_total = total - old + cand
            new_squares = squares - old * old + cand * cand
            mean, std = _mean_std_from_sums(new_total, new_squares, size)
            costs = _fast_cost(mean, std, targets[group_index], mean_weight, std_weight)
            pick = int(np.argmin(costs))
            delta = float(costs[pick] - current_cost)
            if delta < best_delta:
                best_delta = delta
                best = ("replace", group_index, pos, int(unused[pick]), float(costs[pick]))
    return best_delta, best


def _swap_move(
    groups: list[list[int]],
    values: np.ndarray,
    targets: list[tuple[float, float]],
    mean_weight: float,
    std_weight: float,
) -> tuple[float, tuple | None]:
    best_delta = 0.0
    best: tuple | None = None
    n_groups = len(groups)
    for left in range(n_groups):
        a = values[np.asarray(groups[left], dtype=int)]
        cost_a = group_cost(a, targets[left], mean_weight, std_weight)
        sum_a, sq_a = _sums(a)
        for right in range(left + 1, n_groups):
            b = values[np.asarray(groups[right], dtype=int)]
            cost_b = group_cost(b, targets[right], mean_weight, std_weight)
            sum_b, sq_b = _sums(b)
            size_a = a.size
            size_b = b.size
            new_sum_a = sum_a - a[:, None] + b[None, :]
            new_sum_b = sum_b - b[None, :] + a[:, None]
            new_sq_a = sq_a - a[:, None] ** 2 + b[None, :] ** 2
            new_sq_b = sq_b - b[None, :] ** 2 + a[:, None] ** 2
            mean_a, std_a = _mean_std_from_sums(new_sum_a, new_sq_a, size_a)
            mean_b, std_b = _mean_std_from_sums(new_sum_b, new_sq_b, size_b)
            costs = _fast_cost(mean_a, std_a, targets[left], mean_weight, std_weight)
            costs = costs + _fast_cost(mean_b, std_b, targets[right], mean_weight, std_weight)
            flat = int(np.argmin(costs))
            pos_a, pos_b = divmod(flat, b.size)
            delta = float(costs[pos_a, pos_b] - cost_a - cost_b)
            if delta < best_delta:
                best_delta = delta
                best = ("swap", left, int(pos_a), right, int(pos_b))
    return best_delta, best


def _apply(groups: list[list[int]], move: tuple) -> None:
    if move[0] == "replace":
        _kind, group_index, pos, new_index, _cost = move
        groups[group_index][pos] = new_index
        return
    _kind, left, pos_a, right, pos_b = move
    groups[left][pos_a], groups[right][pos_b] = groups[right][pos_b], groups[left][pos_a]


def _exact_total(groups: list[list[int]], values: np.ndarray, targets, mean_weight: float, std_weight: float) -> float:
    total = 0.0
    for group, target in zip(groups, targets):
        total += group_cost(values[np.asarray(group, dtype=int)], target, mean_weight, std_weight)
    return total


def polish(
    groups: list[list[int]],
    values: np.ndarray,
    targets: list[tuple[float, float]],
    mean_weight: float,
    std_weight: float,
) -> list[list[int]]:
    """Hill-climb by replacing or swapping indices. Exact cost decides whether a move stays."""
    current = [list(group) for group in groups]
    current_cost = _exact_total(current, values, targets, mean_weight, std_weight)
    for _sweep in range(200):
        replace_delta, replace = _replacement_move(current, values, targets, mean_weight, std_weight)
        swap_delta, swap = _swap_move(current, values, targets, mean_weight, std_weight)
        candidates = []
        if replace is not None:
            candidates.append((replace_delta, replace))
        if swap is not None:
            candidates.append((swap_delta, swap))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        accepted = False
        for _delta, move in candidates:
            trial = [list(group) for group in current]
            _apply(trial, move)
            trial_cost = _exact_total(trial, values, targets, mean_weight, std_weight)
            if trial_cost < current_cost - 1e-9:
                current = trial
                current_cost = trial_cost
                accepted = True
                break
        if not accepted:
            break
    for group in current:
        group.sort()
    return current


def _signature(groups: list[list[int]]) -> tuple:
    return tuple(tuple(group) for group in groups)


def optimize(
    values: np.ndarray,
    targets: list[tuple[float, float]],
    group_size: int,
    restarts: int,
    seed: int,
    mean_weight: float,
    std_weight: float,
) -> tuple[list[list[int]], str, float]:
    values = np.asarray(values, dtype=np.float64)
    n_groups = len(targets)
    if group_size < 1:
        raise ValueError("每組至少要 1 筆。")
    if values.size < n_groups * group_size:
        raise ValueError(
            f"可用資料 {values.size} 筆，不足 {n_groups} 組 × {group_size} 筆。"
        )
    if group_size == 1:
        groups, _total = _hungarian(values, targets, mean_weight, std_weight)
        for group in groups:
            group.sort()
        return groups, "hungarian", _exact_total(groups, values, targets, mean_weight, std_weight)
    if n_groups == 1 and combination_count(int(values.size), group_size) <= 200_000:
        groups, total = _exact_one_group(values, targets[0], group_size, mean_weight, std_weight)
        return groups, "exact", total

    rng = np.random.default_rng(seed)
    orders = [list(range(n_groups))]
    by_std = sorted(range(n_groups), key=lambda index: (-targets[index][1], targets[index][0], index))
    if by_std != orders[0]:
        orders.append(by_std)
    best_groups: list[list[int]] | None = None
    best_cost = None
    best_signature = None
    attempts = max(int(restarts), 1)
    for attempt in range(attempts):
        if attempt < len(orders):
            start = _greedy_init(values, targets, group_size, orders[attempt])
        else:
            start = _random_init(int(values.size), n_groups, group_size, rng)
        polished = polish(start, values, targets, mean_weight, std_weight)
        cost = _exact_total(polished, values, targets, mean_weight, std_weight)
        signature = _signature(polished)
        if best_cost is None or cost < best_cost - 1e-9 or (abs(cost - best_cost) <= 1e-9 and (best_signature is None or signature < best_signature)):
            best_cost = cost
            best_groups = polished
            best_signature = signature
    assert best_groups is not None and best_cost is not None
    return best_groups, "local", best_cost


def _usable(rows: list[dict[str, str]], point: str) -> tuple[np.ndarray, list[dict]]:
    values: list[float] = []
    kept: list[dict] = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get(point, "")).strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if not np.isfinite(value):
            continue
        item = dict(row)
        item["index"] = str(index)
        values.append(value)
        kept.append(item)
    return np.asarray(values, dtype=np.float64), kept


def _format(value: float) -> str:
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def write_reports(
    summary_path: Path,
    long_path: Path,
    reports: list[dict],
) -> None:
    summary_fields = [
        "point",
        "group",
        "indices",
        "n",
        "mean",
        "std",
        "target_mean",
        "target_std",
        "mean_error",
        "std_error",
        "cost",
        "point_cost",
        "method",
    ]
    meta = ["index", "count", "loop", "cycle", "session"]
    long_fields = ["point", "group", *meta, "value", "mean", "std", "target_mean", "target_std"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            for row in report["summary"]:
                writer.writerow(row)
    with long_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields, extrasaction="ignore", restval="")
        writer.writeheader()
        for report in reports:
            for row in report["long"]:
                writer.writerow(row)


def match_point(
    point: str,
    rows: list[dict[str, str]],
    targets: list[tuple[float, float]],
    group_size: int,
    restarts: int,
    seed: int,
    mean_weight: float,
    std_weight: float,
) -> dict:
    values, kept = _usable(rows, point)
    groups, method, point_cost = optimize(
        values,
        targets,
        group_size,
        restarts,
        seed,
        mean_weight,
        std_weight,
    )
    summary = []
    long_rows = []
    for group_index, (members, target) in enumerate(zip(groups, targets), start=1):
        chosen = [kept[index] for index in members]
        data = values[np.asarray(members, dtype=int)]
        mean, std = sample_mean_std(data)
        target_mean, target_std = target
        summary.append(
            {
                "point": point,
                "group": str(group_index),
                "indices": ";".join(item["index"] for item in chosen),
                "n": str(len(chosen)),
                "mean": _format(mean),
                "std": _format(std),
                "target_mean": _format(target_mean),
                "target_std": _format(target_std),
                "mean_error": _format(mean - target_mean),
                "std_error": _format(std - target_std),
                "cost": _format(group_cost(data, target, mean_weight, std_weight)),
                "point_cost": _format(point_cost),
                "method": method,
            }
        )
        for item, value in zip(chosen, data):
            long_rows.append(
                {
                    "point": point,
                    "group": str(group_index),
                    "index": item["index"],
                    "count": item.get("count", ""),
                    "loop": item.get("loop", ""),
                    "cycle": item.get("cycle", ""),
                    "session": item.get("session", ""),
                    "value": _format(float(value)),
                    "mean": _format(mean),
                    "std": _format(std),
                    "target_mean": _format(target_mean),
                    "target_std": _format(target_std),
                }
            )
    return {"point": point, "summary": summary, "long": long_rows, "cost": point_cost, "method": method}


def self_test() -> None:
    values = np.array([0, 1, 2, 10, 11, 12, 20, 21, 22], dtype=np.float64)
    targets = [(1.0, 1.0), (11.0, 1.0), (21.0, 1.0)]
    groups, method, cost = optimize(values, targets, 3, restarts=4, seed=1, mean_weight=1, std_weight=1)
    if cost > 1e-8 or [sorted(group) for group in groups] != [[0, 1, 2], [3, 4, 5], [6, 7, 8]]:
        raise SystemExit(f"分組自我測試失敗：cost={cost} method={method} groups={groups}")
    singles = np.array([3.0, 1.0, 2.0])
    groups, method, cost = optimize(
        singles,
        [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        1,
        restarts=2,
        seed=1,
        mean_weight=1,
        std_weight=1,
    )
    if method != "hungarian" or cost > 1e-8 or groups != [[1], [2], [0]]:
        raise SystemExit(f"單筆分組自我測試失敗：cost={cost} method={method} groups={groups}")
    print("自我測試通過。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 index 不重複的條件下，找出最接近預期平均與標準差的分組。")
    parser.add_argument("csv", nargs="?", type=Path, help="hdr_batch_distance.py 寫出的距離 CSV")
    parser.add_argument("--group-size", type=int, default=None, help="同一點位取幾筆當成一組")
    parser.add_argument("--targets", type=Path, default=None, help="目標 CSV，欄位 mean,std，可加 point 與 group")
    parser.add_argument("--target", action="append", default=None, help="mean,std，可重複。組數等於重複次數")
    parser.add_argument("--point", action="append", default=None, help="只處理這些點位欄名，可重複")
    parser.add_argument("--output", type=Path, default=None, help="摘要 CSV。未指定時寫在距離 CSV 旁邊")
    parser.add_argument("--restarts", type=int, default=12, help="多起點次數。預設 12。每組 1 筆時不使用")
    parser.add_argument("--seed", type=int, default=1, help="隨機起點的種子。預設 1")
    parser.add_argument("--mean-weight", type=float, default=1.0, help="平均誤差的平方權重。預設 1")
    parser.add_argument("--std-weight", type=float, default=1.0, help="標準差誤差的平方權重。預設 1")
    parser.add_argument("--self-test", action="store_true", help="只跑內建的分組檢查，不讀 CSV")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.csv is None or args.group_size is None:
        parser.error("請提供距離 CSV 和 --group-size")
    if args.group_size < 1:
        parser.error("--group-size 至少為 1")
    if args.restarts < 1:
        parser.error("--restarts 至少為 1")
    if args.mean_weight < 0 or args.std_weight < 0:
        parser.error("權重不能小於 0")
    return args


def main(argv: list[str] | None = None) -> None:
    _stdio_utf8()
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    distance_path = args.csv.expanduser().resolve()
    points, rows = load_distance(distance_path)
    targets_by_point = load_targets(args.targets, args.target)
    selected = points if not args.point else args.point
    missing = [point for point in selected if point not in points]
    if missing:
        raise SystemExit("距離 CSV 沒有這些點位：" + "、".join(missing))
    if None not in targets_by_point:
        unknown = [point for point in targets_by_point if point not in points]
        if unknown:
            raise SystemExit("目標檔的點位不在距離 CSV：" + "、".join(str(point) for point in unknown))
        selected = [point for point in selected if point in targets_by_point]
    summary_path = (
        args.output.expanduser().resolve()
        if args.output
        else distance_path.with_name(distance_path.stem + "_groups.csv")
    )
    long_path = summary_path.with_name(summary_path.stem + "_long.csv")
    reports = []
    for point in selected:
        targets = targets_by_point.get(point, targets_by_point.get(None))
        if not targets:
            print(f"{point}：沒有目標，略過", flush=True)
            continue
        try:
            report = match_point(
                point,
                rows,
                targets,
                args.group_size,
                args.restarts,
                args.seed,
                args.mean_weight,
                args.std_weight,
            )
        except ValueError as exc:
            print(f"{point}：{exc}", flush=True)
            continue
        reports.append(report)
        print(
            f"{point}  {len(targets)} 組 × {args.group_size} 筆  "
            f"cost {report['cost']:.6g}  （{report['method']}）",
            flush=True,
        )
    if not reports:
        raise SystemExit("沒有點位完成分組。")
    write_reports(summary_path, long_path, reports)
    print(f"摘要 {summary_path}", flush=True)
    print(f"明細 {long_path}", flush=True)


if __name__ == "__main__":
    main()
