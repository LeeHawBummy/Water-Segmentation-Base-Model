#!/usr/bin/env python3
"""
统计标签里「前景」像素占比（按列表中的样本，逐张读 *_label.tif）。

默认二分类：前景 = 标签值 == 1；背景 = 0。可用 --foreground_values 指定多值前景并集。
可选 --ignore_index（如 255）：这些像素不参与分母（与 MMSeg ignore 一致）。

示例:
  python tools/stats_label_foreground_ratio.py \\
    --data_dir data/Synapse_all \\
    --list_dir lists/lists_Synapse_37 \\
    --splits train,val

  python tools/stats_label_foreground_ratio.py \\
    --data_dir data/Synapse_all \\
    --list_paths lists/a.txt,lists/b.txt \\
    --max_samples 2000 --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile

def _read_label_tiff(path: str) -> np.ndarray:
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        pass
    last = None
    for kwargs in ({}, {"maxworkers": 1}):
        try:
            arr = tifffile.imread(path, **kwargs)
            if arr is not None and getattr(arr, "size", 0) > 0:
                return arr
        except Exception as e:
            last = e
    try:
        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
            if arr is not None and getattr(arr, "size", 0) > 0:
                return arr
    except Exception as e:
        last = e
    if last is not None:
        raise last
    raise ValueError("unreadable: %s" % path)


def _read_stems(path: str) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].strip())
    return out


def _one_file(
    data_dir: str,
    stem: str,
    fg_set: set[int],
    ignore_index: int | None,
) -> tuple[str, float | None, int | None, int | None, str | None]:
    """返回 (stem, fg_ratio, n_fg, n_valid, err)。"""
    p = os.path.join(data_dir, "%s_label.tif" % stem)
    if not os.path.isfile(p):
        return stem, None, None, None, "missing"
    try:
        lab = _read_label_tiff(p)
        lab = np.squeeze(lab)
        if lab.ndim != 2:
            return stem, None, None, None, "bad_ndim"
        x = lab.astype(np.int64, copy=False).ravel()
        if ignore_index is not None:
            valid = x != ignore_index
        else:
            valid = np.ones_like(x, dtype=bool)
        x = x[valid]
        if x.size == 0:
            return stem, None, None, 0, "no_valid"
        fg = np.zeros_like(x, dtype=bool)
        for v in fg_set:
            fg |= x == v
        n_fg = int(fg.sum())
        n_valid = int(x.size)
        return stem, n_fg / n_valid, n_fg, n_valid, None
    except Exception as e:
        return stem, None, None, None, str(e)[:80]


def main():
    ap = argparse.ArgumentParser(description="统计标签前景像素占比")
    ap.add_argument("--data_dir", required=True, help="*_label.tif 所在目录")
    ap.add_argument("--list_dir", default=None, help="与 train.txt 等同级目录")
    ap.add_argument(
        "--splits",
        default="train,val",
        help="与 --list_dir 联用，逗号分隔，如 train,val,test",
    )
    ap.add_argument(
        "--list_paths",
        default=None,
        help="直接指定多个列表文件，逗号分隔（与 list_dir 二选一或合并）",
    )
    ap.add_argument(
        "--foreground_values",
        default="1",
        help="前景标签取值，逗号分隔，如 1 或 1,2,3",
    )
    ap.add_argument("--ignore_index", type=int, default=None, help="不计入分母的像素值，如 255")
    ap.add_argument("--max_samples", type=int, default=0, help="最多统计多少条（0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isdir(data_dir):
        raise SystemExit("data_dir 不存在: %s" % data_dir)

    fg_set = set(int(x.strip()) for x in args.foreground_values.split(",") if x.strip())

    list_files = []
    if args.list_paths:
        for p in args.list_paths.split(","):
            p = p.strip()
            if p:
                list_files.append(os.path.abspath(os.path.expanduser(p)))
    if args.list_dir:
        ld = os.path.abspath(os.path.expanduser(args.list_dir))
        for sp in args.splits.split(","):
            sp = sp.strip()
            if not sp:
                continue
            fp = os.path.join(ld, "%s.txt" % sp)
            if os.path.isfile(fp):
                list_files.append(fp)

    if not list_files:
        raise SystemExit("请提供 --list_dir 或 --list_paths")

    stems: list[str] = []
    seen = set()
    for fp in list_files:
        for s in _read_stems(fp):
            if s not in seen:
                seen.add(s)
                stems.append(s)

    if args.max_samples > 0 and len(stems) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(stems), size=args.max_samples, replace=False)
        stems = [stems[i] for i in sorted(idx)]

    print("data_dir: %s" % data_dir)
    print("列表文件数: %d, 唯一样本数: %d" % (len(list_files), len(stems)))
    print("前景取值: %s, ignore_index: %s" % (sorted(fg_set), args.ignore_index))

    ratios = []
    total_fg = 0
    total_valid = 0
    n_err = 0

    if args.workers <= 1:
        for st in stems:
            _, r, nf, nv, err = _one_file(data_dir, st, fg_set, args.ignore_index)
            if err:
                n_err += 1
                continue
            ratios.append(r)
            total_fg += nf
            total_valid += nv
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(_one_file, data_dir, st, fg_set, args.ignore_index): st
                for st in stems
            }
            for f in as_completed(futs):
                _, r, nf, nv, err = f.result()
                if err:
                    n_err += 1
                    continue
                ratios.append(r)
                total_fg += nf
                total_valid += nv

    if not ratios:
        print("无有效样本（错误数: %d）" % n_err)
        sys.exit(1)

    arr = np.array(ratios, dtype=np.float64)
    print("-" * 50)
    print("按像素全局合并: 前景 %d / 有效 %d = %.4f%%" % (
        total_fg, total_valid, 100.0 * total_fg / max(total_valid, 1)))
    print("按图像平均占比:  mean=%.4f%%  median=%.4f%%  std=%.4f%%" % (
        100.0 * arr.mean(), 100.0 * np.median(arr), 100.0 * arr.std()))
    print("  min=%.4f%%  max=%.4f%%  p25=%.4f%%  p75=%.4f%%" % (
        100.0 * arr.min(), 100.0 * arr.max(),
        100.0 * np.percentile(arr, 25), 100.0 * np.percentile(arr, 75)))
    print("成功: %d, 失败/跳过: %d" % (len(ratios), n_err))


if __name__ == "__main__":
    main()
