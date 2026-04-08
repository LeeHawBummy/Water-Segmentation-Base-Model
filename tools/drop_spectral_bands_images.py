#!/usr/bin/env python3
"""
从多波段影像（GeoTIFF）中删除指定波段，原地覆盖写回（先写临时文件再替换）。

默认按「第 1 波段 = 索引 0」；--bands 6,7,8 表示删除第 6、7、8 波段（0-based 即 5,6,7）。

支持 (H,W,C) 与 (C,H,W)（C<=64 且 C 在最后一维时视为 HWC）。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile


def _to_chw(arr: np.ndarray) -> tuple[np.ndarray, str]:
    if arr.ndim != 3:
        raise ValueError("须为 3 维，当前 shape=%s" % (arr.shape,))
    h, w, c = arr.shape[-2], arr.shape[-1], None
    if arr.shape[-1] <= 64 and arr.shape[-1] != arr.shape[0]:
        return arr.transpose(2, 0, 1), "hwc"
    if arr.shape[0] <= 64:
        return arr, "chw"
    raise ValueError("无法判断通道维: shape=%s" % (arr.shape,))


def _from_chw(arr: np.ndarray, order: str) -> np.ndarray:
    if order == "hwc":
        return arr.transpose(1, 2, 0)
    return arr


def _drop_bands_chw(chw: np.ndarray, drop_idx: list[int]) -> np.ndarray:
    c = chw.shape[0]
    for i in sorted(drop_idx, reverse=True):
        if i < 0 or i >= c:
            raise ValueError("波段索引越界: %d 不在 [0,%d)" % (i, c))
    keep = [i for i in range(c) if i not in set(drop_idx)]
    return chw[keep, :, :]


def process_one(path: str, bands_one_based: list[int]) -> tuple[str, str | None]:
    drop0 = [b - 1 for b in bands_one_based]
    try:
        arr = tifffile.imread(path)
        chw, order = _to_chw(np.asarray(arr))
        out_chw = _drop_bands_chw(chw, drop0)
        out = _from_chw(out_chw, order)
        d = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(suffix=".tif", dir=d)
        os.close(fd)
        try:
            tifffile.imwrite(tmp, out.astype(arr.dtype, copy=False))
            os.replace(tmp, path)
        except Exception:
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise
        return path, None
    except Exception as e:
        return path, str(e)[:200]


def main():
    ap = argparse.ArgumentParser(description="从 images 目录的 TIFF 中删除指定波段（原地）")
    ap.add_argument("--images_dir", required=True, help="含 .tif/.tiff 的目录")
    ap.add_argument(
        "--bands",
        type=str,
        default="6,7,8",
        help="要删除的波段编号（从 1 开始），逗号分隔，默认 6,7,8",
    )
    ap.add_argument("--workers", type=int, default=8, help="并行进程数")
    ap.add_argument("--dry_run", action="store_true", help="只打印将处理的文件数")
    args = ap.parse_args()

    bands = [int(x.strip()) for x in args.bands.split(",") if x.strip()]
    if any(b < 1 for b in bands):
        raise SystemExit("--bands 须为从 1 开始的波段编号")

    img_dir = os.path.abspath(os.path.expanduser(args.images_dir))
    if not os.path.isdir(img_dir):
        raise SystemExit("目录不存在: %s" % img_dir)

    paths = []
    for name in sorted(os.listdir(img_dir)):
        low = name.lower()
        if low.endswith(".tif") or low.endswith(".tiff"):
            paths.append(os.path.join(img_dir, name))

    if not paths:
        raise SystemExit("未找到 tif/tiff: %s" % img_dir)

    print("目录: %s" % img_dir)
    print("删除第 %s 波段（1-based），共 %d 个文件" % (bands, len(paths)))
    if args.dry_run:
        return

    bad = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_one, p, bands): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            p, err = f.result()
            if err:
                bad.append((p, err))
            if i % 2000 == 0 or i == len(paths):
                print("  已处理 %d / %d" % (i, len(paths)))

    if bad:
        print("失败 %d 个，示例:" % len(bad))
        for p, e in bad[:15]:
            print(" ", p, e)
        sys.exit(1)
    print("全部完成。")


if __name__ == "__main__":
    main()
