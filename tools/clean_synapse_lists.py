#!/usr/bin/env python3
"""
彻底清洗 Synapse 列表：扫描 data_dir 下 train.txt / val.txt（可选 test.txt）中的 stem，
校验 {stem}_img.tif 与 {stem}_label.tif 可读、数值有限、空间尺寸一致、标签类别合法，
剔除坏样本后写回列表（默认先备份）。

用法:
  cd /path/to/Swin-Unet-main
  python tools/clean_synapse_lists.py \\
    --data_dir data/Synapse_all \\
    --list_dir lists/lists_Synapse_37 \\
    --target_channels 3 \\
    --num_classes 2

  # 只统计、不写文件
  python tools/clean_synapse_lists.py --data_dir ... --list_dir ... --dry_run

  # 安装 imagecodecs 可减少 JPEG 瓦片 TIFF 误杀: pip install imagecodecs
"""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import tifffile

# 与 datasets/dataset_synapse 一致，避免本工具依赖 torch
def _imread_tiff_robust(path):
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
    raise ValueError("empty or unreadable tiff: %s" % path)


def _check_finite_array(name, arr):
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        raise ValueError("%s 为空" % name)
    if arr.dtype.kind in "fc":
        if not np.isfinite(arr).all():
            raise ValueError("%s 含 NaN/Inf" % name)


def _normalize_image_to_chw(image: np.ndarray, nch: int) -> tuple[np.ndarray, str | None]:
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
    if len(image.shape) != 3:
        return image, "image 须为 2D 或 3D，当前 %s" % (image.shape,)
    if image.shape[-1] <= 64 and image.shape[-1] != image.shape[0]:
        image = image.transpose(2, 0, 1)
    c, h, w = image.shape
    if c != nch:
        if c < nch:
            pad = np.zeros((nch - c, h, w), dtype=image.dtype)
            image = np.concatenate([image, pad], axis=0)
        else:
            image = image[:nch, :, :]
    return image, None


def validate_stem(
    data_dir: str,
    stem: str,
    target_channels: int | None,
    num_classes: int | None,
    ignore_index: int | None,
) -> tuple[bool, str]:
    """返回 (是否保留, 原因说明)。"""
    img_p = os.path.join(data_dir, "%s_img.tif" % stem)
    lab_p = os.path.join(data_dir, "%s_label.tif" % stem)
    if not os.path.isfile(img_p):
        return False, "缺 img"
    if not os.path.isfile(lab_p):
        return False, "缺 label"
    try:
        image = _imread_tiff_robust(img_p)
        label = _imread_tiff_robust(lab_p)
    except Exception as e:
        return False, "读取失败: %s" % (str(e)[:120],)

    label = np.squeeze(label)
    if label.ndim != 2:
        return False, "label 非 HxW: %s" % (label.shape,)

    try:
        _check_finite_array("image", image)
        _check_finite_array("label", label.astype(np.float32, copy=False))
    except ValueError as e:
        return False, str(e)

    nch = target_channels if target_channels is not None else 6
    image, err = _normalize_image_to_chw(image, nch)
    if err:
        return False, err

    _, ih, iw = image.shape
    if (ih, iw) != tuple(label.shape):
        return False, "HW 不一致 img(%d,%d) lab%s" % (ih, iw, label.shape)

    if num_classes is not None:
        u = np.unique(label)
        for v in u:
            vi = int(v)
            if ignore_index is not None and vi == ignore_index:
                continue
            if vi < 0 or vi >= num_classes:
                return False, "标签越界 %d 不在 [0,%d)" % (vi, num_classes)

    return True, ""


def _read_stems(list_path: str) -> list[str]:
    stems = []
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            stems.append(s.split()[0].strip())
    return stems


def _backup_list(list_path: str, tag: str) -> str | None:
    if not os.path.isfile(list_path):
        return None
    dst = "%s.bak.%s" % (list_path, tag)
    shutil.copy2(list_path, dst)
    return dst


def _clean_one_file(
    data_dir: str,
    list_path: str,
    target_channels: int | None,
    num_classes: int | None,
    ignore_index: int | None,
    workers: int,
    dry_run: bool,
    tag: str,
) -> dict:
    stems = _read_stems(list_path)
    if not stems:
        return {"file": list_path, "total": 0, "kept": 0, "dropped": 0, "errors": []}

    bad: list[tuple[str, str]] = []

    def task(st: str):
        ok, msg = validate_stem(data_dir, st, target_channels, num_classes, ignore_index)
        return st, ok, msg

    if workers <= 1:
        results = [task(st) for st in stems]
    else:
        results = [None] * len(stems)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(task, st): i for i, st in enumerate(stems)}
            for f in as_completed(fut):
                i = fut[f]
                results[i] = f.result()
        results = list(results)

    kept = []
    for st, ok, msg in results:
        if ok:
            kept.append(st)
        else:
            bad.append((st, msg))

    out = {
        "file": list_path,
        "total": len(stems),
        "kept": len(kept),
        "dropped": len(bad),
        "errors": bad,
    }

    if dry_run:
        return out

    bak = _backup_list(list_path, tag)
    if bak:
        print("  已备份: %s -> %s" % (list_path, os.path.basename(bak)))
    with open(list_path, "w", encoding="utf-8") as f:
        for st in kept:
            f.write(st + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description="清洗 Synapse train/val/test 列表（剔除不可读或非法样本）")
    ap.add_argument("--data_dir", required=True, help="含 *_img.tif / *_label.tif 的目录（与训练 --single_dir 一致）")
    ap.add_argument("--list_dir", required=True, help="train.txt / val.txt 所在目录")
    ap.add_argument("--target_channels", type=int, default=None, help="与 MODEL.SWIN.IN_CHANS 一致；默认 6")
    ap.add_argument("--num_classes", type=int, default=None, help="若指定则检查标签是否在 [0,n)（可配合 --ignore_index）")
    ap.add_argument("--ignore_index", type=int, default=None, help="跳过的标签值（如 255）")
    ap.add_argument("--splits", type=str, default="train,val", help="逗号分隔: train,val,test")
    ap.add_argument("--workers", type=int, default=8, help="并行线程数（I/O 为主）")
    ap.add_argument("--dry_run", action="store_true", help="只统计与打印坏样本，不写回列表")
    ap.add_argument(
        "--bad_log",
        type=str,
        default=None,
        help="将剔除的 stem 与原因写入该文件（默认 list_dir/rejected_stems_<timestamp>.txt）",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    list_dir = os.path.abspath(os.path.expanduser(args.list_dir))
    if not os.path.isdir(data_dir):
        raise SystemExit("data_dir 不存在: %s" % data_dir)
    if not os.path.isdir(list_dir):
        raise SystemExit("list_dir 不存在: %s" % list_dir)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_bad: list[tuple[str, str, str]] = []

    summary = []
    for sp in splits:
        name = "%s.txt" % sp
        lp = os.path.join(list_dir, name)
        if not os.path.isfile(lp):
            print("跳过（无文件）: %s" % lp)
            continue
        print("处理 %s (%d 条) ..." % (name, len(_read_stems(lp))))
        r = _clean_one_file(
            data_dir,
            lp,
            args.target_channels,
            args.num_classes,
            args.ignore_index,
            args.workers,
            args.dry_run,
            tag,
        )
        summary.append(r)
        for st, msg in r["errors"]:
            all_bad.append((sp, st, msg))
        print(
            "  -> 保留 %d / %d，剔除 %d"
            % (r["kept"], r["total"], r["dropped"])
        )

    if args.bad_log:
        bad_path = os.path.abspath(os.path.expanduser(args.bad_log))
    else:
        bad_path = os.path.join(list_dir, "rejected_stems_%s.txt" % tag)

    if all_bad and not args.dry_run:
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("# split\tstem\treason\n")
            for sp, st, msg in all_bad:
                f.write("%s\t%s\t%s\n" % (sp, st, msg))
        print("剔除明细: %s" % bad_path)
    elif all_bad and args.dry_run:
        print("\n--dry_run：以下为首条剔除示例（完整请去重后自行保存）:")
        for line in all_bad[:20]:
            print("  ", line)
        if len(all_bad) > 20:
            print("  ... 共 %d 条" % len(all_bad))

    if args.dry_run:
        print("\n未写回任何列表。去掉 --dry_run 后执行将备份并覆盖 train.txt / val.txt。")


if __name__ == "__main__":
    main()
