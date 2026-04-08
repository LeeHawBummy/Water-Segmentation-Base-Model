#!/usr/bin/env python3
"""
将 Swin-Unet 的 data 另存为三通道版本：
  原数据的 第3、第2、第1 通道 → 新数据的 第1、第2、第3 通道
（即新 ch1=原 ch3，新 ch2=原 ch2，新 ch3=原 ch1；0-based 为 new[0]=old[2], new[1]=old[1], new[2]=old[0]）

用法:
  python tools/convert_to_3ch.py --src_root data/Synapse --out_root data/Synapse_3ch
  python tools/convert_to_3ch.py --src_root data/Synapse --out_root data/Synapse_3ch --splits train_tif test_vol_tif
"""
import os
import argparse
import shutil
import numpy as np
import tifffile


def ensure_cwh(img):
    """确保为 (C, H, W)。"""
    if len(img.shape) != 3:
        return None
    if img.shape[-1] <= 64 and img.shape[-1] != img.shape[0]:
        img = img.transpose(2, 0, 1)
    return img


def convert_one_img(src_path, out_path, channels_old_1based=(3, 2, 1)):
    """
    读取多通道 TIFF，取原第 3,2,1 通道作为新第 1,2,3 通道并保存。
    channels_old_1based: 原图 1-based 通道号顺序，对应新图 ch1,ch2,ch3。
    若文件损坏或读取失败，返回 (False, 错误信息)，不抛异常。
    """
    try:
        img = tifffile.imread(src_path)
    except (tifffile.TiffFileError, OSError, IOError, ValueError) as e:
        return False, "读取失败: %s" % (str(e)[:80])
    img = ensure_cwh(img)
    if img is None:
        return False, "shape not 3d"
    c, h, w = img.shape
    # 1-based -> 0-based index
    idx = [int(x) - 1 for x in channels_old_1based]
    if any(i < 0 or i >= c for i in idx):
        return False, "channel index out of range (max %d)" % c
    try:
        out = np.stack([img[idx[0]], img[idx[1]], img[idx[2]]], axis=0)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tifffile.imwrite(out_path, out, photometric='minisblack')
    except Exception as e:
        return False, "写入失败: %s" % (str(e)[:80])
    return True, None


def main():
    parser = argparse.ArgumentParser(description="多通道转三通道：原 3,2,1 → 新 1,2,3")
    parser.add_argument("--src_root", type=str, default="data/Synapse", help="源数据根目录，下有 train_tif、test_vol_tif 等")
    parser.add_argument("--out_root", type=str, default="data/Synapse_3ch", help="输出根目录")
    parser.add_argument("--splits", type=str, nargs="+", default=["train_tif", "test_vol_tif"], help="要转换的子目录名")
    parser.add_argument("--channels", type=int, nargs=3, default=[3, 2, 1], help="原图 1-based 通道号，依次对应新图 ch1,ch2,ch3")
    args = parser.parse_args()
    src_root = os.path.abspath(args.src_root)
    out_root = os.path.abspath(args.out_root)
    if not os.path.isdir(src_root):
        print("错误: 源目录不存在:", src_root)
        return 1
    channels = tuple(args.channels)
    for split in args.splits:
        src_dir = os.path.join(src_root, split)
        out_dir = os.path.join(out_root, split)
        if not os.path.isdir(src_dir):
            print("跳过（不存在）:", src_dir)
            continue
        os.makedirs(out_dir, exist_ok=True)
        names = [f[:-len("_img.tif")] for f in os.listdir(src_dir) if f.endswith("_img.tif")]
        n_ok, n_fail = 0, 0
        for base in names:
            src_img = os.path.join(src_dir, base + "_img.tif")
            src_lbl = os.path.join(src_dir, base + "_label.tif")
            out_img = os.path.join(out_dir, base + "_img.tif")
            out_lbl = os.path.join(out_dir, base + "_label.tif")
            if not os.path.isfile(src_img):
                n_fail += 1
                continue
            ok, err = convert_one_img(src_img, out_img, channels_old_1based=channels)
            if not ok:
                print("  [%s] 图像转换失败: %s" % (base, err))
                n_fail += 1
                continue
            n_ok += 1
            if os.path.isfile(src_lbl):
                shutil.copy2(src_lbl, out_lbl)
        print("[%s] 图像: %d 成功, %d 失败; 标签已复制" % (split, n_ok, n_fail))
    print("输出目录:", out_root)
    return 0


if __name__ == "__main__":
    exit(main())
