#!/usr/bin/env python3
"""
批量推理：对一批大小、格式不一的图片做分割预测。
- 支持格式：.tif, .tiff, .png, .jpg, .jpeg
- 自动缩放到模型输入尺寸，推理后再还原为原图尺寸保存。
- 输出为单通道 PNG（0=背景，255=前景），与输入同名 + _pred.png，或指定 --out_dir。

用法示例:
  # 三通道模型，输入目录，输出到 predictions_batch
  python tools/batch_inference.py \\
    --cfg configs/swin_large_patch4_window7_224_water.yaml \\
    --checkpoint output_water_large_mmseg40k3ch/best_model.pth \\
    --input_dir /path/to/images \\
    --output_dir predictions_batch \\
    --opts MODEL.SWIN.IN_CHANS 3

  # 六通道模型
  python tools/batch_inference.py \\
    --cfg configs/swin_large_patch4_window7_224_water.yaml \\
    --checkpoint output_water_large_mmseg40k/best_model.pth \\
    --input_dir /path/to/images \\
    --output_dir predictions_batch
"""
import os
import sys
import argparse
import numpy as np
import torch

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 支持的图片后缀
IMG_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


def load_image(path, in_chans, dtype=np.float32):
    """加载一张图，返回 (C, H, W)，通道数 = in_chans（不足则填充，多则截断）。"""
    path = os.path.abspath(path)
    low = path.lower()
    if low.endswith(('.tif', '.tiff')):
        import tifffile
        img = tifffile.imread(path)
    else:
        try:
            from PIL import Image
            img = np.array(Image.open(path))
        except Exception:
            import cv2
            img = cv2.imread(path)
            if img is None:
                raise IOError("无法读取: %s" % path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 统一 (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        if img.shape[-1] <= 64 and img.shape[-1] != img.shape[0]:
            img = np.transpose(img, (2, 0, 1))
    if img.shape[0] > in_chans:
        img = img[:in_chans]
    elif img.shape[0] < in_chans:
        pad = np.zeros((in_chans - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
        img = np.concatenate([img, pad], axis=0)
    return img.astype(dtype)


def collect_images(input_dir, recursive=False):
    """收集目录下所有支持格式的图片路径。"""
    input_dir = os.path.abspath(input_dir)
    paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(IMG_SUFFIXES):
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            if f.lower().endswith(IMG_SUFFIXES):
                paths.append(os.path.join(input_dir, f))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="批量推理：大小格式不一的图片")
    parser.add_argument("--cfg", type=str, required=True, help="config yaml（与训练一致）")
    parser.add_argument("--checkpoint", type=str, required=True, help="权重 best_model.pth")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, default="predictions_batch", help="预测图保存目录")
    parser.add_argument("--img_size", type=int, default=224, help="模型输入尺寸")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数")
    parser.add_argument("--threshold", type=float, default=None, help="二分类前景阈值，不设则 argmax")
    parser.add_argument("--recursive", action="store_true", help="递归子目录")
    parser.add_argument("--opts", nargs="+", default=None, help="覆盖配置，如 MODEL.SWIN.IN_CHANS 3")
    parser.add_argument("--ext", type=str, default="_pred.png", help="输出文件名后缀，默认 _pred.png")
    args = parser.parse_args()
    if not os.path.isabs(args.cfg):
        args.cfg = os.path.join(ROOT, args.cfg)
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(ROOT, args.checkpoint)

    # 加载 config 与模型（get_config 需要 args 带 cfg/opts 等属性）
    from config import get_config
    _args = argparse.Namespace(
        cfg=args.cfg, opts=args.opts,
        batch_size=None, zip=False, cache_mode=None, resume=None,
        accumulation_steps=None, use_checkpoint=False, amp_opt_level=None,
        tag=None, eval=False, throughput=False,
    )
    config = get_config(_args)
    in_chans = getattr(config.MODEL.SWIN, 'IN_CHANS', 6)

    from networks.vision_transformer import SwinUnet as ViT_seg
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    ckpt = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    net.load_state_dict(ckpt, strict=True)
    net.eval()

    from scipy.ndimage import zoom as scipy_zoom
    os.makedirs(args.output_dir, exist_ok=True)
    paths = collect_images(args.input_dir, recursive=args.recursive)
    if not paths:
        print("未找到图片，请检查 --input_dir 及格式（支持 %s）" % str(IMG_SUFFIXES))
        return 1
    print("共 %d 张，输出目录: %s" % (len(paths), args.output_dir))
    for i, ip in enumerate(paths):
        try:
            img = load_image(ip, in_chans)
            C, H, W = img.shape
            need_resize = (H != args.img_size or W != args.img_size)
            if need_resize:
                scale_h, scale_w = args.img_size / H, args.img_size / W
                img_224 = np.zeros((C, args.img_size, args.img_size), dtype=img.dtype)
                for c in range(C):
                    img_224[c] = scipy_zoom(img[c], (scale_h, scale_w), order=3)
            else:
                img_224 = img
            x = torch.from_numpy(img_224[np.newaxis, ...].astype(np.float32)).cuda()
            with torch.no_grad():
                logits = net(x)
                probs = torch.softmax(logits, dim=1)
            # 在原图尺寸上做插值再二值化，可减轻锯齿（先放大概率图再 argmax/阈值）
            probs_np = probs[0].cpu().numpy()  # (num_classes, 224, 224)
            if need_resize:
                # 用双线性插值放大概率图到 (num_classes, H, W)，再在原尺寸上 argmax/阈值，边界更平滑
                scale_h, scale_w = H / args.img_size, W / args.img_size
                probs_hr = np.zeros((probs_np.shape[0], H, W), dtype=np.float32)
                for c in range(probs_np.shape[0]):
                    probs_hr[c] = scipy_zoom(probs_np[c], (scale_h, scale_w), order=1)
                if args.num_classes == 2 and args.threshold is not None:
                    pred = (probs_hr[1] >= args.threshold).astype(np.uint8)
                else:
                    pred = np.argmax(probs_hr, axis=0).astype(np.uint8)
            else:
                if args.num_classes == 2 and args.threshold is not None:
                    pred = (probs_np[1] >= args.threshold).astype(np.uint8)
                else:
                    pred = np.argmax(probs_np, axis=0).astype(np.uint8)
            # 保存为 0/255 的单通道 PNG
            out_name = os.path.splitext(os.path.basename(ip))[0] + args.ext
            out_path = os.path.join(args.output_dir, out_name)
            pred_255 = (pred * 255) if pred.max() <= 1 else pred
            from PIL import Image
            Image.fromarray(pred_255).save(out_path)
            if (i + 1) % 50 == 0 or i == 0:
                print("  %d/%d %s -> %s" % (i + 1, len(paths), os.path.basename(ip), out_name))
        except Exception as e:
            print("  [失败] %s: %s" % (ip, e))
    print("完成.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
