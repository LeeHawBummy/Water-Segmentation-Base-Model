from os.path import split
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_synapse import Synapse_dataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import test_single_volume, calculate_segmentation_metrics, get_confusion_matrix, calculate_global_metrics_from_cm
from visualization import SegmentationVisualizer, visualize_training_results

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/water',
                    help='数据根目录；默认优先按“单目录”推理（目录内直接包含 tif/png/jpg 等）；若未设 --volume_path 且非单目录，则用 root_path/split_name')
parser.add_argument('--single_dir', action='store_true',
                    help='单目录 + 列表：推理数据与列表在同一逻辑下，不再按默认子目录名拼接')
parser.add_argument('--dataset', type=str,
                    default='water', help='数据集名称（兼容 Synapse；建议用 water）')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_water', help='list dir（默认 lists/lists_water）')
parser.add_argument('--output_dir', type=str, help='output dir (used to look for best_model.pth if --checkpoint not set)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to checkpoint .pth (e.g. ./model_out/xxx/best_model.pth); if set, overrides output_dir/best_model.pth')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--split_name", default="test_vol_h5", help="root_path 下的数据子目录名（未设 --volume_path 时生效）")
parser.add_argument("--volume_path", type=str, default=None,
                    help="direct path to test data (e.g. data/water/test_vol_tif); if set, overrides root_path/split_name")
parser.add_argument("--threshold", type=float, default=None,
                    help="二分类时前景类判定阈值，>0.5 可减少假阳性、提高 Precision/Dice（如 0.55 或 0.6）；不设则用 argmax")
parser.add_argument("--visualize", action="store_true", help="Enable visualization of segmentation results")
parser.add_argument("--vis_save_dir", type=str, default="./visualization_results", help="Directory to save visualization results")
parser.add_argument("--vis_num_samples", type=int, default=10, help="Number of samples to visualize")

args = parser.parse_args()

# 兼容旧命名：Synapse / water
if isinstance(args.dataset, str) and args.dataset.lower() == "synapse":
    args.dataset = "water"

# 设置 volume_path：若命令行传入了 --volume_path 则直接使用，否则自动推导（默认支持单目录）
if args.volume_path is None:
    rp = os.path.abspath(os.path.expanduser(args.root_path))

    def _looks_like_single_dir(p: str) -> bool:
        try:
            if not os.path.isdir(p):
                return False
            for name in os.listdir(p):
                low = name.lower()
                if low.endswith(("_img.tif", "_label.tif", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
                    return True
        except Exception:
            return False
        return False

    if args.single_dir or _looks_like_single_dir(rp):
        args.volume_path = rp
    else:
        args.volume_path = os.path.join(rp, args.split_name)
config = get_config(args)


def _label_to_int_hw(label_np):
    """标签转为整型 HxW，避免 float 与混淆矩阵边界问题。"""
    a = np.asarray(label_np)
    a = np.squeeze(a)
    if a.dtype.kind in "fc":
        a = np.rint(a).astype(np.int64)
    else:
        a = a.astype(np.int64, copy=False)
    return a


def _align_pred_to_label(prediction, th, tw, num_classes):
    """将网络输出的 2D 预测缩放到与标签一致 (th,tw)，order=0。"""
    from scipy.ndimage import zoom

    prediction = np.asarray(prediction)
    ph, pw = int(prediction.shape[0]), int(prediction.shape[1])
    if ph == th and pw == tw:
        return prediction.astype(np.int64, copy=False)
    zh, zw = th / ph, tw / pw
    out = zoom(prediction.astype(np.float64), (zh, zw), order=0)
    out = np.rint(out).astype(np.int64)
    if out.shape[0] != th or out.shape[1] != tw:
        tmp = np.zeros((th, tw), dtype=np.int64)
        mh, mw = min(th, out.shape[0]), min(tw, out.shape[1])
        tmp[:mh, :mw] = out[:mh, :mw]
        out = tmp
    hi = max(0, int(num_classes) - 1)
    return np.clip(out, 0, hi)


def _log_inference_diag(i_batch, probs, pred_small, label_int, args, case_name):
    """首个样本：打印 softmax 与前景占比，便于判断「全 0 预测」或阈值过高。"""
    if i_batch != 0:
        return
    gt_fg = float((label_int == 1).mean()) if args.num_classes >= 2 else float("nan")
    pr_fg = float((pred_small == 1).mean()) if args.num_classes >= 2 else float("nan")
    if args.num_classes == 2:
        m1 = float(probs[0, 1].mean().cpu().item())
        m0 = float(probs[0, 0].mean().cpu().item())
    else:
        m0 = m1 = float("nan")
    thr = getattr(args, "threshold", None)
    logging.info(
        "[diag] case=%s | GT 前景占比=%.4f | 输入分辨率下 pred 前景占比=%.4f | "
        "mean softmax cls0=%.4f cls1=%.4f | threshold=%s",
        case_name,
        gt_fg,
        pr_fg,
        m0,
        m1,
        str(thr),
    )
    if args.num_classes == 2 and gt_fg > 1e-6 and pr_fg < 1e-6:
        logging.info(
            "[diag] GT 含前景但预测几乎全为背景：请核对 --cfg 与训练时 IN_CHANS 是否一致、"
            "--checkpoint 是否为该实验的 best_model、--n_class 是否为 2；"
            "若设了 --threshold，过高会导致全判为背景。"
        )


def inference(args, model, test_save_path=None):
    in_chans = getattr(args, 'in_chans', None)
    db_test = Synapse_dataset(
        base_dir=args.volume_path,
        split=args.split_name,
        list_dir=args.list_dir,
        img_size=args.img_size,
        target_channels=in_chans,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    
    # 初始化可视化器
    visualizer = None
    if args.visualize:
        visualizer = SegmentationVisualizer(num_classes=args.num_classes)
        os.makedirs(args.vis_save_dir, exist_ok=True)
    
    # 用于累积总体指标（每张图算一次，再取平均）
    all_metrics = {
        'overall_accuracy': [],
        'mean_precision': [],
        'mean_recall': [],
        'mean_f1': [],
        'mean_iou': [],
        'mean_dice': []
    }
    # 按像素聚合的混淆矩阵（不忽略任何像素），用于与 MMSeg 的 aAcc / mIoU 对齐
    total_cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        if args.dataset == "datasets":
            case_name = split(case_name.split(",")[0])[-1]
        
        # 获取预测结果用于计算指标
        model.eval()
        with torch.no_grad():
            # 处理6通道数据：[B, C, H, W] -> [C, H, W]
            if len(image.shape) == 4:  # [B, C, H, W]
                image_np = image.squeeze(0).cpu().numpy()  # [C, H, W]
            else:  # [C, H, W]
                image_np = image.cpu().numpy()
                
            if len(label.shape) == 3:  # [B, H, W]
                label_np = label.squeeze(0).cpu().numpy()  # [H, W]
            else:  # [H, W]
                label_np = label.cpu().numpy()

            label_np = _label_to_int_hw(label_np)
            orig_h, orig_w = int(label_np.shape[0]), int(label_np.shape[1])

            # 调整图像大小到模型输入尺寸
            C, H, W = image_np.shape
            if H != args.img_size or W != args.img_size:
                from scipy.ndimage import zoom
                resized_image = np.zeros((C, args.img_size, args.img_size))
                for c in range(C):
                    resized_image[c] = zoom(image_np[c], (args.img_size / H, args.img_size / W), order=3)
                image_np = resized_image

            # 转换为tensor并添加batch维度 [1, C, H, W]
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            if args.num_classes == 2 and getattr(args, 'threshold', None) is not None:
                # 二分类：仅当前景概率 >= threshold 判为 1，可减少 FP、提高 Precision/Dice
                prediction = (probs[:, 1] >= args.threshold).long().squeeze(0).cpu().numpy()
            else:
                prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

            _log_inference_diag(i_batch, probs, prediction, label_np, args, case_name)

            # 与标签空间对齐（不依赖可能被 resize 前的 H,W，避免图/标尺寸不一致时错缩放）
            if prediction.shape[0] != orig_h or prediction.shape[1] != orig_w:
                prediction = _align_pred_to_label(prediction, orig_h, orig_w, args.num_classes)
        
        # 计算指标（不忽略任何像素，与 MMSeg aAcc/mIoU 一致，OA 与 mIoU 会正确区分）
        metrics = calculate_segmentation_metrics(
            prediction, label_np,
            num_classes=args.num_classes,
            ignore_index=255
        )
        
        # 累积指标
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])
        
        # 累积混淆矩阵（ignore_index=255 表示不排除任何像素，与 MMSeg 一致）
        cm = get_confusion_matrix(prediction, label_np, args.num_classes, ignore_index=255)
        total_cm += cm
        
        # 输出每个样本的指标（参考格式）
        dice = metrics['mean_dice']
        iou = metrics['mean_iou']
        f1 = metrics['mean_f1']
        oa = metrics['overall_accuracy']
        logging.info(f'idx {i_batch} case {case_name} - Dice: {dice:.4f}, IoU: {iou:.4f}, F1: {f1:.4f}, OA: {oa:.4f}')
        
        # 保留原有的test_single_volume调用用于保存结果
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        
        # 可视化结果
        if args.visualize and visualizer and i_batch < args.vis_num_samples:
            # 获取预测结果进行可视化
            # 处理6通道数据：[B, C, H, W] -> [C, H, W]
            if len(image.shape) == 4:  # [B, C, H, W]
                image_np = image.squeeze(0).cpu().numpy()  # [C, H, W]
            else:  # [C, H, W]
                image_np = image.cpu().numpy()
                
            if len(label.shape) == 3:  # [B, H, W]
                label_np = label.squeeze(0).cpu().numpy()  # [H, W]
            else:  # [H, W]
                label_np = label.cpu().numpy()
            
            print(f"Debug: image shape = {image.shape}")
            print(f"Debug: label shape = {label.shape}")
            print(f"Debug: image_np shape = {image_np.shape}")
            print(f"Debug: label_np shape = {label_np.shape}")
            
            # 获取模型预测
            model.eval()
            with torch.no_grad():
                if len(image_np.shape) == 3:  # [C, H, W] - 6通道数据
                    # 调整图像大小到模型输入尺寸
                    C, H, W = image_np.shape
                    if H != args.img_size or W != args.img_size:
                        from scipy.ndimage import zoom
                        resized_image = np.zeros((C, args.img_size, args.img_size))
                        for c in range(C):
                            resized_image[c] = zoom(image_np[c], (args.img_size / H, args.img_size / W), order=3)
                        image_np = resized_image
                    
                    # 转换为tensor并添加batch维度 [1, C, H, W]
                    input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    if args.num_classes == 2 and getattr(args, 'threshold', None) is not None:
                        prediction = (probs[:, 1] >= args.threshold).long().squeeze(0).cpu().numpy()
                    else:
                        prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
                    # 如果原始尺寸不同，缩放回原始尺寸
                    if H != args.img_size or W != args.img_size:
                        from scipy.ndimage import zoom
                        prediction = zoom(prediction, (H / args.img_size, W / args.img_size), order=0)
                else:
                    # 处理2D数据
                    input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().cuda()
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    if args.num_classes == 2 and getattr(args, 'threshold', None) is not None:
                        prediction = (probs[:, 1] >= args.threshold).long().squeeze(0).cpu().numpy()
                    else:
                        prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            
            # 保存可视化结果
            vis_save_path = os.path.join(args.vis_save_dir, f'{case_name}_visualization.png')
            
            # 确保图像和标签尺寸一致用于可视化
            if image_np.shape[1:] != label_np.shape:
                from scipy.ndimage import zoom
                C, H, W = image_np.shape
                target_H, target_W = label_np.shape
                resized_image = np.zeros((C, target_H, target_W))
                for c in range(C):
                    resized_image[c] = zoom(image_np[c], (target_H / H, target_W / W), order=3)
                image_np = resized_image
            
            visualizer.visualize_prediction(
                image_np, prediction, label_np, 
                save_path=vis_save_path, 
                show_overlay=True, 
                show_side_by_side=True
            )
            logging.info(f'Visualization saved for case: {case_name}')
    
    # 计算总体指标（参考格式：每张图指标再取平均）
    overall_metrics = {
        'overall_accuracy': np.mean(all_metrics['overall_accuracy']),
        'mean_precision': np.mean(all_metrics['mean_precision']),
        'mean_recall': np.mean(all_metrics['mean_recall']),
        'mean_f1': np.mean(all_metrics['mean_f1']),
        'mean_iou': np.mean(all_metrics['mean_iou']),
        'mean_dice': np.mean(all_metrics['mean_dice'])
    }
    
    # 按像素聚合的全局指标（与 MMSeg aAcc / mIoU 计算方式一致，可直接对比）
    global_metrics = calculate_global_metrics_from_cm(total_cm, args.num_classes)
    
    # 输出总体指标（参考格式）
    logging.info('=' * 50)
    logging.info('遥感分割评估指标结果：')
    logging.info('=' * 50)
    logging.info(f'overall_accuracy (每图平均): {overall_metrics["overall_accuracy"]:.4f}')
    logging.info(f'mean_precision: {overall_metrics["mean_precision"]:.4f}')
    logging.info(f'mean_recall: {overall_metrics["mean_recall"]:.4f}')
    logging.info(f'mean_f1: {overall_metrics["mean_f1"]:.4f}')
    logging.info(f'mean_iou (每图平均): {overall_metrics["mean_iou"]:.4f}')
    logging.info(f'mean_dice: {overall_metrics["mean_dice"]:.4f}')
    logging.info('-' * 50)
    logging.info('与 MMSeg 对齐的全局指标（按像素聚合，可直接与 mmseg 的 aAcc / mIoU 对比）：')
    logging.info(f'  OA_global (aAcc): {global_metrics["overall_accuracy_global"]:.4f}')
    logging.info(f'  mIoU_global:      {global_metrics["mean_iou_global"]:.4f}')
    logging.info('=' * 50)
    
    # 保留原有的输出格式（兼容性）
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    if args.visualize:
        logging.info(f'Visualization results saved to: {args.vis_save_dir}')
    
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': f'./lists/lists_{args.dataset}',  # 修正路径为 lists/lists_Synapse
            'num_classes': args.n_class,
            "z_spacing": 1
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # 不要重新设置volume_path，保持之前根据split_name设置的路径
    # args.volume_path = dataset_config[dataset_name]['root_path']
    # args.Dataset = dataset_config[dataset_name]['Dataset']
    # 保留用户传入的list_dir，不要覆盖（用户通过命令行传入的值应该优先）
    # args.list_dir = dataset_config[dataset_name]['list_dir']  # 注释掉，保留用户传入的值
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    if getattr(args, 'checkpoint', None):
        snapshot = args.checkpoint
    else:
        snapshot = os.path.join(args.output_dir, 'best_model.pth')
        if not os.path.exists(snapshot):
            snapshot = os.path.join(args.output_dir, 'epoch_' + str(args.max_epochs - 1) + '.pth')
    if not os.path.exists(snapshot):
        raise FileNotFoundError(
            f'Checkpoint not found: {snapshot}\n'
            '请用 --checkpoint /path/to/your/best_model.pth 指定权重路径，或确认 --output_dir 下存在 best_model.pth / epoch_*.pth'
        )
    try:
        msg = net.load_state_dict(torch.load(snapshot), strict=True)
        print("self trained swin unet", msg)
    except RuntimeError as e:
        err = str(e)
        if "size mismatch" in err:
            raise RuntimeError(
                err
                + "\n\n提示：权重张量形状与当前 --cfg 构造的网络不一致。"
                " 若 swin_unet.output 第二维不同（如 [2,192,...] vs [2,96,...]），"
                "通常是 **骨干不同**：checkpoint 为 Swin-Large 而当前用了 Tiny/Base 等更小 yaml（或相反）。"
                "请用 **与训练完全相同的** `--cfg`（例如 Large 对应 "
                "`configs/swin_large_patch4_window7_224_water.yaml`）。"
                " 另请核对 `MODEL.SWIN.IN_CHANS` 与训练一致。"
                "\n仅在 **output 首维**（类别数）与 checkpoint 不一致时，才需要改 `--n_class`。"
            ) from e
        raise
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # 与 yaml / 权重首层 conv 一致；不设则 dataset 默认 6 通道，3ch 模型会报 channel mismatch
    args.in_chans = getattr(config.MODEL.SWIN, 'IN_CHANS', 6)
    inference(args, net, test_save_path)

# python train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE
# python train.py --output_dir './model_out/datasets' --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split
# python test.py --output_dir ./model_out/datasets --dataset datasets --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/test --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
