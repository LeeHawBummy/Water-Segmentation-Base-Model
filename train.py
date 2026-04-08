import argparse
import os
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/water', help='数据根目录：默认优先按“单目录”读取（目录内直接包含 *_img.tif / *_label.tif）；若是两级目录结构也可自动兼容（train_tif）')
parser.add_argument('--single_dir', action='store_true',
                    help='单目录 + 列表划分：所有样本在同一文件夹，仅用 list_dir 下 train.txt / val.txt 区分，不再拼接 train_tif')
parser.add_argument('--val_root_path', type=str, default=None,
                    help='验证集数据目录；不设则与 root_path 相同。双目录场景（如 train_tif / test_vol_tif）时单独指定')
parser.add_argument('--dataset', type=str,
                    default='water', help='数据集名称（兼容 Synapse；建议用 water）')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_water', help='list dir（默认 lists/lists_water）')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
# parser.add_argument("--dataset_name", default="datasets")
parser.add_argument("--n_class", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_interval", default=1, type=int)
# 迭代训练相关参数
parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay（默认 0.0005）")
parser.add_argument("--lr_schedule", type=str, default="poly", choices=["cosine", "poly"],
                    help="学习率策略：poly（多项式衰减）或 cosine（余弦退火）")
parser.add_argument("--max_iters", type=int, default=None,
                    help="总迭代数（iter-based training）。设置后会根据 dataloader 长度自动推算 max_epochs")
parser.add_argument("--class_weight", type=str, default=None,
                    help="类别权重，逗号分隔，如 '1,3' 表示背景=1、水体=3，用于缓解类别不平衡、提升水体 mIoU；不设则两类等权")
parser.add_argument("--fast", action="store_true",
                    help="尽量提速：cudnn.benchmark、不限制 DataLoader worker、降低 TensorBoard/有限性检查/梯度扫描频率（略损可复现性）")
parser.add_argument("--persistent_workers", action="store_true",
                    help="DataLoader 长驻 worker（略快，但部分环境+tifffile/scipy 下可能段错误；默认关闭）")

args = parser.parse_args()

# 兼容旧命名：Synapse / water 走同一套数据逻辑
if isinstance(args.dataset, str) and args.dataset.lower() == "synapse":
    args.dataset = "water"

if args.dataset == "water":
    rp = os.path.abspath(os.path.expanduser(args.root_path))
    vrp = os.path.abspath(os.path.expanduser(args.val_root_path)) if args.val_root_path is not None else None

    def _looks_like_single_dir(p: str) -> bool:
        # 单目录：直接含 *_img.tif / *_label.tif（只做轻量判定）
        try:
            if not os.path.isdir(p):
                return False
            for name in os.listdir(p):
                if name.endswith("_img.tif") or name.endswith("_label.tif"):
                    return True
        except Exception:
            return False
        return False

    # 默认支持单目录：不传 --single_dir 时也会自动识别
    if args.single_dir or _looks_like_single_dir(rp):
        args.root_path = rp
        if vrp is not None:
            args.val_root_path = vrp
    else:
        # 兼容旧两级目录：root_path 下有 train_tif
        if os.path.basename(os.path.normpath(rp)) == "train_tif":
            args.root_path = rp
        else:
            args.root_path = os.path.join(rp, "train_tif")
        if args.val_root_path is None:
            args.val_root_path = args.root_path
        else:
            args.val_root_path = vrp
if getattr(args, 'class_weight', None):
    args.class_weight = [float(x.strip()) for x in args.class_weight.split(',')]
    assert len(args.class_weight) == args.n_class, "class_weight 数量需等于 n_class"
else:
    args.class_weight = None
# 若指定 max_iters（如 40000），在 trainer 内根据 dataloader 长度推算 max_epochs
config = get_config(args)

if __name__ == "__main__":
    if getattr(args, "fast", False):
        cudnn.benchmark = True
        cudnn.deterministic = False
    elif not args.deterministic:
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
    # 保留用户传入的 list_dir，不要覆盖
    list_dir = args.list_dir  # 使用用户传入的值
    
    dataset_config = {
        args.dataset: {
            'root_path': args.root_path,
            'list_dir': list_dir,
            'num_classes': args.n_class,
        },
    }

    # 按 batch 缩放学习率（与 24 对齐）；Large 小 batch 时设下限，避免有效 lr 过小导致不如 Tiny
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    if getattr(args, 'cfg', '') and 'large' in args.cfg.lower() and args.batch_size <= 6:
        args.base_lr = max(args.base_lr, 0.002)
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_chans = getattr(config.MODEL.SWIN, 'IN_CHANS', 3)  # 输入通道数由 config/--opts 指定；未设时默认 3

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    # 强制模型在CPU上运行避免CUDA兼容性问题
    net = net.cuda()
    net.load_from(config)

    # trainer = {'Synapse': trainer_synapse}
    
# 初始化混合精度训练
scaler = GradScaler()
trainer_synapse(args, net, args.output_dir, scaler=scaler)



# python train.py --output_dir ./model_out/datasets --dataset datasets --img_size 224 --batch_size 32 --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/Dataset001_mm/nnUNetPlans_2d_split