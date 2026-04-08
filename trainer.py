import logging
import os
import random
import sys
import math

import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss, calculate_batch_metrics
from custom_collate import custom_collate


def trainer_synapse(args, model, snapshot_path, scaler=None):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    val_base = getattr(args, 'val_root_path', None) or args.root_path
    # 保留原分割（不移动文件）时，train/val 列表为 5:5 但样本仍在 train_tif / test_vol_tif，用 fallback_dir 在另一目录查找
    fallback_for_train = val_base if (val_base != args.root_path) else None
    fallback_for_val = args.root_path if (val_base != args.root_path) else None
    in_chans = getattr(args, 'in_chans', 3)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size, target_channels=in_chans,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                               fallback_dir=fallback_for_train)
    db_val = Synapse_dataset(base_dir=val_base, list_dir=args.list_dir, split="val", img_size=args.img_size, target_channels=in_chans,
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                             fallback_dir=fallback_for_val)
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        # 多进程 + numpy/scipy/tiff 时避免每 worker 再开满 BLAS 线程，降低偶发段错误概率
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    fast = getattr(args, "fast", False)
    num_workers = max(0, int(args.num_workers))
    _dl_common = dict(
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=custom_collate,
    )
    if num_workers > 0:
        _dl_common["prefetch_factor"] = 2
        if getattr(args, "persistent_workers", False):
            _dl_common["persistent_workers"] = True
    train_loader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, **_dl_common
    )
    val_loader = DataLoader(
        db_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, **_dl_common
    )
    if num_workers > 0:
        logging.info(
            "DataLoader: num_workers=%d, prefetch_factor=%s, persistent_workers=%s"
            % (
                num_workers,
                _dl_common.get("prefetch_factor", "default"),
                _dl_common.get("persistent_workers", False),
            )
        )
    # --fast 时降低同步与日志频率（数据已清洗时可安全使用）
    _metric_every = 50 if fast else 10
    _tb_img_every = 500 if fast else 20
    _finite_every = 256 if fast else 1
    _grad_nan_every = 8 if fast else 1
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # 类别权重：缓解背景多、水体少导致 OA 高但 mIoU 低
    ce_weight = None
    dice_weight = None
    if getattr(args, 'class_weight', None) is not None:
        ce_weight = torch.tensor(args.class_weight, dtype=torch.float32).cuda()
        dice_weight = args.class_weight
        logging.info("Using class_weight for CE and Dice: %s" % dice_weight)
    ce_loss = CrossEntropyLoss(weight=ce_weight)
    dice_loss = DiceLoss(num_classes)
    weight_decay = getattr(args, 'weight_decay', 0.0001)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    if getattr(args, 'max_iters', None) is not None:
        max_iterations = args.max_iters
        max_epoch = max(1, (max_iterations + len(train_loader) - 1) // len(train_loader))
        logging.info("MMSeg 对齐: max_iters=%d -> max_epoch=%d" % (max_iterations, max_epoch))
    else:
        max_epoch = args.max_epochs
        max_iterations = max_epoch * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    # 接续训练：从 checkpoint 恢复 model / iter_num / best_miou
    start_epoch = 0
    best_miou = 0.0
    if getattr(args, 'resume', None) and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cuda', weights_only=False)
        model_to_load = model.module if hasattr(model, 'module') else model
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model_to_load.load_state_dict(ckpt['model'], strict=True)
            iter_num = int(ckpt.get('iter_num', 0))
            best_miou = float(ckpt.get('best_miou', 0.0))
            start_epoch = iter_num // len(train_loader)
            logging.info("Resume from %s: iter_num=%d, best_miou=%.4f, start_epoch=%d" % (args.resume, iter_num, best_miou, start_epoch))
        else:
            model_to_load.load_state_dict(ckpt, strict=True)
            logging.info("Resume model weights from %s (no iter_num/best_miou, training from iter 0)" % args.resume)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70, initial=start_epoch, total=max_epoch)
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        
        # 用于累积评估指标
        epoch_train_metrics = {
            'overall_accuracy': 0,
            'mean_precision': 0,
            'mean_recall': 0,
            'mean_f1': 0,
            'mean_iou': 0,
            'mean_dice': 0
        }
        for i in range(num_classes):
            epoch_train_metrics['iou_class_%d' % i] = 0
            epoch_train_metrics['dice_class_%d' % i] = 0
            epoch_train_metrics['f1_class_%d' % i] = 0
        
        # 接续训练时，从 start_epoch 的第一个 epoch 要跳过已跑过的 batch
        skip_batches = max(0, iter_num - epoch_num * len(train_loader))
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            if i_batch < skip_batches:
                continue
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)
            if _finite_every == 1 or (iter_num % _finite_every == 0):
                if not torch.isfinite(image_batch).all() or not torch.isfinite(label_batch.float()).all():
                    logging.warning(
                        "iteration %d: batch 含 NaN/Inf，跳过（多为损坏 TIFF 或异常标签）" % iter_num)
                    optimizer.zero_grad(set_to_none=True)
                    continue
            # 标签越界会导致 CE 产生 NaN，先过滤或跳过
            label_flat = label_batch.long().view(-1)
            if (label_flat < 0).any() or (label_flat >= num_classes).any():
                logging.warning(f"iteration {iter_num}: label out of [0, {num_classes-1}], skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            with autocast('cuda'):
                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, weight=dice_weight, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                
                # 损坏/异常样本可能导致 NaN loss，跳过本 batch 继续训（长期全 NaN 请清洗数据或降 lr）
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("iteration %d: NaN/Inf loss，跳过本 batch" % iter_num)
                    optimizer.zero_grad(set_to_none=True)
                    continue

            def _has_nan_inf_grad(model):
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        return True
                return False

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if _grad_nan_every == 1 or (iter_num % _grad_nan_every == 0):
                    if _has_nan_inf_grad(model):
                        logging.warning(f"iteration {iter_num}: NaN/Inf in gradients, skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()  # 已 unscale_，需 update 以保持 scaler 状态
                        continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                if _grad_nan_every == 1 or (iter_num % _grad_nan_every == 0):
                    if _has_nan_inf_grad(model):
                        logging.warning(f"iteration {iter_num}: NaN/Inf in gradients, skipping step")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 学习率调度：poly 与 MMSeg PolyLR 一致，cosine 为原余弦退火
            lr_schedule = getattr(args, 'lr_schedule', 'cosine')
            if lr_schedule == 'poly':
                eta_min = 1e-4  # 与 MMSeg schedule_40k 一致
                power = 0.9
                ratio = max(0.0, 1.0 - iter_num / max_iterations)  # 避免负数的非整数幂产生复数
                lr_ = eta_min + (base_lr - eta_min) * (ratio ** power)
                lr_ = max(lr_, eta_min)
            else:
                min_lr = base_lr * 0.01
                lr_ = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * iter_num / max_iterations))
                lr_ = max(lr_, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            if iter_num >= max_iterations:
                break

            if iter_num % _metric_every == 0:
                with torch.no_grad():
                    pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    batch_metrics = calculate_batch_metrics(pred_labels, label_batch, num_classes)
                    for key in epoch_train_metrics.keys():
                        epoch_train_metrics[key] += batch_metrics[key]

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % _tb_img_every == 0:
                image = image_batch[1, 0:1, :, :]
                lo, hi = image.min(), image.max()
                image = (image - lo) / (hi - lo).clamp(min=1e-8) if hi > lo else torch.zeros_like(image)
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        if iter_num >= max_iterations:
            logging.info("Reached max_iterations=%d, stopping." % max_iterations)
            break
        # 计算epoch平均指标（与 _metric_every 对齐的近似次数）
        num_metric_calculations = max(1, len(train_loader) // _metric_every)
        if num_metric_calculations > 0:
            for key in epoch_train_metrics.keys():
                epoch_train_metrics[key] /= num_metric_calculations
        
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        
        # 记录训练指标到TensorBoard
        writer.add_scalar('train/overall_accuracy', epoch_train_metrics['overall_accuracy'], epoch_num)
        writer.add_scalar('train/mean_precision', epoch_train_metrics['mean_precision'], epoch_num)
        writer.add_scalar('train/mean_recall', epoch_train_metrics['mean_recall'], epoch_num)
        writer.add_scalar('train/mean_f1', epoch_train_metrics['mean_f1'], epoch_num)
        writer.add_scalar('train/mean_iou', epoch_train_metrics['mean_iou'], epoch_num)
        writer.add_scalar('train/mean_dice', epoch_train_metrics['mean_dice'], epoch_num)
        
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f, OA: %.4f, mIoU: %.4f, mDice: %.4f, mF1: %.4f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss, 
            epoch_train_metrics['overall_accuracy'], epoch_train_metrics['mean_iou'], 
            epoch_train_metrics['mean_dice'], epoch_train_metrics['mean_f1']))
        if num_classes == 2:
            logging.info('  -> IoU_bg: %.4f, IoU_water: %.4f  |  Dice_water: %.4f, F1_water: %.4f' % (
                epoch_train_metrics['iou_class_0'], epoch_train_metrics['iou_class_1'],
                epoch_train_metrics['dice_class_1'], epoch_train_metrics['f1_class_1']))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            
            # 用于累积验证评估指标
            epoch_val_metrics = {
                'overall_accuracy': 0,
                'mean_precision': 0,
                'mean_recall': 0,
                'mean_f1': 0,
                'mean_iou': 0,
                'mean_dice': 0
            }
            for i in range(num_classes):
                epoch_val_metrics['iou_class_%d' % i] = 0
                epoch_val_metrics['dice_class_%d' % i] = 0
                epoch_val_metrics['f1_class_%d' % i] = 0
            
            val_batches_ok = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch = image_batch.cuda(non_blocking=True)
                    label_batch = label_batch.cuda(non_blocking=True)
                    label_flat = label_batch.long().view(-1)
                    if (label_flat < 0).any() or (label_flat >= num_classes).any():
                        continue
                    # 与训练一致：loss 须在 autocast 内计算；若在 autocast 外对 fp16 logits 算 CE 易溢出 -> val loss 全 nan，而 argmax 指标仍可能正常
                    with autocast('cuda'):
                        outputs = model(image_batch)
                        loss_ce = ce_loss(outputs, label_batch[:].long())
                        loss_dice = dice_loss(outputs, label_batch, weight=dice_weight, softmax=True)
                    if not torch.isfinite(loss_ce) or not torch.isfinite(loss_dice):
                        continue
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()
                    val_batches_ok += 1

                    # 计算验证评估指标
                    pred_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    batch_metrics = calculate_batch_metrics(pred_labels, label_batch, num_classes)
                    for key in epoch_val_metrics.keys():
                        epoch_val_metrics[key] += batch_metrics[key]

                denom = max(val_batches_ok, 1)
                batch_ce_loss /= denom
                batch_dice_loss /= denom
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss

                # 与 loss 一致：仅对实际参与前向的 batch 取平均（跳过的 batch 不计入）
                for key in epoch_val_metrics.keys():
                    epoch_val_metrics[key] /= denom
                if val_batches_ok == 0:
                    logging.warning("Val: 无有效 batch（标签越界或 loss 全 NaN），请检查数据与混合精度设置")
                
                # 记录验证指标到TensorBoard
                writer.add_scalar('val/overall_accuracy', epoch_val_metrics['overall_accuracy'], epoch_num)
                writer.add_scalar('val/mean_precision', epoch_val_metrics['mean_precision'], epoch_num)
                writer.add_scalar('val/mean_recall', epoch_val_metrics['mean_recall'], epoch_num)
                writer.add_scalar('val/mean_f1', epoch_val_metrics['mean_f1'], epoch_num)
                writer.add_scalar('val/mean_iou', epoch_val_metrics['mean_iou'], epoch_num)
                writer.add_scalar('val/mean_dice', epoch_val_metrics['mean_dice'], epoch_num)
                
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f, OA: %.4f, mIoU: %.4f, mDice: %.4f, mF1: %.4f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss,
                    epoch_val_metrics['overall_accuracy'], epoch_val_metrics['mean_iou'],
                    epoch_val_metrics['mean_dice'], epoch_val_metrics['mean_f1']))
                if num_classes == 2:
                    logging.info('  -> IoU_bg: %.4f, IoU_water: %.4f（与修改前「只算前景」时的 mIoU 约等于 IoU_water）' % (
                        epoch_val_metrics['iou_class_0'], epoch_val_metrics['iou_class_1']))
                    logging.info('  -> Dice_bg: %.4f, Dice_water: %.4f  |  F1_bg: %.4f, F1_water: %.4f（旧版 mDice/mF1 仅水体类，约等于 Dice_water/F1_water）' % (
                        epoch_val_metrics['dice_class_0'], epoch_val_metrics['dice_class_1'],
                        epoch_val_metrics['f1_class_0'], epoch_val_metrics['f1_class_1']))
                
                # 使用mIoU作为模型保存的评判标准
                current_miou = epoch_val_metrics['mean_iou']
                if current_miou > best_miou:  # 这里用mIoU替代loss作为评判标准
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save((model.module if hasattr(model, 'module') else model).state_dict(), save_mode_path)
                    best_miou = current_miou
                    logging.info("New best model saved with mIoU: %.4f" % current_miou)
                state_to_save = (model.module if hasattr(model, 'module') else model).state_dict()
                last_path = os.path.join(snapshot_path, 'last_model.pth')
                torch.save({'model': state_to_save, 'iter_num': iter_num, 'best_miou': best_miou}, last_path)
                logging.info("save model to {}".format(last_path))

    writer.close()
    return "Training Finished!"
