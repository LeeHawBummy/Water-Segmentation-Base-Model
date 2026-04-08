import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 处理6通道数据：[B, C, H, W] -> [C, H, W]
    if len(image.shape) == 4:  # [B, C, H, W]
        image = image.squeeze(0).cpu().detach().numpy()  # [C, H, W]
    else:  # [C, H, W]
        image = image.cpu().detach().numpy()
        
    if len(label.shape) == 3:  # [B, H, W]
        label = label.squeeze(0).cpu().detach().numpy()  # [H, W]
    else:  # [H, W]
        label = label.cpu().detach().numpy()
    # 对于6通道数据，直接处理整个图像
    if len(image.shape) == 3:  # [C, H, W] - 6通道数据
        # 调整图像大小到patch_size
        C, H, W = image.shape
        if H != patch_size[0] or W != patch_size[1]:
            # 对每个通道进行缩放
            resized_image = np.zeros((C, patch_size[0], patch_size[1]))
            for c in range(C):
                resized_image[c] = zoom(image[c], (patch_size[0] / H, patch_size[1] / W), order=3)
            image = resized_image
        
        # 转换为tensor并添加batch维度
        input_tensor = torch.from_numpy(image).unsqueeze(0).float().cuda()  # [1, C, H, W]
        
        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  # [H, W]
            prediction = out.cpu().detach().numpy()
            
            # 如果原始尺寸不同，缩放回原始尺寸
            if H != patch_size[0] or W != patch_size[1]:
                prediction = zoom(prediction, (H / patch_size[0], W / patch_size[1]), order=0)
    else:
        # 2D数据的情况
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def calculate_segmentation_metrics(pred, gt, num_classes, ignore_index=255):
    """
    计算遥感图像分割的常用评估指标。
    默认 ignore_index=255：不忽略任何像素，OA 与 mIoU 按全部类别计算，可与 MMSeg 的 aAcc/mIoU 对齐。
    若设为 0：会排除背景像素再算指标，二分类时 OA 会与 mIoU 相等（仅在前景上的准确率）。
    
    Args:
        pred: 预测结果 [H, W] 或 [B, H, W]
        gt: 真实标签 [H, W] 或 [B, H, W]
        num_classes: 类别数量
        ignore_index: 忽略的标签值，默认 255（不忽略任何像素）
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 确保输入是numpy数组
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    
    # 展平为一维数组
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # 只排除 ignore_index 的像素（255 表示不忽略任何像素）
    valid_mask = (gt_flat != ignore_index) & (gt_flat >= 0) & (gt_flat < num_classes)
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    
    # 计算混淆矩阵
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    
    # 计算各类别的指标
    metrics = {}
    
    # 总体准确率 (Overall Accuracy)
    cm_sum = np.sum(cm)
    if cm_sum > 0:
        overall_accuracy = np.trace(cm) / cm_sum
    else:
        overall_accuracy = 0.0  # 如果没有有效像素，准确率为0
    metrics['overall_accuracy'] = overall_accuracy
    
    # 计算每个类别的指标（先算各类，再按需要挑选前景类做平均）
    class_metrics = {}
    for i in range(num_classes):
        # True Positive, False Positive, False Negative, True Negative
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Precision (精确率)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_metrics[f'precision_class_{i}'] = precision
        
        # Recall (召回率)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_metrics[f'recall_class_{i}'] = recall
        
        # F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_metrics[f'f1_class_{i}'] = f1
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        class_metrics[f'iou_class_{i}'] = iou
        
        # Dice Coefficient
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        class_metrics[f'dice_class_{i}'] = dice
    
    # 计算平均指标
    # 默认：所有类别参与（与 MMSeg mIoU 一致）
    valid_classes = list(range(num_classes))
    # 若是二分类，很多场景只关心前景（类别1），
    # 此时将均值指标改为仅在前景类上计算，背景类指标仍可从 class_metrics 中单独查看。
    if num_classes == 2:
        valid_classes = [1]
    
    metrics['mean_precision'] = np.mean([class_metrics[f'precision_class_{i}'] for i in valid_classes])
    metrics['mean_recall'] = np.mean([class_metrics[f'recall_class_{i}'] for i in valid_classes])
    metrics['mean_f1'] = np.mean([class_metrics[f'f1_class_{i}'] for i in valid_classes])
    metrics['mean_iou'] = np.mean([class_metrics[f'iou_class_{i}'] for i in valid_classes])
    metrics['mean_dice'] = np.mean([class_metrics[f'dice_class_{i}'] for i in valid_classes])
    
    # 添加类别指标
    metrics.update(class_metrics)
    
    return metrics


def get_confusion_matrix(pred, gt, num_classes, ignore_index=255):
    """
    计算单张图的混淆矩阵，用于后续按像素聚合得到与 MMSeg 一致的全局指标。
    使用 ignore_index=255 时不会排除任何像素（假设标签为 0,1,...,num_classes-1），
    与 MMSeg 的 aAcc / mIoU 计算方式一致。

    Args:
        pred: 预测 [H, W]
        gt: 标签 [H, W]
        num_classes: 类别数
        ignore_index: 忽略的标签值，默认 255（不忽略任何像素）

    Returns:
        np.ndarray: 混淆矩阵 [num_classes, num_classes]
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.cpu().numpy()
    pred_flat = np.clip(pred.flatten().astype(np.int64), 0, num_classes - 1)
    gt_flat = gt.flatten().astype(np.int64)
    valid_mask = (gt_flat != ignore_index) & (gt_flat >= 0) & (gt_flat < num_classes)
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]
    if pred_flat.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    return np.array(cm, dtype=np.int64)


def calculate_global_metrics_from_cm(total_cm, num_classes):
    """
    从「按像素聚合」的混淆矩阵计算全局 OA 与 mIoU，与 MMSeg 的 aAcc / mIoU 一致。
    适用于多张图累加后的 total_cm。

    Args:
        total_cm: 累加后的混淆矩阵 [num_classes, num_classes]
        num_classes: 类别数

    Returns:
        dict: overall_accuracy_global, mean_iou_global, 以及 per-class iou（可选）
    """
    total_cm = np.asarray(total_cm, dtype=np.float64)
    cm_sum = np.sum(total_cm)
    if cm_sum <= 0:
        return {
            'overall_accuracy_global': 0.0,
            'mean_iou_global': 0.0,
        }
    overall_accuracy_global = np.trace(total_cm) / cm_sum
    iou_list = []
    for i in range(num_classes):
        tp = total_cm[i, i]
        union = np.sum(total_cm[i, :]) + np.sum(total_cm[:, i]) - tp
        iou_i = (tp / union) if union > 0 else 0.0
        iou_list.append(iou_i)
    mean_iou_global = np.mean(iou_list)
    return {
        'overall_accuracy_global': float(overall_accuracy_global),
        'mean_iou_global': float(mean_iou_global),
    }


def calculate_batch_metrics(predictions, targets, num_classes, ignore_index=255):
    """
    计算一个batch的评估指标。默认不忽略任何像素，与 MMSeg aAcc/mIoU 一致。

    Args:
        predictions: 预测结果 [B, H, W]
        targets: 真实标签 [B, H, W]
        num_classes: 类别数量
        ignore_index: 忽略的标签值，默认 255（不忽略）

    Returns:
        dict: 平均评估指标
    """
    batch_metrics = []
    
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        gt = targets[i]
        metrics = calculate_segmentation_metrics(pred, gt, num_classes, ignore_index)
        batch_metrics.append(metrics)
    
    # 计算batch平均指标（含每类 IoU/Dice/F1，便于与「只算前景」时的数值对照）
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        if (key.startswith('mean_') or key == 'overall_accuracy' or
            key.startswith('iou_class_') or key.startswith('dice_class_') or key.startswith('f1_class_')):
            avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
    
    return avg_metrics