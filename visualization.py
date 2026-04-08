import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图像
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import torch
import cv2
from PIL import Image
import seaborn as sns


class SegmentationVisualizer:
    """分割结果可视化工具类"""
    
    def __init__(self, num_classes=9, class_names=None):
        """
        初始化可视化工具
        
        Args:
            num_classes: 分割类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        
        # 定义颜色映射
        self.colors = self._generate_colors(num_classes)
        
    def _generate_colors(self, num_classes):
        """生成类别颜色"""
        # 使用seaborn的调色板生成颜色
        colors = sns.color_palette("husl", num_classes)
        # 转换为RGB值
        colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
        return colors
    
    def visualize_prediction(self, image, prediction, ground_truth=None, save_path=None, 
                           show_overlay=True, show_side_by_side=True):
        """
        可视化分割结果
        
        Args:
            image: 输入图像 [C, H, W] 或 [H, W]
            prediction: 预测结果 [H, W]
            ground_truth: 真实标签 [H, W] (可选)
            save_path: 保存路径 (可选)
            show_overlay: 是否显示叠加图像
            show_side_by_side: 是否并排显示
        """
        # 处理输入图像
        if len(image.shape) == 3:
            # 多通道图像，选择前3个通道进行RGB显示
            if image.shape[0] >= 3:
                rgb_image = image[:3].transpose(1, 2, 0)  # [H, W, 3]
            else:
                # 如果通道数不足3，重复通道
                rgb_image = np.repeat(image[0:1], 3, axis=0).transpose(1, 2, 0)
        else:
            # 单通道图像
            rgb_image = np.stack([image] * 3, axis=-1)
        
        # 归一化到0-1
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        
        # 创建颜色映射的预测结果
        pred_colored = self._apply_colormap(prediction)
        
        if ground_truth is not None:
            gt_colored = self._apply_colormap(ground_truth)
            
            if show_side_by_side:
                # 并排显示
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # 第一行：原图、预测、真实标签
                axes[0, 0].imshow(rgb_image)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(pred_colored)
                axes[0, 1].set_title('Prediction')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(gt_colored)
                axes[0, 2].set_title('Ground Truth')
                axes[0, 2].axis('off')
                
                # 第二行：叠加图像
                if show_overlay:
                    overlay_pred = self._create_overlay(rgb_image, pred_colored, alpha=0.6)
                    overlay_gt = self._create_overlay(rgb_image, gt_colored, alpha=0.6)
                    
                    axes[1, 0].imshow(overlay_pred)
                    axes[1, 0].set_title('Image + Prediction')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(overlay_gt)
                    axes[1, 1].set_title('Image + Ground Truth')
                    axes[1, 1].axis('off')
                    
                    # 差异图
                    diff = self._create_difference_map(prediction, ground_truth)
                    axes[1, 2].imshow(diff, cmap='RdBu')
                    axes[1, 2].set_title('Prediction vs GT Difference')
                    axes[1, 2].axis('off')
                else:
                    axes[1, 0].axis('off')
                    axes[1, 1].axis('off')
                    axes[1, 2].axis('off')
                
                # 添加图例
                legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                                 for color, name in zip(self.colors, self.class_names)]
                fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(self.class_names)))
                
            else:
                # 单独显示
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(rgb_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(pred_colored)
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                
                axes[2].imshow(gt_colored)
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                
                # 添加图例
                legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                                 for color, name in zip(self.colors, self.class_names)]
                fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(self.class_names)))
        else:
            # 只有预测结果
            if show_overlay:
                overlay = self._create_overlay(rgb_image, pred_colored, alpha=0.6)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(rgb_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(pred_colored)
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                
                axes[2].imshow(overlay)
                axes[2].set_title('Image + Prediction')
                axes[2].axis('off')
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                axes[0].imshow(rgb_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(pred_colored)
                axes[1].set_title('Prediction')
                axes[1].axis('off')
            
            # 添加图例
            legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                             for color, name in zip(self.colors, self.class_names)]
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(self.class_names)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.close()  # 关闭图形，释放内存
        return fig
    
    def _apply_colormap(self, mask):
        """应用颜色映射到分割掩码"""
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            if class_id == 0:  # 背景类
                continue
            mask_class = (mask == class_id)
            colored_mask[mask_class] = self.colors[class_id]
        
        return colored_mask
    
    def _create_overlay(self, image, colored_mask, alpha=0.6):
        """创建叠加图像"""
        overlay = image.copy()
        mask_indices = np.any(colored_mask > 0, axis=2)
        overlay[mask_indices] = alpha * colored_mask[mask_indices] + (1 - alpha) * image[mask_indices]
        return overlay
    
    def _create_difference_map(self, pred, gt):
        """创建预测和真实标签的差异图"""
        diff = np.zeros_like(pred, dtype=np.float32)
        diff[pred != gt] = 1.0
        return diff
    
    def visualize_batch(self, images, predictions, ground_truths=None, save_dir=None, max_samples=4):
        """
        可视化批次数据
        
        Args:
            images: 批次图像 [B, C, H, W]
            predictions: 批次预测 [B, H, W]
            ground_truths: 批次真实标签 [B, H, W] (可选)
            save_dir: 保存目录 (可选)
            max_samples: 最大显示样本数
        """
        batch_size = min(len(images), max_samples)
        
        for i in range(batch_size):
            image = images[i]
            prediction = predictions[i]
            ground_truth = ground_truths[i] if ground_truths is not None else None
            
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'sample_{i:03d}.png')
            
            self.visualize_prediction(image, prediction, ground_truth, save_path)
    
    def create_comparison_grid(self, images, predictions, ground_truths, save_path=None, grid_size=(2, 3)):
        """
        创建比较网格图
        
        Args:
            images: 图像列表
            predictions: 预测列表
            ground_truths: 真实标签列表
            save_path: 保存路径
            grid_size: 网格大小 (rows, cols)
        """
        num_samples = min(len(images), grid_size[0] * grid_size[1])
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i in range(num_samples):
            image = images[i]
            prediction = predictions[i]
            ground_truth = ground_truths[i]
            
            # 处理图像
            if len(image.shape) == 3:
                if image.shape[0] >= 3:
                    rgb_image = image[:3].transpose(1, 2, 0)
                else:
                    rgb_image = np.repeat(image[0:1], 3, axis=0).transpose(1, 2, 0)
            else:
                rgb_image = np.stack([image] * 3, axis=-1)
            
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            
            # 创建叠加图像
            pred_colored = self._apply_colormap(prediction)
            overlay = self._create_overlay(rgb_image, pred_colored, alpha=0.6)
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        # 添加图例
        legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                         for color, name in zip(self.colors, self.class_names)]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(self.class_names)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison grid saved to: {save_path}")
        
        plt.close()  # 关闭图形，释放内存
        return fig


def visualize_training_results(model, test_loader, device, save_dir=None, num_samples=5):
    """
    可视化训练结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化样本数量
    """
    model.eval()
    visualizer = SegmentationVisualizer()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 获取预测结果
            outputs = model(images)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # 转换为numpy
            images_np = images.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # 可视化
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'test_sample_{i:03d}.png')
            
            visualizer.visualize_prediction(
                images_np[0], predictions_np[0], labels_np[0], 
                save_path=save_path, show_overlay=True
            )


if __name__ == "__main__":
    # 测试可视化功能
    import numpy as np
    
    # 创建测试数据
    image = np.random.rand(6, 224, 224)  # 6通道图像
    prediction = np.random.randint(0, 9, (224, 224))  # 预测结果
    ground_truth = np.random.randint(0, 9, (224, 224))  # 真实标签
    
    # 创建可视化器
    visualizer = SegmentationVisualizer(num_classes=9)
    
    # 可视化
    visualizer.visualize_prediction(image, prediction, ground_truth, show_overlay=True)
