#!/usr/bin/env python3
"""
检查所有图像的通道数
"""
import os
import tifffile
from collections import Counter
from tqdm import tqdm

def check_image_channels(data_dir, list_file):
    """检查列表中所有图像的通道数"""
    print(f"正在检查图像通道数...")
    print(f"数据目录: {data_dir}")
    print(f"列表文件: {list_file}\n")
    
    if not os.path.exists(list_file):
        print(f"错误: 列表文件不存在: {list_file}")
        return
    
    with open(list_file, 'r') as f:
        sample_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"总共 {len(sample_names)} 个样本需要检查\n")
    
    channel_counts = []
    channel_details = {}
    error_samples = []
    
    for sample_name in tqdm(sample_names, desc="检查进度"):
        img_path = os.path.join(data_dir, f"{sample_name}_img.tif")
        
        if not os.path.exists(img_path):
            error_samples.append((sample_name, "文件不存在"))
            continue
        
        try:
            image = tifffile.imread(img_path)
            
            # 处理维度
            if len(image.shape) == 3:
                # 判断维度顺序
                if image.shape[-1] <= 64 and image.shape[-1] != image.shape[0]:
                    # 可能是(H, W, C)，需要转置
                    image = image.transpose(2, 0, 1)
                # 现在应该是(C, H, W)
                C = image.shape[0]
            elif len(image.shape) == 2:
                C = 1
            else:
                C = image.shape[0] if image.shape[0] < 100 else image.shape[-1]
            
            channel_counts.append(C)
            
            if C not in channel_details:
                channel_details[C] = []
            channel_details[C].append(sample_name)
            
        except Exception as e:
            error_samples.append((sample_name, str(e)))
    
    # 统计结果
    channel_counter = Counter(channel_counts)
    
    print("\n" + "="*60)
    print("通道数统计:")
    print("="*60)
    print(f"总样本数: {len(sample_names)}")
    print(f"成功检查: {len(channel_counts)}")
    print(f"检查失败: {len(error_samples)}")
    print("\n通道数分布:")
    for channels in sorted(channel_counter.keys()):
        count = channel_counter[channels]
        percentage = count / len(channel_counts) * 100 if channel_counts else 0
        print(f"  {channels} 通道: {count} 个样本 ({percentage:.2f}%)")
    
    # 显示非6通道的样本
    if 6 in channel_details:
        print(f"\n✅ 6通道样本: {len(channel_details[6])} 个")
    
    non_6_channels = {k: v for k, v in channel_details.items() if k != 6}
    if non_6_channels:
        print(f"\n⚠️  非6通道样本:")
        for channels in sorted(non_6_channels.keys()):
            samples = non_6_channels[channels]
            print(f"  {channels} 通道: {len(samples)} 个样本")
            if len(samples) <= 10:
                print(f"    样本列表: {', '.join(samples)}")
            else:
                print(f"    前10个样本: {', '.join(samples[:10])}...")
    
    # 保存非6通道样本列表
    if non_6_channels:
        output_file = os.path.join(os.path.dirname(list_file), "non_6channel_samples.txt")
        with open(output_file, 'w') as f:
            for channels in sorted(non_6_channels.keys()):
                f.write(f"# {channels} 通道样本:\n")
                for sample in sorted(non_6_channels[channels]):
                    f.write(f"{sample}\n")
        print(f"\n非6通道样本列表已保存到: {output_file}")
    
    # 显示错误样本
    if error_samples:
        print(f"\n❌ 检查失败的样本 ({len(error_samples)} 个):")
        for sample, error in error_samples[:10]:
            print(f"  {sample}: {error[:50]}")
        if len(error_samples) > 10:
            print(f"  ... 还有 {len(error_samples) - 10} 个失败样本")
    
    print("="*60)
    
    return channel_details, error_samples

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检查图像通道数')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--list_file', type=str, required=True, help='样本列表文件路径')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isabs(args.list_file):
        args.list_file = os.path.abspath(args.list_file)
    
    check_image_channels(args.data_dir, args.list_file)
