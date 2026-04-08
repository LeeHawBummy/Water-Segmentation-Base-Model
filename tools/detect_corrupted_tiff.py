#!/usr/bin/env python3
"""
检测损坏的TIFF文件
支持相对路径和绝对路径
"""
import os
import sys
import tifffile
from tqdm import tqdm


def check_tiff_file(file_path):
    """检查单个TIFF文件是否损坏"""
    try:
        # 尝试读取文件
        data = tifffile.imread(file_path)
        # 检查数据是否为空
        if data.size == 0:
            return False, "empty"
        return True, None
    except tifffile.TiffFileError as e:
        return False, f"TiffFileError: {str(e)}"
    except OSError as e:
        return False, f"OSError: {str(e)}"
    except IOError as e:
        return False, f"IOError: {str(e)}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"


def detect_corrupted_files(data_dir, list_file):
    """检测列表文件中的损坏TIFF文件"""
    # 转换为绝对路径
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    if not os.path.isabs(list_file):
        list_file = os.path.abspath(list_file)
    
    print(f"正在检测损坏的TIFF文件...")
    print(f"数据目录: {data_dir}")
    print(f"列表文件: {list_file}")
    
    # 读取列表文件
    if not os.path.exists(list_file):
        print(f"错误: 列表文件不存在: {list_file}")
        return [], {}
    
    with open(list_file, 'r') as f:
        sample_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"总共 {len(sample_names)} 个样本需要检测\n")
    
    corrupted_files = []
    stats = {
        'total': len(sample_names),
        'valid': 0,
        'corrupted': 0,
        'missing': 0,
        'empty': 0,
        'tiff_error': 0,
        'other_error': 0
    }
    
    # 检测每个样本
    for sample_name in tqdm(sample_names, desc="检测进度"):
        img_path = os.path.join(data_dir, f"{sample_name}_img.tif")
        label_path = os.path.join(data_dir, f"{sample_name}_label.tif")
        
        # 检查图像文件
        if not os.path.exists(img_path):
            corrupted_files.append({
                'sample': sample_name,
                'file': 'img',
                'path': img_path,
                'error': 'missing'
            })
            stats['missing'] += 1
            continue
        
        if not os.path.exists(label_path):
            corrupted_files.append({
                'sample': sample_name,
                'file': 'label',
                'path': label_path,
                'error': 'missing'
            })
            stats['missing'] += 1
            continue
        
        # 检查图像文件是否损坏
        is_valid, error = check_tiff_file(img_path)
        if not is_valid:
            corrupted_files.append({
                'sample': sample_name,
                'file': 'img',
                'path': img_path,
                'error': error
            })
            stats['corrupted'] += 1
            if 'empty' in error.lower():
                stats['empty'] += 1
            elif 'TiffFileError' in error:
                stats['tiff_error'] += 1
            else:
                stats['other_error'] += 1
            continue
        
        # 检查标签文件是否损坏
        is_valid, error = check_tiff_file(label_path)
        if not is_valid:
            corrupted_files.append({
                'sample': sample_name,
                'file': 'label',
                'path': label_path,
                'error': error
            })
            stats['corrupted'] += 1
            if 'empty' in error.lower():
                stats['empty'] += 1
            elif 'TiffFileError' in error:
                stats['tiff_error'] += 1
            else:
                stats['other_error'] += 1
            continue
        
        # 文件正常
        stats['valid'] += 1
    
    return corrupted_files, stats


def save_corrupted_list(corrupted_files, output_file):
    """保存损坏文件列表"""
    # 获取所有损坏的样本名称（去重）
    corrupted_samples = set()
    for item in corrupted_files:
        corrupted_samples.add(item['sample'])
    
    with open(output_file, 'w') as f:
        for sample_name in sorted(corrupted_samples):
            f.write(f"{sample_name}\n")
    
    print(f"\n损坏样本列表已保存到: {output_file}")
    print(f"共 {len(corrupted_samples)} 个损坏的样本")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检测损坏的TIFF文件')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径（支持相对路径和绝对路径）')
    parser.add_argument('--list_file', type=str, required=True, help='样本列表文件路径（支持相对路径和绝对路径）')
    parser.add_argument('--output_file', type=str, default=None, help='输出损坏样本列表文件路径（可选）')
    parser.add_argument('--filter', action='store_true', help='是否自动过滤损坏的样本')
    parser.add_argument('--inplace', action='store_true',
                        help='配合 --filter：直接覆盖 list_file（会先备份为 .bak）；不加则输出为 *_clean.txt')
    
    args = parser.parse_args()
    
    # 检测损坏文件
    corrupted_files, stats = detect_corrupted_files(args.data_dir, args.list_file)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("检测统计:")
    print("="*60)
    print(f"总样本数: {stats['total']}")
    print(f"有效文件: {stats['valid']}")
    print(f"损坏文件: {stats['corrupted']}")
    print(f"缺失文件: {stats['missing']}")
    print(f"  - 空文件: {stats['empty']}")
    print(f"  - TIFF错误: {stats['tiff_error']}")
    print(f"  - 其他错误: {stats['other_error']}")
    print("="*60)
    
    if corrupted_files:
        print(f"\n发现 {len(corrupted_files)} 个损坏的文件:")
        # 显示前10个损坏文件的详细信息
        for i, item in enumerate(corrupted_files[:10]):
            error_msg = item['error'][:80] if len(item['error']) > 80 else item['error']
            print(f"  {i+1}. {item['sample']} ({item['file']}): {error_msg}")
        if len(corrupted_files) > 10:
            print(f"  ... 还有 {len(corrupted_files) - 10} 个损坏文件")
        
        # 保存损坏文件列表
        if args.output_file:
            output_file = args.output_file
        else:
            # 默认输出到列表文件同目录
            list_dir = os.path.dirname(args.list_file)
            list_basename = os.path.basename(args.list_file)
            list_name = os.path.splitext(list_basename)[0]
            output_file = os.path.join(list_dir, f"{list_name}_corrupted.txt")
        
        save_corrupted_list(corrupted_files, output_file)
        
        # 如果需要，自动过滤
        if args.filter:
            list_dir = os.path.dirname(args.list_file)
            list_basename = os.path.basename(args.list_file)
            list_name = os.path.splitext(list_basename)[0]
            output_list_file = args.list_file if args.inplace else os.path.join(list_dir, f"{list_name}_clean.txt")
            
            # 读取损坏样本列表
            corrupted_samples = set()
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    corrupted_samples = {line.strip() for line in f.readlines() if line.strip()}
            
            # 读取原始列表
            with open(args.list_file, 'r') as f:
                all_samples = [line.strip() for line in f.readlines() if line.strip()]
            
            # 过滤掉损坏的样本
            filtered_samples = [s for s in all_samples if s not in corrupted_samples]
            
            # 备份原始文件
            backup_file = f"{args.list_file}.bak"
            if os.path.exists(args.list_file):
                import shutil
                shutil.copy(args.list_file, backup_file)
                print(f"原始列表文件已备份到: {backup_file}")
            
            # 保存过滤后的列表
            with open(output_list_file, 'w') as f:
                for sample in filtered_samples:
                    f.write(f"{sample}\n")
            
            removed_count = len(all_samples) - len(filtered_samples)
            print(f"\n过滤完成:")
            print(f"  原始样本数: {len(all_samples)}")
            print(f"  移除样本数: {removed_count}")
            print(f"  保留样本数: {len(filtered_samples)}")
            if args.inplace:
                print(f"  已更新: {output_list_file}")
            else:
                print(f"  过滤后的列表已保存到: {output_list_file}")
    else:
        print("\n✅ 没有发现损坏的文件！")
