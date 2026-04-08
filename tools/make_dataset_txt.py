from collections import defaultdict
from itertools import chain
from os.path import join, split, exists
import numpy as np
import os
import sys

import pandas as pd
from argparse import ArgumentParser
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--name", default="datasets", type=str)
parser.add_argument("--n_jobs", default=10, type=int)
parser.add_argument("--data", default=".npz", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--nnunet",
                    default="/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/")

# 使用 --water / --synapse 时由 __main__ 单独解析，避免与上方 parser 冲突
if ('--water' not in sys.argv) and ('--synapse' not in sys.argv):
    args = parser.parse_args()
else:
    args = None

seed = 1234


def chain(lst: list[list]):
    out = []
    for l in lst:
        out.extend(l)
    return out


def npz_csv():
    from deep_utils import DirUtils
    datasets_config = {
        # 'CT_CORONARY': {
        #     'data_dir': f'{args.nnunet}/Dataset002_china_narco/nnUNetPlans_2d',
        #     'num_classes': 3 + 1,  # plus background
        #     'predict_head': 1
        # },
        'MRI_MM': {
            'data_dir': f'{args.nnunet}/Dataset001_mm/nnUNetPlans_2d',
            'num_classes': 3 + 1,  # plus background
            'predict_head': 0
        },
    }

    samples = []
    columns = ["data_dir", "predict_head", "n_classes"]

    for dataset_name, config in datasets_config.items():
        data_files = DirUtils.list_dir_full_path(config['data_dir'], interest_extensions=args.data)
        split_path = config['data_dir'] + "_split"
        if exists(split_path):
            data = DirUtils.list_dir_full_path(split_path, return_dict=True, interest_extensions=".npz")
            seg_img_samples = dict()
            for key, val in tqdm(data.items(), desc="getting data"):
                item = key.replace("_seg", "").replace("_img", "")
                seg_img_samples[item] = val

            file_samples = defaultdict(list)
            for key, val in tqdm(seg_img_samples.items(), desc="Getting final data"):
                item = "_".join(k for k in key.split("_")[:-1])
                file_samples[item].append(val)
        else:
            file_samples = []
        if args.split:
            split_path = DirUtils.split_extension(config['data_dir'], suffix="_split")
            os.makedirs(split_path, exist_ok=True)
        else:
            split_path = None
        print("Getting ready for the data splitting!")
        samples_ = Parallel(n_jobs=args.n_jobs)(
            delayed(process_file)(config, split_path, filepath, file_samples) for filepath in tqdm(data_files))
        samples.extend(samples_)

    # 训练:验证 = 5:5（test_size=0.5 表示 50% 为验证集）
    train, val = train_test_split(samples, test_size=0.5, random_state=seed)
    csv_file_path = f'./lists/{args.name}/'

    train = chain(train)
    val = chain(val)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    pd.DataFrame(train, columns=columns).to_csv(csv_file_path + "/train.txt", index=False)
    pd.DataFrame(val, columns=columns).to_csv(csv_file_path + "/val.txt", index=False)


def process_file(config, split_path, filepath, file_samples):
    filename = split(filepath)[-1].replace(".npz", "")
    if split_path and filename not in file_samples:
        # print(filename)
        samples = []
        file_data = np.load(filepath)
        img = file_data['data']
        seg = file_data['seg']
        for z_index in range(img.shape[1]):
            img_ = img[:, z_index, ...]
            seg_ = seg[:, z_index, ...]
            img_path = join(split_path,
                            f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}')}")
            # seg_path = join(split_path,
            #                 f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}_seg')}")
            if not exists(img_path):
                seg_ = seg_.squeeze(0)
                seg_[seg_ < 0] = 0
                np.savez(img_path, image=img_.squeeze(0), label=seg_)
            samples.append(
                [img_path,
                 config['predict_head'],
                 config['num_classes'],
                 ]
            )
            # np.savez(seg_path, seg_)
    else:
        samples = [[
            filepath,
            config['predict_head'],
            config['num_classes'],
        ]]

    return samples

def process_tiff_files(data_dir, split_path):
    # 遍历所有多光谱TIFF文件
    tiff_files = [f for f in os.listdir(data_dir) if f.endswith('.tiff') and '_label' not in f]
    with open(join(split_path, 'train.txt'), 'w') as f:
        for tiff in tiff_files:
            # 写入不带后缀的文件名（如"case0001"）
            f.write(tiff.replace('.tiff', '') + '\n')


def _backup_list_files(list_dir):
    """若存在 train.txt / val.txt，先复制为 train.txt.bak、val.txt.bak 保留原分割副本。"""
    import shutil
    for name in ('train.txt', 'val.txt'):
        src = join(list_dir, name)
        if os.path.isfile(src):
            dst = src + '.bak'
            shutil.copy2(src, dst)
            print("  已备份: %s -> %s" % (name, name + '.bak'))


def _collect_tif_names(data_dir):
    """从目录收集所有 *_img.tif 且存在 *_label.tif 的样本名。"""
    data_dir = os.path.abspath(data_dir)
    names = []
    for f in os.listdir(data_dir):
        if f.endswith('_img.tif'):
            base = f[:-len('_img.tif')]
            if os.path.isfile(os.path.join(data_dir, f"{base}_label.tif")):
                names.append(base)
    return sorted(names)


def _move_sample(sample_name, from_dir, to_dir):
    """将单个样本的 *_img.tif 与 *_label.tif 从 from_dir 移到 to_dir。"""
    import shutil
    for suf in ('_img.tif', '_label.tif'):
        src = os.path.join(from_dir, sample_name + suf)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(to_dir, sample_name + suf)
        os.makedirs(to_dir, exist_ok=True)
        shutil.move(src, dst)


def merge_synapse_dirs_into_one(out_dir, source_dirs, mode='copy'):
    """
    将多个目录中的 Synapse 成对文件（{stem}_img.tif / {stem}_label.tif）合并到同一目录。
    后出现的源目录若与先前 stem 重复则跳过并打印提示。目标目录中已存在完整一对则跳过该 stem。
    mode: copy | symlink | hardlink | move
    返回合并写入的样本数（不含已跳过/已存在）。
    """
    import shutil
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            '无法在目标路径创建目录（权限不足）: %s\n'
            '请把 --merge_into 改为你有写权限的路径，例如：\n'
            '  ~/Synapse_all  或  %s/Synapse_all'
            % (out_dir, os.path.expanduser('~'))
        ) from e
    stems_done = set()
    n_written = 0
    for d in source_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            raise FileNotFoundError('源目录不存在: %s' % d)
        for stem in _collect_tif_names(d):
            if stem in stems_done:
                print('  跳过重复 stem（已由先前列处理）: %s' % stem)
                continue
            img_dst = os.path.join(out_dir, stem + '_img.tif')
            lab_dst = os.path.join(out_dir, stem + '_label.tif')
            if os.path.isfile(img_dst) and os.path.isfile(lab_dst):
                stems_done.add(stem)
                continue
            for suf in ('_img.tif', '_label.tif'):
                src = os.path.join(d, stem + suf)
                if not os.path.isfile(src):
                    raise FileNotFoundError('缺少文件: %s' % src)
                dst = os.path.join(out_dir, stem + suf)
                if mode == 'copy':
                    shutil.copy2(src, dst)
                elif mode == 'symlink':
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
                elif mode == 'hardlink':
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.link(src, dst)
                elif mode == 'move':
                    shutil.move(src, dst)
                else:
                    raise ValueError('merge mode: %s' % mode)
            stems_done.add(stem)
            n_written += 1
    return n_written


def make_synapse_lists(data_dir, list_dir, val_ratio=0.5, random_state=1234,
                       train_tif_dir=None, test_vol_tif_dir=None, merge_split=False, redistribute=False):
    """
    生成 train.txt 和 val.txt。
    - 若 merge_split 且指定了 train_tif_dir 与 test_vol_tif_dir：合并两目录的样本名单，打乱后按 val_ratio 划分（如 0.5 即 5:5）。要求 data_dir 为合并后的目录（包含全部文件）。
    - 若未 merge_split 且指定 test_vol_tif_dir：训练集名单来自 train_tif_dir，验证集名单来自 test_vol_tif。
    - 否则：仅在 data_dir 内按 val_ratio 划分，默认 5:5。
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    os.makedirs(list_dir, exist_ok=True)
    _backup_list_files(list_dir)

    if merge_split and train_tif_dir and test_vol_tif_dir and os.path.isdir(os.path.abspath(test_vol_tif_dir)):
        # 合并 train_tif + test_vol_tif 名单，打乱后按 val_ratio 划分（如 5:5）
        train_dir = os.path.abspath(train_tif_dir)
        test_dir = os.path.abspath(test_vol_tif_dir)
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"训练目录不存在: {train_dir}")
        n1 = _collect_tif_names(train_dir)
        n2 = _collect_tif_names(test_dir)
        all_names = sorted(set(n1) | set(n2))
        if not all_names:
            raise FileNotFoundError(f"在 {train_dir} 与 {test_dir} 中未找到 *_img.tif 及对应 *_label.tif")
        # 若 data_dir 与 train_dir/test_dir 不同，视为“合并目录”，检查是否包含全部样本（训练时 root_path 指此目录）
        if os.path.abspath(data_dir) not in (train_dir, test_dir):
            in_data = _collect_tif_names(data_dir)
            missing = set(all_names) - set(in_data)
            if missing:
                print("警告: data_dir 中缺少以下样本（训练时请用已合并的目录作 root_path）: %s ..." % list(missing)[:5])
        rng = np.random.default_rng(random_state)
        rng.shuffle(all_names)
        n_val = max(1, int(len(all_names) * val_ratio))
        val_names = all_names[:n_val]
        train_names = all_names[n_val:]
        with open(join(list_dir, 'train.txt'), 'w') as f:
            for name in train_names:
                f.write(name + '\n')
        with open(join(list_dir, 'val.txt'), 'w') as f:
            for name in val_names:
                f.write(name + '\n')
        print(f"列表已生成: {list_dir}（合并 train_tif + test_vol_tif 后按 {100*(1-val_ratio):.0f}:{100*val_ratio:.0f} 划分）")
        print(f"  总样本: {len(all_names)}, 训练: {len(train_names)}, 验证: {len(val_names)}")
        if redistribute:
            set_train, set_val = set(train_names), set(val_names)
            moved_to_train = moved_to_val = 0
            for name in n1:
                if name in set_val:
                    _move_sample(name, train_dir, test_dir)
                    moved_to_val += 1
            for name in n2:
                if name in set_train:
                    _move_sample(name, test_dir, train_dir)
                    moved_to_train += 1
            print(f"  已按列表移动文件: {moved_to_val} 个从 train_tif -> test_vol_tif, {moved_to_train} 个从 test_vol_tif -> train_tif。")
        else:
            print("  训练时请将 root_path 指向包含全部样本的目录，或使用 --redistribute 将文件挪到 train_tif/test_vol_tif 以匹配列表。")
        return
    if test_vol_tif_dir and os.path.isdir(os.path.abspath(test_vol_tif_dir)) and not merge_split:
        # 使用 test_vol_tif 作为验证集：train = train_tif，val = test_vol_tif（不合并、不重划）
        train_dir = os.path.abspath(train_tif_dir) if train_tif_dir else data_dir
        test_dir = os.path.abspath(test_vol_tif_dir)
        train_names = _collect_tif_names(train_dir)
        val_names = _collect_tif_names(test_dir)
        if not train_names:
            raise FileNotFoundError(f"在 {train_dir} 中未找到 *_img.tif 及对应 *_label.tif")
        if not val_names:
            raise FileNotFoundError(f"在 {test_dir} 中未找到 *_img.tif 及对应 *_label.tif")
        with open(join(list_dir, 'train.txt'), 'w') as f:
            for name in train_names:
                f.write(name + '\n')
        with open(join(list_dir, 'val.txt'), 'w') as f:
            for name in val_names:
                f.write(name + '\n')
        print(f"列表已生成: {list_dir}（训练=train_tif，验证=test_vol_tif）")
        print(f"  训练: {len(train_names)}（来自 {train_dir}）")
        print(f"  验证: {len(val_names)}（来自 {test_dir}）")
        return
    # 仅在 data_dir 内按比例划分
    names = _collect_tif_names(data_dir)
    if not names:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 *_img.tif 及对应 *_label.tif")
    rng = np.random.default_rng(random_state)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    val_names = names[:n_val]
    train_names = names[n_val:]
    with open(join(list_dir, 'train.txt'), 'w') as f:
        for name in train_names:
            f.write(name + '\n')
    with open(join(list_dir, 'val.txt'), 'w') as f:
        for name in val_names:
            f.write(name + '\n')
    print(f"列表已生成: {list_dir}")
    print(f"  总样本: {len(names)}, 训练: {len(train_names)}, 验证: {len(val_names)} (约 {100*(1-val_ratio):.0f}:{100*val_ratio:.0f})")


if __name__ == '__main__':
    import argparse
    main_parser = argparse.ArgumentParser(description='生成 train/val 列表（默认 5:5）')
    # 历史兼容：--synapse 是旧名字；建议改用 --water
    main_parser.add_argument('--water', action='store_true', help='water 模式：从目录生成 train.txt 与 val.txt')
    main_parser.add_argument('--synapse', action='store_true', help='(兼容) 同 --water')
    main_parser.add_argument('--data_dir', type=str, default='data/water',
                            help='数据目录：单文件夹模式下放全部 *_img.tif / *_label.tif；配合 train.py 与 --list_dir')
    main_parser.add_argument('--list_dir', type=str, default='lists/lists_water', help='列表输出目录')
    main_parser.add_argument('--val_ratio', type=float, default=0.5,
                            help='验证集占比。训练:验证=3:7 时设为 0.7；5:5 为 0.5。指定 --test_vol_tif 且未 --merge_split 时无效')
    main_parser.add_argument('--merge_into', type=str, default=None,
                            help='将 --train_tif 与 --test_vol_tif 合并到该单一目录（见 --merge_mode），再按 --val_ratio 写 train.txt/val.txt')
    main_parser.add_argument('--merge_mode', type=str, default='copy', choices=('copy', 'symlink', 'hardlink', 'move'),
                            help='合并方式：默认 copy；省空间可用 symlink/hardlink；move 会从源目录移走文件')
    main_parser.add_argument('--train_tif', type=str, default=None, help='训练集目录（与 --test_vol_tif 一起用时生效）')
    main_parser.add_argument('--test_vol_tif', type=str, default=None, help='验证集目录（如 data/Synapse/test_vol_tif）')
    main_parser.add_argument('--merge_split', action='store_true',
                            help='合并 train_tif 与 test_vol_tif 的样本后按 --val_ratio 重新划分（如 8:2 改为 5:5）')
    main_parser.add_argument('--redistribute', action='store_true',
                            help='与 --merge_split 同用：按新列表移动文件，使 train_tif 仅含训练样本、test_vol_tif 仅含验证样本')
    main_parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    main_args = main_parser.parse_args()
    if main_args.water or main_args.synapse:
        data_dir = main_args.data_dir
        test_vol = main_args.test_vol_tif
        merge_split = main_args.merge_split
        redistribute = main_args.redistribute
        train_tif_arg = main_args.train_tif

        if main_args.merge_into:
            if not train_tif_arg or not test_vol:
                raise SystemExit('--merge_into 需要同时指定 --train_tif 与 --test_vol_tif')
            if merge_split or redistribute:
                raise SystemExit('--merge_into 与 --merge_split/--redistribute 不要同时使用（合并后只按单目录划分）')
            srcs = [os.path.abspath(train_tif_arg), os.path.abspath(test_vol)]
            merged = os.path.abspath(os.path.expanduser(main_args.merge_into))
            nw = merge_synapse_dirs_into_one(merged, srcs, mode=main_args.merge_mode)
            data_dir = merged
            print('已合并到单目录: %s（本步新写入样本数: %d）' % (data_dir, nw))
            test_vol = None
            merge_split = False
            redistribute = False

        make_synapse_lists(
            data_dir,
            main_args.list_dir,
            val_ratio=main_args.val_ratio,
            random_state=main_args.seed,
            train_tif_dir=train_tif_arg or data_dir,
            test_vol_tif_dir=test_vol,
            merge_split=merge_split,
            redistribute=redistribute,
        )
    else:
        data_dir = '../SentinelCut'
        split_path = 'lists/lists_Multispectral'
        os.makedirs(split_path, exist_ok=True)
        process_tiff_files(data_dir, split_path)