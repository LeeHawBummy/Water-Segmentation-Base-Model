import os
import random
import tifffile
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def _imread_tiff_robust(path):
    """
    多策略读取 GeoTIFF / 瓦片压缩 TIFF。JPEG/LZW 等瓦片偶发解码失败时可换路径重试。
    若仍失败，可安装: pip install imagecodecs
    """
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        pass
    last = None
    for kwargs in ({}, {'maxworkers': 1}):
        try:
            arr = tifffile.imread(path, **kwargs)
            if arr is not None and getattr(arr, 'size', 0) > 0:
                return arr
        except Exception as e:
            last = e
    try:
        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
            if arr is not None and getattr(arr, 'size', 0) > 0:
                return arr
    except Exception as e:
        last = e
    if last is not None:
        raise last
    raise ValueError('empty or unreadable tiff: %s' % path)


def _check_finite_array(name, arr):
    if arr is None or (hasattr(arr, 'size') and arr.size == 0):
        raise ValueError('%s 为空' % name)
    if arr.dtype.kind in 'fc':
        if not np.isfinite(arr).all():
            raise ValueError('%s 含 NaN/Inf' % name)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']  # image shape: [C, H, W]
        C, H, W = image.shape

        # 随机旋转和翻转（所有通道同步）
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            # 对每个通道旋转
            image = np.stack([np.rot90(chan, k) for chan in image], axis=0)
            label = np.rot90(label, k)
            axis = np.random.randint(1, 3)  # 沿H或W轴翻转（避开通道轴0）
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis-1).copy()  # 标签维度是[H, W]，轴减1
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            # 对每个通道旋转
            image = np.stack([ndimage.rotate(chan, angle, order=3, reshape=False) for chan in image], axis=0)
            label = ndimage.rotate(label, angle, order=0, reshape=False)

        # 调整大小（所有通道同步缩放）
        if H != self.output_size[0] or W != self.output_size[1]:
            scale = (self.output_size[0]/H, self.output_size[1]/W)
            image = np.stack([
                zoom(chan, scale, order=3) for chan in image
            ], axis=0)
            label = zoom(label, scale, order=0)

        # 转为Tensor（保持通道维度）
        image = torch.from_numpy(image.astype(np.float32))  # 已包含通道维度[C, H, W]
        label = torch.from_numpy(label.astype(np.float32)).long()
        return {'image': image, 'label': label}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, target_channels=None, transform=None, fallback_dir=None):
        self.transform = transform
        self.split = split
        self.img_size = img_size
        # 对于训练集，使用train.txt
        if split == 'train':
            list_file = os.path.join(list_dir, 'train.txt')
        else:
            list_file = os.path.join(list_dir, self.split + '.txt')
        with open(list_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        # 每行一个样本 stem；跳过空行与 # 注释；兼容一行多列时取第一列
        self.sample_list = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            stem = s.split()[0].strip()
            if stem:
                self.sample_list.append(stem + '\n')
        if not self.sample_list:
            raise FileNotFoundError('列表为空或无效: %s' % list_file)
        self.target_channels = target_channels
        self.data_dir = base_dir  # 多光谱TIFF图像存放路径
        self.fallback_dir = os.path.abspath(fallback_dir) if fallback_dir else None  # 保留原分割时，样本可能在另一目录

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 最多尝试10次读取有效样本，避免无限循环
        max_retries = 10
        for attempt in range(max_retries):
            tiff_name = '?'
            try:
                # 读取TIFF文件名（假设列表中是不带后缀的文件名）
                tiff_name = self.sample_list[idx].strip('\n')
                tiff_path = os.path.join(self.data_dir, f"{tiff_name}_img.tif")  # 多光谱TIFF图像路径
                if not os.path.exists(tiff_path) and self.fallback_dir:
                    tiff_path = os.path.join(self.fallback_dir, f"{tiff_name}_img.tif")
                # 检查文件是否存在
                if not os.path.exists(tiff_path):
                    idx = (idx + 1) % len(self.sample_list)
                    continue
                # 标签与图像同目录（base_dir 或 fallback_dir）
                label_path = os.path.join(self.data_dir, f"{tiff_name}_label.tif")
                if not os.path.exists(label_path) and self.fallback_dir:
                    label_path = os.path.join(self.fallback_dir, f"{tiff_name}_label.tif")
                if not os.path.exists(label_path):
                    idx = (idx + 1) % len(self.sample_list)
                    continue
                
                image = _imread_tiff_robust(tiff_path)
                label = _imread_tiff_robust(label_path)
                label = np.squeeze(label)
                if label.ndim != 2:
                    raise ValueError('label 须为 HxW，当前 shape=%s' % (label.shape,))

                _check_finite_array('image', image)
                _check_finite_array('label', label.astype(np.float32, copy=False))

                # 确保图像维度正确（C, H, W），若TIFF读取为(H, W, C)，需转置
                # 通道数由 target_channels 指定（与 config MODEL.SWIN.IN_CHANS 一致）
                nch = self.target_channels if self.target_channels is not None else 6
                if len(image.shape) == 2:
                    image = image[np.newaxis, ...]
                if len(image.shape) == 3:
                    # 判断维度顺序：如果最后一维较小（<=64），可能是通道维，需要转置
                    if image.shape[-1] <= 64 and image.shape[-1] != image.shape[0]:
                        image = image.transpose(2, 0, 1)  # 转为(C, H, W)
                    C, H, W = image.shape
                    if C != nch:
                        if C < nch:
                            padding = np.zeros((nch - C, H, W), dtype=image.dtype)
                            image = np.concatenate([image, padding], axis=0)
                        else:
                            image = image[:nch, :, :]
                elif len(image.shape) != 3:
                    raise ValueError('image 维度异常: shape=%s' % (image.shape,))

                _check_finite_array('image', image)

                sample = {'image': image, 'label': label}
                if self.transform:
                    sample = self.transform(sample)
                sample['case_name'] = tiff_name
                return sample
                
            except Exception as e:
                # 含 TiffFileError、瓦片解码 RuntimeError、磁盘错误等
                print(f"Warning: Failed to read {tiff_name}: {str(e)}. Trying next sample...")
                idx = (idx + 1) % len(self.sample_list)
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "连续 %d 次读样本失败。请检查 --root_path / --single_dir 是否指向含 {stem}_img.tif 的目录；"
                        "Synapse 非 single_dir 时不要传 .../train_tif 再被拼成 .../train_tif/train_tif。"
                        "当前 data_dir=%s" % (max_retries, self.data_dir)
                    )
        
        raise RuntimeError(
            "读样本失败。当前 data_dir=%s（同上检查路径与列表 stem 是否一致）" % self.data_dir
        )
