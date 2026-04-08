# Water Segmentation Base Model（核心用法）

本仓库提供水体分割基础模型的训练/测试脚本，并配套多通道 TIFF（默认 3 通道，可改 6/11/…）的数据读取与列表生成工具。

## 1. 环境

```bash
pip install -r requirements.txt
```

## 2. 数据组织

- **训练/验证/测试**：
  - 目录示例：`data/water/`
- **文件命名**：
  - 图像：`{stem}_img.tif`
  - 标签：`{stem}_label.tif`
- **列表目录**：`lists/lists_water/`
  - `train.txt`、`val.txt`：每行一个 `stem`

如果你想用两级目录结构，也兼容：
- `data/water/train_tif/`、`data/water/test_vol_tif/`

## 3. 生成 train/val 列表

仅用 `data_dir`（单目录）按比例划分：

```bash
python tools/make_dataset_txt.py --water \
  --data_dir data/water \
  --list_dir lists/lists_water \
  --val_ratio 0.2
```

## 4. 输入通道数（默认 3）

- **默认**：3 通道（无需额外参数）
- **改为 N 通道**：用配置或命令行覆盖，例如 6 通道：

```bash
python train.py ... --opts MODEL.SWIN.IN_CHANS 6
```

## 5. 训练

```bash
python train.py \
  --dataset water \
  --cfg configs/swin_large_patch4_window7_224_water.yaml \
  --root_path data/water \
  --list_dir lists/lists_water \
  --output_dir model_out/water_large \
  --img_size 224 \
  --max_epochs 150 \
  --batch_size 6 \
  --use-checkpoint
```

### 迭代训练（按 max_iters）

```bash
python train.py \
  --dataset water \
  --cfg configs/swin_large_patch4_window7_224_water.yaml \
  --root_path data/water \
  --list_dir lists/lists_water \
  --output_dir model_out/water_large_iter40k \
  --img_size 224 \
  --max_iters 40000 \
  --base_lr 0.005 \
  --lr_schedule poly \
  --weight_decay 0.0005 \
  --batch_size 6 \
  --use-checkpoint
```

## 6. 测试 / 推理

```bash
python test.py \
  --dataset water \
  --cfg configs/swin_large_patch4_window7_224_water.yaml \
  --root_path data/water \
  --list_dir lists/lists_water \
  --output_dir model_out/water_large \
  --max_epochs 150 \
  --batch_size 6 \
  --is_savenii
```

## References
- [TransUnet](https://github.com/Beckschen/TransUNet)
- [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
# Water-Segmentation-Base-Model
