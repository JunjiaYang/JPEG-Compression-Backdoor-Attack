# JPEG-Compression-Backdoor-Attack

本仓库整理了 JPEG 压缩后门攻击的核心思路，并给出了在 MNIST 数据集上的复现脚本。

## 内容

- `analysis/paper_summary.md`：论文要点与实验流程总结。
- `src/mnist_jpeg_backdoor.py`：MNIST 实验脚本，支持训练、评估与指标保存。
- `experiments/`：默认的实验结果输出目录（运行脚本后生成）。

## 快速开始

1. 安装依赖（需要 Python 3.9+，并确保能够安装 PyTorch 与 TorchVision）：

   ```bash
   pip install torch torchvision
   ```

2. 推荐配置（以在单张 GPU 或性能良好的 CPU 上 30~40 分钟内完成训练为例）：

   ```text
   训练轮数（epochs）      : 15
   批量大小（batch-size） : 128
   学习率（lr）            : 1e-3（Adam 优化器）
   投毒比例（poison-rate）: 0.1
   JPEG 质量因子          : 10
   目标标签（target-label）: 0
   ```

   以上配置能够在干净准确率与攻击成功率之间取得较稳定的平衡，训练完成后可在 `experiments` 目录下查看对应的日志与模型权重。

3. 运行复现脚本：

   ```bash
   python src/mnist_jpeg_backdoor.py \
     --data-dir data \
     --output-dir experiments \
     --epochs 15 \
     --poison-rate 0.1 \
     --jpeg-quality 10 \
     --target-label 0
   ```

4. 完成训练后，可在 `experiments/mnist_jpeg_backdoor_results.json` 查看每个 epoch 的干净准确率、攻击成功率等指标。

## 参考

- 论文：*JPEG Compression Backdoor Attack*（原文请参考公开发表的版本）。
