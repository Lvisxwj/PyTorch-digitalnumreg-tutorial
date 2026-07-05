Another summer. Same heat. Same humming fluorescence bleeding through the lab, casting everything in that sickly pallor that makes the living look dead and the dead look like they're just resting.

Last summer I drank the pain in shot by shot, watching the ice melt faster than my reasons for staying sober. The hurt blurred the days—Monday bled into Thursday, Thursday into another lonely Friday. Because time doesn't move when you're counting the hours since she left. It just sits there, heavy and thick, like formaldehyde fumes in a sealed room.

That silence stays. Like a scar. Like summer. Like the flicker of those lights that never quite turn off.

# 手写数字识别系统

## env

```powershell
pip install -r requirements.txt
```

## How2use

```powershell
python GUI.py      // Eng
python GUI.py -zh  // Chinese
```

cli：

```powershell
python train.py --dataset private --epochs 10
python train.py --dataset mnist --epochs 10
python train.py --dataset private+mnist --epochs 10

python test.py models/mnist_cnn.pth dataset/mnist/mnist_test.csv --save
```

## reminder

MNIST 训练与测试文件位于 `dataset/mnist/mnist_train.csv` 与 `dataset/mnist/mnist_test.csv`，均包含标签和可选表头。`mnist_predict.csv` 是无标签 Kaggle 预测集，不用于指标评估。私有图片由制作界面保存并转换，也可使用 `0` 至 `9` 标签子目录组织外部图片。

`models/mnist_cnn.pth` 是已有 MNIST 模型，可直接用于测试和实时识别；`models/smoke_private.pth` 仅由合成数据生成，用于证明训练和 checkpoint 流程可运行，不用于识别。

## 测试

```powershell
python -m unittest discover -s tests -v
```
