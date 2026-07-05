# 手写数字识别系统

## 环境

```powershell
conda activate yolov11
pip install -r requirements.txt
```

## 使用

综合界面：

```powershell
python GUI.py
```

The GUI defaults to English. Use `python GUI.py -zh` for Chinese.

主界面一次只显示一个功能页；每个子页面使用“返回上一级”回到主菜单。“训练与测试”页训练后手动输入或选择 PTH，并按 checkpoint 元数据或文件名中的 `private`、`mnist`、`private_mnist` 自动匹配测试集。“验证泛化性”页用于选择外部带标签数据集与本项目 PTH。测试图直接嵌入当前页面。

命令行训练：

```powershell
python train.py --dataset private --epochs 10
python train.py --dataset mnist --epochs 10
python train.py --dataset private+mnist --epochs 10
```

命令行测试：

```powershell
python test.py models/mnist_cnn.pth dataset/mnist/mnist_test.csv --save
```

MNIST 训练与测试文件位于 `dataset/mnist/mnist_train.csv` 与 `dataset/mnist/mnist_test.csv`，均包含标签和可选表头。`mnist_predict.csv` 是无标签 Kaggle 预测集，不用于指标评估。私有图片由制作界面保存并转换，也可使用 `0` 至 `9` 标签子目录组织外部图片。

`models/mnist_cnn.pth` 是已有 MNIST 模型，可直接用于测试和实时识别；`models/smoke_private.pth` 仅由合成数据生成，用于证明训练和 checkpoint 流程可运行，不用于识别。

## 测试

```powershell
python -m unittest discover -s tests -v
```
