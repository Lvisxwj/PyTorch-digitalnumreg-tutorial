# 手写数字识别系统实施方案

## 总体方案

构建基于 PyTorch CNN 的手写数字识别系统，同时支持命令行与 Tkinter 图形界面。系统覆盖私有样本采集、MNIST 格式转换、数据增强、模型训练、交叉测试、结果可视化和实时手写识别。

## 实施内容

1. 建立 `dataset/private/snapshot`、`dataset/private/csv`、`dataset/mnist`、`models` 和 `results` 目录，并声明 PyTorch、NumPy、Pillow、Matplotlib、scikit-learn 依赖。
2. 使用统一的两层卷积 CNN，保持训练、测试和实时识别模型一致，并兼容已有纯 `state_dict` 模型。
3. 提供私有数字绘制、0-9 标签校验、分类连续命名、画布清空、保存提示及增量 CSV 转换功能。
4. 支持 MNIST CSV与标签子目录图片，统一执行灰度化、反色判断、笔迹裁剪、缩放和 28×28 居中处理；private 数据固定种子分层 80/20，训练集执行轻量增强。
5. 训练支持 private、MNIST 和混合数据、CLI 参数、后台 GUI 训练、CPU/CUDA 自动选择及带元数据 checkpoint 保存。
6. 测试支持 CSV及图片目录，输出 accuracy、macro precision、macro recall、macro F1 和混淆矩阵，并生成红绿标注结果图。
7. 实时识别支持选择并加载 `.pth`、绘制时动态推理、概率柱状图与预测数字显示。
8. `GUI.py` 作为综合 GUI 唯一入口，各功能模块同时保留独立入口或可调用接口。

## 接口与格式

- CSV 为无表头的 `label,pixel1,...,pixel784`，像素范围 0-255。
- 图片数据为 `<数据集目录>/<0-9>/<图片文件>`。
- checkpoint 包含模型权重、架构标识、数据集名称、类别映射、训练配置和历史；加载器兼容旧版纯权重文件。
- 私有 PNG 使用黑底白字，外部图片自动判断反色。

## 验收

- 验证图片编号、标签校验、画布清空、状态提示及重复 Convert 不重复写入。
- 验证 CSV 785 列、分层拆分可复现、三种数据组合可训练并保存模型。
- 验证新旧 checkpoint 均可加载，CSV和不同尺寸图片均可测试。
- 验证指标、混淆矩阵、结果图、实时概率和 GUI 后台线程行为。
