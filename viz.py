"""Evaluation result visualization."""
from pathlib import Path
import numpy as np


def create_result_figure(images, labels, predictions, metrics, model_name, dataset_name, max_items=12, zh=False):
    import matplotlib.pyplot as plt
    correct=np.where(labels==predictions)[0][:max_items//2]; wrong=np.where(labels!=predictions)[0][:max_items//2]; selected=np.concatenate((correct,wrong))
    if zh:
        heading=f"模型: {model_name} | 数据集: {dataset_name}\n准确率 {metrics['accuracy']:.3f}  精确率 {metrics['precision']:.3f}  召回率 {metrics['recall']:.3f}  F1 {metrics['f1']:.3f}"
    else:
        heading=f"Model: {model_name} | Dataset: {dataset_name}\nAccuracy {metrics['accuracy']:.3f}  Precision {metrics['precision']:.3f}  Recall {metrics['recall']:.3f}  F1 {metrics['f1']:.3f}"
    figure=plt.figure(figsize=(12,8)); figure.suptitle(heading)
    if not len(selected): return figure
    for pos,index in enumerate(selected,1):
        ax=figure.add_subplot(3,4,pos); ok=labels[index]==predictions[index]; color="green" if ok else "red"
        sample_text = f"标签: {labels[index]}\n预测: {predictions[index]}" if zh else f"label: {labels[index]}\npredict: {predictions[index]}"
        ax.imshow(images[index],cmap="gray",vmin=0,vmax=1); ax.set_title(sample_text,color=color)
        for spine in ax.spines.values(): spine.set_edgecolor(color); spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    figure.tight_layout(rect=(0,0,1,.92)); return figure


def save_result(figure, output_dir="results", model_name="model", dataset_name="dataset"):
    from datetime import datetime
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); path=output/f"{Path(model_name).stem}_{Path(dataset_name).stem}_{datetime.now():%Y%m%d_%H%M%S}.png"
    figure.savefig(path,dpi=160,bbox_inches="tight"); return path
