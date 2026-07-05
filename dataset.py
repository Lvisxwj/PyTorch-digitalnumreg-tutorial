"""Dataset loading, preprocessing, augmentation and deterministic splitting."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageChops, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def preprocess_image(image, auto_invert=True):
    """Return an MNIST-like uint8 28x28 black-background image."""
    image = image.convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    if auto_invert and float(arr.mean()) > 127:
        image = ImageOps.invert(image)
    bbox = ImageChops.difference(image, Image.new("L", image.size)).getbbox()
    if bbox:
        image = image.crop(bbox)
        image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        output = Image.new("L", (28, 28))
        output.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    else:
        output = Image.new("L", (28, 28))
    return np.asarray(output, dtype=np.uint8)


def read_mnist_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 不存在：{path}")
    rows, labels = [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for line_no, row in enumerate(csv.reader(handle), 1):
            if line_no == 1 and row and row[0].strip().lower() == "label":
                if len(row) != 785:
                    raise ValueError(f"{path} 表头应为 785 列，实际为 {len(row)} 列")
                continue
            if len(row) != 785:
                raise ValueError(f"{path} 第 {line_no} 行应为 785 列，实际为 {len(row)} 列")
            try:
                label = int(row[0]); pixels = np.asarray(row[1:], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_no} 行包含非数字内容") from exc
            if label not in range(10) or np.any((pixels < 0) | (pixels > 255)):
                raise ValueError(f"{path} 第 {line_no} 行标签或像素越界")
            labels.append(label); rows.append(pixels.reshape(28, 28).astype(np.uint8))
    if not rows:
        raise ValueError(f"数据集为空：{path}")
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def read_image_directory(path):
    root = Path(path); images, labels = [], []
    if not root.is_dir():
        raise NotADirectoryError(f"图片目录不存在：{root}")
    for label in range(10):
        folder = root / str(label)
        if not folder.exists():
            continue
        for item in sorted(folder.iterdir()):
            if item.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(item) as source:
                    images.append(preprocess_image(source))
            except Exception as exc:
                raise ValueError(f"无法读取图片：{item} ({exc})") from exc
            labels.append(label)
    if not images:
        raise ValueError(f"目录中没有按 0-9 子目录组织的图片：{root}")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def load_arrays(path):
    return read_mnist_csv(path) if Path(path).suffix.lower() == ".csv" else read_image_directory(path)


def stratified_split(images, labels, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split
    counts = np.unique(labels, return_counts=True)[1]
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError("分层划分要求至少两个类别，且每个类别至少两个样本")
    indexes = np.arange(len(labels))
    train, test = train_test_split(indexes, test_size=test_size, random_state=seed, stratify=labels)
    return (images[train], labels[train]), (images[test], labels[test])


def _torch_dataset(images, labels, augment=False):
    import torch
    from torch.utils.data import Dataset
    import random

    class ArrayDataset(Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, index):
            image = Image.fromarray(images[index])
            if augment:
                angle, scale = random.uniform(-10, 10), random.uniform(0.9, 1.1)
                resized = image.resize((max(1, int(28 * scale)),) * 2, Image.Resampling.BILINEAR)
                stage = Image.new("L", (32, 32)); stage.paste(resized, ((32-resized.width)//2, (32-resized.height)//2))
                stage = stage.rotate(angle, resample=Image.Resampling.BILINEAR)
                x, y = random.randint(0, 4), random.randint(0, 4); image = stage.crop((x, y, x+28, y+28))
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).unsqueeze(0) / 255.0
            return tensor, torch.tensor(int(labels[index]), dtype=torch.long)
    return ArrayDataset()


def build_dataloaders(selection="private", batch_size=64, seed=42, root="dataset"):
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    root = Path(root); train_sets, test_sets = [], []
    choices = selection.split("+")
    if "private" in choices:
        images, labels = read_mnist_csv(root / "private" / "csv" / "private.csv")
        train, test = stratified_split(images, labels, seed=seed)
        train_sets.append(_torch_dataset(*train, augment=True)); test_sets.append(_torch_dataset(*test))
    if "mnist" in choices:
        train_sets.append(_torch_dataset(*read_mnist_csv(root / "mnist" / "mnist_train.csv")))
        test_sets.append(_torch_dataset(*read_mnist_csv(root / "mnist" / "mnist_test.csv")))
    if not train_sets: raise ValueError(f"未知数据集选择：{selection}")
    generator = torch.Generator().manual_seed(seed)
    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    test_ds = test_sets[0] if len(test_sets) == 1 else ConcatDataset(test_sets)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))


def build_test_arrays(selection, root="dataset", seed=42):
    """Build the automatic labeled test set associated with a trained model."""
    root = Path(root); image_parts, label_parts = [], []
    choices = selection.split("+")
    if "private" in choices:
        images, labels = read_mnist_csv(root / "private" / "csv" / "private.csv")
        _, private_test = stratified_split(images, labels, seed=seed)
        image_parts.append(private_test[0]); label_parts.append(private_test[1])
    if "mnist" in choices:
        images, labels = read_mnist_csv(root / "mnist" / "mnist_test.csv")
        image_parts.append(images); label_parts.append(labels)
    if not image_parts: raise ValueError(f"未知数据集选择：{selection}")
    return np.concatenate(image_parts), np.concatenate(label_parts)
