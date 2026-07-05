"""Split a labeled Kaggle-style MNIST CSV into stratified train/validation files."""
import argparse
import csv
import os
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_csv(source, train_output, val_output, seed=42):
    source, train_output, val_output = map(Path, (source, train_output, val_output))
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 785 or header[0].lower() != "label":
            raise ValueError("输入必须是含 label 表头的 785 列 MNIST CSV")
        rows = list(reader)
    if any(len(row) != 785 for row in rows):
        raise ValueError("输入包含列数不是 785 的数据行")
    labels = [int(row[0]) for row in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=1 / 8, random_state=seed, stratify=labels
    )
    temporary_outputs = []
    try:
        for output, selected in ((train_output, train_rows), (val_output, val_rows)):
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header); writer.writerows(selected)
            temporary_outputs.append((temporary, output))
        for temporary, output in temporary_outputs:
            os.replace(temporary, output)
    finally:
        for temporary, _ in temporary_outputs:
            if temporary.exists(): temporary.unlink()
    return len(train_rows), len(val_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source"); parser.add_argument("train_output"); parser.add_argument("val_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_count, val_count = split_csv(args.source, args.train_output, args.val_output, args.seed)
    print(f"train={train_count} val={val_count}")


if __name__ == "__main__": main()
