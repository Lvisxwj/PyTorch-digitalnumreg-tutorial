import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from dataset import preprocess_image, read_mnist_csv, stratified_split
from label import convert_snapshots, migrate_legacy_snapshots, next_snapshot_name, snapshot_statistics
from checkpoint import infer_dataset_selection
from i18n import gui_error, tr


class CoreTests(unittest.TestCase):
    def test_preprocess_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); image=Image.new("L",(100,60),255); ImageDraw.Draw(image).rectangle((40,5,55,55),fill=0)
            pixels=preprocess_image(image); self.assertEqual(pixels.shape,(28,28)); self.assertEqual(int(pixels.max()),255)
            path=root/"data.csv"
            with path.open("w",newline="") as handle:
                writer=csv.writer(handle); writer.writerow(["label",*[f"pixel{i}" for i in range(784)]]); writer.writerow([3,*pixels.reshape(-1)])
            images,labels=read_mnist_csv(path); self.assertEqual(images.shape,(1,28,28)); self.assertEqual(labels.tolist(),[3])

    def test_incremental_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); snapshots=root/"snap"; snapshots.mkdir(); csv_path=root/"private.csv"; manifest=root/"done.txt"
            Image.new("L",(28,28),255).save(snapshots/"0_0000.png")
            self.assertEqual(next_snapshot_name("0",snapshots),"0_0001.png")
            self.assertEqual(convert_snapshots(snapshots,csv_path,manifest),(1,0,0)); self.assertEqual(convert_snapshots(snapshots,csv_path,manifest),(0,1,0))
            self.assertEqual(len(csv_path.read_text().strip().split(",")),785)
            stats=snapshot_statistics(snapshots,manifest)
            self.assertEqual((stats[0]["snapshot"],stats[0]["converted"]),(1,1))
            self.assertEqual([path.name for path in stats[0]["latest"]],["0_0000.png"])

    def test_legacy_migration_updates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); snapshots=root/"snap"; snapshots.mkdir(); manifest=root/"converted.txt"
            Image.new("L",(28,28),255).save(snapshots/"023.png"); manifest.write_text("023.png\n",encoding="utf-8")
            self.assertEqual(migrate_legacy_snapshots(snapshots,manifest),1)
            self.assertTrue((snapshots/"0_0023.png").is_file()); self.assertFalse((snapshots/"023.png").exists())
            self.assertEqual(manifest.read_text(encoding="utf-8"),"0_0023.png\n")

    def test_stratified_split_is_reproducible(self):
        images=np.zeros((20,28,28),dtype=np.uint8); labels=np.repeat([0,1],10)
        a=stratified_split(images,labels,seed=7); b=stratified_split(images,labels,seed=7)
        self.assertTrue(all(np.array_equal(x,y) for pair_a,pair_b in zip(a,b) for x,y in zip(pair_a,pair_b)))

    def test_checkpoint_dataset_inference(self):
        self.assertEqual(infer_dataset_selection("mnist_cnn.pth"),"mnist")
        self.assertEqual(infer_dataset_selection("cnn_private_2026.pth"),"private")
        self.assertEqual(infer_dataset_selection("cnn_private_mnist_2026.pth"),"private+mnist")
        self.assertEqual(infer_dataset_selection("anything.pth",{"dataset":"private+mnist"}),"private+mnist")

    def test_gui_language_switch(self):
        self.assertEqual(tr("back"),"Back"); self.assertEqual(tr("back",True),"返回上一级")
        self.assertNotIn("模型不存在",gui_error(FileNotFoundError("模型不存在: x.pth")))


if __name__ == "__main__": unittest.main()
