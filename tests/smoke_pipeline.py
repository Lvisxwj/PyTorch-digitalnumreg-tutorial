"""End-to-end remote smoke test using disposable synthetic data."""
import csv
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root=Path(sys.argv[1]); csv_path=root/"dataset/private/csv/private.csv"; csv_path.parent.mkdir(parents=True,exist_ok=True)
with csv_path.open("w",newline="") as handle:
    writer=csv.writer(handle)
    for label in range(10):
        for sample in range(10):
            image=Image.new("L",(28,28)); draw=ImageDraw.Draw(image)
            x=3+label*2; draw.line((x,3,x,24),fill=255,width=2); draw.line((3,4+sample%8,24,4+sample%8),fill=255,width=1)
            writer.writerow([label,*np.asarray(image,dtype=np.uint8).reshape(-1)])

from train import train_model
model_path=train_model("private",epochs=1,batch_size=32,output_dir=root/"models",data_root=root/"dataset")
from test import evaluate
metrics,images,labels,predictions=evaluate(model_path,csv_path)
from viz import create_result_figure,save_result
figure=create_result_figure(images,labels,predictions,metrics,Path(model_path).name,csv_path.name)
result=save_result(figure,root/"results",Path(model_path).name,csv_path.name)
assert Path(model_path).is_file() and Path(result).is_file()
print(f"checkpoint={model_path}")
print(f"result={result}")
