"""CLI and Tkinter training entry points."""
from datetime import datetime, timedelta
from pathlib import Path
import argparse, queue, threading, time, tkinter as tk
from tkinter import messagebox, ttk


def train_model(selection="private", epochs=10, batch_size=64, learning_rate=0.001, seed=42,
                output_dir="models", data_root="dataset", callback=print):
    import torch
    from torch import nn, optim
    from dataset import build_dataloaders
    from model import DigitCNN
    torch.manual_seed(seed)
    train_loader, test_loader = build_dataloaders(selection, batch_size, seed, data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = datetime.now(); training_started = time.perf_counter()
    callback(f"Started: {started_at:%Y-%m-%d %H:%M:%S}")
    callback(f"Dataset: {selection}")
    callback(f"Device: {device}")
    callback(f"Learning rate: {learning_rate:g} | Batch size: {batch_size} | Epochs: {epochs}")
    model = DigitCNN().to(device); criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train(); loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad()
            output = model(images); loss = criterion(output, labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * labels.size(0); total += labels.size(0); correct += int((output.argmax(1) == labels).sum().item())
        history["loss"].append(loss_sum / total); history["accuracy"].append(correct / total)
        epoch_seconds = time.perf_counter() - epoch_started
        average_seconds = (time.perf_counter() - training_started) / (epoch + 1)
        eta_seconds = average_seconds * (epochs - epoch - 1)
        expected_end = datetime.now() + timedelta(seconds=eta_seconds)
        eta_text = str(timedelta(seconds=max(0, int(eta_seconds))))
        callback(f"Epoch {epoch+1}/{epochs} | Loss {history['loss'][-1]:.4f} | Accuracy {history['accuracy'][-1]:.2%} | Time {epoch_seconds:.1f}s | ETA {eta_text} | Expected end {expected_end:%Y-%m-%d %H:%M:%S}")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cnn_{selection.replace('+','_')}_{datetime.now():%Y%m%d_%H%M%S}.pth"
    torch.save({"model_state_dict": model.state_dict(), "architecture": model.architecture,
                "dataset": selection, "class_mapping": list(range(10)),
                "config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed},
                "history": history}, str(path))
    callback(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S} | Total time: {timedelta(seconds=int(time.perf_counter()-training_started))}")
    callback(f"Saved: {path}"); return path


class TrainWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master); self.title("Train Model"); self.geometry("620x500"); self.messages = queue.Queue()
        self.private, self.mnist = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        row = ttk.Frame(self); row.pack(pady=12); ttk.Checkbutton(row,text="Private",variable=self.private).pack(side="left"); ttk.Checkbutton(row,text="MNIST",variable=self.mnist).pack(side="left")
        self.epochs=tk.StringVar(value="10"); ttk.Label(row,text="Epochs").pack(side="left",padx=(20,2)); ttk.Entry(row,textvariable=self.epochs,width=6).pack(side="left")
        self.button=ttk.Button(self,text="Start",command=self.start); self.button.pack(); self.bar=ttk.Progressbar(self,mode="indeterminate",length=500); self.bar.pack(pady=10)
        self.log=tk.Text(self,height=20,width=75); self.log.pack(); self.after(100,self.poll)
    def start(self):
        selected=[name for name,var in (("private",self.private),("mnist",self.mnist)) if var.get()]
        if not selected: messagebox.showerror("Error","Select at least one dataset"); return
        try: epochs=int(self.epochs.get()); assert epochs > 0
        except Exception: messagebox.showerror("Error","Epochs must be a positive integer"); return
        self.button.config(state="disabled"); self.bar.start()
        def worker():
            try: train_model("+".join(selected),epochs=epochs,callback=self.messages.put)
            except Exception as exc: self.messages.put(f"ERROR: {exc}")
            finally: self.messages.put(None)
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                item=self.messages.get_nowait()
                if item is None: self.bar.stop(); self.button.config(state="normal")
                else: self.log.insert("end",item+"\n"); self.log.see("end")
        except queue.Empty: pass
        self.after(100,self.poll)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["private","mnist","private+mnist"],default="private"); p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--learning-rate",type=float,default=.001); p.add_argument("--seed",type=int,default=42); p.add_argument("--output-dir",default="models"); p.add_argument("--data-root",default="dataset"); return p.parse_args()
def main(): a=parse_args(); train_model(a.dataset,a.epochs,a.batch_size,a.learning_rate,a.seed,a.output_dir,a.data_root)
if __name__ == "__main__": main()
