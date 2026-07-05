"""Model evaluation APIs and the external generalization-validation GUI."""
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from i18n import gui_error, tr


def evaluate_arrays(model_path, images, labels, batch_size=128):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from checkpoint import load_model
    from dataset import _torch_dataset
    model, device, payload = load_model(model_path)
    loader = torch.utils.data.DataLoader(_torch_dataset(images, labels), batch_size=batch_size)
    predictions = []
    with torch.no_grad():
        for inputs, _ in loader:
            predictions.extend(model(inputs.to(device)).argmax(1).cpu().numpy().tolist())
    predictions = np.asarray(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(labels, predictions), "precision": precision,
        "recall": recall, "f1": f1,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(10))),
    }
    return metrics, images.astype("float32") / 255, labels, predictions, payload


def evaluate(model_path, dataset_path, batch_size=128):
    from dataset import load_arrays
    images, labels = load_arrays(dataset_path)
    return evaluate_arrays(model_path, images, labels, batch_size)[:4]


def evaluate_model_testset(model_path, data_root="dataset", batch_size=128):
    from checkpoint import infer_dataset_selection, load_model
    from dataset import build_test_arrays
    _, _, payload = load_model(model_path)
    selection = infer_dataset_selection(model_path, payload)
    images, labels = build_test_arrays(selection, data_root)
    result = evaluate_arrays(model_path, images, labels, batch_size)
    return (*result[:4], selection)


def format_metrics(metrics, zh=False):
    names = ("准确率", "精确率", "召回率", "F1", "混淆矩阵") if zh else ("Accuracy", "Precision", "Recall", "F1", "Confusion matrix")
    return (f"{names[0]}: {metrics['accuracy']:.4f}    {names[1]}: {metrics['precision']:.4f}    "
            f"{names[2]}: {metrics['recall']:.4f}    {names[3]}: {metrics['f1']:.4f}\n"
            f"{names[4]}:\n{metrics['confusion_matrix']}")


def embed_figure(parent, figure, previous=None):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    if previous is not None: previous.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(figure, master=parent); canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return canvas


class ValidationWindow(tk.Toplevel):
    """Validate generalization on another person's labeled CSV/image directory."""
    def __init__(self, master=None, on_back=None, zh=False):
        super().__init__(master); self.zh=zh; self.title(tr("validation",zh)); self.geometry("1050x850")
        self.on_back=on_back; self.protocol("WM_DELETE_WINDOW",self.back); self.figure=self.figure_canvas=None
        self.model=tk.StringVar(); self.data=tk.StringVar()
        for title,var,command in (("PTH",self.model,self.browse_model),(tr("external_dataset",zh),self.data,self.browse_data)):
            row=ttk.Frame(self); row.pack(fill="x",padx=12,pady=5); ttk.Label(row,text=title,width=12).pack(side="left")
            ttk.Entry(row,textvariable=var).pack(side="left",fill="x",expand=True); ttk.Button(row,text=tr("browse",zh),command=command).pack(side="left",padx=5)
        actions=ttk.Frame(self); actions.pack(pady=4); ttk.Button(actions,text=tr("validate",zh),command=self.run).pack(side="left",padx=5)
        ttk.Button(actions,text=tr("save_result",zh),command=self.save).pack(side="left",padx=5); ttk.Button(actions,text=tr("back",zh),command=self.back).pack(side="left",padx=5)
        self.output=tk.Text(self,height=9); self.output.pack(fill="x",padx=12,pady=5)
        self.figure_frame=ttk.Frame(self); self.figure_frame.pack(fill="both",expand=True,padx=12,pady=5)
    def browse_model(self): self.model.set(filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")]))
    def browse_data(self):
        selected=filedialog.askopenfilename(filetypes=[("CSV","*.csv")]); self.data.set(selected or filedialog.askdirectory())
    def run(self):
        try:
            metrics,images,labels,predictions=evaluate(self.model.get(),self.data.get()); self.metrics=metrics
            self.output.delete("1.0","end"); self.output.insert("end",format_metrics(metrics))
            from viz import create_result_figure
            self.figure=create_result_figure(images,labels,predictions,metrics,Path(self.model.get()).name,Path(self.data.get()).name)
            self.figure_canvas=embed_figure(self.figure_frame,self.figure,self.figure_canvas)
        except Exception as exc: messagebox.showerror(tr("validation_failed",self.zh),gui_error(exc,self.zh))
    def save(self):
        if self.figure is None: messagebox.showwarning(tr("notice",self.zh),tr("validate_first",self.zh)); return
        from viz import save_result
        path=save_result(self.figure,model_name=self.model.get(),dataset_name=self.data.get()); messagebox.showinfo(tr("saved",self.zh),str(path))
    def back(self):
        self.destroy()
        if self.on_back: self.on_back()


TestWindow = ValidationWindow


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("dataset"); parser.add_argument("--save",action="store_true"); args=parser.parse_args()
    metrics,images,labels,predictions=evaluate(args.model,args.dataset); print(format_metrics(metrics))
    if args.save:
        from viz import create_result_figure,save_result
        figure=create_result_figure(images,labels,predictions,metrics,Path(args.model).name,Path(args.dataset).name)
        print(save_result(figure,model_name=args.model,dataset_name=args.dataset))
if __name__ == "__main__": main()
