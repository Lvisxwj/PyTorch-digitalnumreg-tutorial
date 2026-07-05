"""Combined model training and automatic matching test GUI."""
from pathlib import Path
import queue, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from i18n import gui_error, tr


class TrainTestWindow(tk.Toplevel):
    def __init__(self, master=None, on_back=None, zh=False):
        super().__init__(master); self.zh=zh; self.title(tr("train_test",zh)); self.geometry("1100x900")
        self.on_back=on_back; self.protocol("WM_DELETE_WINDOW",self.back); self.messages=queue.Queue(); self.figure=self.figure_canvas=None
        train_box=ttk.LabelFrame(self,text=tr("training",zh)); train_box.pack(fill="x",padx=12,pady=8)
        self.private,self.mnist=tk.BooleanVar(value=True),tk.BooleanVar(value=False); self.epochs=tk.StringVar(value="10")
        ttk.Checkbutton(train_box,text="Private",variable=self.private).pack(side="left",padx=8); ttk.Checkbutton(train_box,text="MNIST",variable=self.mnist).pack(side="left")
        ttk.Label(train_box,text="Epochs").pack(side="left",padx=(20,3)); ttk.Entry(train_box,textvariable=self.epochs,width=6).pack(side="left")
        self.train_button=ttk.Button(train_box,text=tr("start_training",zh),command=self.start_train); self.train_button.pack(side="left",padx=12)
        self.progress=ttk.Progressbar(train_box,mode="indeterminate",length=240); self.progress.pack(side="left",padx=8)
        self.train_log=tk.Text(self,height=6); self.train_log.pack(fill="x",padx=12)
        test_box=ttk.LabelFrame(self,text=tr("test_auto",zh)); test_box.pack(fill="x",padx=12,pady=8)
        self.model_path=tk.StringVar(); ttk.Entry(test_box,textvariable=self.model_path).pack(side="left",fill="x",expand=True,padx=6,pady=8)
        ttk.Button(test_box,text=tr("browse",zh),command=self.browse).pack(side="left"); ttk.Button(test_box,text="Test",command=self.run_test).pack(side="left",padx=6)
        ttk.Button(test_box,text=tr("save_result",zh),command=self.save).pack(side="left",padx=6); ttk.Button(test_box,text=tr("back",zh),command=self.back).pack(side="left",padx=6)
        self.test_output=tk.Text(self,height=8); self.test_output.pack(fill="x",padx=12)
        self.figure_frame=ttk.Frame(self); self.figure_frame.pack(fill="both",expand=True,padx=12,pady=6)
        self.after(100,self.poll)
    def start_train(self):
        selected=[name for name,var in (("private",self.private),("mnist",self.mnist)) if var.get()]
        if not selected: messagebox.showerror(tr("error",self.zh),tr("select_dataset",self.zh)); return
        try: epochs=int(self.epochs.get()); assert epochs>0
        except Exception: messagebox.showerror(tr("error",self.zh),tr("epochs_error",self.zh)); return
        self.train_button.config(state="disabled"); self.progress.start()
        def worker():
            try:
                from train import train_model
                path=train_model("+".join(selected),epochs=epochs,callback=lambda text:self.messages.put(("log",text)))
                self.messages.put(("done",str(path)))
            except Exception as exc: self.messages.put(("error",str(exc)))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                kind,value=self.messages.get_nowait()
                if kind=="log": self.train_log.insert("end",value+"\n"); self.train_log.see("end")
                else:
                    self.progress.stop(); self.train_button.config(state="normal")
                    if kind=="done": self.model_path.set(value); messagebox.showinfo(tr("training_complete",self.zh),f'{tr("model_saved",self.zh)}\n{value}')
                    else: messagebox.showerror(tr("training_failed",self.zh),gui_error(value,self.zh))
        except queue.Empty: pass
        self.after(100,self.poll)
    def browse(self): self.model_path.set(filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")]))
    def run_test(self):
        try:
            from test import embed_figure,evaluate_model_testset,format_metrics
            metrics,images,labels,predictions,selection=evaluate_model_testset(self.model_path.get()); self.metrics=metrics
            self.test_output.delete("1.0","end"); self.test_output.insert("end",f"Automatic test set: {selection}\n"+format_metrics(metrics))
            from viz import create_result_figure
            self.figure=create_result_figure(images,labels,predictions,metrics,Path(self.model_path.get()).name,selection)
            self.figure_canvas=embed_figure(self.figure_frame,self.figure,self.figure_canvas)
        except Exception as exc: messagebox.showerror(tr("test_failed",self.zh),gui_error(exc,self.zh))
    def save(self):
        if self.figure is None: messagebox.showwarning(tr("notice",self.zh),tr("test_first",self.zh)); return
        from viz import save_result
        path=save_result(self.figure,model_name=self.model_path.get(),dataset_name="auto_test"); messagebox.showinfo(tr("saved",self.zh),str(path))
    def back(self):
        self.destroy()
        if self.on_back: self.on_back()
