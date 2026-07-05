"""Real-time drawing recognition GUI."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw
from i18n import gui_error, tr


class PlayWindow(tk.Toplevel):
    def __init__(self, master=None, default_model="", on_back=None, zh=False):
        super().__init__(master); self.zh=zh; self.title(tr("play",zh)); self.geometry("900x570")
        self.on_back=on_back; self.protocol("WM_DELETE_WINDOW",self.back)
        self.model_path=tk.StringVar(value=default_model); self.model=None; self.device=None; self.pending=None
        row=ttk.Frame(self); row.pack(fill="x",padx=10,pady=8); ttk.Entry(row,textvariable=self.model_path).pack(side="left",fill="x",expand=True)
        ttk.Button(row,text=tr("browse",zh),command=self.browse).pack(side="left",padx=5); ttk.Button(row,text="Load",command=self.load).pack(side="left")
        body=ttk.Frame(self); body.pack(); self.canvas=tk.Canvas(body,width=420,height=420,bg="black"); self.canvas.grid(row=0,column=0,padx=15)
        self.chart=tk.Canvas(body,width=400,height=420,bg="white"); self.chart.grid(row=0,column=1,padx=15)
        self.canvas.bind("<Button-1>",self.start); self.canvas.bind("<B1-Motion>",self.paint); self.image=Image.new("L",(420,420)); self.draw=ImageDraw.Draw(self.image); self.last=None
        controls=ttk.Frame(self); controls.pack(); ttk.Button(controls,text=tr("clear",zh),command=self.clear).pack(side="left"); self.prediction=ttk.Label(controls,text="Prediction: -",font=("Arial",18)); self.prediction.pack(side="left",padx=20); ttk.Button(controls,text=tr("back",zh),command=self.back).pack(side="left"); self.render(np.zeros(10))
    def browse(self): self.model_path.set(filedialog.askopenfilename(filetypes=[("PyTorch","*.pth")]))
    def load(self):
        try:
            from checkpoint import load_model; self.model,self.device,_=load_model(self.model_path.get()); self.prediction.config(text="Prediction: ready")
        except Exception as exc: messagebox.showerror(tr("load_failed",self.zh),gui_error(exc,self.zh))
    def start(self,event): self.last=(event.x,event.y)
    def paint(self,event):
        if self.last:
            self.canvas.create_line(*self.last,event.x,event.y,fill="white",width=26,capstyle="round",smooth=True); self.draw.line((*self.last,event.x,event.y),fill=255,width=26); self.last=(event.x,event.y)
            if self.pending: self.after_cancel(self.pending)
            self.pending=self.after(40,self.recognize)
    def recognize(self):
        if self.model is None: return
        import torch
        from dataset import preprocess_image
        pixels=preprocess_image(self.image,auto_invert=False).astype("float32")/255
        if not pixels.any(): self.render(np.zeros(10)); self.prediction.config(text="Prediction: -"); return
        with torch.no_grad(): probabilities=torch.softmax(self.model(torch.from_numpy(pixels).reshape(1,1,28,28).to(self.device)),dim=1)[0].cpu().numpy()
        self.render(probabilities); self.prediction.config(text=f"Prediction: {int(probabilities.argmax())}")
    def render(self,values):
        self.chart.delete("all"); base=370
        for digit,value in enumerate(values):
            x=15+digit*38; height=float(value)*300; self.chart.create_rectangle(x,base-height,x+25,base,fill="#3578c9",outline="")
            self.chart.create_text(x+12,base-height-10,text=f"{value:.2f}",font=("Arial",8)); self.chart.create_text(x+12,base+18,text=str(digit))
    def clear(self): self.canvas.delete("all"); self.image=Image.new("L",(420,420)); self.draw=ImageDraw.Draw(self.image); self.last=None; self.render(np.zeros(10)); self.prediction.config(text="Prediction: -")
    def back(self):
        self.destroy()
        if self.on_back: self.on_back()


def main(): root=tk.Tk(); root.withdraw(); PlayWindow(root); root.mainloop()
if __name__ == "__main__": main()
