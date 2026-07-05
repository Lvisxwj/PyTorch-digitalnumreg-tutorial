"""Single graphical entry point with one-page-at-a-time navigation."""
import tkinter as tk
from tkinter import ttk
import argparse
from i18n import tr


class MainWindow(tk.Tk):
    def __init__(self, zh=False):
        super().__init__(); self.zh=zh; self.title(tr("app_title",zh)); self.geometry("520x470"); self.resizable(False,False)
        ttk.Label(self,text=tr("app_title",zh),font=("Microsoft YaHei",24,"bold")).pack(pady=40)
        ttk.Button(self,text=tr("private_dataset",zh),command=self.open_label,width=32).pack(pady=7)
        ttk.Button(self,text=tr("train_test",zh),command=self.open_train_test,width=32).pack(pady=7)
        ttk.Button(self,text=tr("validation",zh),command=self.open_validation,width=32).pack(pady=7)
        ttk.Button(self,text=tr("play",zh),command=self.open_play,width=32).pack(pady=7)
        ttk.Label(self,text="made by lvisxwj",foreground="#777777").pack(side="bottom",pady=16)
    def show_main(self): self.deiconify(); self.lift()
    def open_child(self,factory): self.withdraw(); factory(on_back=self.show_main)
    def open_label(self):
        from label import LabelWindow; self.open_child(lambda on_back:LabelWindow(self,on_back,zh=self.zh))
    def open_train_test(self):
        from train_test import TrainTestWindow; self.open_child(lambda on_back:TrainTestWindow(self,on_back,zh=self.zh))
    def open_validation(self):
        from test import ValidationWindow; self.open_child(lambda on_back:ValidationWindow(self,on_back,zh=self.zh))
    def open_play(self):
        from play import PlayWindow; self.open_child(lambda on_back:PlayWindow(self,on_back=on_back,zh=self.zh))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("-zh",action="store_true",help="use Chinese GUI text"); args=parser.parse_args()
    MainWindow(zh=args.zh).mainloop()
if __name__ == "__main__": main()
