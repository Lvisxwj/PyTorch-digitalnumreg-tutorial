"""Private digit drawing and incremental CSV conversion."""
from pathlib import Path
import csv
import re
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
from dataset import preprocess_image
from i18n import gui_error, tr

SNAPSHOT_DIR = Path("dataset/private/snapshot")
CSV_PATH = Path("dataset/private/csv/private.csv")
MANIFEST_PATH = Path("dataset/private/csv/converted.txt")
NEW_NAME_RE = re.compile(r"^(?P<label>[0-9])_(?P<index>\d{4,})\.png$", re.IGNORECASE)
LEGACY_NAME_RE = re.compile(r"^(?P<label>[0-9])(?P<index>\d{2})\.png$", re.IGNORECASE)


def parse_snapshot_name(name):
    """Return (label, index) for the current naming convention, else None."""
    match = NEW_NAME_RE.fullmatch(Path(name).name)
    return (int(match.group("label")), int(match.group("index"))) if match else None


def migrate_legacy_snapshots(snapshot_dir=SNAPSHOT_DIR, manifest_path=MANIFEST_PATH):
    """Safely migrate 023.png to 0_0023.png and update the manifest."""
    snapshot_dir, manifest_path = Path(snapshot_dir), Path(manifest_path)
    migrations = []
    if snapshot_dir.exists():
        for source in snapshot_dir.iterdir():
            match = LEGACY_NAME_RE.fullmatch(source.name)
            if not match:
                continue
            target = snapshot_dir / f'{match.group("label")}_{int(match.group("index")):04d}.png'
            migrations.append((source, target))
    conflicts = [target for source, target in migrations if target.exists() and target != source]
    if conflicts:
        raise FileExistsError(f"迁移目标已存在，未修改任何文件：{conflicts[0]}")
    name_map = {source.name: target.name for source, target in migrations}
    for source, target in migrations:
        source.rename(target)
    if manifest_path.exists() and name_map:
        original = manifest_path.read_text(encoding="utf-8").splitlines()
        updated = [name_map.get(name, name) for name in original]
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text("".join(name + "\n" for name in updated), encoding="utf-8")
        temporary.replace(manifest_path)
    return len(migrations)


def next_snapshot_name(label, directory=SNAPSHOT_DIR):
    directory = Path(directory); used = set()
    for path in directory.glob(f"{label}_*.png"):
        parsed = parse_snapshot_name(path.name)
        if parsed and parsed[0] == int(label): used.add(parsed[1])
    index = 0
    while index in used: index += 1
    return f"{label}_{index:04d}.png"


def snapshot_statistics(snapshot_dir=SNAPSHOT_DIR, manifest_path=MANIFEST_PATH):
    """Return latest images and snapshot/converted counts grouped by label."""
    snapshot_dir, manifest_path = Path(snapshot_dir), Path(manifest_path)
    converted = set(manifest_path.read_text(encoding="utf-8").splitlines()) if manifest_path.exists() else set()
    result = {}
    for label in range(10):
        files = [(path, parse_snapshot_name(path.name)) for path in snapshot_dir.glob(f"{label}_*.png")]
        files = [item for item in files if item[1] and item[1][0] == label]
        files = [path for path, parsed in sorted(files, key=lambda item: item[1][1])]
        result[label] = {
            "latest": files[-3:][::-1],
            "snapshot": len(files),
            "converted": sum(1 for path in files if path.name in converted),
        }
    return result


def convert_snapshots(snapshot_dir=SNAPSHOT_DIR, csv_path=CSV_PATH, manifest_path=MANIFEST_PATH, progress=None):
    snapshot_dir, csv_path, manifest_path = map(Path, (snapshot_dir, csv_path, manifest_path))
    migrate_legacy_snapshots(snapshot_dir, manifest_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(manifest_path.read_text(encoding="utf-8").splitlines()) if manifest_path.exists() else set()
    files = [(path, parse_snapshot_name(path.name)) for path in snapshot_dir.glob("*.png")]
    files = [item for item in files if item[1] is not None]
    files = [path for path, parsed in sorted(files, key=lambda item: (item[1][0], item[1][1]))]
    converted, skipped, failed, newly_done = 0, 0, 0, []
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for idx, path in enumerate(files, 1):
            key = path.name
            if key in done: skipped += 1
            else:
                try:
                    with Image.open(path) as image: pixels = preprocess_image(image, auto_invert=False).reshape(-1)
                    writer.writerow([parse_snapshot_name(path.name)[0], *map(int, pixels)])
                    converted += 1; newly_done.append(key)
                except Exception: failed += 1
            if progress: progress(idx, len(files))
    if newly_done:
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write("".join(name + "\n" for name in newly_done))
    return converted, skipped, failed


class LabelWindow(tk.Toplevel):
    def __init__(self, master=None, on_back=None, zh=False):
        super().__init__(master); self.zh=zh; self.title(tr("private_dataset",zh)); self.geometry("620x900")
        self.on_back = on_back; self.protocol("WM_DELETE_WINDOW", self.back)
        migrated = migrate_legacy_snapshots()
        self.image = Image.new("L", (280, 280)); self.draw = ImageDraw.Draw(self.image); self.last = None
        self.canvas = tk.Canvas(self, width=280, height=280, bg="black"); self.canvas.pack(pady=15)
        self.canvas.bind("<Button-1>", self._start); self.canvas.bind("<B1-Motion>", self._paint)
        row = ttk.Frame(self); row.pack(); ttk.Label(row, text="Label (0-9):").pack(side="left")
        self.label = ttk.Entry(row, width=5); self.label.pack(side="left", padx=6)
        ttk.Button(row, text=tr("save",zh), command=self.save).pack(side="left"); ttk.Button(row, text=tr("clear",zh), command=self.clear).pack(side="left", padx=6)
        self.status = ttk.Label(self, text=tr("migrated",zh,count=migrated) if migrated else tr("ready",zh)); self.status.pack(pady=10)
        self.bar = ttk.Progressbar(self, length=360); self.bar.pack()
        ttk.Button(self, text=tr("convert",zh), command=self.convert).pack(pady=8)
        self.stats_frame = ttk.LabelFrame(self, text="Private dataset statistics")
        self.stats_frame.pack(fill="x", padx=20, pady=(2, 12))
        ttk.Label(self.stats_frame, text="Label", width=7).grid(row=0, column=0)
        ttk.Label(self.stats_frame, text="Latest snapshots", width=24).grid(row=0, column=1)
        ttk.Label(self.stats_frame, text="Snapshot / Converted", width=22).grid(row=0, column=2)
        self.stat_images = []
        self.stat_preview_frames, self.stat_count_labels = {}, {}
        for digit in range(10):
            ttk.Label(self.stats_frame, text=str(digit), width=7, anchor="center").grid(row=digit+1, column=0, pady=1)
            preview = ttk.Frame(self.stats_frame, width=120, height=36); preview.grid(row=digit+1, column=1, pady=1); preview.grid_propagate(False)
            count = ttk.Label(self.stats_frame, anchor="center"); count.grid(row=digit+1, column=2)
            self.stat_preview_frames[digit], self.stat_count_labels[digit] = preview, count
        ttk.Separator(self.stats_frame).grid(row=11, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Label(self.stats_frame, text="Total", width=7, anchor="center").grid(row=12, column=0)
        self.total_label = ttk.Label(self.stats_frame, anchor="center"); self.total_label.grid(row=12, column=2)
        self.refresh_statistics(); self.after(1000, self._periodic_refresh)
        ttk.Button(self, text=tr("back",zh), command=self.back).pack(pady=(0, 10))
    def _start(self, event): self.last = (event.x, event.y)
    def _paint(self, event):
        if self.last:
            self.canvas.create_line(*self.last, event.x, event.y, fill="white", width=20, capstyle="round", smooth=True)
            self.draw.line((*self.last, event.x, event.y), fill=255, width=20); self.last = (event.x, event.y)
    def clear(self): self.canvas.delete("all"); self.image = Image.new("L", (280, 280)); self.draw = ImageDraw.Draw(self.image); self.last = None
    def save(self):
        value = self.label.get().strip()
        if value not in tuple(map(str, range(10))): messagebox.showerror(tr("error",self.zh),tr("label_error",self.zh)); return
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True); name = next_snapshot_name(value)
        self.image.resize((28, 28), Image.Resampling.LANCZOS).save(SNAPSHOT_DIR / name)
        self.status.config(text=f"{name} saved to {SNAPSHOT_DIR}")
        self.clear(); self.label.delete(0, "end"); self.label.focus_set(); self.refresh_statistics()
    def convert(self):
        result = convert_snapshots(progress=lambda n, total: (self.bar.configure(maximum=max(total, 1), value=n), self.update_idletasks()))
        self.status.config(text=tr("convert_done",self.zh,converted=result[0],skipped=result[1],failed=result[2]))
        self.refresh_statistics()
    def refresh_statistics(self):
        stats = snapshot_statistics(); self.stat_images = []
        total_snapshot = total_converted = 0
        for digit, values in stats.items():
            frame = self.stat_preview_frames[digit]
            for child in frame.winfo_children(): child.destroy()
            for path in values["latest"]:
                try:
                    with Image.open(path) as source:
                        preview = ImageTk.PhotoImage(source.convert("L").resize((32, 32), Image.Resampling.NEAREST))
                    self.stat_images.append(preview); ttk.Label(frame, image=preview).pack(side="left", padx=2)
                except Exception: pass
            self.stat_count_labels[digit].config(text=f'{values["snapshot"]} / {values["converted"]}')
            total_snapshot += values["snapshot"]; total_converted += values["converted"]
        self.total_label.config(text=f"{total_snapshot} / {total_converted}")
    def _periodic_refresh(self):
        if self.winfo_exists():
            self.refresh_statistics(); self.after(1000, self._periodic_refresh)
    def back(self):
        self.destroy()
        if self.on_back: self.on_back()


def main(): root = tk.Tk(); root.withdraw(); LabelWindow(root); root.mainloop()
if __name__ == "__main__": main()
