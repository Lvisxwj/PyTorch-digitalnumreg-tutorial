"""Compatible checkpoint loading for legacy and project models."""
from pathlib import Path
import inspect


def load_model(path, device=None):
    import torch
    from model import DigitCNN
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"模型不存在：{path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_options = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    payload = torch.load(str(path), **load_options)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload.state_dict()
    model = DigitCNN().to(device)
    try: model.load_state_dict(state)
    except Exception as exc: raise ValueError(f"模型结构不兼容：{path} ({exc})") from exc
    model.eval()
    return model, device, payload


def infer_dataset_selection(path, payload=None):
    """Infer private/mnist/private+mnist from checkpoint metadata or filename."""
    if isinstance(payload, dict) and payload.get("dataset") in {"private", "mnist", "private+mnist"}:
        return payload["dataset"]
    name = Path(path).stem.lower()
    has_private, has_mnist = "private" in name, "mnist" in name or "minst" in name
    if has_private and has_mnist: return "private+mnist"
    if has_private: return "private"
    if has_mnist: return "mnist"
    raise ValueError("无法从 PTH 名称判断数据集，请将文件名包含 private、mnist 或 private_mnist")
