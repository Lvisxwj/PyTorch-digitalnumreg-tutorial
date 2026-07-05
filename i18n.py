"""Small GUI-only English/Chinese translation helper."""

TEXT = {
    "app_title": ("Handwritten Digit Recognition", "手写数字识别系统"),
    "private_dataset": ("Create Private Dataset", "制作私有数据集"),
    "train_test": ("Train and Test", "训练与测试"),
    "validation": ("Generalization Validation", "验证泛化性"),
    "play": ("Real-time Recognition", "实时识别"),
    "back": ("Back", "返回上一级"), "browse": ("Browse", "浏览"),
    "save_result": ("Save Result", "保存结果"), "saved": ("Saved", "已保存"),
    "error": ("Error", "错误"), "notice": ("Notice", "提示"),
    "ready": ("Ready", "就绪"), "clear": ("Clear", "清空"), "save": ("Save", "保存"),
    "convert": ("Convert", "转换"), "training": ("Training", "训练"),
    "start_training": ("Start Training", "开始训练"),
    "training_complete": ("Training Complete", "训练完成"),
    "training_failed": ("Training Failed", "训练失败"),
    "model_saved": ("Model saved to:", "模型已保存到："),
    "test_auto": ("Test (automatically matches the dataset from PTH metadata or filename)", "测试（根据 PTH 名称或元数据自动匹配测试集）"),
    "test_failed": ("Test Failed", "测试失败"), "auto_testset": ("Automatic test set", "自动测试集"),
    "select_dataset": ("Select at least one dataset", "至少选择一个数据集"),
    "epochs_error": ("Epochs must be a positive integer", "Epochs 必须为正整数"),
    "test_first": ("Run a test first", "请先测试"),
    "external_dataset": ("External Dataset", "外部数据集"),
    "validate": ("Validate", "验证"), "validation_failed": ("Validation Failed", "验证失败"),
    "validate_first": ("Run validation first", "请先验证"),
    "load_failed": ("Load Failed", "加载失败"),
    "label_error": ("Label must be a digit from 0 to 9", "标签必须是 0-9"),
    "migrated": ("Migrated {count} legacy snapshots", "已迁移 {count} 张旧命名图片"),
    "convert_done": ("Done: converted {converted}, skipped {skipped}, failed {failed}", "完成：转换 {converted}，跳过 {skipped}，失败 {failed}"),
}


def tr(key, zh=False, **values):
    text = TEXT[key][1 if zh else 0]
    return text.format(**values)


def gui_error(error, zh=False):
    if zh: return str(error)
    text = str(error)
    replacements = {
        "模型不存在": "Model does not exist", "模型结构不兼容": "Incompatible model architecture",
        "无法从 PTH 名称判断数据集": "Cannot infer the dataset from the PTH filename",
        "请将文件名包含": "Include one of these names in the filename:",
        "CSV 不存在": "CSV does not exist", "数据集为空": "Dataset is empty",
        "图片目录不存在": "Image directory does not exist", "无法读取图片": "Cannot read image",
        "目录中没有按 0-9 子目录组织的图片": "No images were found in 0-9 label subdirectories",
        "未知数据集选择": "Unknown dataset selection", "分层划分要求至少两个类别，且每个类别至少两个样本": "Stratified splitting requires at least two classes and two samples per class",
        "标签或像素越界": "label or pixel value is out of range", "包含非数字内容": "contains non-numeric data",
        "行应为": "row must contain", "列，实际为": "columns; actual:", "表头应为": "header must contain",
        "迁移目标已存在，未修改任何文件": "Migration target already exists; no files were changed",
    }
    for source, target in replacements.items(): text = text.replace(source, target)
    return text
