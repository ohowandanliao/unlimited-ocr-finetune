"""JSONL -> 统一 sample dict。只做读取/路径解析/字段校验，不做任何图像 tensor 变换。"""
import json
from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset

from .constants import MODE_PRESETS, DEFAULT_SINGLE_PROMPT, DEFAULT_MULTI_PROMPT


class UOCRJsonlDataset(Dataset):
    """每行一个样本。支持单页(`image`)和多页(`images`)。

    统一 schema（见 docs/train_uocr_design 第 6 节）：
      {id, source, image|images, mode, prompt, target, target_format, task, meta}
    输出 sample dict：{id, mode, images:[abs_path,...], prompt, target, meta}
    """

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None):
        self.jsonl_path = Path(jsonl_path)
        # 相对图片路径的基准目录，默认取 jsonl 所在目录
        self.image_root = Path(image_root) if image_root else self.jsonl_path.parent
        self.rows: List[dict] = []
        with open(self.jsonl_path) as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                self.rows.append(self._normalize(json.loads(line), ln))

    def _resolve(self, p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else (self.image_root / pp))

    def _normalize(self, row: dict, ln: int) -> dict:
        mode = row.get("mode", "single_gundam")
        if mode not in MODE_PRESETS:
            raise ValueError(f"line {ln}: unknown mode {mode!r}, expected one of {list(MODE_PRESETS)}")
        # 图片：单页 image 或多页 images
        if "images" in row and row["images"]:
            images = [self._resolve(x) for x in row["images"]]
        elif row.get("image"):
            images = [self._resolve(row["image"])]
        else:
            raise ValueError(f"line {ln}: no image/images field")
        if mode.startswith("single") and len(images) != 1:
            raise ValueError(f"line {ln}: mode {mode} expects 1 image, got {len(images)}")
        target = row.get("target")
        if target is None or str(target).strip() == "":
            raise ValueError(f"line {ln}: empty target")
        default_prompt = DEFAULT_MULTI_PROMPT if mode.startswith("multi") else DEFAULT_SINGLE_PROMPT
        return dict(
            id=row.get("id", f"row_{ln}"),
            mode=mode,
            images=images,
            prompt=row.get("prompt", default_prompt),
            target=str(target),
            meta=row.get("meta", {}),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]
