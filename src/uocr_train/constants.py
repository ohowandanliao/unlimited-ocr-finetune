"""固定路径、特殊 token、各 mode 的图像预处理预设。

image-token 构造与常量对齐 Unlimited-OCR 官方 forward（infer / infer_multi）。
"""
import os
from pathlib import Path

# 本地 Unlimited-OCR 权重目录（baidu/Unlimited-OCR）。用环境变量指定：
#   export UOCR_MODEL_DIR=/path/to/baidu_Unlimited-OCR
MODEL_DIR = Path(os.environ.get("UOCR_MODEL_DIR", "/path/to/baidu_Unlimited-OCR"))

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
PAGE_TOKEN = "<PAGE>"

# 视觉 token 数公式的两个常量（见 processor.num_queries）
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4

DEFAULT_SINGLE_PROMPT = "<image>document parsing."
DEFAULT_MULTI_PROMPT = "<image>Multi page parsing."   # UOCR 原生(infer_multi):逐页 <PAGE> 输出
# 北极星新任务:多页 -> 一份跨页合并的连续 markdown。用独立 prompt,不覆盖原生逐页行为。
# 核实过 "Multi page parsing." 是 UOCR 专属(DeepSeek-OCR 用的是 Free OCR./Convert to markdown.),
# 故可安全另起一个合并 prompt 让模型学新行为,推理时按 prompt 选合并/逐页。
DEFAULT_MERGE_PROMPT = "<image>Multi page merge."

# 每个 mode 的图像预处理预设（对齐 model.infer / infer_multi）
MODE_PRESETS = {
    "single_gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "single_base": dict(base_size=1024, image_size=1024, crop_mode=False),
    "multi_base": dict(base_size=1024, image_size=1024, crop_mode=False),
}

# HF 环境默认不强制。国内直连 huggingface.co 被墙时自行 export：
#   HF_ENDPOINT=https://hf-mirror.com  HF_HUB_DISABLE_XET=1
HF_ENV = {}
