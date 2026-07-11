import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer


DEFAULT_ROOT = Path("/mnt1/yixuan/unlimited-ocr-posttrain")
DEFAULT_MODEL = DEFAULT_ROOT / "models" / "baidu_Unlimited-OCR"
DEFAULT_IMAGE = DEFAULT_ROOT / "debug" / "test_images" / "sample_receipt.png"
DEFAULT_OUTPUT = DEFAULT_ROOT / "debug" / "outputs" / "single_image"


def make_sample_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (960, 640), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        body_font = ImageFont.truetype("DejaVuSansMono.ttf", 28)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    lines = [
        ("UNLIMITED OCR DEBUG", title_font),
        ("Invoice No: UOCR-2026-0706", body_font),
        ("Date: 2026-07-06", body_font),
        ("Item                Qty    Price", body_font),
        ("Notebook paper       2     12.50", body_font),
        ("Black pen            5      8.75", body_font),
        ("Total                     21.25", body_font),
    ]
    y = 60
    for text, font in lines:
        draw.text((70, y), text, fill="black", font=font)
        y += 64
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug baidu/Unlimited-OCR with transformers model.infer.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-save-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", str(DEFAULT_ROOT / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_ROOT / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not image_path.exists():
        print(f"Sample image not found; creating {image_path}")
        make_sample_image(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
    else:
        base_size, image_size, crop_mode = 1024, 1024, False

    print("model_dir:", model_dir)
    print("image:", image_path)
    print("output_dir:", output_dir)
    print("gpu:", args.gpu)
    print("mode:", args.mode, "base_size:", base_size, "image_size:", image_size, "crop_mode:", crop_mode)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), torch.version.cuda)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    print(f"tokenizer loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    print(f"model loaded in {time.time() - t1:.1f}s")

    t2 = time.time()
    result = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=args.max_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        temperature=args.temperature,
        save_results=not args.no_save_results,
    )
    print(f"infer finished in {time.time() - t2:.1f}s")
    print("result type:", type(result))
    if result is not None:
        print("result:", result)
    print("output files:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(" ", path, path.stat().st_size)


if __name__ == "__main__":
    main()
