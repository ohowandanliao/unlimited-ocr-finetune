import argparse
import importlib
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer


ROOT = Path("/mnt1/yixuan/unlimited-ocr-posttrain")
MODEL_DIR = ROOT / "models" / "baidu_Unlimited-OCR"
DEBUG_DIR = ROOT / "debug"
IMAGE_DIR = DEBUG_DIR / "test_images"
OUTPUT_DIR = DEBUG_DIR / "outputs" / "batch2_forward"

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815


def make_sample_image(path: Path, title: str, invoice: str, total: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (560, 360), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        body_font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    lines = [
        (title, title_font),
        (f"Invoice: {invoice}", body_font),
        ("Date: 2026-07-07", body_font),
        ("Item           Qty  Price", body_font),
        ("Paper           2   12.50", body_font),
        ("Pen             5    8.75", body_font),
        (f"Total              {total}", body_font),
    ]
    y = 32
    for text, font in lines:
        draw.text((36, y), text, fill="black", font=font)
        y += 42
    image.save(path)


def ensure_debug_images() -> tuple[Path, Path]:
    img_a = IMAGE_DIR / "batch2_a.png"
    img_b = IMAGE_DIR / "batch2_b.png"
    if not img_a.exists():
        make_sample_image(img_a, "BATCH SAMPLE A", "A-2026-0001", "21.25")
    if not img_b.exists():
        make_sample_image(img_b, "BATCH SAMPLE B", "B-2026-0002", "34.80")
    return img_a, img_b


def build_sample(
    tokenizer,
    remote_mod,
    image_path: Path,
    target_text: str,
    prompt: str = "<image>document parsing.",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
) -> dict:
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": [str(image_path)]},
        {"role": "<|Assistant|>", "content": target_text},
    ]
    prompt_only_conversation = [
        {"role": "<|User|>", "content": prompt, "images": [str(image_path)]},
        {"role": "<|Assistant|>", "content": ""},
    ]

    full_prompt = remote_mod.format_messages(conversation, sft_format="plain", system_prompt="")
    prompt_only = remote_mod.format_messages(prompt_only_conversation, sft_format="plain", system_prompt="")
    images = remote_mod.load_pil_images(conversation)

    image_transform = remote_mod.BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True,
    )
    patch_size = 16
    downsample_ratio = 4

    def encode_formatted_text(formatted_text: str):
        text_splits = formatted_text.split(IMAGE_TOKEN)
        tokenized_str = []
        images_seq_mask = []
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        for text_sep, image in zip(text_splits, images):
            tokenized_sep = remote_mod.text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                    images_crop_raw = []
                else:
                    images_crop_raw, crop_ratio = remote_mod.dynamic_preprocess(
                        image,
                        image_size=image_size,
                    )

                global_view = ImageOps.pad(
                    image,
                    (base_size, base_size),
                    color=tuple(int(x * 255) for x in image_transform.mean),
                )
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                width_crop_num, height_crop_num = crop_ratio
                images_spatial_crop.append([int(width_crop_num), int(height_crop_num)])

                if width_crop_num > 1 or height_crop_num > 1:
                    for crop_img in images_crop_raw:
                        images_crop_list.append(image_transform(crop_img).to(torch.bfloat16))

                num_queries = (image_size // patch_size + downsample_ratio - 1) // downsample_ratio
                num_queries_base = (base_size // patch_size + downsample_ratio - 1) // downsample_ratio
                tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
                tokenized_image += [IMAGE_TOKEN_ID]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += (
                        [IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]
                    ) * (num_queries * height_crop_num)
            else:
                if image_size <= 640:
                    image = image.resize((image_size, image_size))
                global_view = ImageOps.pad(
                    image,
                    (image_size, image_size),
                    color=tuple(int(x * 255) for x in image_transform.mean),
                )
                images_list.append(image_transform(global_view).to(torch.bfloat16))
                images_spatial_crop.append([1, 1])

                num_queries = (image_size // patch_size + downsample_ratio - 1) // downsample_ratio
                tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
                tokenized_image += [IMAGE_TOKEN_ID]

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

        tokenized_sep = remote_mod.text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        tokenized_str = [0] + tokenized_str
        images_seq_mask = [False] + images_seq_mask
        return tokenized_str, images_seq_mask, images_list, images_crop_list, images_spatial_crop

    tokenized_str, images_seq_mask, images_list, images_crop_list, images_spatial_crop = encode_formatted_text(full_prompt)
    prompt_tokenized_str, *_ = encode_formatted_text(prompt_only)
    prompt_len = len(prompt_tokenized_str)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size), dtype=torch.bfloat16)
        images_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)
        spatial_crop = [1, 1]
    else:
        images_ori = torch.stack(images_list, dim=0)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)
        spatial_crop = images_spatial_crop[0]

    input_ids = torch.tensor(tokenized_str, dtype=torch.long)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[images_seq_mask] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
        "images_seq_mask": images_seq_mask,
        "images": (images_crop, images_ori),
        "images_spatial_crop": spatial_crop,
        "prompt_len": prompt_len,
        "image_tokens": int(images_seq_mask.sum().item()),
    }


def collate_batch(samples: list[dict], tokenizer) -> dict:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = pad_sequence([s["input_ids"] for s in samples], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([s["labels"] for s in samples], batch_first=True, padding_value=-100)
    images_seq_mask = pad_sequence(
        [s["images_seq_mask"] for s in samples],
        batch_first=True,
        padding_value=False,
    )
    attention_mask = input_ids.ne(pad_id)

    # Important: this list length is the batch size. Each element belongs to one conversation/sample.
    images = [s["images"] for s in samples]
    images_spatial_crop = [s["images_spatial_crop"] for s in samples]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images_seq_mask": images_seq_mask,
        "images": images,
        "images_spatial_crop": images_spatial_crop,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real batch_size=2 forward example for Unlimited-OCR.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    parser.add_argument("--no-labels", action="store_true", help="Skip labels/loss and only return logits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_a, img_b = ensure_debug_images()
    target_a = "BATCH SAMPLE A\nInvoice: A-2026-0001\nDate: 2026-07-07\nTotal 21.25"
    target_b = "BATCH SAMPLE B\nInvoice: B-2026-0002\nDate: 2026-07-07\nTotal 34.80"

    print("Loading tokenizer/model from", MODEL_DIR)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    print(f"loaded in {time.time() - t0:.1f}s")

    remote_mod = importlib.import_module(model.__class__.__module__)

    if args.mode == "gundam":
        base_size, image_size, crop_mode = 1024, 640, True
    else:
        base_size, image_size, crop_mode = 1024, 1024, False

    samples = [
        build_sample(tokenizer, remote_mod, img_a, target_a, base_size=base_size, image_size=image_size, crop_mode=crop_mode),
        build_sample(tokenizer, remote_mod, img_b, target_b, base_size=base_size, image_size=image_size, crop_mode=crop_mode),
    ]
    batch = collate_batch(samples, tokenizer)

    print("batch_size:", batch["input_ids"].shape[0])
    print("input_ids:", tuple(batch["input_ids"].shape))
    print("labels:", tuple(batch["labels"].shape))
    print("attention_mask:", tuple(batch["attention_mask"].shape), batch["attention_mask"].sum(dim=1).tolist())
    print("images_seq_mask:", tuple(batch["images_seq_mask"].shape), batch["images_seq_mask"].sum(dim=1).tolist())
    print("len(images):", len(batch["images"]))
    print("image tensor shapes:", [(tuple(crop.shape), tuple(ori.shape)) for crop, ori in batch["images"]])
    print("images_spatial_crop:", batch["images_spatial_crop"])
    print("prompt_lens:", [s["prompt_len"] for s in samples])
    print("target_token_counts:", [(s["labels"] != -100).sum().item() for s in samples])

    images_cuda = [(crop.cuda(), ori.cuda()) for crop, ori in batch["images"]]
    labels = None if args.no_labels else batch["labels"].cuda()

    # Put breakpoints below this line when debugging forward.
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                labels=labels,
                images=images_cuda,
                images_seq_mask=batch["images_seq_mask"].cuda(),
                images_spatial_crop=batch["images_spatial_crop"],
                use_cache=False,
                return_dict=True,
            )

    print("logits:", tuple(outputs.logits.shape), outputs.logits.dtype)
    if outputs.loss is not None:
        print("loss:", float(outputs.loss.detach().cpu()))
    print("batch2_forward_ok")


if __name__ == "__main__":
    main()
