"""UOCR-native processor：把一个 sample dict 编码成 Unlimited-OCR forward 需要的单样本 tensor。

single_gundam / single_base 的 image-token 构造与 labels mask 逐行对齐已实测的
debug/debug_batch2_forward.py。multi_base 按 infer_multi 复刻，标注 [UNVERIFIED]，
等有多页样本再对 infer_multi 的 token 数核验。
"""
import torch
from PIL import ImageOps

from .constants import IMAGE_TOKEN, IMAGE_TOKEN_ID, PATCH_SIZE, DOWNSAMPLE_RATIO, MODE_PRESETS


def _num_queries(size: int) -> int:
    # (size // patch + downsample - 1) // downsample  —— 与 debug 完全一致
    return (size // PATCH_SIZE + DOWNSAMPLE_RATIO - 1) // DOWNSAMPLE_RATIO


class UOCRProcessor:
    def __init__(self, tokenizer, remote_mod):
        self.tokenizer = tokenizer
        self.remote = remote_mod
        self.image_transform = remote_mod.BasicImageTransform(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )

    # ---- 一张图 -> (token_ids, ori_tensor, crop_tensors, [W,H]) ----
    def _encode_one_image(self, image, base_size, image_size, crop_mode):
        tokens = []
        ori = None
        crops = []
        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
                crop_raw = []
            else:
                crop_raw, crop_ratio = self.remote.dynamic_preprocess(image, image_size=image_size)
            global_view = ImageOps.pad(
                image, (base_size, base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            ori = self.image_transform(global_view).to(torch.bfloat16)
            w_crop, h_crop = int(crop_ratio[0]), int(crop_ratio[1])
            if w_crop > 1 or h_crop > 1:
                crops = [self.image_transform(c).to(torch.bfloat16) for c in crop_raw]
            nq = _num_queries(image_size)
            nq_base = _num_queries(base_size)
            tokens = ([IMAGE_TOKEN_ID] * nq_base + [IMAGE_TOKEN_ID]) * nq_base
            tokens += [IMAGE_TOKEN_ID]
            if w_crop > 1 or h_crop > 1:
                tokens += ([IMAGE_TOKEN_ID] * (nq * w_crop) + [IMAGE_TOKEN_ID]) * (nq * h_crop)
            spatial = [w_crop, h_crop]
        else:
            if image_size <= 640:
                image = image.resize((image_size, image_size))
            global_view = ImageOps.pad(
                image, (image_size, image_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            ori = self.image_transform(global_view).to(torch.bfloat16)
            nq = _num_queries(image_size)
            tokens = ([IMAGE_TOKEN_ID] * nq + [IMAGE_TOKEN_ID]) * nq
            tokens += [IMAGE_TOKEN_ID]
            spatial = [1, 1]
        return tokens, ori, crops, spatial

    # ---- 单图路径（single_gundam / single_base），逐行对齐 debug ----
    def _encode_single(self, formatted_text, pil_images, base_size, image_size, crop_mode):
        splits = formatted_text.split(IMAGE_TOKEN)
        assert len(splits) == len(pil_images) + 1, (
            f"<image> 占位数({len(splits) - 1}) 与图片数({len(pil_images)}) 不匹配；"
            f"single 模式 prompt 应恰有 1 个 <image> 且 1 张图"
        )
        ids, seq_mask, oris, all_crops, spatials = [], [], [], [], []
        for text_sep, image in zip(splits, pil_images):
            tsep = self.remote.text_encode(self.tokenizer, text_sep, bos=False, eos=False)
            ids += tsep
            seq_mask += [False] * len(tsep)
            tok, ori, crops, spatial = self._encode_one_image(image, base_size, image_size, crop_mode)
            oris.append(ori)
            all_crops.extend(crops)
            spatials.append(spatial)
            ids += tok
            seq_mask += [True] * len(tok)
        tlast = self.remote.text_encode(self.tokenizer, splits[-1], bos=False, eos=False)
        ids += tlast
        seq_mask += [False] * len(tlast)
        ids = [0] + ids
        seq_mask = [False] + seq_mask
        return ids, seq_mask, oris, all_crops, spatials

    # ---- 多页 base 路径（对齐 infer_multi 的 token 布局 + forward no-crop 分支：每页 273 token）----
    def _encode_multi_base(self, formatted_text, pil_images, image_size):
        splits = formatted_text.split(IMAGE_TOKEN)  # 期望恰好一个 <image> -> 两段
        assert len(splits) == 2, "multi_base expects exactly one <image> placeholder"
        ids, seq_mask, oris, spatials = [], [], [], []
        pre = self.remote.text_encode(self.tokenizer, splits[0], bos=False, eos=False)
        ids += pre
        seq_mask += [False] * len(pre)
        for image in pil_images:  # 一个 <image> 位置塞入全部页
            tok, ori, _crops, spatial = self._encode_one_image(image, image_size, image_size, crop_mode=False)
            oris.append(ori)
            spatials.append(spatial)
            ids += tok
            seq_mask += [True] * len(tok)
        post = self.remote.text_encode(self.tokenizer, splits[1], bos=False, eos=False)
        ids += post
        seq_mask += [False] * len(post)
        ids = [0] + ids
        seq_mask = [False] + seq_mask
        return ids, seq_mask, oris, [], spatials

    def _stack_images(self, oris, crops, base_size, image_size):
        if len(oris) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size), dtype=torch.bfloat16)
            images_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)
        else:
            images_ori = torch.stack(oris, dim=0)
            if crops:
                images_crop = torch.stack(crops, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size), dtype=torch.bfloat16)
        return images_crop, images_ori

    def encode(self, sample: dict) -> dict:
        mode = sample["mode"]
        preset = MODE_PRESETS[mode]
        base_size, image_size, crop_mode = preset["base_size"], preset["image_size"], preset["crop_mode"]

        conv = [
            {"role": "<|User|>", "content": sample["prompt"], "images": list(sample["images"])},
            {"role": "<|Assistant|>", "content": sample["target"]},
        ]
        conv_prompt_only = [
            {"role": "<|User|>", "content": sample["prompt"], "images": list(sample["images"])},
            {"role": "<|Assistant|>", "content": ""},
        ]
        full = self.remote.format_messages(conv, sft_format="plain", system_prompt="")
        prompt_only = self.remote.format_messages(conv_prompt_only, sft_format="plain", system_prompt="")
        pil_images = self.remote.load_pil_images(conv)

        if mode.startswith("multi"):
            ids, seq_mask, oris, crops, spatials = self._encode_multi_base(full, pil_images, image_size)
            p_ids, *_ = self._encode_multi_base(prompt_only, pil_images, image_size)
            spatial_out = spatials  # 多页：per-page [1,1] 列表
        else:
            ids, seq_mask, oris, crops, spatials = self._encode_single(full, pil_images, base_size, image_size, crop_mode)
            p_ids, *_ = self._encode_single(prompt_only, pil_images, base_size, image_size, crop_mode)
            spatial_out = spatials[0]  # 单图：[W,H]（对齐 debug）

        prompt_len = len(p_ids)
        images_crop, images_ori = self._stack_images(oris, crops, base_size, image_size)

        input_ids = torch.tensor(ids, dtype=torch.long)
        images_seq_mask = torch.tensor(seq_mask, dtype=torch.bool)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[images_seq_mask] = -100

        return dict(
            id=sample.get("id"),
            mode=mode,
            input_ids=input_ids,
            labels=labels,
            images_seq_mask=images_seq_mask,
            images=(images_crop, images_ori),
            images_spatial_crop=spatial_out,
            prompt_len=prompt_len,
            image_tokens=int(images_seq_mask.sum().item()),
        )
