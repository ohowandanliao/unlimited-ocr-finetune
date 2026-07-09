"""把多个已编码单样本 pad 成 batch。逐行对齐 debug/debug_batch2_forward.py: collate_batch。

关键：images 必须保持 list（长度=batch），每元素 (images_crop, images_ori)；
images_spatial_crop 也保持 list，不要盲目 cat（要和 UnlimitedOCRModel.forward 的 zip 行为一致）。
"""
import torch
from torch.nn.utils.rnn import pad_sequence


class UOCRCollator:
    def __init__(self, tokenizer):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.pad_id = pad_id

    def __call__(self, samples: list) -> dict:
        input_ids = pad_sequence([s["input_ids"] for s in samples], batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence([s["labels"] for s in samples], batch_first=True, padding_value=-100)
        images_seq_mask = pad_sequence(
            [s["images_seq_mask"] for s in samples], batch_first=True, padding_value=False
        )
        attention_mask = input_ids.ne(self.pad_id)
        images = [s["images"] for s in samples]                      # len == batch
        images_spatial_crop = [s["images_spatial_crop"] for s in samples]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images_seq_mask=images_seq_mask,
            images=images,
            images_spatial_crop=images_spatial_crop,
        )
