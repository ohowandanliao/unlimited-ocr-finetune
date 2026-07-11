# Unlimited-OCR 后训练代码调研记录

日期：2026-07-06

远端根目录：

```text
/mnt1/yixuan/unlimited-ocr-posttrain
```

仓库调研目录：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/research_repos
```

## 目标

目标是基于已经 release 的 `baidu/Unlimited-OCR` 权重做继续 post-training。

目前 `baidu/Unlimited-OCR` 没有公开训练代码，所以调研路线是：

1. 拉取 Unlimited-OCR 官方推理仓库，确认源码和权重接口。
2. 拉取 DeepSeek-OCR / DeepSeek-OCR-2 官方代码，确认 backbone、预处理和推理接口。
3. 搜索社区里真正有训练逻辑的 DeepSeek-OCR 微调仓库。
4. 从可复用训练代码里选一个主模板，后续改成能加载 `baidu/Unlimited-OCR` 权重继续训练。

## 当前保留的活跃仓库

这些仓库保留在 `research_repos/` 下：

| 目录 | 来源 | 作用 | 当前判断 |
| --- | --- | --- | --- |
| `baidu_Unlimited-OCR` | https://github.com/baidu/Unlimited-OCR | Unlimited-OCR 官方仓库 | 已用 GitHub zip 补齐工作区，`git status` 干净；没有训练代码。 |
| `DeepSeek-OCR` | https://github.com/deepseek-ai/DeepSeek-OCR | DeepSeek-OCR 官方仓库 | 保留作模型结构、预处理和推理参考。 |
| `DeepSeek-OCR-2` | https://github.com/deepseek-ai/DeepSeek-OCR-2 | DeepSeek-OCR-2 官方仓库 | `MolSeek-OCR` 基于 OCR-2，保留作对照。 |
| `HaCTang_MolSeek-OCR` | https://github.com/HaCTang/MolSeek-OCR | 主要训练代码候选 | 质量最高，包含 LoRA SFT、full SFT、progressive SFT、merge、config、dataset 等。 |
| `Princeton-AI2-Lab_DeepOCR` | https://github.com/Princeton-AI2-Lab/DeepOCR | 大规模 DeepOCR 复现工程 | 有完整训练栈和 OCR 预训练/SFT 脚本，但工程太重，不适合作为第一版 Unlimited-OCR LoRA 模板。 |
| `baggie11_Finetuning_Deepseek_OCR` | https://github.com/baggie11/Finetuning_Deepseek_OCR | 轻量 LoRA/collator 参考 | 有简单 Trainer/Unsloth/LoRA 流程，质量一般，只作辅助参考。 |
| `WT-ever_deepseek-ocr-lora` | https://github.com/WT-ever/deepseek-ocr-lora | 轻量 DeepSeek-OCR LoRA 参考 | 有 image token 和 label masking collator，质量一般，只作辅助参考。 |

## 归档与删除

低优先级参考目录位于：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/research_repos/archive/low_value_repos_20260706
```

现在只保留 5 个仍有 notebook、配置或代码痕迹的仓库：

```text
DinhXuanKhuong_Finetune-DeepseekOCR-with-Vietnamese-Dataset
ductai05_NLP-Finetune
elihoole_deepseek-ocr-finetutning
ichthyosaur_Post-training-of-DeepSeek-OCR
ricyoung_Justitia-Selective_Vision_Token_Masking_for_PHI-Compliant_OCR
```

以下 10 个归档项已经删除，因为它们为空仓库、工作区没有可读文件，或只有 README，后续参考价值很低：

```text
alphaXiv_DeepSeek-OCR-Dataset
alphaXiv_DeepSeek-OCR-OmniDocBench
DeepSeekOCR-dududuck00
dududuck00_DeepSeekOCR
hannguyen2880_Finetune-DeepSeek-OCR
HoangOnGIT_FinetuneDeepseekOCR
moreWax_DeepseekOCR_finetune
ZooTi9er_unsloth-finetune-deepseek_ocr
Thinh313_FinetuneDeepseekOCR
Thinh59_DeepSeek-OCR-Finetune
```

删除原则：

- 空仓库或实际文件数为 0 的仓库直接删除。
- README-only 仓库删除。
- notebook-only 或有少量训练/配置痕迹的仓库暂时保留，但不作为主线。

## 主要发现

1. `baidu/Unlimited-OCR` 官方仓库没有训练代码。
   目前只有 `infer.py`、README、PDF、assets 和 sglang wheel。

2. `deepseek-ai/DeepSeek-OCR` 和 `deepseek-ai/DeepSeek-OCR-2` 官方仓库也没有训练入口。
   它们主要提供模型、预处理和推理代码。

3. `HaCTang/MolSeek-OCR` 是最适合作为主模板的仓库。
   关键文件包括：

```text
lora_sft.py
misc/full_sft.py
progressive_sft.py
merge_lora_weight.py
lora_sft_config.yaml
DeepSeek_OCR_2.py
dataset.py
```

4. `baggie11_Finetuning_Deepseek_OCR` 和 `WT-ever_deepseek-ocr-lora` 可以用于对照 collator、image token 构造和 label masking，但不建议作为主实现。

5. `Princeton-AI2-Lab/DeepOCR` 有更完整的大规模训练工程，适合后面研究数据配方和训练栈，不适合作为第一版快速 LoRA baseline。

## 建议下一步

新建正式实现目录，不直接改第三方参考仓库：

```text
/mnt1/yixuan/unlimited-ocr-posttrain/uocr_train
```

建议路线：

1. 以 `HaCTang_MolSeek-OCR` 的 LoRA SFT 代码为主模板。
2. 把模型加载从 DeepSeek-OCR-2 改为 `baidu/Unlimited-OCR`。
3. 对齐 Unlimited-OCR 的 remote code 中的图像预处理、image token 构造和 forward labels。
4. 先实现只训 assistant/output token 的 label masking。
5. 检查 Unlimited-OCR 的模块名后再确定 LoRA target modules。
6. 先用 synthetic/toy OCR 样本做 smoke training，再开始设计真实数据生成。

默认先做 LoRA SFT，不先做 full-parameter SFT。
