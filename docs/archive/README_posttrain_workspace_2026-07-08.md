# unlimited-ocr-posttrain

Unlimited-OCR 权重继续 post-training 的工作区。

## 这是什么
基于 DeepSeek-OCR 的训练思路，做一套能加载 `baidu/Unlimited-OCR` 权重后训练的干净工程。
关键事实：Unlimited-OCR 没改模型结构，只把 decoder 注意力从 MLA 换成 128 滑窗 MHA ring-buffer。
详见 `docs/train_uocr_design_2026-07-08.md`（权威设计）。

## 目录地图
- `models/baidu_Unlimited-OCR/` — [CANONICAL] 权重 + remote code。模型代码只认这里。
- `train_uocr/` — 训练工程（实现中）。
- `debug/` — processor/collator 事实源头 `debug_batch2_forward.py` + 测试图（已实测 batch=2 forward/backward 通过）。
- `docs/` — 设计与状态文档；`docs/archive/` 为历史 plan/audit。
- `research_repos/` — [REFERENCE] 参考仓库 clone（DeepSeek-OCR、MolSeek 等），只作参考不在此改动。
- `hf_cache/` — [CACHE] transformers remote-code 自动缓存，可重建。
- `envs/uocr-debug/` — 工作 conda 环境。

## 当前状态
见 `docs/server_status_2026-07-08.md`。

## 约定
- 加载模型：`attn_implementation="eager"`（禁 FlashAttention2，mha 分支没注册 FA2）。
- 拉 HF：`HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（直连被墙）。
- 不改 `models/` 下 remote code；训练代码全在 `train_uocr/`。
