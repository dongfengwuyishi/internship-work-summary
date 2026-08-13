# 用 AWQ 和 GPTQ 方法量化新模型

## 做了什么

在 MiniCPM-o 4.5 上分别跑通 AWQ 和 GPTQ 的 4 bit 量化，并把方法扩展到 MiniCPM-V、CPMOCR 等模型结构。基本原则是只量化 LLM 主干，视觉、音频、resampler、projector、TTS 和 `lm_head` 等模块保持原精度。

## 通用流程

1. 先确认 LLM 子模块路径和骨干类型，例如 MiniCPM-o 4.5 使用 `model.llm` 和 Qwen3。
2. 在 AutoAWQ/AutoGPTQ 中注册或复用对应的模型适配类，明确 transformer 层位置及不能量化的模块。
3. 准备校准数据。本机默认使用 128 条 Alpaca 文本，主要配置为 W4、group size 128。
4. 只对 LLM 执行量化，随后将未量化的多模态权重、配置、tokenizer 和自定义模型代码合并回输出目录。
5. 重新加载量化 checkpoint，并用 transformers、vLLM 或短生成进行 smoke test。

## AWQ

AWQ 根据校准时的激活分布保护重要通道。为 MiniCPM-o 增加模型适配后，量化阶段切换到 `model.llm`，使用 W4、group size 128、GEMM 配置，保存后再合并非权重配置文件。

## GPTQ

GPTQ 对 LLM 各层进行逐层误差补偿量化。实现先提取 `full_model.llm`，按 Qwen3 结构做 4 bit GPTQ，再把量化 LLM 与原模型中未量化的多模态权重合并，最终生成可供 Transformers 和 vLLM 加载的 checkpoint。

## 新模型适配要点

- MiniCPM-V 4.6、CPMOCR 的 LLM 是 Qwen3.5，需要处理 full attention 和 linear attention 两类层。
- `in_proj_a`、`in_proj_b` 等小维度层通常参与 scaling，但不直接量化。
- GPTQ 合并权重时需要检查 state dict 前缀，避免 `llm.`、`model.language_model.` 或 tied `lm_head` 重复。
- 校准阶段如果 KV cache 持续增长导致 OOM，需要关闭 cache 或清理 `past_key_values`。
- CPMOCR 的 AWQ/GPTQ 适配与验证流程已经完成，但最终量化 checkpoint 未保留，因此不把它表述为完整交付结果。

## OCR 量化中的静默错误

在 MiniCPM-V OCR checkpoint 上发现，Transformers 5.x 加载后可能不报告 missing keys，但 `vpm.*` 视觉塔仍保留随机初始化值。如果直接量化，LLM 会基于错误的视觉特征做校准；仅在保存后覆盖视觉塔权重也无法修复已经完成的量化过程。

处理方式是在量化前从原始 safetensors 恢复视觉塔、resampler 和 merger，并逐张量确认加载值与 checkpoint 一致。校准数据目录不存在或 OCR 文本数量过少时直接报错，避免静默退化到少量固定 prompt。

## 量化产物验证

量化完成后不能只检查模型能否保存，还需要验证：

1. 视觉塔、resampler 和 merger 与 FP checkpoint 位级一致；
2. LLM 中存在数量匹配的 `qweight`、`qzeros` 和 `scales`；
3. packed 模块旁没有残留的全精度 `.weight`；
4. 可选执行一次前向，确认 logits 有限且视觉塔不是初始化状态。
