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

AWQ 根据校准时的激活分布保护重要通道。本机 MiniCPM-o 4.5 的入口是：

```text
/cache/shitong/autoAWQ/use_awq/quantized.py
```

模型适配位于：

```text
/cache/shitong/autoAWQ/AutoAWQ/awq/models/minicpmo.py
```

量化配置为 W4、group size 128、GEMM。适配代码在量化阶段切换到 `model.llm`，保存后再复制非权重配置文件。

## GPTQ

GPTQ 对 LLM 各层进行逐层误差补偿量化。本机 MiniCPM-o 4.5 的入口是：

```text
/cache/shitong/autoGPTQ/use_gptq/quantize_minicpmo_o45.py
```

脚本先提取 `full_model.llm`，按 Qwen3 结构做 4 bit GPTQ，再把量化 LLM 与原模型中未量化的多模态权重合并，最终生成可供 transformers 和 vLLM 加载的 checkpoint。

## 新模型适配要点

- MiniCPM-V 4.6、CPMOCR 的 LLM 是 Qwen3.5，需要处理 full attention 和 linear attention 两类层。
- `in_proj_a`、`in_proj_b` 等小维度层通常参与 scaling，但不直接量化。
- GPTQ 合并权重时需要检查 state dict 前缀，避免 `llm.`、`model.language_model.` 或 tied `lm_head` 重复。
- 校准阶段如果 KV cache 持续增长导致 OOM，需要关闭 cache 或清理 `past_key_values`。
- CPMOCR 的 AWQ/GPTQ 与验证脚本已经存在，但本机未找到对应量化输出目录，因此只能确认适配代码完成，不能确认最终模型已经跑完。

相关脚本：

```text
/cache/shitong/CPMOCR/quantize_awq_cpmoocr.py
/cache/shitong/CPMOCR/quantize_gptq_cpmoocr.py
/cache/shitong/CPMOCR/verify_quantized_cpmoocr.py
/cache/shitong/MiniCPM-V-4_6/latest/quantize_awq_instruct.py
/cache/shitong/MiniCPM-V-4_6/latest/quantize_gptq_instruct.py
```
