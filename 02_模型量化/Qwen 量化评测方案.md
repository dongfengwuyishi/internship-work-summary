# Qwen 量化评测方案

# Qwen 量化评测方案

## 模型来源说明

- **BF16 / 原始模型**：`Qwen/Qwen3.5-35B-A3B`。

- **llama\.cpp GGUF**：`unsloth/Qwen3.5-35B-A3B-GGUF`，直接下载的 GGUF 模型，包含 BF16、Q8\_0、Q4\_K\_M 三档。

- **Qwen3\-8B BF16 GGUF**：`unsloth/Qwen3-8B-GGUF`，下载 `Qwen3-8B-BF16.gguf`，用于 BF16 权重下的 KV cache 量化基线。

- **Qwen3\-8B 量化 GGUF**：`Qwen/Qwen3-8B-GGUF`，下载 `Qwen3-8B-Q8_0.gguf` 和 `Qwen3-8B-Q4_K_M.gguf`，用于权重量化后再做 KV cache 量化的补测。

- **vLLM GPTQ\-Int8**：`JunHowie/Qwen3.5-35B-A3B-GPTQ-Int8`，社区 GPTQ 模型；vLLM 普通 `gptq` 后端要求 `dtype=float16`，且模型包缺少多模态 processor 小配置，测试时从同基座补齐。

- **vLLM GPTQ\-Int4**：`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`，Qwen 官方 GPTQ\-Int4；vLLM 使用 MoE GPTQ 专用 `moe_wna16` 后端。

---

## Task 1：Qwen3\.5\-35B\-A3B 量化对比

**目的**：对比量化前后的**显存**和**速度**。

### llama\.cpp

**测试参数**：prompt=2048 tokens, gen=256 tokens, KV cache=f16/f16, batch=1, 4 卡均分 \(`-ts 1/1/1/1`\)，3 轮平均。

|Model|Framework|显存|Prefill \(t/s\)|Decode \(t/s\)|
|---|---|---|---|---|
|GGUF BF16 \(原始\)|llama\.cpp|66\.5 GiB|2557\.0|102\.5|
|GGUF Q8\_0 \(量化\)|llama\.cpp|36\.6 GiB|8363\.8|156\.6|
|GGUF Q4\_K\_M \(量化\)|llama\.cpp|22\.7 GiB|8532\.6|171\.3|

**少卡复测（在可放入显存的前提下减少 GPU 数）**：

|Model|GPU|显存|Prefill \(t/s\)|Decode \(t/s\)|
|---|---|---|---|---|
|GGUF Q4\_K\_M \(量化\)|1× RTX 4090|21\.0 GiB|7682\.8|181\.3|
|GGUF Q8\_0 \(量化\)|2× RTX 4090|35\.5 GiB|9763\.0|163\.7|

> **显存**：`nvidia-smi` 在 `llama-bench` 运行期间采集的 4 卡峰值之和，实测值。
> 
> 



### vllm

**测试参数**：prompt=2048 tokens, gen=256 tokens, batch=1, TP=4, `max_model_len=4096`, CUDA Graphs 开启

|Model|Framework|`gpu_memory_utilization`|显存 \(权重\)|显存 \(vLLM 预占\)|Prefill \(t/s\)|Decode \(t/s\)|
|---|---|---|---|---|---|---|
|BF16 \(原始\)|vLLM|0\.80|66\.08 GiB|\~76\.8 GiB|17884\.4|219\.4|
|GPTQ\-Int8 \(量化\)|vLLM|0\.60|35\.40 GiB|\~57\.6 GiB|9190\.4|219\.6|
|GPTQ\-Int4 \(量化\)|vLLM|0\.45|21\.56 GiB|\~43\.2 GiB|17363\.5|216\.3|

> **显存测量方法**：
> 
> - **权重显存**：以 vLLM 内部 `Model loading took X GiB` 日志为准，乘以 TP=4 后得到 BF16 66\.08 GiB、GPTQ\-Int8 35\.40 GiB、GPTQ\-Int4 21\.56 GiB。
> 
> - **vLLM 预占显存**：vLLM 实际运行时会预占到 `gpu_memory_utilization` 设定值附近；本次为了开启 CUDA Graphs，分别下调到 BF16 0\.80、INT8 0\.60、GPTQ\-Int4 0\.45。
> 
> 

---

## Task 2：KV Cache 量化对比（llama\.cpp）

**目的**：在 llama\.cpp 里测 KV cache 量化的显存/速度影响。

**上下文长度**：2K / 8K / 32K 各测一次。

**测试参数**：gen=256 tokens, prompt 填充到接近 ctx 长度（用 `-d` 参数），batch=1, Flash Attention 开启，3 轮平均。

### Qwen3\-8B（纯 GQA，36 层全 full attention，单卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|17\.0 GiB|18\.0 GiB|21\.8 GiB|57\.5 / 54\.5 / 45\.0|
|q8\_0 / q8\_0|17\.0 GiB|17\.6 GiB|20\.3 GiB|55\.9 / 53\.7 / 46\.2|
|q4\_0 / q4\_0|16\.9 GiB|17\.4 GiB|19\.2 GiB|55\.9 / 53\.6 / 46\.1|

### Qwen3\-8B\-Q8\_0（权重量化 \+ KV Cache 量化，单卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|8\.6 GiB|9\.4 GiB|12\.8 GiB|100\.2 / 91\.2 / 67\.5|
|q8\_0 / q8\_0|8\.5 GiB|9\.0 GiB|10\.9 GiB|96\.0 / 89\.9 / 70\.3|
|q4\_0 / q4\_0|8\.4 GiB|8\.7 GiB|9\.7 GiB|96\.2 / 90\.1 / 69\.8|

### Qwen3\-8B\-Q4\_K\_M（权重量化 \+ KV Cache 量化，单卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|5\.4 GiB|6\.3 GiB|9\.6 GiB|154\.3 / 134\.5 / 88\.6|
|q8\_0 / q8\_0|5\.3 GiB|5\.8 GiB|7\.7 GiB|144\.7 / 131\.3 / 93\.4|
|q4\_0 / q4\_0|5\.3 GiB|5\.5 GiB|6\.6 GiB|145\.0 / 132\.2 / 92\.6|

> 显存为 `nvidia-smi` 4 卡峰值之和，实测值。Decode t/s 为 3 轮平均，实测值。
> 
> 



### Qwen3\.5\-35B\-A3B（混合架构：\(3\*linear\_attention\+1\*full\_attention\)\*10，4 卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|66\.5 GiB|66\.7 GiB|67\.7 GiB|101\.8 / 98\.8 / 93\.6|
|q8\_0 / q8\_0|66\.5 GiB|66\.6 GiB|67\.5 GiB|99\.5 / 97\.2 / 90\.5|
|q4\_0 / q4\_0|66\.5 GiB|66\.6 GiB|67\.4 GiB|99\.6 / 97\.2 / 89\.7|

### Qwen3\.5\-35B\-A3B\-Q8\_0（权重量化 \+ KV Cache 量化，4 卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|36\.7 GiB|36\.9 GiB|37\.9 GiB|156\.2 / 150\.8 / 138\.7|
|q8\_0 / q8\_0|36\.8 GiB|37\.0 GiB|38\.0 GiB|151\.3 / 148\.1 / 133\.1|
|q4\_0 / q4\_0|36\.8 GiB|37\.0 GiB|37\.8 GiB|150\.5 / 147\.0 / 131\.1|

### Qwen3\.5\-35B\-A3B\-Q4\_K\_M（权重量化 \+ KV Cache 量化，4 卡）

|KV 配置|ctx=2K 显存|ctx=8K 显存|ctx=32K 显存|Decode t/s \(2K / 8K / 32K\)|
|---|---|---|---|---|
|f16 / f16|22\.8 GiB|23\.1 GiB|24\.0 GiB|172\.2 / 164\.6 / 150\.1|
|q8\_0 / q8\_0|22\.9 GiB|23\.2 GiB|24\.1 GiB|165\.4 / 160\.7 / 144\.2|
|q4\_0 / q4\_0|22\.9 GiB|23\.1 GiB|23\.9 GiB|164\.3 / 159\.6 / 141\.1|

> 显存为 `nvidia-smi` 4 卡峰值之和，实测值。Decode t/s 为 3 轮平均，实测值。
> 
> 



---



## 关键结论

### 权重量化（Task 1）

1. **显存节省显著**：Q4 量化将 66\.5 GiB 压缩到 22\.7 GiB（\-66%），INT8 压缩到 35–36\.6 GiB（\-45\~47%）。

2. **Decode 提速**（llama\.cpp）：Q4\_K\_M 比 BF16 快 **67%**（171 vs 102 t/s），主因是权重更小减轻了显存带宽压力。

3. **vLLM 量化收益体现在 CUDA Graph 可用性和服务余量**：下调 `gpu_memory_utilization` 后三档都能开启 CUDA Graphs，但 BF16 的 KV 余量很小；INT8/INT4 在明显更低预占下仍保留约 95K–96K tokens 的 KV 容量。

### KV Cache 量化（Task 2）

4. **对纯 dense 模型（Qwen3\-8B）有效**：ctx=32K 下 q4\_0 比 f16 省 **2\.6 GiB**（21\.8 → 19\.2 GiB），符合理论预期。

5. **对混合架构模型（Qwen3\.5\-35B\-A3B）收效甚微**：ctx=32K 下 q4\_0 仅省 **0\.3 GiB**。因为 40 层中只有 10 层是 full attention（`full_attention_interval=4`），其余 30 层 DeltaNet 的 state 大小固定，不受 KV 量化影响。

6. **KV 量化不提速**：在所有配置下 decode 速度基本持平或略降（dequant 开销抵消了带宽收益）。



