# MiniCPM\-Llama\.cpp调优须知

# 算子编写

在 llama\.cpp 里写一个算子并测试的流程如下：

假设我们要编写的算子在python里面的实现如下方代码：

```Python
def **make_pad_mask**(*lengths*: torch.Tensor, *max_len*: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        *>>> *lengths = [5, 3, 2]
        *>>> *make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = *lengths*.size(0)
    *max_len* = *max_len* *if* *max_len* > 0 *else* *lengths*.max().item()
    seq_range = torch.arange(0,
                             *max_len*,
                             *dtype*=torch.int64,
                             *device*=*lengths*.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, *max_len*)
    seq_length_expand = *lengths*.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    *return* mask
```

那我们在llama\.cpp中的实现流程如下：

## 1\.1 声明接口

在相关头文件（如omni\.h）里面声明所写函数借口

```Plain Text
ggml_tensor * make_pad_mask(
        struct ggml_context * *ctx*,
        struct ggml_tensor  * *lengths*,
        int64_t               *max_len*);
```

声明算子函数原型及测试函数。

## 1\.2 实现算子

在cpp文件里面实现算子，如make\_pad\_mask这个算子

```C++
ggml_tensor * make_pad_mask(
        struct ggml_context * *ctx*,
        struct ggml_tensor  * *lengths*,
        int64_t               *max_len*) {
    GGML_ASSERT(lengths != nullptr);
    GGML_ASSERT(ggml_n_dims(lengths) == 1);*  // lengths 应该是一维: (batch_size,)*

    const int64_t batch_size = lengths->ne[0];
    
*    // 如果 max_len 为 0 或负数，需要从 lengths 计算*
    if (max_len <= 0) {
        LOG_ERR("%s: max_len must be > 0, wrong\n", **func**);
        return nullptr;
    }
    
    *// 创建序列 [0, 1, 2, ..., max_len-1]*
    ggml_tensor * seq_range = **ggml_arange**(*ctx*, 0.0f, (float)*max_len*, 1.0f);
    
*    // seq_range reshape 为 PyTorch (1, max_len) → GGML ne=[max_len, 1]*
    ggml_tensor * seq_range_2d = **ggml_reshape_2d**(*ctx*, seq_range, *max_len*, 1);
    
*    // 沿批次维度重复 seq_range 以得到 PyTorch (batch_size, max_len) → GGML ne=[max_len, batch_size]*
    ggml_tensor * seq_range_expand = **ggml_repeat_4d**(*ctx*, seq_range_2d, *max_len*, batch_size, 1, 1);
    
*    // 将 lengths reshape 为 PyTorch (batch_size, 1) → GGML ne=[1, batch_size]*
    ggml_tensor * lengths_2d = **ggml_reshape_2d**(*ctx*, *lengths*, 1, batch_size);
    
*    // 沿序列维度重复 lengths 以得到 PyTorch (batch_size, max_len) → GGML ne=[max_len, batch_size]*
    ggml_tensor * lengths_expand = **ggml_repeat_4d**(*ctx*, lengths_2d, *max_len*, batch_size, 1, 1);
    
    //后面还有一些逻辑……
}

核心思路: 不写kernel，而是组合ggml基础操作（ggml_arange, ggml_reshape_2d, ggml_repeat_4d等）来实现。
```

## 1\.3 编写测试

在cpp文件里面实现算子的测试，如make\_pad\_mask这个算子

```Java
void test_make_pad_mask() {
*    // 1. 加载后端*
    ggml_backend_load_all();
    ggml_backend * backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    
*    // 2. 创建context (no_alloc=true)*
    ggml_init_params params = {.mem_size=16*1024*1024, .no_alloc=true};
    ggml_context * ctx = ggml_init(params);
    
*    // 3. 创建输入tensor*
    ggml_tensor * lengths = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batch_size);
    ggml_set_input(lengths);
    
*    // 4. 调用算子构建计算图*
    ggml_tensor * mask = make_pad_mask(ctx, lengths, max_len);
    
*    // 5. 构建前向传播图*
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mask);
    
*    // 6. 分配buffer*
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    
*    // 7. 设置输入数据*
    ggml_backend_tensor_set(lengths, lengths_data, 0, size);
    
*    // 8. 执行计算*
    ggml_backend_graph_compute(backend, gf);
    
*    // 9. 读取输出验证*
    ggml_backend_tensor_get(mask, mask_data.data(), 0, ggml_nbytes(mask));
    
    //验证……
}
```

## 1\.4 添加CLI入口，进行测试

如对make\_pad\_mask算子的测试，在omni\-cli\.cpp的main\(\)函数里面加入下方内容可实现快速测试。

```Plain Text
*// 测试 make_pad_mask 算子*
    if (argc > 1 && strcmp(argv[1], "--test-pad-mask") == 0) {
        test_make_pad_mask();
        return 0;
    }
```

编译并测试的指令如下：

```Plain Text
打开llamacpp文件夹

# 如果需要在gpu后端上测试，需要配置CMake，启用CUDA
cmake -B build -DGGML_CUDA=ON

cmake --build build --config Release -j$(nproc)
./build/bin/llama-omni-cli --test-pad-mask
```

# 调试方法

## 2\.1 GGML 执行模式

GGML 采用延迟执行模式——调用 ggml\_add\(\) 等函数时不会立即计算，只是构建计算图。这导致：

1. 无法直接打印中间值：算子函数执行时，张量里还没有数据

2. 必须等图执行完：只有 ggml\_backend\_graph\_compute\(\) 之后才能读取结果

3. 需要特殊手段：必须把中间张量"标记"为输出，计算完成后才能获取

## 2\.2 调试方法

既然无法在算子内部直接读取数据，解决思路是：把中间张量"标记"为输出，等计算完成后再统一读取。具体步骤：

1. 构图阶段：在算子的关键位置，将中间张量复制一份并标记为输出节点，同时保存到一个列表中

2. 执行阶段：正常执行 ggml\_backend\_graph\_compute\(\)，此时所有被标记的张量都会被计算

3. 读取阶段：遍历保存的张量列表，用 ggml\_backend\_tensor\_get\(\) 把数据从后端（GPU/CPU）拷贝到主机内存，然后打印查看

这种方法的本质是：用空间换调试能力——额外复制中间结果，牺牲一点内存和性能，换取对中间状态的可观测性。

## 2\.3 调试样例

以make\-pad\-mask算子编写为例：
1、在omni\-impl\.h中添加工具函数:

- print\_tensor\_info：打印张量元信息（维度、名称、大小、形状、类型）

- print\_tensor\_shape：简洁打印张量形状，如 tensor\.shape = \[512, 64\]

- print\_tensor\_data：打印张量实际数据，支持省略中间部分避免输出过长

- debug\_add\_tensor：将中间张量复制并标记为输出，用于计算完成后读取调试

```C++
*//*
*// debugging*
*//*

*// Print basic tensor information*
static void **print_tensor_info**(const ggml_tensor * *tensor*, const char * *prefix* = "") {
    size_t tensor_size = **ggml_nbytes**(*tensor*);
    LOG_INF("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
            *prefix*, **ggml_n_dims**(*tensor*), *tensor*->name, tensor_size,
            *tensor*->ne[0], *tensor*->ne[1], *tensor*->ne[2], *tensor*->ne[3], **ggml_type_name**(*tensor*->type));
}

static void **print_tensor_shape**(ggml_tensor * *t*) {
    **printf**("%s.shape = [", *t*->name);
    for (int i = 0; i < **ggml_n_dims**(*t*); ++i) {
        **printf**("%" PRId64, *t*->ne[i]);
        if (i < **ggml_n_dims**(*t*) - 1) {
            **printf**(", ");
        }
    }
    **printf**("]\n");
}

static void **print_tensor_data**(ggml_tensor * *t*, uint8_t * *data*, int64_t *n*) {
    ggml_type type = *t*->type;
    int64_t * ne = *t*->ne;
    size_t * nb = *t*->nb;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        **printf**("%s.data: [\n", *t*->name);
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == *n* && ne[2] > 2**n*) {
                **printf**("     ..., \n");
                i2 = ne[2] - *n*;
            }
            **printf**("     [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == *n* && ne[1] > 2**n*) {
                    **printf**("      ..., \n");
                    i1 = ne[1] - *n*;
                }
                **printf**("      [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == *n* && ne[0] > 2**n*) {
                        **printf**("..., ");
                        i0 = ne[0] - *n*;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = **ggml_fp16_to_fp32**(*(ggml_fp16_t *) &*data*[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &*data*[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &*data*[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &*data*[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &*data*[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    **printf**("%8.4f", v);
                    if (i0 < ne[0] - 1) **printf**(", ");
                }
                **printf**("],\n");
            }
            **printf**("     ],\n");
        }
        **printf**("    ]\n");
    }
}

static void **debug_add_tensor**(
    struct ggml_context * *ctx0*,
    struct ggml_cgraph * *gf*,
    std::vector<struct ggml_tensor*> * *debug_tensors*,
    const std::set<std::string> & *enabled_debug_tensors*,
    struct ggml_tensor * *cur0*,
    const char * *name*,
    int *il* = -1
) {
    if (!*debug_tensors*) {
        return;
    }

*    // Check if name is in enabled set, if not return*
*    // if (enabled_debug_tensors.find(name) == enabled_debug_tensors.end()) {*
*    //     return;*
*    // }*

    ggml_tensor * cur = **ggml_cpy**(*ctx0*, *cur0*, **ggml_dup_tensor**(*ctx0*, *cur0*));
    std::string cur_name = *il* >= 0 ? std::string(*name*) + "_" + std::**to_string**(*il*) : *name*;
    **ggml_set_name**(cur, cur_name.c_str());
    **ggml_set_output**(cur);
    **ggml_build_forward_expand**(*gf*, cur);
    *debug_tensors*->push_back(cur);
}
```

2、在omni\.cpp中添加全局变量，用于收集调试内容:

```C++
std::vector<ggml_tensor *> debug_tensors;
std::set<std::string> enabled_debug_tensors = {};//用于调试点很多时，选择调试点进行输出
```

3、在所编写算子中插入调试点，并将测试函数中创建的图gf传入算子函数，即算子函数增加一个参数：

```C++
ggml_tensor * make_pad_mask(ggml_context * *ctx*, ggml_tensor * *lengths*, int64_t *max_len*, ggml_cgraph * *gf*) {
    //上方省略
    *// 创建序列 [0, 1, 2, ..., max_len-1]*
    ggml_tensor * seq_range = **ggml_arange**(*ctx*, 0.0f, (float)*max_len*, 1.0f);
    **debug_add_tensor**(*ctx*, *gf*, &debug_print_tensors, enabled_debug_tensors, seq_range, "seq_range");
    
*    // seq_range reshape 为 PyTorch (1, max_len) → GGML ne=[max_len, 1]*
    ggml_tensor * seq_range_2d = **ggml_reshape_2d**(*ctx*, seq_range, *max_len*, 1);
    **debug_add_tensor**(*ctx*, *gf*, &debug_print_tensors, enabled_debug_tensors, seq_range_2d, "seq_range_2d");
    
*    // 沿批次维度重复 seq_range 以得到 PyTorch (batch_size, max_len) → GGML ne=[max_len, batch_size]*
    ggml_tensor * seq_range_expand = **ggml_repeat_4d**(*ctx*, seq_range_2d, *max_len*, batch_size, 1, 1);
    **debug_add_tensor**(*ctx*, *gf*, &debug_print_tensors, enabled_debug_tensors, seq_range_expand, "seq_range_expand");
    
*    // 将 lengths reshape 为 PyTorch (batch_size, 1) → GGML ne=[1, batch_size]*
    ggml_tensor * lengths_2d = **ggml_reshape_2d**(*ctx*, *lengths*, 1, batch_size);
    **debug_add_tensor**(*ctx*, *gf*, &debug_print_tensors, enabled_debug_tensors, lengths_2d, "lengths_2d");
    
*    // 沿序列维度重复 lengths 以得到 PyTorch (batch_size, max_len) → GGML ne=[max_len, batch_size]*
    ggml_tensor * lengths_expand = **ggml_repeat_4d**(*ctx*, lengths_2d, *max_len*, batch_size, 1, 1);
    **debug_add_tensor**(*ctx*, *gf*, &debug_print_tensors, enabled_debug_tensors, lengths_expand, "lengths_expand");
*    // ... 更多操作 ...*
    return mask;
}
```

4、在测试函数中，计算完成后再读取:

```C++
//上方省略
*// 执行计算*
ggml_backend_graph_compute(backend, gf);

*// 读取并打印调试张量*
for (auto * t : debug_tensors) {
    print_tensor_info(t, "DEBUG");
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    **print_tensor_info**(t, t->name);
    **print_tensor_shape**(t);
    **print_tensor_data**(t, data.data(), 4);
}
//下方省略
```

5、效果：

![image\.png](图片和附件/image.png)

如图，我们就可以看到每一步的输出，以便找到每一步操作后得到的结果。

# ggml特点

## 3\.1 维度顺序：与 PyTorch 相反

这是最关键的区别。GGML 的 ne 数组是 PyTorch shape 的逆序。

具体对应关系：

创建张量示例：

```Plain Text
*// PyTorch: torch.zeros(3, 4)  → shape = (3, 4) 即 3 行 4 列*

*// GGML:*
ggml_tensor * mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
*// mat->ne[0] = 4 (列数，最内层)*
*// mat->ne[1] = 3 (行数)*
```

## 3\.2 计算图与延迟执行

GGML 采用声明式编程——调用算子函数时不执行计算，只构建图。

```Plain Text
ggml_add(a, b)     →  只创建节点，不计算
        ↓
ggml_build_forward_expand(graph, result)  →  构建依赖关系
        ↓
ggml_backend_graph_compute(backend, graph)  →  真正执行计算
```

好处：

- 后端可进行图优化（算子融合、内存复用）

- 同一套代码可运行在 CPU/GPU/Metal 等不同硬件



