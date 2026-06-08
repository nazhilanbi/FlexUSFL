# FlexUSFL

[English](README.md) | 中文

FlexUSFL 是一个用于 **U-Shape Federated Split Learning for LLM** 的研究型框架。它将大语言模型按 Transformer 层切分成三段：

- **Head Model**：运行在客户端，包含 embedding 和前若干层 Transformer。
- **Server Model**：运行在服务端，包含中间 Transformer 层，承担主要计算。
- **Tail Model**：运行在客户端，包含最后若干层 Transformer 和 LM head。

训练过程中，原始数据始终保留在客户端。本项目通过 socket 在客户端和服务端之间传输中间激活、反向梯度和聚合参数，用于评估不同 split learning 服务端实现、不同模型、不同数据集和不同客户端数量下的训练效率。

## 1. 项目结构

```text
FlexUSFL/
├── experiment/
│   ├── server_run.py        # 单次实验服务端入口
│   ├── client_run.py        # 单次实验客户端入口，会 spawn 多个 client 进程
│   ├── server_run.sh        # 批量评估服务端脚本
│   ├── client_run.sh        # 批量评估客户端脚本
│   └── test.sh              # 简单测试脚本
├── usfl/
│   ├── client/              # 客户端训练逻辑
│   ├── server/              # 服务端版本：v1、v2、v3、merge
│   ├── llm/                 # GPT/LLaMA/Qwen 的模型切分实现
│   ├── socket/              # TCP socket 通信工具
│   └── utils/               # 数据集、日志、模型加载、profiling 等工具
├── vis/
│   ├── dcp.py               # 合并 server/client profiling JSON
│   └── vis.py               # 绘制 timeline 可视化图
├── setup.py
├── README.md
└── README_CN.md
```

## 2. 环境安装

### 2.1 硬件环境

当前脚本默认在多 GPU 机器上运行：

- 服务端默认使用 `cuda:0`。
- 客户端进程默认轮流使用 `cuda:1` 到 `cuda:7`。
- 客户端和服务端默认通过本机 `localhost` 通信。

如果 GPU 编号不同，需要修改：

- 服务端 GPU：运行参数 `-SD` / `--server_device`，默认 `cuda:0`。
- 客户端 GPU：`experiment/client_run.py` 中的 `available_gpus = [1, 2, 3, 4, 5, 6, 7]`。

### 2.2 Python 环境

推荐使用 conda：

```bash
conda create -n flexusfl python=3.10 -y
conda activate flexusfl
```

安装 PyTorch 时请根据本机 CUDA 版本选择合适命令。之后安装项目依赖：

```bash
pip install torch transformers datasets peft bitsandbytes accelerate
pip install numpy pandas matplotlib nltk sentencepiece protobuf
pip install -e .
```

`setup.py` 只负责安装本地 `usfl` 包，不会自动安装完整第三方依赖。

### 2.3 模型和数据集路径

代码默认从 `/share/models` 加载模型：

```text
/share/models/<model_name>
```

例如：

```text
/share/models/qwen/qwen3-0.6b
/share/models/meta-llama/llama3.2-1b
/share/models/qwen/qwen3-1.7b
```

数据集缓存目录在 `usfl/env.py` 中配置为：

```text
/share/datasets/
```

当前批量脚本主要使用：

- `gsm8k`
- `dialogsum`
- `e2e`

## 3. 快速评估

本项目主要通过 `experiment/` 下的两个 shell 脚本评估：

- 终端 1：运行 `experiment/server_run.sh`
- 终端 2：运行 `experiment/client_run.sh`

两个脚本的循环顺序和端口递增规则必须一致。当前脚本都从端口 `8000` 开始，每跑完一个配置后端口加 1。

### 3.1 启动服务端

在第一个终端运行：

```bash
cd /home/lzh/projects/FlexUSFL
conda activate flexusfl
bash experiment/server_run.sh
```

服务端脚本会逐个启动 server，并等待对应客户端连接。

### 3.2 启动客户端

在第二个终端运行：

```bash
cd /home/lzh/projects/FlexUSFL
conda activate flexusfl
bash experiment/client_run.sh
```

`client_run.sh` 不需要手动开多个客户端终端。每个实验配置下，`experiment/client_run.py` 会根据 `-NC` 自动创建多个 client 进程。

### 3.3 必须保持一致的参数

服务端和客户端必须匹配以下参数：

| 参数 | 含义 |
| --- | --- |
| `-NC` | 客户端数量 |
| `-V` | 方法版本，例如 `v1`、`v2`、`v3` |
| `-SP` | 模型切分点 |
| `-M` | 模型名称 |
| `-P` | 通信端口 |
| `-DS` | 数据集名称 |
| `-LAG` | 异构延迟配置 |
| `-QO` | 队列调度策略 |

如果修改了 `server_run.sh` 的数据集、模型、客户端数量、版本或端口，也需要同步修改 `client_run.sh`。

## 4. 当前批量评估配置

当前 `server_run.sh` 和 `client_run.sh` 的主循环配置如下：

| 配置项 | 当前取值 |
| --- | --- |
| 数据集 | `gsm8k`, `dialogsum`, `e2e` |
| 方法版本 | `v1`, `v2`, `v3` |
| 模型 | `qwen/qwen3-0.6b`, `meta-llama/llama3.2-1b`, `qwen/qwen3-1.7b` |
| 客户端数量 | `1`, `2`, `4`, `8`, `16`, `32` |
| 异构延迟 index | `0` |
| 队列策略 | `fifo` |
| 起始端口 | `8000` |

模型对应的 split point：

| 模型 | split point |
| --- | ---: |
| `qwen/qwen3-0.6b` | `4` |
| `meta-llama/llama3.2-1b` | `3` |
| `qwen/qwen3-1.7b` | `4` |

数据集对应的最大序列长度：

| 数据集 | `max_seq_len` |
| --- | ---: |
| `gsm8k` | `256` |
| `dialogsum` | `512` |
| `e2e` | `128` |

当前客户端脚本默认：

- 开启 LoRA：`-L`
- 每客户端 batch size：`-B=4`
- 数据划分：`random_overlap`
- 采样比例：`-SR=0.2`

当前服务端脚本默认：

- checkpoint mode：`-CKPT=all`

`-CKPT=all` 主要由 `v3` 使用；`v1` 和 `v2` 不使用 checkpoint 逻辑。

## 5. 单次实验示例

如果只想测试一个配置，可以分别运行下面两个命令。

服务端：

```bash
python experiment/server_run.py \
    -NC=4 \
    -V=v3 \
    -SP=4 \
    -M=qwen/qwen3-0.6b \
    -P=8000 \
    -CKPT=all \
    -DS=gsm8k \
    -LAG=0 \
    -QO=fifo
```

客户端：

```bash
python experiment/client_run.py \
    -NC=4 \
    -V=v3 \
    -L \
    -SP=4 \
    -M=qwen/qwen3-0.6b \
    -P=8000 \
    -B=4 \
    -DS=gsm8k \
    -SL=256 \
    -LAG=0 \
    -QO=fifo
```

运行顺序仍然是先启动服务端，再启动客户端。

## 6. 输入说明

### 6.1 模型输入

`server_run.py` 和 `client_run.py` 会根据 `-M` 拼出本地模型路径：

```text
/share/models/<-M 参数>
```

例如：

```bash
-M=qwen/qwen3-0.6b
```

对应模型路径：

```text
/share/models/qwen/qwen3-0.6b
```

客户端加载完整模型后切出 head 和 tail；服务端加载完整模型后切出 server middle layers。

### 6.2 数据输入

客户端根据 `-DS` 加载数据集，将样本格式化为文本 prompt，再通过 tokenizer 转成张量。训练 batch 中常见字段包括：

- `input_ids`
- `attention_mask`
- `labels`
- `input_text`

默认数据划分方式为：

```text
partition_mode = random_overlap
sample_ratio = 0.2
shuffle = False
```

如需互斥划分，可以给客户端添加：

```bash
-PM=exclusive
```

训练过程中，服务端接收的是中间激活、mask、position embedding 和梯度等信息，不接收原始训练文本。

## 7. 输出说明

### 7.1 训练日志

当前代码会把日志写入 `logs` 目录：

```text
logs/exp_main/<version>/<model>/<dataset>/lag_<lag>/client_number_<num_clients>/<queue_order>/
├── server/
│   ├── training_steps.log
│   └── training_metrics.log
└── client/
    ├── client_0.log
    ├── client_1.log
    └── ...
```

例如：

```text
logs/exp_main/v3/qwen/qwen3-0.6b/gsm8k/lag_0/client_number_4/fifo/
```

文件含义：

| 文件 | 内容 |
| --- | --- |
| `server/training_steps.log` | 服务端详细事件，例如接收激活、发送结果、聚合、更新等 |
| `server/training_metrics.log` | 聚合/训练指标，包括显存占用和平均 loss |
| `client/client_<id>.log` | 每个客户端的 batch loss、聚合事件、显存和耗时 |

`training_metrics.log` 的表头为：

```text
step | mem alloc(GB) | mem reserved(GB) | avg_loss
```

## 8. 主要超参数

### 8.1 通用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-P`, `--port` | `8000` | 服务端监听端口，客户端必须一致 |
| `-NC`, `--num_clients` | server: `1`, client: `2` | 客户端数量 |
| `-V`, `--version` | `v1` | 方法版本，支持 `v1`、`v2`、`v3`、`merge` |
| `-SP`, `--split_point` | `3` | head 和 tail 各自包含的层数 |
| `-M`, `--model` | `meta-llama/llama3.2-1b` | 模型名 |
| `-DS`, `--dataset` | `gsm8k` | 数据集名 |
| `-LR`, `--learning_rate` | `5e-4` | 学习率 |
| `-LAG`, `--lag_ratio` | `0` | 异构延迟配置 index |
| `-QO`, `--queue_order` | `fifo` | 调度策略 |
| `-BPS`, `--batch_per_sync` | `20` | 每隔多少 batch 聚合一次 |

### 8.2 服务端参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-SD`, `--server_device` | `cuda:0` | 服务端 GPU |
| `-BF`, `--buffer_size` | `4096` | socket buffer size |
| `-CKPT`, `--checkpoint_mode` | `no` | checkpoint 模式，`v3` 使用 |
| `-AVG`, `--use_avg` | 关闭 | 服务端梯度平均相关选项 |
| `-Q4`, `--use_qlora_4bit` | 关闭 | 4-bit QLoRA |
| `-Q8`, `--use_qlora_8bit` | 关闭 | 8-bit QLoRA |

`v3` 中 `-CKPT` 支持：

| 取值 | 行为 |
| --- | --- |
| `no` | 不使用 checkpoint |
| `all` | 所有 server forward 使用 checkpoint |
| `selective` | 根据显存占用选择性使用 checkpoint |

### 8.3 客户端参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-B`, `--batch_size` | `4` | 每个客户端的 batch size |
| `-SL`, `--max_seq_len` | `256` | 最大序列长度 |
| `-E`, `--epoch` | `1` | 本地 epoch 数 |
| `-PM`, `--partition_mode` | `random_overlap` | 数据划分方式 |
| `-SR`, `--sample_ratio` | `0.2` | `random_overlap` 下每个客户端采样比例 |
| `-L`, `--use_lora` | 关闭 | 是否使用 LoRA |
| `-Q4`, `--use_qlora_4bit` | 关闭 | 4-bit QLoRA |
| `-Q8`, `--use_qlora_8bit` | 关闭 | 8-bit QLoRA |

### 8.4 异构延迟

`-LAG` 会从 `experiment/client_run.py` 中选择一组 lag ratios：

| index | lag ratios |
| ---: | --- |
| `0` | `[1.0, 1.0, 1.0, 1.0]` |
| `1` | `[1.0, 1.2, 1.4, 1.6]` |
| `2` | `[1.0, 2.0, 3.0, 4.0]` |
| `3` | `[1.0, 1.0, 2.0, 10.0]` |
| `4` | `[2, 4, 6, 8]` |
| `5` | `[3, 6, 9, 12]` |

当前批量脚本只跑 `-LAG=0`，即无异构延迟。

## 9. Profiling 可视化

实验结束后可以进入 `vis/` 目录合并 profiling JSON：

```bash
cd /home/lzh/projects/FlexUSFL/vis
python dcp.py \
    -V=v3 \
    -LAG=0 \
    -NC=4 \
    -M=qwen/qwen3-0.6b \
    -DS=gsm8k \
    -QO=fifo
```

然后绘制 timeline：

```bash
python vis.py \
    -V=v3 \
    -LAG=0 \
    -NC=4 \
    -M=qwen/qwen3-0.6b \
    -DS=gsm8k \
    -QO=fifo \
    -SB=1 \
    -EB=20
```

输出图片为：

```text
training_timeline.png
```
