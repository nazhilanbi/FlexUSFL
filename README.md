# FlexUSFL

English | [中文](README_CN.md)

FlexUSFL is a research framework for **U-Shape Federated Split Learning for LLMs**. It splits a large language model into three parts along Transformer layer boundaries:

- **Head Model**: runs on clients and usually contains the embedding layer and the first several Transformer layers.
- **Server Model**: runs on the server and contains the middle Transformer layers, which are the main compute-heavy part.
- **Tail Model**: runs on clients and contains the final Transformer layers and the LM head.

During training, raw data always stays on clients. The project uses socket communication between clients and the server to transmit intermediate activations, backward gradients, and aggregation parameters. It is mainly used to evaluate different split learning server implementations, models, datasets, and client counts.

## 1. Project Structure

```text
FlexUSFL/
├── experiment/
│   ├── server_run.py        # Server entry point for one experiment
│   ├── client_run.py        # Client entry point; spawns multiple client processes
│   ├── server_run.sh        # Batch evaluation script for the server
│   ├── client_run.sh        # Batch evaluation script for clients
│   └── test.sh              # Simple test script
├── usfl/
│   ├── client/              # Client-side training logic
│   ├── server/              # Server versions: v1, v2, v3, merge
│   ├── llm/                 # Model splitting implementations for GPT/LLaMA/Qwen
│   ├── socket/              # TCP socket communication utilities
│   └── utils/               # Dataset, logging, model loading, and profiling utilities
├── vis/
│   ├── dcp.py               # Merge server/client profiling JSON files
│   └── vis.py               # Draw timeline visualizations
├── requirements.txt         # Pinned artifact dependencies
├── setup.py
├── README.md
└── README_CN.md
```

## 2. Installation

### 2.1 Hardware Assumptions

The current scripts assume a multi-GPU machine:

- The server uses `cuda:0` by default.
- Client processes use `cuda:1` to `cuda:7` in round-robin order.
- Clients and the server communicate through local `localhost` sockets by default.

If your GPU layout is different, update:

- Server GPU: pass `-SD` / `--server_device`, whose default value is `cuda:0`.
- Client GPUs: edit `available_gpus = [1, 2, 3, 4, 5, 6, 7]` in `experiment/client_run.py`.

### 2.2 Python Environment

Using conda is recommended:

```bash
conda create -n flexusfl python=3.10.18 -y
conda activate flexusfl
```

Install the pinned artifact dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`setup.py` only installs the local `usfl` package. It does not automatically install all third-party dependencies.

The reference environment was import-checked with:

| Component | Version |
| --- | --- |
| Python | `3.10.18` |
| PyTorch | `2.6.0+cu126` |
| CUDA runtime bundled with PyTorch | `12.6` |
| Non-quantized model dtype | `float32` |
| Transformers | `4.51.1` |
| Datasets | `3.2.0` |
| PEFT | `0.14.0` |
| Accelerate | `1.10.1` |

The main artifact scripts use LoRA but not QLoRA. Therefore, `bitsandbytes` was not installed in the reference environment and is not included in `requirements.txt`. The `-Q4` and `-Q8` options require a separately installed `bitsandbytes` build compatible with the local CUDA and PyTorch versions.

LoRA is applied to the split head, server, and tail with PEFT's generic `PeftModel`. These components intentionally do not use `TaskType.CAUSAL_LM`, because an individual split component is not a complete generation model and does not implement generation methods.

### 2.3 Model and Dataset Paths

The code first looks for models under `/share/models`:

```text
/share/models/<model_name>
```

For example:

```text
/share/models/qwen/qwen3-0.6b
/share/models/meta-llama/llama3.2-1b
/share/models/qwen/qwen3-1.7b
```

Models used by the current evaluation scripts:

| `-M` value | Hugging Face model card | Expected local directory |
| --- | --- | --- |
| `qwen/qwen3-0.6b` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `/share/models/qwen/qwen3-0.6b` |
| `meta-llama/llama3.2-1b` | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | `/share/models/meta-llama/llama3.2-1b` |
| `qwen/qwen3-1.7b` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `/share/models/qwen/qwen3-1.7b` |

The models can be downloaded with the Hugging Face CLI:

```bash
hf download Qwen/Qwen3-0.6B \
    --local-dir /share/models/qwen/qwen3-0.6b

# Llama 3.2 is gated. Accept its license on Hugging Face and run
# `hf auth login` before downloading it.
hf download meta-llama/Llama-3.2-1B \
    --local-dir /share/models/meta-llama/llama3.2-1b

hf download Qwen/Qwen3-1.7B \
    --local-dir /share/models/qwen/qwen3-1.7b
```

The dataset cache directory is configured in `usfl/env.py` as:

```text
/share/datasets/
```

The current batch scripts mainly use:

| CLI name | Hugging Face dataset |
| --- | --- |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), configuration `main` |
| `dialogsum` | [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum) |
| `e2e` | [GEM/e2e_nlg](https://huggingface.co/datasets/GEM/e2e_nlg) |

For the three models and datasets above, FlexUSFL uses local-first resolution:

- If the expected directory exists under `/share/models` or `/share/datasets`, it is loaded directly for offline artifact evaluation.
- If the directory does not exist, the official namespaced Hugging Face ID in the tables above is used.
- The Hub version of E2E uses `meaning_representation` and `target`; the formatter accepts both those fields and the local artifact's `context` and `completion` fields.

For a fully frozen artifact, record a Hugging Face revision or commit hash when downloading each model and dataset, then use the same revision for future downloads.

## 3. Quick Evaluation

The main evaluation workflow uses two shell scripts under `experiment/`:

- Terminal 1: run `experiment/server_run.sh`
- Terminal 2: run `experiment/client_run.sh`

The loop order and port increment rule in the two scripts must stay aligned. Both scripts currently start from port `8000`, and the port is increased by 1 after each configuration.

### 3.1 Start the Server

Run this in the first terminal:

```bash
cd /home/lzh/projects/FlexUSFL
conda activate flexusfl
bash experiment/server_run.sh
```

The server script starts one server configuration at a time and waits for the corresponding clients to connect.

### 3.2 Start the Clients

Run this in the second terminal:

```bash
cd /home/lzh/projects/FlexUSFL
conda activate flexusfl
bash experiment/client_run.sh
```

You do not need to manually open one terminal per client. For each experiment configuration, `experiment/client_run.py` creates multiple client processes according to `-NC`.

### 3.3 Parameters That Must Match

The server and clients must use matching values for the following parameters:

| Argument | Meaning |
| --- | --- |
| `-NC` | Number of clients |
| `-V` | Method version, such as `v1`, `v2`, or `v3` |
| `-SP` | Model split point |
| `-M` | Model name |
| `-P` | Communication port |
| `-DS` | Dataset name |
| `-LAG` | Heterogeneous lag setting |
| `-QO` | Queue scheduling policy |

If you change datasets, models, client counts, versions, or ports in `server_run.sh`, make the same change in `client_run.sh`.

## 4. Current Batch Evaluation Settings

The main loops in `server_run.sh` and `client_run.sh` currently use the following settings:

| Item | Current values |
| --- | --- |
| Datasets | `gsm8k`, `dialogsum`, `e2e` |
| Method versions | `v1`, `v2`, `v3` |
| Models | `qwen/qwen3-0.6b`, `meta-llama/llama3.2-1b`, `qwen/qwen3-1.7b` |
| Number of clients | `1`, `2`, `4`, `8`, `16`, `32` |
| Lag index | `0` |
| Queue policy | `fifo` |
| Initial port | `8000` |

Model-specific split points:

| Model | Split point |
| --- | ---: |
| `qwen/qwen3-0.6b` | `4` |
| `meta-llama/llama3.2-1b` | `3` |
| `qwen/qwen3-1.7b` | `4` |

Dataset-specific maximum sequence lengths:

| Dataset | `max_seq_len` |
| --- | ---: |
| `gsm8k` | `256` |
| `dialogsum` | `512` |
| `e2e` | `128` |

The client script currently uses:

- LoRA enabled: `-L`
- Batch size per client: `-B=4`
- Data partition mode: `random_overlap`
- Sample ratio: `-SR=0.2`

The server script currently uses:

- Checkpoint mode: `-CKPT=all`

`-CKPT=all` is mainly used by `v3`; `v1` and `v2` do not use the checkpoint logic.

## 5. Single-Configuration Example

To test a single configuration, run the following two commands separately.

Server:

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

Clients:

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

Start the server first, then start the clients.

## 6. Inputs

### 6.1 Model Input

`server_run.py` and `client_run.py` build the local model path from `-M`:

```text
/share/models/<-M value>
```

For example:

```bash
-M=qwen/qwen3-0.6b
```

maps to:

```text
/share/models/qwen/qwen3-0.6b
```

The client loads the full model and extracts the head and tail parts. The server loads the full model and extracts the middle server layers.

### 6.2 Dataset Input

Clients load the dataset according to `-DS`, format each sample as a text prompt, and tokenize it into tensors. Common fields in a training batch include:

- `input_ids`
- `attention_mask`
- `labels`
- `input_text`

The default partition settings are:

```text
partition_mode = random_overlap
sample_ratio = 0.2
shuffle = False
```

To use exclusive partitioning, add this argument to the client command:

```bash
-PM=exclusive
```

During training, the server receives intermediate activations, masks, position embeddings, and gradients. It does not receive the raw training text.

## 7. Outputs

### 7.1 Training Logs

The code writes logs to the `logs` directory:

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

For example:

```text
logs/exp_main/v3/qwen/qwen3-0.6b/gsm8k/lag_0/client_number_4/fifo/
```

File meanings:

| File | Content |
| --- | --- |
| `server/training_steps.log` | Detailed server events, such as receiving activations, sending results, aggregation, and updates |
| `server/training_metrics.log` | Aggregation/training metrics, including GPU memory usage and average loss |
| `client/client_<id>.log` | Batch loss, aggregation events, GPU memory usage, and elapsed time for each client |

The header of `training_metrics.log` is:

```text
step | mem alloc(GB) | mem reserved(GB) | avg_loss
```

## 8. Main Hyperparameters

### 8.1 Common Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-P`, `--port` | `8000` | Server listening port; clients must use the same value |
| `-NC`, `--num_clients` | server: `1`, client: `2` | Number of clients |
| `-V`, `--version` | `v1` | Method version; supports `v1`, `v2`, `v3`, `merge` |
| `-SP`, `--split_point` | `3` | Number of layers in both the head and tail |
| `-M`, `--model` | `meta-llama/llama3.2-1b` | Model name |
| `-DS`, `--dataset` | `gsm8k` | Dataset name |
| `-LR`, `--learning_rate` | `5e-4` | Learning rate |
| `-LAG`, `--lag_ratio` | `0` | Lag configuration index |
| `-QO`, `--queue_order` | `fifo` | Scheduling policy |
| `-BPS`, `--batch_per_sync` | `20` | Number of batches between aggregations |

### 8.2 Server Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-SD`, `--server_device` | `cuda:0` | Server GPU |
| `-BF`, `--buffer_size` | `4096` | Socket buffer size |
| `-CKPT`, `--checkpoint_mode` | `no` | Checkpoint mode used by `v3` |
| `-AVG`, `--use_avg` | disabled | Server-side gradient averaging option |
| `-Q4`, `--use_qlora_4bit` | disabled | 4-bit QLoRA |
| `-Q8`, `--use_qlora_8bit` | disabled | 8-bit QLoRA |

`v3` supports the following `-CKPT` values:

| Value | Behavior |
| --- | --- |
| `no` | Do not use checkpointing |
| `all` | Use checkpointing for all server forward passes |
| `selective` | Use checkpointing selectively based on GPU memory usage |

### 8.3 Client Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-B`, `--batch_size` | `4` | Batch size per client |
| `-SL`, `--max_seq_len` | `256` | Maximum sequence length |
| `-E`, `--epoch` | `1` | Number of local epochs |
| `-PM`, `--partition_mode` | `random_overlap` | Dataset partition mode |
| `-SR`, `--sample_ratio` | `0.2` | Per-client sample ratio in `random_overlap` mode |
| `-L`, `--use_lora` | disabled | Enable LoRA |
| `-Q4`, `--use_qlora_4bit` | disabled | 4-bit QLoRA |
| `-Q8`, `--use_qlora_8bit` | disabled | 8-bit QLoRA |

### 8.4 Heterogeneous Lag

`-LAG` selects a list of lag ratios from `experiment/client_run.py`:

| Index | Lag ratios |
| ---: | --- |
| `0` | `[1.0, 1.0, 1.0, 1.0]` |
| `1` | `[1.0, 1.2, 1.4, 1.6]` |
| `2` | `[1.0, 2.0, 3.0, 4.0]` |
| `3` | `[1.0, 1.0, 2.0, 10.0]` |
| `4` | `[2, 4, 6, 8]` |
| `5` | `[3, 6, 9, 12]` |

The current batch scripts only use `-LAG=0`, which means no heterogeneous lag.

## 9. Profiling Visualization

After an experiment finishes, enter the `vis/` directory and merge profiling JSON files:

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

Then draw the timeline:

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

The output image is:

```text
training_timeline.png
```
