# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ms-swift** (Scalable lightWeight Infrastructure for Fine-Tuning) is a ModelScope framework for training, inference, evaluation, quantization, and deployment of 600+ LLMs and 400+ multimodal models. Entry point: `swift` CLI dispatched through `swift/cli/main.py`.

## Commands

### Install
```bash
pip install -e .
# or with uv
uv pip install -e . --torch-backend=auto
```

### Lint
```bash
pre-commit install
pre-commit run --all-files
# or
make linter
```

Enforced via `.pre-commit-config.yaml`: **flake8** (max 120 chars), **yapf** (formatting), **isort** (imports). Config lives in `setup.cfg`.

### Tests
```bash
# Full suite
python tests/run.py --subprocess
# or
make test

# Single test file
python tests/llm/test_sft.py
python -m pytest tests/general/test_utils.py
```

Test dependencies: `pip install -r requirements/tests.txt`. Tests are in `tests/` organized by category (`llm/`, `general/`, `app/`, `deploy/`). Uses `unittest` framework.

---

## Architecture

### Top-level data flow

```
swift <command>                         # swift/cli/main.py routes to submodule
  └─ swift/cli/<cmd>.py                 # arg parsing, calls *_main()
       └─ swift/pipelines/              # SwiftPipeline subclass (.run() method)
            ├─ swift/model/             # model + processor loading
            ├─ swift/template/          # tokenization / chat format
            ├─ swift/dataset/           # dataset loading + encoding
            ├─ swift/tuners/            # PEFT wrapper (LoRA, etc.)
            └─ swift/trainers/ or       # HF Trainer subclass
               swift/rlhf_trainers/     # TRL-based RLHF trainers
```

---

### 1. Pipeline layer (`swift/pipelines/`)

**`SwiftPipeline`** (`swift/pipelines/base.py`) is the abstract base for all operations.

```python
class SwiftPipeline(ABC, ProcessorMixin):
    args_class = BaseArguments        # override in subclass

    def main()                        # entry point; wraps run() with logging
    def run()                         # ABSTRACT — implement all logic here
    def _parse_args()                 # parses CLI or programmatic args
    def _set_seed()                   # rank-aware seed initialization
```

**`SwiftSft`** (`swift/pipelines/train/sft.py`) is the canonical concrete example:
```python
class SwiftSft(SwiftPipeline, TunerMixin):
    args_class = SftArguments

    def run()                           # orchestrates the steps below
    def _prepare_model_tokenizer()      # loads model via args.get_model_processor()
    def _prepare_template()             # initializes Template, checks padding-free
    def _prepare_dataset()              # loads + encodes datasets
    def _post_process_datasets()        # packing / lazy / streaming wrappers
    def _get_trainer_kwargs()           # override to inject custom trainer kwargs
    def train(trainer)                  # trainer.train() + checkpoint handling
```

To add a new pipeline: subclass `SwiftPipeline`, set `args_class`, implement `run()`. Follow the `_prepare_*` method decomposition pattern.

---

### 2. Trainer layer

#### SFT trainers (`swift/trainers/`)

**Class hierarchy:**
```
HfSeq2SeqTrainer (transformers)
  └─ Seq2SeqTrainer(SwiftMixin, DataLoaderMixin, HfSeq2SeqTrainer)
```

**`SwiftMixin`** (`swift/trainers/mixin.py`) adds 100+ methods to the HF Trainer. Key extension points:

| Method | When to override |
|---|---|
| `compute_loss()` | Custom loss: loss_scale, per-token loss, aux router loss |
| `create_loss_and_eval_metric()` | Custom metrics or loss functions |
| `_save_model()` / `_save_checkpoint()` | Custom serialization or checkpoint rotation |
| `_patch_tasks()` | Register forward hooks for new task types (seq_cls, reranker, embedding) |
| `train()` | Pre/post hooks around training loop |

**`DataLoaderMixin`** (`swift/trainers/mixin.py`) handles sequence-parallel and packed dataloaders. Override `get_train_dataloader()` to plug in a custom sampler.

**`Seq2SeqTrainer`** (`swift/trainers/seq2seq_trainer.py`) adds:
- `prediction_step()` — switches between loss-only and `infer_engine.infer()` generate mode
- `compute_loss()` — MOE aux loss, per-token loss, accuracy metrics with SP support

#### RLHF trainers (`swift/rlhf_trainers/`)

These extend TRL trainers, not HF Trainer directly:
```
HFGRPOTrainer (trl)
  └─ GRPOTrainer(RolloutTrainerMixin, SwiftMixin, HFGRPOTrainer)
```

`GRPOTrainer` adds a rollout generation loop, reward computation (via `reward_funcs` or reward models), and async generation with `ThreadPoolExecutor`. Use `RolloutTrainerMixin` when you need rollout capabilities in a new RLHF trainer.

**To add a new trainer:**
1. Inherit from `Seq2SeqTrainer` (for SFT-style) or an RLHF trainer.
2. Override `compute_loss()` and/or `prediction_step()`.
3. Add a custom `TrainingArguments` subclass if new args are needed.
4. Create a pipeline in `swift/pipelines/train/` that instantiates your trainer.

---

### 3. Arguments system (`swift/arguments/`)

All arguments are `@dataclass` with a deep inheritance chain:

```
SftArguments
  └─ SwanlabArguments → TunerArguments → BaseArguments → Seq2SeqTrainingArguments
```

`__post_init__()` handles lazy initialization: resolves checkpoint paths, sets default learning rates by `tuner_type`, validates `padding_free`/`packing` compatibility, initializes DeepSpeed/FSDP configs.

Arguments serialize to JSON/YAML and are saved in checkpoints. Config files (`--config foo.yaml`) are parsed before CLI flags in the router. When adding new args, inherit from the appropriate base and add fields to `__post_init__()` for cross-field validation.

---

### 4. Model registry (`swift/model/`)

```python
# swift/model/register.py
register_model(ModelMeta(
    model_type='mymodel-7b',          # unique ID used everywhere
    model_groups=[ModelGroup(         # one group per model family
        models=[Model(ms_model_id='org/mymodel-7b', hf_model_id='org/mymodel-7b')]
    )],
    template='mytpl',                 # default template_type
    model_arch=ModelArch(...),        # optional: vision tower names, etc.
    is_multimodal=False,
    task_type='causal_lm',            # 'causal_lm' | 'seq_cls' | 'embedding' | ...
))
```

`swift/model/patcher.py` applies model-specific monkey patches at load time. `swift/model/model_arch.py` defines architecture metadata for multimodal models (which module is the vision tower, how to find projectors, etc.).

To add a new model: create a file in `swift/model/models/`, call `register_model()` at module level, and import it from `swift/model/models/__init__.py`.

---

### 5. Template system (`swift/template/`)

Templates convert `List[Message]` → token IDs (for training) and token IDs → `str` (for inference). They also handle multimodal encoding.

```python
class Template(ProcessorMixin):
    is_encoder_decoder: bool = False
    task_type: str                    # 'causal_lm', 'seq_cls', 'embedding', ...
    mode: str                         # 'train' | 'rlhf' | 'kto' | 'vllm' | ...
    padding_free: bool
    sequence_parallel_size: int
    packing: bool

    def encode(inputs: TemplateInputs) → Dict   # messages → input_ids, labels, etc.
    def decode(output_ids, ...) → str            # token IDs → text
    def _post_encode(model, inputs) → Dict       # post-processing hook (e.g. embed images)
    def packing_row(rows) → Dict                 # merge examples for sequence packing
    def data_collator(batch, ...) → Dict         # pad + batch
```

Registration:
```python
register_template(TemplateMeta(
    template_type='mytpl',
    template_cls=MyTemplate,
    default_system='You are a helpful assistant.',
    is_thinking=False,          # True for models with <think> tokens
))
```

To add a new template: create a file in `swift/template/templates/`, subclass `Template`, override `encode()` for custom tokenization, and call `register_template()`.

---

### 6. Dataset pipeline (`swift/dataset/`)

```python
train_ds, val_ds = load_dataset(
    datasets=['mydata#1000'],       # "dataset_id#n_samples" syntax
    split_dataset_ratio=0.01,
    seed=42,
)
```

Internally: `DatasetLoader.load()` → raw HF dataset → `EncodePreprocessor` (applies `Template.encode()`) → optional `PackingDataset` / `LazyLLMDataset` / `IterablePackingDataset`.

To register a built-in dataset: add a `DatasetMeta` to `swift/dataset/dataset_meta.py` with a `preprocess_func` for row-level transformation. For purely local/custom datasets, pass a file path or a `HfDataset` directly.

---

### 7. Tuner system (`swift/tuners/`)

`SwiftModel` wraps any `nn.Module` with named adapters:

```python
class SwiftModel(nn.Module):
    adapters: Dict[str, SwiftOutput]  # named adapters
    active_adapters: Set[str]

    def activate_adapter(name)
    def deactivate_adapter(name, offload=False)
    def save_pretrained(save_dir, adapter_name)
    def from_pretrained(model, model_id, adapter_name) → SwiftModel
```

Each tuner is a `SwiftAdapter` subclass with a single static method:
```python
class MyTuner(SwiftAdapter):
    @staticmethod
    def prepare_model(model, config, adapter_name) → SwiftOutput:
        # patch model layers in-place
        # return SwiftOutput with callbacks:
        return SwiftOutput(
            config=config,
            state_dict_callback=...,       # how to extract adapter weights
            mark_trainable_callback=...,   # which params to unfreeze
            optimizer_group_callback=...,  # per-group LR (e.g. LoRA+ B matrix)
        )
```

To add a new tuner: subclass `SwiftAdapter`, implement `prepare_model()`, register in `SwiftTuners` enum and `SWIFT_MAPPING` dict in `swift/tuners/__init__.py`.

---

### 8. Inference engine abstraction (`swift/infer_engine/`)

All backends implement the same interface:
```python
class BaseInferEngine(ABC):
    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        **kwargs
    ) → List[ChatCompletionResponse]

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        **kwargs
    ) → ChatCompletionResponse
```

`InferRequest` holds `messages`, multimodal IDs (`image_ids`, `video_ids`, `audio_ids`), and `tools`. `RequestConfig` holds `max_tokens`, `temperature`, `top_p`, `do_sample`. Return type is OpenAI-compatible `ChatCompletionResponse`.

The Seq2SeqTrainer instantiates a `TransformersEngine` for generate-mode evaluation. GRPO uses `GrpoVllmEngine` for async rollout generation.

To add a new backend: subclass `BaseInferEngine`, implement both `infer()` and `infer_async()`.

---

### Key conventions

- **Naming**: `*_main()` for pipeline entrypoints, `Swift*` for pipeline classes, `*Args` for argument dataclasses, `*Trainer` for trainer classes.
- **Logger**: `from swift.utils import get_logger; logger = get_logger()` — never use `logging` directly.
- **Lazy loading**: `swift/__init__.py` uses `_LazyModule`. When adding a new top-level export, follow the `TYPE_CHECKING` + `_LazyModule` pattern — don't eagerly import heavy modules.
- **Registration at import time**: model, template, dataset, and tuner files call their `register_*()` function at module level, so they self-register when imported. Their `__init__.py` imports them all.
- **Multi-GPU**: set `NPROC_PER_NODE` env var. DeepSpeed configs: `swift/trainers/ds_config/`. Megatron parallelism: `swift/megatron/`. Sequence parallelism (Ulysses/Ring): `swift/sequence_parallel/`.
- **Argument validation**: cross-field validation belongs in `__post_init__()`, not in the pipeline. Single-field constraints go as `field(metadata=...)`.

---

## Megatron Integration (`swift/megatron/`)

This section covers everything needed to understand or port the Megatron integration to another project.

### Overview and dependency

The Megatron path is a parallel implementation that replaces the HF Trainer stack with Megatron-Core (`megatron-core`) + the `mcore_bridge` adapter library (min version 1.2.0). `mcore_bridge` is the critical glue layer: it converts HF model configs to MCore configs and handles weight translation in both directions.

```
megatron CLI                         # swift/cli/_megatron/main.py
  └─ MegatronSft / MegatronPretrain / MegatronRLHF / MegatronExport
       └─ BaseMegatronTrainer         # swift/megatron/trainers/base.py
            ├─ get_mcore_model()      # swift/megatron/model/utils.py
            ├─ mcore_bridge           # external: HF↔MCore weight bridge
            └─ Megatron-Core          # external: actual parallel training engine
```

### Initialization sequence (critical order)

`MegatronArguments.__post_init__()` calls `initialize_megatron()` which calls `_initialize_mpu()`. **This must happen before any model or optimizer is created.** The MPU (Model Parallel Unit) sets up all process groups:

```python
# swift/megatron/utils/megatron_lm_utils.py
_initialize_mpu():
  1. torch.distributed.init_process_group()  # if not already initialized
  2. mpu.initialize_model_parallel(
       tensor_model_parallel_size=TP,
       pipeline_model_parallel_size=PP,
       virtual_pipeline_model_parallel_size=VPP,
       context_parallel_size=CP,
       expert_model_parallel_size=EP,
       expert_tensor_parallel_size=ETP,
     )
  3. random seed setup (TP-aware via mcore_bridge)
  4. MoE aux loss auto-scaler init
```

`data_parallel_size` is **derived**, not configured: `world_size / (TP × PP × CP)`.

### Environment patching (`swift/megatron/init.py`)

`init_megatron_env()` is called at `import swift.megatron` and applies monkey patches before any training code runs:

| Patch | What it fixes |
|---|---|
| `_patch_unified_memory()` | Disables managed alloc for mcore ≥ 0.15 |
| `_patch_mcore_bridge()` | Hooks `GPTBridge.save_weights()` to also save HF config, processor, and PEFT state |
| `_patch__batched_p2p_ops()` | Sets `group=None` in P2P ops to fix pipeline parallel group sync |
| `_patch_torch_FileSystemReader()` | Parallel DCP loading via `ThreadPoolExecutor` |
| `_patch_validate_non_overlapping_shards_metadata()` | Skips slow DCP shard validation |

When porting: replicate these patches or upgrade `megatron-core` to versions where they are no longer needed.

### Parallelism strategies

#### Tensor Parallelism (TP)
Vocabulary logits are sharded across TP ranks. All loss/metric computation must all-reduce across TP before logging. Helper functions in `swift/megatron/trainers/vocab_parallel_utils.py`:
- `vocab_parallel_log_softmax()` — numerically stable log_softmax with all-reduce of max/sum
- `vocab_parallel_gather_logps()` — gather log-probs for target tokens from sharded vocab
- `vocab_parallel_kl_div()` — KL divergence over sharded vocab
- `compute_logps_and_entropy_from_logits()` — combined logps + entropy in one pass

#### Pipeline Parallelism (PP)
Set `pipeline_model_parallel_size > 1`. Megatron's `get_forward_backward_func()` returns an interleaved schedule that handles microbatch forwarding and gradient accumulation across pipeline stages automatically. Virtual PP (`virtual_pipeline_model_parallel_size`) splits each stage into multiple model chunks.

#### Context / Sequence Parallelism (CP/SP)
`context_parallel_size` enables ring-attention across CP group. `sequence_parallel=True` shards activations along the sequence dimension within the TP group (reduces activation memory). In `_prepare_batch()`, `get_batch_on_this_cp_rank()` slices the batch for the local CP rank.

#### Expert Parallelism (EP) — MoE
`expert_model_parallel_size` distributes experts across EP ranks. Key args:
- `moe_router_load_balancing_type`: `'aux_loss'` | `'seq_aux_loss'` | `'global_aux_loss'`
- `moe_token_dispatcher_type`: `'allgather'` | `'alltoall'` | `'flex'`
- `moe_enable_deepep`: enable DeepExpertParallel
- `moe_grouped_gemm`: fused grouped GEMM for expert computation
- Router Replay modes (`'disabled'` | `'R2'` | `'R3'`): re-use routing decisions across microbatches to reduce load-balancing variance.

### HuggingFace ↔ MCore checkpoint conversion (`swift/megatron/convert.py`)

```
HF checkpoint
  → mcore_bridge.hf_to_mcore_config()   # translate HF config → ModelConfig
  → get_mcore_model(model_config)        # instantiate sharded MCore model
  → bridge.load_weights(hf_model)        # copy weights HF → MCore
  → save_mcore_checkpoint()              # write distributed checkpoint (DCP)

MCore checkpoint (DCP)
  → load shards
  → bridge.save_weights()                # reassemble + write safetensors
  → HF config + processor saved alongside
```

DCP format: one file per tensor-parallel shard, stored under `mp_rank_XX/` directories. Cannot be loaded by HF directly — must go through `MegatronExport` (`megatron export`) first.

### Arguments hierarchy (`swift/megatron/arguments/`)

```
MegatronSftArguments
  ├─ MegatronArguments          # parallelism, optimizer, precision, fusions, checkpointing
  ├─ RLHFMegatronArgumentsMixin # DPO/GRPO/GKD/KTO/RM specific
  ├─ MegatronTunerMixin         # LoRA config for MCore
  └─ BaseArguments              # shared with standard swift training
```

Key argument groups unique to Megatron (not present in `SftArguments`):

| Group | Key args |
|---|---|
| Batch | `micro_batch_size`, `global_batch_size` (num_microbatches = global/micro/DP) |
| Parallelism | `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `virtual_pipeline_model_parallel_size` |
| Precision | `fp16`, `bf16`, `fp8_format`, `fp8_recipe` |
| Recompute | `recompute_granularity` (`'selective'`/`'full'`/`'none'`), `recompute_modules` |
| Fusions | `masked_softmax_fusion`, `bias_dropout_fusion`, `gradient_accumulation_fusion`, `apply_rope_fusion`, `tp_comm_overlap` |
| Optimizer | type: `'adam'`/`'sgd'`/`'muon'`/`'dist_muon'` |
| Checkpoints | `save_steps`, `save_total_limit`, `no_save_optim`, `no_save_rng`, `async_save`, `finetune` |
| Attention | `attention_backend`: `'flash'`/`'fused'`/`'unfused'`/`'local'`/`'auto'` |

### `BaseMegatronTrainer` — the training engine (`swift/megatron/trainers/base.py`)

```python
class BaseMegatronTrainer(ABC):
    def prepare_model()               # create MCore model(s) via get_mcore_model(), apply LoRA, wrap for DDP
    def get_optimizer_and_scheduler() # OptimizerConfig → DistributedOptimizer + ParamScheduler
    def train()                       # main loop: step → log → eval → save
    def train_step()                  # zero_grad → get_forward_backward_func() → optimizer.step()
    def save_checkpoint()             # DCP save (optionally async); rotate by save_total_limit
    def evaluate()                    # run eval loop, aggregate metrics across TP/PP/DP
    def _prepare_dataloader()         # MegatronPretrainingRandomSampler, consumed_samples tracking
    def _prepare_batch()              # move to device, CP slicing via get_batch_on_this_cp_rank()
```

`get_forward_backward_func()` is Megatron-Core's schedule function — it handles the interleaved pipeline schedule, microbatch forwarding, and gradient accumulation internally. Your forward function receives a single microbatch and returns `(loss, metrics_dict)`.

The optimizer is a `DistributedOptimizer` (gradient sharding across DP group) with per-parameter-group learning rates. To add component-specific LR (e.g. `vit_lr`, `aligner_lr` for multimodal), override `_patch_get_param_groups()`.

### RLHF trainers (`swift/megatron/trainers/`)

All Megatron RLHF trainers inherit from `MegatronRLHFTrainer(BaseMegatronTrainer)`. The reference model is handled differently based on tuner type:
- **Full fine-tuning**: a second `get_mcore_model()` call creates a separate ref model (double memory).
- **LoRA**: reference is obtained by calling the model with `disable_adapter()` context (no extra memory).

`MegatronGRPOTrainer` integrates a vLLM engine for rollout generation. It sets up a separate inference process group, sends model weights after each optimizer step, and collects completions asynchronously.

### Callback system (`swift/megatron/callbacks/base.py`)

```python
class MegatronCallback:
    def on_train_begin(args, state, control, **kwargs)
    def on_train_end(args, state, control, **kwargs)
    def on_step_begin / on_step_end
    def on_log(args, state, control, logs, **kwargs)
    def on_eval_begin / on_eval_end / on_eval_step
    def on_save(args, state, control, output_dir, **kwargs)
```

Note: `on_save` fires when the checkpoint is *queued*, not when async save completes. If you need post-save actions that depend on the files being present, check `AsyncCallsQueue` status first.

### Porting checklist for other projects

When integrating this Megatron stack into a new project, the minimum required pieces are:

1. **Dependencies**: `megatron-core`, `mcore_bridge >= 1.2.0`, `transformer-engine` (for FP8/fusions).
2. **Patches**: Copy `swift/megatron/init.py` patches and apply them before any MCore import.
3. **MPU init**: Call `_initialize_mpu()` from `megatron_lm_utils.py` before model creation. This must happen after `torch.distributed` is up.
4. **Model creation**: Use `get_mcore_model(model_config)` from `swift/megatron/model/utils.py`. `ModelConfig` is built via `mcore_bridge.hf_to_mcore_config(hf_config, **kwargs)`.
5. **Forward function**: Write a function `(data_iterator) → (loss, metrics)` and pass it to `get_forward_backward_func()`. Loss must be reduced across TP group.
6. **Vocab-parallel loss**: If using TP > 1, replace `F.cross_entropy` with `vocab_parallel_log_softmax` + gather from `vocab_parallel_utils.py`.
7. **Checkpoint format**: Use `save_mcore_checkpoint()` / `load_mcore_checkpoint()` — not HF `save_pretrained`. Export to HF via the convert path when needed.
8. **Data sampling**: Use `MegatronPretrainingRandomSampler` with `consumed_samples` for resume-correct distributed sampling.
