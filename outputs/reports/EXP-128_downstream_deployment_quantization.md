# EXP-128: Deployment metrics and quantisation of the downstream heads only

## Status
**Pre-registered 2026-10-29, before any run** (DEC-027). The user asked: "there should be section in the appendix where we measure deployments metrics and quantization effect of just the downstream model". It is queued in `scripts/run_queue16.sh` before the EXP-127 trainings, after queue15 and queue14.

## Question
Every application shares one frozen foundation backbone and one core algorithm and trains only a small downstream head: one tree per branch, plus a translation block for 6-D.
- What does the head cost to store and run?
- How much accuracy does it lose when its inference precision drops from fp32 to bf16, fp16 or int8?

The backbone is frozen, shared, and not quantised here. Its cost is quoted once, from EXP-112, for context.

## Heads (validation-selected configuration per application; checkpoints reused unchanged)
| Application | Head | Source checkpoints | Reference bf16 evaluation (existing) |
|---|---|---|---|
| SPEED+ (spacecraft), PoE | distilled PoE students (EXP-117, teacher_variant=PoE) | EXP-117 runs | `EXP-117/RUN-20261020-114253` |
| SPEED+, DINOv3 only | distilled DINOv3 students (teacher_variant=DINO) | EXP-117 runs | `EXP-117/RUN-20261020-113318` |
| T-LESS (industrial parts) | A2 multi-path trees | EXP-106 (label=multi) | `EXP-106/RUN-20261028-221625` |
| SYMSOL-T (symmetric solids) | A2 multi-path trees | EXP-106 (label=multi) | `EXP-106/RUN-20261028-120353` |
| YCB-V (household objects) | synthetic-trained trees | EXP-122 | `EXP-122/RUN-20261027-200852` |

## Precisions (`src/tfpose/precision.py`; evaluators get `--precision`)
- **fp32:** no autocast. This is the reference.
- **bf16:** bf16 autocast, the precision of every earlier result. The existing evaluations above are reused; the evaluator code is unchanged apart from the precision switch.
- **fp16:** fp16 autocast.
- **int8w:** weight-only int8, simulated.
  - Every matmul weight is rounded to symmetric per-output-channel int8 and de-quantised: all `nn.Linear` weights, including the attention out-projections, plus the `nn.MultiheadAttention` in-projections. That is 20 tensors per tree, 99.3% of its parameters.
  - Activations, biases, LayerNorms, embeddings and the root token stay fp32.
  - This measures the accuracy of an int8-weight deployment. Its speed is measured separately with real int8 kernels (next section).

## Measurements
1. **Accuracy per precision.** Each application's own evaluator and metric (`scripts/orbit_eval.py` for SPEED+, `scripts/exp064_eval.py` for the others), 3 seeds, validation and test.
   - Inference: greedy and orbit_product.
   - Metrics: Δ mean error and Δ acc@10 against fp32, plus greedy-path agreement. Path agreement is descriptive and derived from the per-image outputs where available.
2. **Deployment cost** (`scripts/exp128_deploy.py`):
   - parameters;
   - size on disk at fp32, fp16 and int8: real dynamic-int8 state dict, plus the ideal int8-weight size;
   - head-only latency, greedy and orbit (4 views × beam 8 + scoring of 32 candidates), at batch 1 and 64:
     - GPU at fp32, bf16 and fp16;
     - CPU at fp32 and real dynamic int8 (`quantize_dynamic` on `nn.Linear`), with 1 thread and all threads;
   - peak transient GPU memory.
   - The same for the translation block where applicable.
   - **Caveat on real dynamic int8:** it converts only `nn.Linear` modules. The attention in-projections stay fp32, so the number of converted modules is reported.
   - **Caveat on the CPU rows:** PyTorch's fused-MHA fast path is disabled for both CPU precisions, because it rejects quantised layers.
   - **Not timed here:** the head architecture is the same for every application, so cost is measured once per head type. Tree vs flat cost is EXP-127's job and is not repeated.

## Criterion (fixed now)
- **Negligible loss:** a precision's degradation is "negligible" if |Δ mean error| against fp32 is no larger than the larger of the two seed stds. This is the project's pass rule, applied per application, per inference mode and per split.
- **Reporting:** every precision is reported whatever the outcome.
- **Cost:** descriptive.

## Not in scope
- Translation-block accuracy under quantisation. Its cost is measured, but its accuracy is evaluated only by EXP-118/126, at bf16.
- Quantising the backbone.
- Quantisation-aware training.

## Results
Not run.

## Results (completed 2026-11-03)
- **Interruptions:** the run was cut short by the 2026-11-01 GPU loss and by shared-GPU out-of-memory errors. It was resumed with completed evaluations skipped (`run_queue16.sh`).
- **Accuracy runs:** `EXP-128_downstream_deployment_quantization`, RUN ids:

  | precision | SPEED+ PoE | SPEED+ DINO | T-LESS | SYMSOL-T | YCB-V |
  |---|---|---|---|---|---|
  | fp32 | 072122 | 073932 | 074839 | 014830 | 030824 |
  | fp16 | 032357 | 035642 | 041348 | 043959 | 052728 |
  | int8w | 053630 | 060756 | 062145 | 065417 | 075706 |

- **Cost run:** `RUN-20261103-080855`.

### Accuracy vs the paper's bf16 results
Real test sets, validation-selected head, 3 seeds. Mean error in degrees, Δ against bf16.

| application | bf16 (± seed std) | fp32 | fp16 | int8 weights |
|---|---|---|---|---|
| SPEED+ distilled PoE + orbit, lightbox | 11.214 ± 0.416 | +0.006 | +0.007 | −0.004 |
| SPEED+ distilled PoE + orbit, sunlamp | 11.088 ± 0.490 | −0.009 | −0.009 | −0.021 |
| SPEED+ DINO student + orbit, lightbox | 16.431 ± 0.751 | +0.003 | +0.002 | −0.000 |
| T-LESS A2 PoE + orbit | 15.187 ± 0.272 | +0.004 | +0.008 | −0.003 |
| SYMSOL-T A2 DINO + orbit | 2.067 ± 0.002 | +0.000 | −0.000 | +0.002 |
| YCB-V DINO greedy | 12.917 ± 6.905 | +0.042 | +0.046 | +0.045 |

**Criterion (negligible loss, |Δ| within the larger seed std): PASS for every precision and application.**

### Size
| head | parameters | fp32 | fp16 | int8 ideal | PyTorch dynamic int8 file |
|---|---|---|---|---|---|
| DINOv3 tree | 2,391,564 | 9.2 MB | 4.6 MB | 2.3 MB | 6.2 MB |
| MoGe-2 tree | 2,128,404 | 8.4 MB | 4.2 MB | 2.1 MB | 6.1 MB |
| translation block | 331,011 | 1.3 MB | 0.6 MB | 0.3 MB | 1.1 MB |

`quantize_dynamic` converted 20 of 20 linear modules per tree, attention out-projections included.

### Head-only latency
In ms per image. Shared RTX 6000 Ada at full load from other jobs; Intel Core Ultra 9 285K.

| system | GPU b1 fp32 / bf16 / fp16 | GPU b64 fp32 / bf16 / fp16 | CPU 1 thread fp32 / int8 |
|---|---|---|---|
| DINO, 1 view | 18.9 / 19.3 / 19.2 | 0.33 / 0.31 / 0.32 | 6.7 / 7.4 |
| DINO, orbit | 111.6 / 111.1 / 109.9 | 4.09 / 2.36 / 2.39 | 129.8 / 126.6 |
| PoE, 1 view | 35.6 / 35.9 / 36.0 | 0.68 / 0.60 / 0.60 | 20.6 / 20.9 |
| PoE, orbit | 191.8 / 191.5 / 190.3 | 13.93 / 6.82 / 6.65 | 577.6 / 567.6 |

- **Peak transient GPU memory** of the head is 1–41 MB.
- **Multi-thread CPU timings** (24 threads) were 0.9–16 s per image. They were dominated by contention with other processes on the shared machine, so they are recorded but not reported.
- **Conclusions:**
  - Precision does not affect batch-1 latency, which is limited by launch overhead in the autoregressive decoder.
  - fp16/bf16 halve the batched orbit cost.
  - Dynamic int8 gives no CPU speed-up for heads this small.
- **Paper:** `tab:deploy_size`, `tab:deploy_latency` and `tab:deploy_quant` in the appendix section "Deployment of the downstream head"; one clause in contribution (1).
