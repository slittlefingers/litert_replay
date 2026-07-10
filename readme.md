# litert_replay — replay the recovered KV cache on the Tensor NPU

Kit to build an android_arm64 `litert_lm_main` that restores a KV cache recovered from a physical
memory dump and answers questions about it (the forensic "what did the model see" read-out).

## Files
| file | what |
|---|---|
| `session_kv.bin` | the recovered KV — KVEX v1, 30 tensors (15 K + 15 V), step=655, 37.7 MB. Goes to the **phone**. |
| `kvex_restore.inc` | KV injection code: parse KVEX → write each `kv_cache_*` into the NPU executor's TensorBuffer map (coherent `TensorBufferScopedLock`+memcpy). |
| `REPLAY_BUILD.md` | exact integration + build + run steps, and the 5 things to verify while building. |

## Build (Mac) — use the SAME LiteRT-LM commit so REPLAY_BUILD.md line numbers match
```bash
git clone https://github.com/google-ai-edge/LiteRT-LM
cd LiteRT-LM && git checkout b516ae3d      # "Add implementation that uses embeddings for NPU based inference."

cp ../litert_replay/kvex_restore.inc runtime/executor/
# then apply the edits in REPLAY_BUILD.md:
#   - #include "kvex_restore.inc" in llm_litert_npu_compiled_model_executor.cc
#   - call RestoreKvexIntoMap(...) before each `return executor;` (Create* funcs) + set current_step_=655
#   - run with --clear_kv_cache_before_prefill=false

bazel build -c opt --config=android_arm64 //runtime/engine:litert_lm_main
```

## Run (phone)
```bash
adb push bazel-bin/runtime/engine/litert_lm_main /data/local/tmp/
adb push session_kv.bin /data/local/tmp/            # or already pushed from the capture box
adb shell su -c "
  LITERT_RESTORE_KV=/data/local/tmp/session_kv.bin LITERT_RESTORE_STEP=655 \
  /data/local/tmp/litert_lm_main --backend=npu --litert_dispatch_lib_dir=/vendor/lib64 \
    --clear_kv_cache_before_prefill=false \
    --model_path=/sdcard/Android/data/com.google.ai.edge.gallery/files/Gemma_4_E2B_it_TPU/<hash>/gemma-4-E2B-it_Google_Tensor_G5.litertlm \
    --input_prompt='Describe the image the user sent.'"
```
**Success** = the model describes a **bed with bedding + bananas** without ever seeing the image →
the KV was correctly recovered from the dump. See REPLAY_BUILD.md §5 if the restore writes 0 tensors
or the answer is garbled (layer-order tuning; no CLI rebuild needed for that).

## Provenance
KV recovered via: `lemon -S <pid>` (dma-buf sg_table) → reassemble from a LEMON physical dump with the
correct vmemmap base (`0xfffffffdfe000000`) → classify at step 655 → clean → KVEX. 120 dma-bufs =
4 banks × 30 tensors, all at step 655.

## Prebuilt binary + source (this fork)
- **Binary** `litert_lm_advanced_main` (android_arm64, NPU, with the KVEX restore hook) is attached to this repo's **GitHub Release**.
- **Source of the integration** = `kvex_restore.inc` + `litert_lm_npu_kvex.patch`.
  Reproduce:
  ```bash
  git clone https://github.com/google-ai-edge/LiteRT-LM && cd LiteRT-LM && git checkout b516ae3d
  cp ../litert_replay/kvex_restore.inc runtime/executor/
  git apply ../litert_replay/litert_lm_npu_kvex.patch
  bazel build -c opt --config=android_arm64 //runtime/engine:litert_lm_advanced_main
  ```
  Run with `run.sh` (uses `kv_kvsep.bin` = layer-reordered KV, see LAYER_REORDER.md).
