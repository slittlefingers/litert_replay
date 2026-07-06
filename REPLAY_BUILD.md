# Replay build — inject the recovered KV and interrogate it (Android NPU)

Goal: build a `litert_lm_main` for android_arm64 that, on the Tensor NPU, restores `session_kv.bin`
into a fresh session and answers questions about it ("what is in the image?" → bed + bananas).

There is **no prebuilt android CLI** (only macOS in the GitHub release), so this must be built on
your Mac (Bazel + Android NDK). Below is exactly what to change and build.

## 0. Copy over
`session_kv.bin` (37.7 MB, 30 tensors) + the model `.litertlm` → your Mac (and later push to the
phone). `session_kv.bin` is at `testexp/sgtable/session_kv.bin` on the Windows box.

## 1. Drop in the restore helper
Copy `kvcacheexplainer/src/kvex_restore.inc` next to the NPU executor and `#include` it in
`runtime/executor/llm_litert_npu_compiled_model_executor.cc` (top, after the other includes):
```cpp
#include "kvex_restore.inc"   // provides RestoreKvexIntoMap / RestoreKvexFromEnv / RestoreStepFromEnv
```
It parses the KVEX container and writes each `kv_cache_k/v_i` tensor into a
`flat_hash_map<string_view, TensorBuffer>` via the executor's own `TensorBufferScopedLock+memcpy`
pattern. It **skips names that aren't in the map**, so pointing it at a mixed input-buffer map only
touches the 30 KV tensors.

## 2. Call it at the end of Create()
Both `CreateForModelHasPerLayerEmbedding` and `CreateForModelWithoutPerLayerEmbedding` end with
`auto executor = absl::WrapUnique(new ...); ... return executor;` (≈ lines 3704/3718 and 3911/3923).
Just before each `return executor;` add:
```cpp
  if (const char* kv = std::getenv("LITERT_RESTORE_KV"); kv && *kv) {
    // Write the recovered KV into the input buffers the model reads during prefill+decode.
    // VERIFY (see §5): these are the maps holding the kv_cache_* input tensors.
    LITERT_ASSIGN_OR_RETURN(int n1,
        RestoreKvexIntoMap(/*blob*/ *ReadFileToString(kv),
                           executor->llm_inference_context_.prefill_input_buffers));
    LITERT_ASSIGN_OR_RETURN(int n2,
        RestoreKvexIntoMap(*ReadFileToString(kv),
                           executor->llm_inference_context_.decode_input_buffers));
    // Restore the position: the KV represents `step` tokens already processed.
    int step = RestoreStepFromEnv();               // e.g. 655
    if (step > 0) {
      executor->current_step_ = step;
      // processed_tokens_ must report `step` tokens so Prefill() doesn't RollBackToStep() fail.
      // If you don't have the exact token ids, pad to `step` (KV drives attention; ids only affect
      // bookkeeping for [0,step)). Ideally extract the real ids from the dump (int32 ProcessedTokens).
      while (executor->processed_tokens_.TokenCount() < step)
        executor->processed_tokens_.AddProcessedTokens({/*pad id*/ 2});
    }
    ABSL_LOG(ERROR) << "KVEX restore: wrote " << (n1 + n2) << " KV tensors, step=" << step;
  }
```
(`RestoreKvexIntoMap` returns a count; `RestoreKvexFromEnv` is a convenience wrapper if you prefer.)
Add a tiny `ReadFileToString(path)->absl::StatusOr<std::string>` or inline the ifstream read.

## 3. Do NOT clear the KV before prefill
The pipeline clears the KV before the first prefill by default. Run with
`--clear_kv_cache_before_prefill=false` (litert_lm_advanced_main exposes it; for litert_lm_main add
the same flag, or hard-set `settings.clear_kv_cache_before_prefill=false`). Otherwise your restored
KV is wiped.

## 4. Build (Mac) + run (phone)
```bash
# Build android_arm64 (adjust to the repo's documented android config / NDK path)
bazel build -c opt --config=android_arm64 //runtime/engine:litert_lm_main

adb push bazel-bin/runtime/engine/litert_lm_main /data/local/tmp/
adb push session_kv.bin /data/local/tmp/
adb shell su -c "
  LITERT_RESTORE_KV=/data/local/tmp/session_kv.bin LITERT_RESTORE_STEP=655 \
  /data/local/tmp/litert_lm_main --backend=npu \
    --litert_dispatch_lib_dir=/vendor/lib64 \
    --clear_kv_cache_before_prefill=false \
    --model_path=/sdcard/Android/data/com.google.ai.edge.gallery/files/Gemma_4_E2B_it_TPU/<hash>/gemma-4-E2B-it_Google_Tensor_G5.litertlm \
    --input_prompt='Describe the image the user sent.'"
```
Success = the model answers about **a bed with bedding + bananas** without ever seeing the image →
the KV recovery (and layer labeling) is correct.

## 5. Known iteration points (expect to tune during the build)
1. **Which map holds the KV inputs.** `llm_inference_context_.{prefill,decode}_input_buffers` is the
   likely target, but the NPU executor also has `cache_update_inference_context_` and separate
   input/output *slice* banks (this is the "4 banks" we saw in the dump). If the restore writes 0
   tensors, log the map keys and point at the one containing `kv_cache_k_0`.
2. **Member access.** `current_step_`, `processed_tokens_`, and the InferenceContext maps are
   private. Either put the restore inside the class (a member `RestoreKvex()` method) or friend it.
3. **Token ids.** If padding `processed_tokens_` gives a rollback/position error, extract the real
   655 int32 ids from the dump (the `ProcessedTokens` buffer) and feed them.
4. **Layer labeling.** Our labels are fd-order heuristic. If the answer is garbled but coherent-ish,
   the KV is right but layer order is off — try reversing/permuting the k/v layer assignment in
   `finish_kv.py` and rebuild `session_kv.bin` (no re-compile of the CLI needed).
5. **String-view keys.** The map keys are `string_view` into the model's input-name table; our
   lookup is content-hashed, so `find(string_view(name))` matches — but confirm the exact key
   spelling (e.g. `kv_cache_k_0` vs a signature-prefixed alias).

This is the last mile: the KV data is recovered and verified structurally; this build turns it into
a semantic read-out.
