#!/bin/bash
# Forensic KV replay on the Pixel 10 Tensor NPU.
# Usage:  bash run.sh ["your prompt"]        # generate a description
#         bash run.sh -s "a caption"         # score a caption (NLL, lower=better fit)
# Everything already lives on the phone (binary, kv_kvsep.bin, dispatch lib).

MODEL=/sdcard/Android/data/com.google.ai.edge.gallery/files/Gemma_4_E2B_it_TPU/3f250541aff494231036164d89603de72cb6dc70/gemma-4-E2B-it_Google_Tensor_G5.litertlm
KV=/data/local/tmp/kv_kvsep.bin
STEP=655

if [ "$1" = "-s" ]; then
  EXTRA="--score_target_text=\"$2\""
  PROMPT="Describe the question I ask you i forgot."
else
  EXTRA="--max_output_tokens=256 --repetition_penalty=1.2 --suppress_tokens=1,50,106"
  PROMPT="${1:- describe the content in the image i sent to you, please give very detail description, and you can also give a title for the image.}"
fi

adb shell "su -c 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp:/system/lib64:/vendor/lib64 LITERT_RESTORE_KV=$KV LITERT_RESTORE_STEP=$STEP ./litert_lm_advanced_main --backend=npu --litert_dispatch_lib_dir=/data/local/tmp --model_path=$MODEL --input_prompt=\"$PROMPT\" $EXTRA 2>&1'" 2>&1 \
  | grep -iE 'Captured model output|Score:|prefill-hook RESTORE' \
  | sed 's/.*Captured model output: /\n模型输出: /; s/.*prefill-hook /[KVEX] /; s/.*Score:/得分(NLL,越低越贴合):/'
