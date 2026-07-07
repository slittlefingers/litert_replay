# Layer reorder for session_kv.bin (fd-order -> true model layer)

The recovered KVEX labels tensors by dma-buf fd-order, which does NOT equal the
model's true layer index. K and V are scrambled by DIFFERENT permutations.
Derived by per-channel fingerprint matching each block against a known-correct KV
(self_kv.bin, produced by round-tripping the model's own post-prefill buffers).

Apply to session_kv.bin to get session_kv_reordered.bin (used with LITERT_RESTORE_KV):

K:  session_idx -> true_idx
  0->2  1->3  2->8  3->10  4->9  5->12  6->6  7->7  8->13  9->4  10->0  11->1  12->11  13->5  14->14
V:  session_idx -> true_idx
  0->3  1->6  2->11 3->0  4->9  5->5  6->10 7->12 8->7  9->4  10->1  11->13 12->2  13->8  14->14

Result (greedy, KV-only, model never saw the image):
  "The image you sent is a photograph of a bed ... heavily covered with bedding and pillows."
  Scoring: bed+bedding+pillows 29.9 < bananas 36.9 < cat 44.4 (no-KV baseline 65.3).
