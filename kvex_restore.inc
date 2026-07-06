// kvex_restore.inc — restore a KVEX container (session_kv.bin) into an NPU executor's KV buffer
// map, then set the step. Self-contained; #include it into
//   LiteRT-LM/runtime/executor/llm_litert_npu_compiled_model_executor.cc
// and call RestoreKvexFromEnv(...) at the end of the executor's Create() (see REPLAY_BUILD.md).
//
// The NPU executor holds KV as absl::flat_hash_map<string_view, ::litert::TensorBuffer>
// (input_kv_cache_buffers_ etc.). We write each named tensor's bytes into the matching buffer via
// the same TensorBufferScopedLock+memcpy the executor itself uses (see its lines ~210/247).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_map.h"      // from @com_google_absl
#include "absl/status/status.h"                // from @com_google_absl
#include "absl/strings/string_view.h"          // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"    // from @litert

namespace litert::lm {

// Write each KVEX tensor into `kv_map` by name. Buffers not present in the map are skipped (the
// NPU executor keeps several banks; only the input bank needs the restored state). Returns the
// number of tensors written, or an error on a size mismatch / malformed container.
inline absl::StatusOr<int> RestoreKvexIntoMap(
    absl::string_view blob,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& kv_map) {
  auto u16 = [&](size_t p) { return (uint16_t)((uint8_t)blob[p] | ((uint8_t)blob[p + 1] << 8)); };
  auto u32 = [&](size_t p) { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= (uint32_t)(uint8_t)blob[p+i] << (8*i); return v; };
  auto u64 = [&](size_t p) { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)(uint8_t)blob[p+i] << (8*i); return v; };

  if (blob.size() < 24 || blob.substr(0, 4) != "KVEX")
    return absl::InvalidArgumentError("KVEX: bad magic/size");
  size_t p = 4 + 4 + 4 + 4 + 4;               // magic, version, num_entries, batch, flags
  uint32_t count = u32(20);
  int written = 0;
  for (uint32_t i = 0; i < count; i++) {
    p += 1;                                    // bank (unused: we target one map)
    uint16_t nl = u16(p); p += 2;
    std::string name(blob.substr(p, nl)); p += nl;
    uint64_t nb = u64(p); p += 8;
    absl::string_view data = blob.substr(p, nb); p += nb;

    auto it = kv_map.find(absl::string_view(name));   // content-hashed lookup
    if (it == kv_map.end()) continue;                 // not this bank's tensor
    ::litert::TensorBuffer& buf = it->second;
    LITERT_ASSIGN_OR_RETURN(size_t sz, buf.PackedSize());
    if (sz != nb)
      return absl::InternalError("KVEX: size mismatch for " + name);
    LITERT_ASSIGN_OR_RETURN(auto lock, ::litert::TensorBufferScopedLock::Create(
                                           buf, ::litert::TensorBuffer::LockMode::kWrite));
    std::memcpy(lock.second, data.data(), nb);        // coherent CPU write via LiteRt API
    written++;
  }
  return written;
}

// Convenience: if the LITERT_RESTORE_KV env var (path to session_kv.bin) is set, load it into
// `kv_map`. Returns written count (0 if env unset). Call from the executor's Create() with the
// INPUT kv buffer map; then set current_step_ and processed_tokens_ (see REPLAY_BUILD.md).
inline absl::StatusOr<int> RestoreKvexFromEnv(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& kv_map) {
  const char* path = std::getenv("LITERT_RESTORE_KV");
  if (!path || !*path) return 0;
  std::ifstream f(path, std::ios::binary);
  if (!f) return absl::NotFoundError(std::string("KVEX: cannot open ") + path);
  std::ostringstream ss; ss << f.rdbuf();
  std::string blob = ss.str();
  LITERT_ASSIGN_OR_RETURN(int n, RestoreKvexIntoMap(blob, kv_map));
  return n;
}

// Step to restore (num tokens the KV represents), from LITERT_RESTORE_STEP, or -1 if unset.
inline int RestoreStepFromEnv() {
  const char* s = std::getenv("LITERT_RESTORE_STEP");
  return (s && *s) ? std::atoi(s) : -1;
}

}  // namespace litert::lm
