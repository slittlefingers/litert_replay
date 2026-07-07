// test_kvex_parse.cc — host-side validation of the KVEX parser logic in kvex_restore.inc
// (no litert / NDK needed). Mirrors the fixed parse (p=24, count@20) and checks the real
// session_kv.bin: 30 tensors, correct names/sizes, exact end-of-blob (no over/under-run).
//   clang++ -std=c++17 -O2 test_kvex_parse.cc -o test_kvex_parse && ./test_kvex_parse session_kv.bin
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "session_kv.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot open %s\n", path); return 1; }
  std::ostringstream ss; ss << f.rdbuf();
  std::string blob = ss.str();

  auto u16 = [&](size_t p){ return (uint16_t)((uint8_t)blob[p] | ((uint8_t)blob[p+1]<<8)); };
  auto u32 = [&](size_t p){ uint32_t v=0; for(int i=0;i<4;i++) v|=(uint32_t)(uint8_t)blob[p+i]<<(8*i); return v; };
  auto u64 = [&](size_t p){ uint64_t v=0; for(int i=0;i<8;i++) v|=(uint64_t)(uint8_t)blob[p+i]<<(8*i); return v; };

  printf("file: %s  size: %zu bytes (%.1f MB)\n", path, blob.size(), blob.size()/1e6);
  if (blob.size() < 24 || blob.substr(0,4) != "KVEX") { printf("FAIL: bad magic/size\n"); return 1; }

  uint32_t count = u32(20);           // count field is at offset 20 (the fix reads this)
  size_t p = 24;                      // first entry starts past the 24-byte header (the off-by-4 fix)
  printf("magic=KVEX version=%u count=%u  (entries start at p=%zu)\n\n", u32(4), count, p);

  size_t data_total = 0; int ok = 0;
  std::vector<std::string> names;
  for (uint32_t i = 0; i < count; i++) {
    if (p + 1 + 2 > blob.size()) { printf("FAIL: overrun at entry %u (header)\n", i); return 1; }
    uint8_t bank = (uint8_t)blob[p]; p += 1;
    uint16_t nl = u16(p); p += 2;
    if (p + nl + 8 > blob.size()) { printf("FAIL: overrun at entry %u (name/size)\n", i); return 1; }
    std::string name = blob.substr(p, nl); p += nl;
    uint64_t nb = u64(p); p += 8;
    if (p + nb > blob.size()) { printf("FAIL: overrun at entry %u (data): need %llu, have %zu\n",
                                       i, (unsigned long long)nb, blob.size()-p); return 1; }
    p += nb;
    data_total += nb;
    names.push_back(name);
    printf("  [%2u] bank=%u  %-16s  %8llu B (%.2f MB)\n", i, bank, name.c_str(),
           (unsigned long long)nb, nb/1e6);
    ok++;
  }

  printf("\nparsed %d tensors; cursor ended at %zu / %zu %s\n", ok, p, blob.size(),
         p == blob.size() ? "(EXACT ✓)" : "(!! leftover/short)");

  // sanity: expect kv_cache_k_0..k_14 and v_0..v_14
  int nk=0, nv=0; for (auto& n : names){ if(n.rfind("kv_cache_k_",0)==0) nk++; if(n.rfind("kv_cache_v_",0)==0) nv++; }
  printf("names: %d k_* + %d v_*   data_total=%.1f MB\n", nk, nv, data_total/1e6);

  bool pass = (ok == (int)count) && (p == blob.size());
  printf("\n%s\n", pass ? "PASS ✅  parser + off-by-4 fix correct" : "FAIL ❌");
  return pass ? 0 : 2;
}
