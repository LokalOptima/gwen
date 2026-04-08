# Post 42: Simplification Sprint and libgwen

Three cleanup commits preparing gwen for orchestrator integration: strip
sampling flags, remove the non-MTP fallback, and expose a library API.

**Date**: 2026-04-08

---

## 1. Replace 34 sampling flags with `--greedy`

Sampling parameters are critical for Qwen3.5-0.8B quality. Exposing them
as CLI flags invites misconfiguration. We hardcoded the Qwen-recommended
values (temp=1.0, top_k=20, presence_penalty=2.0) in `common.h` and
replaced all 34 sampling flags with a single `--greedy` escape hatch.

`--greedy` sets temp=0, all penalties off, and triggers the fast path in
`generate_mtp()`: GPU-side argmax (8 bytes transferred instead of 2MB
logits), empty penalty sampler (`penalty_last_n=0` hits the `is_empty`
short-circuit), full sampler chain bypassed.

The server API is unaffected — it gets sampling params from JSON request
bodies, not CLI flags. Net result: -328 lines from `arg.cpp`.

## 2. Remove the non-MTP code path

Gwen is MTP-only. The single-token decode fallback existed solely for
correctness tests (compare base-no-MTP vs base-with-MTP output). But we
can just compare against upstream llama.cpp instead.

Removed from `completion.cpp`:
- `use_mtp` conditional and `LLAMA_NO_MTP` env var
- Single-token decode loop (sample → accept → push → continue)
- Interactive mode (antiprompt, EOG, stdin input, conversation turns)
- ~15 dead variables (`is_interacting`, `need_insert_eot`, `mtp_skip_decode`, etc.)

The main loop is now: prefill prompt in batches → `generate_mtp()` → break.
~350 lines deleted.

Added an early check: if `!llama_model_has_mtp(model)`, error out with a
message about the missing sidecar.

### Correctness against upstream

`test_correctness.sh` now generates baselines with upstream llama.cpp
(`--temp 0 --presence-penalty 0`) and compares against gwen's MTP output
(`--greedy`). This works because in greedy mode, MTP's accept/reject is
a simple equality check — every emitted token is the main model's argmax
regardless of whether the draft was accepted.

The only divergence source is FP precision from the 2-token verification
batch (batched matmul accumulates differently). Previously we tolerated
500-token failures; against upstream, all 36/36 pass clean.

`config.sh` now resolves `LLAMA_COMPLETION` to the system-installed
upstream binary (or falls back to the vendored `third_party/llama.cpp`
with `LD_LIBRARY_PATH`).

## 3. gwen_lib for orchestrator integration

The orchestrator links paraketto (STT) and rokoko (TTS) as static
libraries via FetchContent. gwen needs the same pattern.

### API (`src/gwen.h`)

```cpp
namespace gwen {
  using TokenCallback = std::function<bool(const char* text, int token_id)>;

  struct Context {
    bool init(const std::string& model_path);  // auto-discovers MTP sidecar
    std::string generate(const std::string& prompt, int n_predict = 100,
                         bool greedy = true, TokenCallback on_token = nullptr);
    Stats last_stats() const;
    void destroy();
  };
}
```

### Implementation (`src/gwen.cpp`)

Self-contained: all sampling helpers (AVX2 top-k, fast_sample,
compute_mtp_q_draft) and the full generate_mtp loop are in this file.
The `Context::Impl` wraps `common_init_from_params` for model loading
and handles prefill + MTP decode internally.

Completion.cpp keeps its own copy of generate_mtp. The ~300 lines of
duplication is acceptable — the CLI has different concerns (arg parsing,
console output, session caching, MTP_STATS on stderr) than the library
(clean callback API, no global state). Dedup can happen later once the
orchestrator integration is proven.

### CMake

```cmake
option(GWEN_BUILD_LIB_ONLY "gwen: only build the library, skip executables" OFF)

add_library(gwen_lib STATIC src/gwen.cpp src/gwen.h)
target_include_directories(gwen_lib PUBLIC src/)
target_link_libraries(gwen_lib PUBLIC llama common)
```

With `GWEN_BUILD_LIB_ONLY=ON`, only gwen_lib + dependencies (llama,
common, ggml) are built. Tools, server, vendor libs are skipped.

The orchestrator CMake:
```cmake
set(GWEN_BUILD_LIB_ONLY ON CACHE BOOL "" FORCE)
FetchContent_Declare(gwen GIT_REPOSITORY ... GIT_TAG main)
FetchContent_MakeAvailable(gwen)
target_link_libraries(orchestrator-server paraketto_lib rokoko_lib gwen_lib)
```

## Results

| Metric | Before | After |
|--------|--------|-------|
| `arg.cpp` sampling flags | 34 | 1 (`--greedy`) |
| `completion.cpp` lines | ~1600 | ~1200 |
| Non-MTP code path | present | removed |
| Library API | none | `gwen_lib` |
| Correctness (36 tests) | 36/36 | 36/36 |
| Baseline source | self (non-MTP) | upstream llama.cpp |
