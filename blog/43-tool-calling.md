# Post 43: Tool Calling

Transcript correction fine-tuning hit a wall. Pivoted to making gwen the
brain of the orchestrator: dispatcher, web search, answer extraction.

**Date**: 2026-04-10

---

## Why not fine-tuning

The plan was to fine-tune Qwen3.5-0.8B for cleaning up STT transcripts
(filler words, punctuation, sentence boundaries). The training pipeline
worked but the model wouldn't converge to anything useful — the
correction task is too close to the model's own generation distribution,
and at 0.8B parameters there isn't enough headroom to learn a reliable
edit function without catastrophic forgetting. Abandoned after multiple
training runs showed no improvement over the base model.

## Tool calling architecture

Gwen now acts as a voice assistant brain. Two LLM calls per query, both
using Qwen3's native tool calling format (`<tool_call>` XML tags):

**Call 1 — Dispatcher.** System prompt defines two tools (`brave_search`,
`spotify_play`). Model classifies the input and emits a tool call. No
direct answers — everything goes through a tool.

**Call 2 — Answer extraction (brave_search only).** Search results from
Brave API are fed back as context. System prompt enforces TTS-friendly
output: plain spoken English, 1-2 sentences, no markdown or URLs.

Both calls clear the KV cache independently — no multi-turn state, no
prompt caching between them. The system prompts are different anyway.

## Implementation

### chat() API in gwen_lib

Added `ChatResult chat(system, user, tools_json, n_predict, greedy, on_token)`
to `gwen::Context`. Internally:

- `format_chat()` builds the ChatML prompt with optional Qwen3 `<tools>`
  block and pre-filled `<think>\n</think>` to skip reasoning
- `generate()` runs prefill + MTP speculative decode (same as before)
- `parse_tool_call()` extracts name/arguments from `<tool_call>` tags
  using nlohmann/json

The old `format_prompt()` in the CLI is gone — everything uses `chat()`.

### Brave Search client

`brave_search.h/cpp` — thin wrapper around the Brave Web Search API using
cpp-httplib (already in vendor/ with OpenSSL support). Returns title +
description for top N results, descriptions capped at 500 chars to bound
the token budget for call 2.

Gotcha: requesting `Accept-Encoding: gzip` from Brave causes cpp-httplib
to fail silently (it doesn't decompress). Removed the header.

### Spotify mock

For now, extracts the query and prints "Now playing: X". Real
implementation will use Brave to find the Spotify track URL (the Spotify
Web API now requires Premium for dev mode since February 2026), then
`xdg-open spotify:track:XXXXX` to play on the local client.

### CMake: one library

Initially split brave_search into a separate CMake target to avoid
pulling cpp-httplib into `GWEN_BUILD_LIB_ONLY` mode. But that's wrong —
the orchestrator needs the full agent, not just inference. Merged
everything into `gwen_lib`. The orchestrator gets inference + chat + tool
calling + web search from a single library link.

```cmake
add_library(gwen_lib STATIC src/gwen.cpp src/gwen.h src/brave_search.cpp src/brave_search.h)
target_link_libraries(gwen_lib PUBLIC llama common cpp-httplib)
```

## Routing accuracy

Tested across diverse inputs:

| Input | Routed to | Result |
|-------|-----------|--------|
| "Who won the 2026 NCAA championship?" | brave_search | Michigan defeated UConn 69-63 |
| "What's the weather in Tokyo?" | brave_search | Light rain, 64F |
| "When did Artemis II launch?" | brave_search | April 1, 2026 |
| "Play Bohemian Rhapsody by Queen" | spotify_play | Bohemian Rhapsody by Queen |
| "Put on some jazz music" | spotify_play | jazz |
| "Play apple seed by aurora" | spotify_play | apple seed by aurora |

The model correctly separates search queries from music requests. Ambiguous
inputs (gibberish, smart home commands) fall through to brave_search,
which is reasonable given only two tools.

## agent() in gwen_lib

The agent pipeline (dispatch → tool → synthesize) was initially in
`tools/gwen/main.cpp` as a standalone `run_agent()` function. Moved the
whole thing into `gwen_lib` as `Context::agent()` so any frontend (CLI,
server, orchestrator) gets the full pipeline from a single library call:

```cpp
AgentResult agent(const std::string& user_input, int n_predict = 500);
```

Returns `AgentResult{text, tool_used}` — the spoken answer and which tool
was invoked. The CLI dropped from ~90 lines of agent logic to 4 lines.

Fallback behavior: if the 0.8B model fails to emit a valid `<tool_call>`
tag (happens with short/ambiguous inputs), the agent defaults to
`brave_search` with the raw user input as query. Better to search and get
a mediocre answer than to say "I don't understand."

## CLI

```
gwen --agent "Who won the 2026 NCAA championship?"
```

Diagnostic output (dispatch decisions, search queries, result counts) goes
to stderr. Answers go to stdout. Ready for piping into the orchestrator's
TTS pipeline.
