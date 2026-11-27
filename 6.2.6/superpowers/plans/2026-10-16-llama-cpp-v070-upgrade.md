# llama.cpp v0.7.0 upgrade for llama_cpp_vendor + llama_ros Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin `llama_cpp_vendor` to llama.cpp release `v0.7.0`, migrate `llama_ros` to the refactored chat API, and expose the new llama.cpp parameters (`--moe-cache-mib`, `--spec-draft-n-ubatch`, `--cache-ram` explicitness) as ROS parameters.

**Architecture:** The vendor package fetches llama.cpp via `FetchContent` and builds `mtmd` from an explicit source list, so the upgrade is a tag change plus one new mtmd source file. `llama_ros` maps ROS parameters onto `common_params` in `llama_params.cpp`; three new mappings are added there and consumed by `Llama` (prompt cache gating, draft context sizing). The main API break is the v0.7.0 chat refactor (`common_chat_session` with `feed()`/`finish()` replaced the old `common_chat_parse(text, ...)` + fat `common_chat_parser_params`), which requires a focused migration of the 6 chat-related files, mirroring what upstream's `tools/server` does.

**Tech Stack:** ROS 2 (ament/colcon), C++17, llama.cpp v0.7.0 (`620d2225cc436ab0b001e3200e590e3b286ef9ab`), gtest.

**Spec:** External: <https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0>. Requirements + exact interface deltas from upstream `b11465 → v0.7.0` are captured in *Global Constraints* (verified against upstream sources during planning).

## Global Constraints

- Vendor tag: `GIT_TAG v0.7.0` in `llama_cpp_vendor/CMakeLists.txt` (was `b11465`). Nightly equivalent: `b11669`.
- New mtmd source file at v0.7.0: `tools/mtmd/models/nemotron-omni.cpp` (upstream list order: after `nemotron-v2-vl.cpp`).
- Upstream API deltas that break current code (all in `common/chat.h` / `llama.h`):
  - `common_chat_params::prompt` is now `common_chat_prompt` (use `.str()`), `::parser` is now `common_peg_arena` (no `.load()/.save()`, it is assignable).
  - `common_chat_parser_params` only has `format`, `generation_prompt` (now `common_chat_input`), `debug`, `parser`. Removed: `reasoning_format`, `reasoning_in_content`, `parse_tool_calls`, `is_continuation`, `echo`.
  - `common_chat_parse(common_chat_input, bool is_partial, const common_chat_parser_params&)`.
  - New `common_chat_session` (ctor: `tmpls, vocab, common_chat_templates_inputs, common_chat_session_params`; methods `prompt()`, `format()`, `msg()`, `grammar()`, `generation_prompt()`, `thinking_start_tag()`, `thinking_end_tags()`, `apply_sampling()`, `has_template()`, `feed()`, `finish()`). `finish()` may be called once; `feed()`/`finish()` assert if called after `finish()`.
  - New helpers `common_sampling_add_preserved_tokens` / `common_sampling_add_grammar_triggers` are used internally by the session; `common_chat_templates_support_enable_thinking()` is the new way to check thinking support.
  - `common_get_decision_type(const std::string & fname)` → `common_get_gguf_info(fname)` (the overload taking `const llama_model*` that llama_ros uses is unchanged).
- New `common_params` / `llama_context_params` fields at v0.7.0: `moe_cache_size` (size_t, bytes, `--moe-cache-mib` × 1 MiB), `speculative.draft.n_ubatch` (int32, 0 = target's n_ubatch), `cache_ram_explicit` (bool). `common_context_params_to_llama()` already copies `moe_cache_size`.
- Upstream semantics to mirror: decision models disable the prompt cache unless `cache_ram_explicit` is set (`server-context.cpp` sets `cache_ram_mib = 0`, `cache_idle_slots = false`); the draft/MTP contexts zero `moe_cache_size` ("only used by the target context") and inherit `n_ubatch` from `speculative.draft.n_ubatch` when > 0.
- Decision types: v0.7.0 adds `COMMON_DECISION_TYPE_VLLM_DECISION2`, `_VLLM_D3`, `_LFM2_D1`, `_LFM2_D1_OMNI`; `"vllm-d3"` GGUFs now parse as `VLLM_D3` (previously `UNKNOWN` + workaround) and behave exactly like `PPLX_DECIDER` (labels A..Z/AA..ZZ; non-causal attention comes from GGUF metadata). `LFM2_D1`, `LFM2_D1_OMNI` and `VLLM_DECISION2` stay unsupported (throw, as today).
- New ROS params (names/defaults are final): `cache.moe_cache_mib` (int32, `0`, MiB; `>0` → `params.moe_cache_size = value * 1024 * 1024`, `<0` treated as 0), `speculative.draft.n_ubatch` (int32, `0`; `<0` treated as 0), `cache.cache_ram_explicit` (bool, `false`).
- Do not bump any `package.xml` version.
- Style: run `clang-format` (repo `.clang-format`) on every edited C++ file; the `ament_clang_format` test must pass.
- Build/verify commands use `colcon` from the workspace root `/home/miguel/ros2_ws` (packages live in `src/llama_ros`).

## Review Focus

- **PEG parse failure while streaming** (grammar permits output the parser rejects): must fall back to raw content, keep diffs empty, and never throw out of `update_chat_msg`. Test: existing `update_chat_msg` fallback path preserved + new slot unit test (Task 4).
- **`update_chat_msg` call pattern** (per-token partial feeds, then one final call, possibly extra calls after the final one): the session's `finish()` must run at most once; later calls must be no-ops. Test: new slot unit test (Task 4).
- **Slot reuse across request types** (plain completion after a chat request): stale chat parser must not leak into completion parsing; `ServerSlot::reset()` must restore a default (content-only) session. Test: slot unit test (Task 4).
- **User grammar + tools together**: explicit `sampling_config.grammar`/`grammar_schema` must still win over the template's tool-call grammar; the parallel-tool-calls grammar workaround must only run when the tool grammar is the effective one. Verified by code review in Task 4 (no model-free test).
- **Non-positive values for the new int params**: `cache.moe_cache_mib < 0` and `speculative.draft.n_ubatch < 0` must clamp to 0/disabled instead of overflowing `size_t`/being forwarded. Test: `test_llama_params.cpp` (Task 2).

---

### Task 1: Bump `llama_cpp_vendor` to v0.7.0

**Files:**
- Modify: `llama_cpp_vendor/CMakeLists.txt`

**Interfaces:**
- Produces: installed targets `llama_cpp_vendor::llama`, `::llama-common`, `::mtmd`, `::cpp-httplib` etc. built from v0.7.0 (used by every later task).

- [ ] **Step 1: Change the fetched tag**

In `llama_cpp_vendor/CMakeLists.txt` change:

```cmake
  GIT_TAG        b11465
```

to:

```cmake
  GIT_TAG        v0.7.0
```

- [ ] **Step 2: Add the new mtmd model source**

In the `add_library(mtmd ...)` source list, add the file after `models/nemotron-v2-vl.cpp` (mirrors upstream `tools/mtmd/CMakeLists.txt`):

```cmake
  ${llama_SOURCE_DIR}/tools/mtmd/models/nemotron-omni.cpp
```

- [ ] **Step 3: Build the vendor package**

Run (from `/home/miguel/ros2_ws`):

```bash
colcon build --packages-select llama_cpp_vendor --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: build succeeds; `nemotron-omni.cpp` appears in the compile commands.

- [ ] **Step 4: Verify the fetched revision**

Run:

```bash
git -C build/llama_cpp_vendor/llama-src describe --tags 2>/dev/null || git -C build/llama_cpp_vendor/_deps/llama-src describe --tags
```

Expected: `v0.7.0`. (If FetchContent placed the clone elsewhere, find it with `find build/llama_cpp_vendor -maxdepth 3 -name llama-src -type d`.)

- [ ] **Step 5: Commit**

```bash
git add llama_cpp_vendor/CMakeLists.txt
git commit -m "feat(llama_cpp_vendor): upgrade llama.cpp to v0.7.0"
```

---

### Task 2: Declare + map the three new parameters

**Files:**
- Modify: `llama_ros/src/llama_utils/llama_params.cpp` (declaration block ~line 112 `context.*`? No — see mapping below; and `get_llama_params`)
- Modify: `llama_ros/test/test_llama_params.cpp`
- Modify: `README.md` (parameter tables)

**Interfaces:**
- Consumes: v0.7.0 `common_params` fields `moe_cache_size` (size_t bytes), `speculative.draft.n_ubatch` (int32), `cache_ram_explicit` (bool).
- Produces: ROS params `cache.moe_cache_mib`, `speculative.draft.n_ubatch`, `cache.cache_ram_explicit`; `LlamaParams.params` carries them for Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `llama_ros/test/test_llama_params.cpp` (inside the existing anonymous `LlamaParamsTest` fixture style):

```cpp
/**
 * @brief Test that verifies the new v0.7.0 parameters and their mapping.
 */
TEST_F(LlamaParamsTest, NewV070Parameters) {
  // Declared with the documented defaults
  EXPECT_EQ(node->get_parameter("cache.moe_cache_mib").as_int(), 0);
  EXPECT_EQ(node->get_parameter("speculative.draft.n_ubatch").as_int(), 0);
  EXPECT_FALSE(node->get_parameter("cache.cache_ram_explicit").as_bool());

  // MiB -> bytes conversion
  node->set_parameter(rclcpp::Parameter("cache.moe_cache_mib", 512));
  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);
  EXPECT_EQ(params.params.moe_cache_size, (size_t)512 * 1024 * 1024);

  // Non-positive values disable / are ignored
  node->set_parameter(rclcpp::Parameter("cache.moe_cache_mib", -1));
  node->set_parameter(rclcpp::Parameter("speculative.draft.n_ubatch", -3));
  params = llama_utils::get_llama_params(node);
  EXPECT_EQ(params.params.moe_cache_size, (size_t)0);
  EXPECT_EQ(params.params.speculative.draft.n_ubatch, 0);

  // Pass-through values
  node->set_parameter(rclcpp::Parameter("speculative.draft.n_ubatch", 128));
  node->set_parameter(rclcpp::Parameter("cache.cache_ram_explicit", true));
  params = llama_utils::get_llama_params(node);
  EXPECT_EQ(params.params.speculative.draft.n_ubatch, 128);
  EXPECT_TRUE(params.params.cache_ram_explicit);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `/home/miguel/ros2_ws`):

```bash
colcon build --packages-select llama_cpp_vendor llama_ros --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select llama_ros --ctest-args -R llama_ros_tests
```

Expected: FAIL — `cache.moe_cache_mib` is not declared (`rclcpp::exceptions::ParameterNotDeclaredException`).

- [ ] **Step 3: Declare the parameters**

In `llama_params.cpp` `declare_llama_params()`:

- In the `cache.*` int32 block (next to `cache_ram_mib`):

```cpp
  node->declare_parameters<int32_t>("cache", {
                                               {"moe_cache_mib", 0},
                                           });
```

(adjust to the surrounding formatting), and in the `cache.*` bool block:

```cpp
                  {"cache_ram_explicit", false},
```

- In the `speculative.draft` int32 block (`n_max`, `n_min`, `n_gpu_layers`) add:

```cpp
                                        {"n_ubatch", 0},
```

- [ ] **Step 4: Map the parameters in `get_llama_params()`**

In the `cache.*` section add:

```cpp
  int32_t moe_cache_mib = 0;
  node->get_parameter("cache.moe_cache_mib", moe_cache_mib);
  if (moe_cache_mib < 0) {
    RCLCPP_WARN(node->get_logger(),
                "cache.moe_cache_mib cannot be negative, disabling the MoE "
                "expert cache");
    moe_cache_mib = 0;
  }
  params.params.moe_cache_size = (size_t)moe_cache_mib * 1024 * 1024;

  node->get_parameter("cache.cache_ram_explicit",
                      params.params.cache_ram_explicit);
```

In the `speculative.draft` section add:

```cpp
  node->get_parameter("speculative.draft.n_ubatch",
                      params.params.speculative.draft.n_ubatch);
  if (params.params.speculative.draft.n_ubatch < 0) {
    RCLCPP_WARN(node->get_logger(),
                "speculative.draft.n_ubatch cannot be negative, using the "
                "target n_ubatch");
    params.params.speculative.draft.n_ubatch = 0;
  }
```

- [ ] **Step 5: Update `README.md`**

Add rows to the parameter tables (match existing column formatting):

- KV Cache (`cache.*`) section, after `cache.cache_ram_mib`:

```
| `cache.cache_ram_explicit`  | `bool`   | `false` | Mark the prompt cache as explicitly configured. Decision models disable the prompt cache unless this is `true` |
| `cache.moe_cache_mib`       | `int32`  | `0`     | GPU cache size in MiB for the MoE experts kept in host memory (split among GPUs like the layers; `0` = disabled) |
```

- Draft Model (`speculative.draft.*`) section, after `speculative.draft.n_gpu_layers`:

```
| `speculative.draft.n_ubatch`              | `int32`    | `0`     | Physical batch size of the draft context (`0` = use the target's `context.n_ubatch`)                                                                                                                     |
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
colcon build --packages-select llama_ros --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select llama_ros --ctest-args -R llama_ros_tests
colcon test-result --verbose
```

Expected: PASS (all `llama_ros_tests`, including `NewV070Parameters`).

- [ ] **Step 7: clang-format and commit**

```bash
clang-format -i llama_ros/src/llama_utils/llama_params.cpp llama_ros/test/test_llama_params.cpp
git add llama_ros/src/llama_utils/llama_params.cpp llama_ros/test/test_llama_params.cpp README.md
git commit -m "feat(llama_ros): expose v0.7.0 params (moe cache, draft n_ubatch, cache_ram_explicit)"
```

---

### Task 3: Consume the new params in `Llama` (draft context + decision cache)

**Files:**
- Modify: `llama_ros/src/llama_ros/llama.cpp` (`Llama::Llama` ~line 266, `Llama::init_speculative` ~line 2049-2106)

**Interfaces:**
- Consumes: `params.moe_cache_size`, `params.speculative.draft.n_ubatch`, `params.cache_ram_explicit`, `is_decision()` from Task 2.
- Produces: correct runtime behavior; no new symbols for later tasks.

- [ ] **Step 1: Gate the host-RAM prompt cache for decision models**

In the `Llama::Llama` constructor, **move** the existing block (currently right after the checkpoints block, before slots are created):

```cpp
  // Host-RAM prompt cache (0 disables, < 0 means unlimited)
  if (this->params.cache_ram_mib != 0) {
    this->prompt_cache_ = std::make_unique<PromptCache>(
        this->params.cache_ram_mib, this->params.n_ctx);
  }
```

to **after** `this->init_decision();` (~line 370) and replace with:

```cpp
  // Host-RAM prompt cache (0 disables, < 0 means unlimited). Decision models
  // rarely repeat a state, so the cache is opt-in for them; mirrors the
  // llama.cpp server (cache_ram_explicit is set by --cache-ram upstream).
  if (this->is_decision() && !this->params.cache_ram_explicit) {
    LLAMA_LOG_INFO("decision model: prompt cache disabled by default, set "
                   "cache.cache_ram_explicit to true to enable it");
    this->params.cache_ram_mib = 0;
    this->params.cache_idle_slots = false;
  }

  // Host-RAM prompt cache (0 disables, < 0 means unlimited)
  if (this->params.cache_ram_mib != 0) {
    this->prompt_cache_ = std::make_unique<PromptCache>(
        this->params.cache_ram_mib, this->params.n_ctx);
  }
```

(Keep only one copy of the creation block; nothing before `init_decision()` reads `prompt_cache_`.)

- [ ] **Step 2: Zero the MoE cache on the draft context and honor `n_ubatch`**

In `Llama::init_speculative()`, inside the `if (has_draft)` branch, next to the other `params_dft` overrides (after the `cache_type_v` lines):

```cpp
    if (spec_params.draft.n_ubatch > 0) {
      params_dft.n_ubatch = spec_params.draft.n_ubatch;
    }
    // the MoE expert cache is only used by the target context
    params_dft.moe_cache_size = 0;
```

In the `else if (is_mtp)` branch, after `cparams_mtp.n_rs_seq = 0;`:

```cpp
    // the MoE expert cache is only used by the target context
    cparams_mtp.moe_cache_size = 0;
```

- [ ] **Step 3: Build and run the test suite**

```bash
colcon build --packages-select llama_ros --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select llama_ros
colcon test-result --verbose
```

Expected: build passes; tests pass at the same rate as before this task (model-dependent tests self-skip without models).

- [ ] **Step 4: clang-format and commit**

```bash
clang-format -i llama_ros/src/llama_ros/llama.cpp
git add llama_ros/src/llama_ros/llama.cpp
git commit -m "feat(llama_ros): apply moe cache, draft n_ubatch and decision prompt-cache semantics"
```

---

### Task 4: Migrate chat handling to `common_chat_session`

**Files:**
- Modify: `llama_ros/include/llama_utils/chat_utils.hpp` (`ChatCompletionsContext`)
- Modify: `llama_ros/src/llama_utils/chat_utils.cpp` (`prepare_chat_completions_call`)
- Modify: `llama_ros/include/llama_utils/chat_formatter.hpp` + `src/llama_utils/chat_formatter.cpp` (`parse_response`)
- Modify: `llama_ros/src/llama_ros/request_handler.cpp` (`ChatCompletionRequestHandler::handle`)
- Modify: `llama_ros/src/llava_ros/llava_request_handler.cpp` (`LlavaChatCompletionRequestHandler::handle`)
- Modify: `llama_ros/include/llama_ros/server_slot.hpp` + `src/llama_ros/server_slot.cpp`
- Test: `llama_ros/test/test_chat_session_slot.cpp` (new) + add it to `CMakeLists.txt` test source list

**Interfaces:**
- Consumes: `llama->get_chat_formatter()->get_templates()`, `llama->get_vocab()` (existing), v0.7.0 `common_chat_session`.
- Produces (exact shapes later steps rely on):
  - `struct ChatCompletionsContext { common_chat_session chat_session; common_params_sampling sparams; };`
  - `ServerSlot` members: `common_chat_session chat_session;`, `size_t chat_fed_len = 0;`, `bool chat_finished = false;`
  - `ChatFormatter::parse_response(const common_chat_input &input, bool is_partial, const common_chat_parser_params &chat_syntax)`

- [ ] **Step 1: Write the failing slot unit test**

Create `llama_ros/test/test_chat_session_slot.cpp` (MIT header copied from an existing test) covering the Review Focus behaviors that need no model:

```cpp
#include <gtest/gtest.h>

#include "llama_ros/server_slot.hpp"

namespace {

// a plain completion is parsed as raw content with a default session
TEST(ChatSessionSlot, DefaultSessionIsContentOnly) {
  llama_ros::ServerSlot slot;
  slot.generated_text = "hello world";
  slot.stop = llama_ros::StopType::NO_STOP;
  std::vector<common_chat_msg_diff> diffs;

  const auto &msg = slot.update_chat_msg(diffs);
  EXPECT_EQ(msg.content, "hello world");
}

// feed() during streaming, finish() once at the end, later calls are no-ops
TEST(ChatSessionSlot, FinishRunsOnce) {
  llama_ros::ServerSlot slot;
  std::vector<common_chat_msg_diff> diffs;

  slot.generated_text = "a";
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  slot.generated_text = "ab";
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  slot.stop = llama_ros::StopType::FULL_STOP;
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  // must not assert/throw on repeated final calls
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  EXPECT_EQ(slot.chat_msg.content, "ab");
}

// reset() restores a fresh session and re-enables feeding
TEST(ChatSessionSlot, ResetRestoresDefaultSession) {
  llama_ros::ServerSlot slot;
  std::vector<common_chat_msg_diff> diffs;

  slot.generated_text = "x";
  slot.stop = llama_ros::StopType::FULL_STOP;
  slot.update_chat_msg(diffs);
  EXPECT_TRUE(slot.chat_finished);

  slot.reset();
  EXPECT_FALSE(slot.chat_finished);
  EXPECT_EQ(slot.chat_fed_len, 0u);

  slot.generated_text = "y";
  slot.stop = llama_ros::StopType::NO_STOP;
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  EXPECT_EQ(slot.chat_msg.content, "y");
}

} // namespace
```

Add `test/test_chat_session_slot.cpp` to the `${PROJECT_NAME}_tests` source list in `llama_ros/CMakeLists.txt`.

- [ ] **Step 2: Run to verify it fails to compile (no `chat_session` member yet)**

```bash
colcon build --packages-select llama_ros --cmake-args -DBUILD_TESTING=ON
```

Expected: compile error (`chat_finished` / `chat_fed_len` unknown; `update_chat_msg` still parses with old params).

- [ ] **Step 3: Migrate `server_slot.hpp` / `server_slot.cpp`**

`server_slot.hpp`: in `struct slot_params` remove `common_chat_parser_params oaicompat_chat_syntax;`. As public members next to `chat_msg` add:

```cpp
  /// @brief Chat session owning parsing for the ongoing generation.
  common_chat_session chat_session;

  /// @brief Bytes of generated_text already fed to the chat session.
  size_t chat_fed_len = 0;

  /// @brief Whether the chat session was finished for this generation.
  bool chat_finished = false;
```

`server_slot.cpp` rewrite of `update_chat_msg` (keep the fallback + diff logic; only the parse call and bookkeeping change):

```cpp
const common_chat_msg &ServerSlot::update_chat_msg(
    std::vector<common_chat_msg_diff> &oaicompat_msg_diffs) {

  // after finish() the session asserts on further feed()/finish() calls
  if (this->chat_finished) {
    oaicompat_msg_diffs.clear();
    return this->chat_msg;
  }

  auto previous_msg = this->chat_msg;
  common_chat_msg new_msg;
  bool parse_fallback = false;

  const bool is_partial = this->stop != StopType::FULL_STOP;
  common_chat_input added(
      this->generated_text.substr(this->chat_fed_len));
  this->chat_fed_len = this->generated_text.size();

  try {
    new_msg = is_partial ? this->chat_session.feed(added)
                         : this->chat_session.finish(added);
  } catch (const std::exception &e) {
    // PEG parser may fail when the grammar allows output the parser cannot
    // fully consume (e.g. multiple <tool_call> blocks with a single-block
    // parser). Fall back to treating everything as raw content; the Python
    // layer will extract all tool calls via regex.
    LLAMA_LOG_WARN("PEG parse failed, falling back to raw content: %s",
                   e.what());
    new_msg.role = "assistant";
    new_msg.content = this->generated_text;
    parse_fallback = true;
  }

  // finish() (successful or not) makes the session final
  this->chat_finished = !is_partial;

  if (!new_msg.empty()) {
    std::function<std::string()> gen_tool_call_id =
        static_cast<std::string (*)()>(llama_utils::random_string);
    new_msg.set_tool_call_ids(this->generated_tool_call_ids, gen_tool_call_id);
    this->chat_msg = new_msg;
    if (!parse_fallback) {
      oaicompat_msg_diffs = common_chat_msg_diff::compute_diffs(
          previous_msg, new_msg.empty() ? previous_msg : new_msg);
    }
  }

  return this->chat_msg;
}
```

In `reset()` add (next to `this->chat_msg = {};`):

```cpp
  this->chat_session = common_chat_session{};
  this->chat_fed_len = 0;
  this->chat_finished = false;
```

- [ ] **Step 4: Migrate `ChatCompletionsContext` + `prepare_chat_completions_call`**

`chat_utils.hpp`: replace the struct with

```cpp
struct ChatCompletionsContext {
  common_chat_session chat_session;
  common_params_sampling sparams;
};
```

`chat_utils.cpp` `prepare_chat_completions_call()` — replace the body from `ctx.prompt_format_config = ...` through the end with (keeping the comment blocks and the reasoning-budget block adapted to session getters):

1. Build inputs as today: `ctx.prompt_format_config = parse_chat_completions_goal(goal);` (keep as a local `auto session_inputs = ...`).
2. Thinking-parser workaround (replaces the `supports_thinking` + parser-copy hack; keep the explanatory comment):

```cpp
  auto tmpls = llama->get_chat_formatter();

  // When thinking is disabled but the model supports it, Jinja templates
  // pre-fill an empty <think></think> block in the generation_prompt. A parser
  // built with reasoning_format=NONE cannot consume this block, leaking the
  // tags into the content. Build the session with reasoning_format=AUTO in
  // that case so the parser consumes the pre-filled block; the prompt is
  // unchanged because templates use enable_thinking (not reasoning_format) to
  // control the pre-fill.
  if (!session_inputs.enable_thinking &&
      common_chat_templates_support_enable_thinking(tmpls->get_templates())) {
    session_inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
  }

  ctx.chat_session = common_chat_session(tmpls->get_templates(),
                                         llama->get_vocab(), session_inputs);
  ctx.sparams = llama_utils::parse_sampling_params(goal->sampling_config,
                                                   llama->get_n_vocab());
```

3. Grammar/sampling rules (replaces the manual preserved-token/trigger/generation-prompt/grammar blocks):

```cpp
  // An explicit user grammar or grammar_schema wins over the template's
  // tool-call grammar.
  const auto user_grammar = ctx.sparams.grammar;
  const bool user_grammar_set = !goal->sampling_config.grammar.empty() ||
                                !goal->sampling_config.grammar_schema.empty();
  const bool user_grammar_lazy = ctx.sparams.grammar_lazy;

  // The session resolves preserved tokens, grammar triggers (WORD -> TOKEN
  // where possible), the tool-call grammar, the generation prompt, and the
  // lazy-grammar flags with the model vocab.
  ctx.chat_session.apply_sampling(ctx.sparams);

  if (user_grammar_set) {
    ctx.sparams.grammar = user_grammar;
  }
  ctx.sparams.grammar_lazy = ctx.sparams.grammar_lazy || user_grammar_lazy;

  // Parallel tool calls grammar workaround for llama.cpp autoparser bug
  // (b8261) ... (keep the existing comment and rewrite block verbatim,
  //  guarded now by:)
  if (!user_grammar_set && goal->tools.size() != 0 &&
      !ctx.sparams.grammar.empty()) {
    /* existing std::string &grammar = ctx.sparams.grammar.grammar; ... */
  }
```

4. Reasoning budget block: replace `ctx.chat_prompt_instance.thinking_end_tags` → `ctx.chat_session.thinking_end_tags()` and `ctx.chat_prompt_instance.thinking_start_tag` → `ctx.chat_session.thinking_start_tag()`; everything else unchanged.
5. Delete the old `ctx.oaicompat_chat_syntax...`, manual preserved-tokens loop, manual trigger loop, `ctx.sparams.generation_prompt` line, and the `parser.load(...)` block (session owns all of it).

`chat_formatter.hpp/.cpp`: change the static helper to the new parse signature:

```cpp
  static common_chat_msg parse_response(const common_chat_input &input,
                                        bool is_partial,
                                        const common_chat_parser_params &chat_syntax);
```

with body `return common_chat_parse(input, is_partial, chat_syntax);`.

- [ ] **Step 5: Migrate the two request handlers**

`request_handler.cpp` `ChatCompletionRequestHandler::handle()`: drop the unused `inputs` local; replace

```cpp
  slot->params.oaicompat_chat_syntax = chat_context.oaicompat_chat_syntax;
  ...
  slot->prompt_tokens = common_tokenize(this->llama_->get_ctx(),
                      chat_context.chat_prompt_instance.prompt, false, true);
  ...
  slot->chat_format = chat_context.chat_prompt_instance.format;
```

with

```cpp
  slot->chat_session = std::move(chat_context.chat_session);
  slot->params.sampling = chat_context.sparams;
  ...
  slot->prompt_tokens = common_tokenize(
      this->llama_->get_ctx(), slot->chat_session.prompt().str(), false, true);
  ...
  slot->chat_format = slot->chat_session.format();
```

(`slot->reset()` runs first, so `chat_fed_len`/`chat_finished`/`chat_msg` are clean.)

`llava_request_handler.cpp` `LlavaChatCompletionRequestHandler::handle()`: same replacements (`slot->chat_session = std::move(chat_context.chat_session);` before tokenization; prompt from `slot->chat_session.prompt().str()`), and because this handler does not call `slot->reset()`, set `slot->chat_msg = {}; slot->chat_fed_len = 0; slot->chat_finished = false;` next to the session assignment.

- [ ] **Step 6: Build and run the tests**

```bash
colcon build --packages-select llama_ros --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select llama_ros
colcon test-result --verbose
```

Expected: clean compile; `ChatSessionSlot.*` tests PASS; other tests pass or self-skip exactly as before.

- [ ] **Step 7: clang-format and commit**

```bash
clang-format -i llama_ros/include/llama_utils/chat_utils.hpp llama_ros/src/llama_utils/chat_utils.cpp \
  llama_ros/include/llama_utils/chat_formatter.hpp llama_ros/src/llama_utils/chat_formatter.cpp \
  llama_ros/src/llama_ros/request_handler.cpp llama_ros/src/llava_ros/llava_request_handler.cpp \
  llama_ros/include/llama_ros/server_slot.hpp llama_ros/src/llama_ros/server_slot.cpp \
  llama_ros/test/test_chat_session_slot.cpp
git add -A && git commit -m "feat(llama_ros): migrate chat parsing to common_chat_session (llama.cpp v0.7.0 API)"
```

---

### Task 5: Fix the vllm-d3 decision-type regression

**Files:**
- Modify: `llama_ros/src/llama_ros/decision_model.cpp` (~line 167-181 init, ~line 486-498 fill_task)
- Modify: `llama_ros/include/llama_ros/decision_model.hpp` (doc comment ~line 126)
- Test: `llama_ros/test/test_decision_model.cpp`

**Interfaces:**
- Consumes: `COMMON_DECISION_TYPE_VLLM_D3` (new upstream enum value).
- Produces: same readout as `PPLX_DECIDER` for vllm-d3 models.

- [ ] **Step 1: Write the failing test**

In `test_decision_model.cpp` add a test (mirror the existing label-config style, e.g. the OPENJEV/LAYA tests):

```cpp
TEST(DecisionModel, VllmD3UsesLabelReadout) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_VLLM_D3;
  config.n_options_max = 255;
  config.labels = {11, 12};

  llama_ros::DecisionModel model(config);
  llama_ros::DecisionTaskMeta meta;
  std::vector<llama_token> tokens = {1, 2, 3};
  model.fill_task(tokens, choice_question(), /* options */ {...}, meta);

  EXPECT_FALSE(model.needs_embeddings());
  EXPECT_EQ(meta.labels, (std::vector<llama_token>{11, 12}));
}
```

Use the existing helpers in that file for the options list (reuse whatever the OPENJEV/LEV tests pass).

- [ ] **Step 2: Run to verify it fails**

Build `llama_ros_tests` and run it; expected: FAIL — `fill_task` throws "unsupported decision model type" for `VLLM_D3`.

- [ ] **Step 3: Implement the mapping**

In `decision_model.cpp` `init_from_model()`:

- Remove the now-dead workaround block mapping `UNKNOWN` + `"vllm-d3"` → `PPLX_DECIDER`, and update the comment to:

```cpp
  // vllm-d3 is a known type since llama.cpp v0.7.0 (COMMON_DECISION_TYPE_VLLM_D3);
  // it behaves like pplx-decider, its non-causal attention is handled by
  // llama.cpp from the GGUF metadata
```

- Extend the label branch condition:

```cpp
  } else if (model_type == COMMON_DECISION_TYPE_LEV ||
             model_type == COMMON_DECISION_TYPE_NIMBLE ||
             model_type == COMMON_DECISION_TYPE_PPLX_DECIDER ||
             model_type == COMMON_DECISION_TYPE_VLLM_D3) {
```

In `fill_task()` extend the first label-based `case`:

```cpp
  case COMMON_DECISION_TYPE_PPLX_DECIDER:
  case COMMON_DECISION_TYPE_VLLM_D3:
```

Update the `decision_model.hpp` class comment (~line 126) to list `VLLM_D3` with the supported types.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
colcon build --packages-select llama_ros --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
colcon test --packages-select llama_ros --ctest-args -R llama_ros_tests
colcon test-result --verbose
```

Expected: PASS.

- [ ] **Step 5: clang-format and commit**

```bash
clang-format -i llama_ros/src/llama_ros/decision_model.cpp llama_ros/test/test_decision_model.cpp
git add -A && git commit -m "fix(decision_model): handle vllm-d3 as a known llama.cpp v0.7.0 decision type"
```

---

### Task 6: Whole-workspace verification

**Files:** none (verification only).

- [ ] **Step 1: Build the whole workspace**

```bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

Expected: all packages build (including `llama_cli`, `llama_bt`, `llama_bringup`).

- [ ] **Step 2: Run the full test suite + linters**

```bash
colcon test
colcon test-result --verbose
```

Expected: no new failures vs. before the change (model-dependent gtest cases self-skip without models; `ament_clang_format`/lint pass).

- [ ] **Step 3: Runtime smoke check (if a GGUF model is available locally)**

```bash
timeout 120 ros2 launch llama_bringup llama_llama.launch.py 2>&1 | head -40
```

Expected: node loads the model and prints `llama.cpp: build = 11669-ish` and no chat-API errors. Skip if no model is cached locally.

- [ ] **Step 4: Report**

Summarize for the user: files changed, new params (`cache.moe_cache_mib`, `speculative.draft.n_ubatch`, `cache.cache_ram_explicit`), the chat-session migration, the vllm-d3 fix, and any test results.

---

## Self-Review Notes

- Spec coverage: vendor tag + mtmd file (Task 1); every new upstream param between `b11465` and `v0.7.0` is covered (`moe_cache_size`, `speculative.draft.n_ubatch`, `cache_ram_explicit`; `port` default is unused by llama_ros — verified no `port` param exists); chat API break (Task 4); decision-type regression (Task 5); docs (Task 2); full build/tests (Task 6).
- Deliberately out of scope: full support for `LFM2_D1`, `LFM2_D1_OMNI`, `VLLM_DECISION2` decision models (were unsupported before; porting upstream's server-decision logic is a separate feature).
- Type consistency: `chat_fed_len`/`chat_finished` names used in tests and implementation match; `chat_session.prompt().str()` used in both handlers.
