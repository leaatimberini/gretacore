#include "gcore/inference/block_scheduler.hpp"
#include "gcore/inference/generator.hpp"
#include "gcore/inference/model_config.hpp"
#include "gcore/inference/tokenizer.hpp"
#include "gcore/inference/weight_loader.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

void print_usage() {
  std::cout
      << "Usage: greta_infer [options]\n"
      << "Options:\n"
      << "  --model <path>      Path to model weights (GGUF format)\n"
      << "  --prompt <text>     Input prompt\n"
      << "  --prompt-file <path> Read prompt from file\n"
      << "  --batch-size <n>    Batch size for inference (default: 1)\n"
      << "  --max-tokens <n>    Maximum tokens to generate (default: 32)\n"
      << "  --temperature <t>   Sampling temperature (default: 1.0)\n"
      << "  --top-k <k>         Top-K sampling (default: 50)\n"
      << "  --greedy            Use greedy decoding\n"
      << "  --seed <n>          Random seed (also reads GRETA_SEED env)\n"
      << "  --kv-aligned <0|1>  KV alignment mode (also reads GRETA_KV_ALIGNED "
         "env)\n"
      << "  --mode <prefill|decode> Execution mode for tracing\n"
      << "  --dump-logits <dir> Dump logits to directory (JSONL.gz + "
         "metadata.json)\n"
      << "  --dump-logits-span <n> Number of tokens to dump (default: 1)\n"
      << "  --demo-tokenizer    Force fallback ASCII tokenizer\n"
      << "  --help              Show this help\n";
}

int main(int argc, char *argv[]) {
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║           GRETA CORE - LLM Inference Engine              ║\n";
  std::cout << "║                    Phase 3 Preview                       ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════╝\n\n";

  // Default parameters
  std::string model_path;
  std::string prompt = "Hello, I am a language model";
  int batch_size = 1;
  gcore::inference::SamplingParams params;
  params.max_tokens = 32;
  params.temperature = 1.0f;
  params.top_k = 50;
  params.greedy = false;

  bool force_demo_tokenizer = false;
  bool enable_alignment = false;

  // B3.68: New flags for equivalence guardrail
  int kv_aligned = -1;   // -1 = not set, read from env
  std::string exec_mode; // prefill or decode
  std::string dump_logits_dir;
  int dump_logits_span = 1; // B3.69: number of tokens to dump
  int seed = -1;            // -1 = not set, read from env

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      prompt = argv[++i];
    } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
      std::ifstream f(argv[++i]);
      if (f.is_open()) {
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = ss.str();
      }
    } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
      batch_size = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      params.max_tokens = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      params.temperature = std::atof(argv[++i]);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      params.top_k = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--greedy") == 0) {
      params.greedy = true;
    } else if (strcmp(argv[i], "--demo-tokenizer") == 0) {
      force_demo_tokenizer = true;
    } else if (strcmp(argv[i], "--alignment") == 0) {
      enable_alignment = true;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      seed = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--kv-aligned") == 0 && i + 1 < argc) {
      kv_aligned = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      exec_mode = argv[++i];
    } else if (strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
      dump_logits_dir = argv[++i];
    } else if (strcmp(argv[i], "--dump-logits-span") == 0 && i + 1 < argc) {
      dump_logits_span = std::atoi(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0) {
      print_usage();
      return 0;
    }
  }

  // Read from environment if not set via args
  if (seed < 0) {
    const char *seed_env = std::getenv("GRETA_SEED");
    if (seed_env)
      seed = std::atoi(seed_env);
  }
  if (kv_aligned < 0) {
    const char *kv_env = std::getenv("GRETA_KV_ALIGNED");
    if (kv_env)
      kv_aligned = std::atoi(kv_env);
  }

  std::cout << "Configuration:\n";
  std::cout << "  Model: " << (model_path.empty() ? "(demo mode)" : model_path)
            << "\n";
  std::cout << "  Prompt: \"" << prompt << "\"\n";
  std::cout << "  Max tokens: " << params.max_tokens << "\n";
  std::cout << "  Temperature: " << params.temperature << "\n";
  std::cout << "  Top-K: " << params.top_k << "\n";
  std::cout << "  Greedy: " << (params.greedy ? "yes" : "no") << "\n";
  if (seed >= 0) {
    std::cout << "  Seed: " << seed << "\n";
  }
  if (kv_aligned >= 0) {
    std::cout << "  KV Aligned: " << kv_aligned << "\n";
  }
  if (!exec_mode.empty()) {
    std::cout << "  Mode: " << exec_mode << "\n";
  }
  if (!dump_logits_dir.empty()) {
    std::cout << "  Dump Logits: " << dump_logits_dir << "\n";
    std::cout << "  Dump Span: " << dump_logits_span << "\n";
  }

  const char *verbose_info = std::getenv("GRETA_VERBOSE_INFO");
  if (verbose_info && std::string(verbose_info) == "1") {
    int hip_ver = 0;
    (void)hipRuntimeGetVersion(&hip_ver);
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);

    const char *graph_env = std::getenv("GRETA_HIP_GRAPH");
    const char *prof_env = std::getenv("GRETA_PROFILE_BLOCKS");

    std::cout << "\nSystem Info (VERBOSE):\n";
    std::cout << "  GPU: " << prop.name << "\n";
    std::cout << "  HIP Runtime Version: " << hip_ver << "\n";
    std::cout << "  GRETA_HIP_GRAPH: " << (graph_env ? graph_env : "0") << "\n";
    std::cout << "  GRETA_PROFILE_BLOCKS: " << (prof_env ? prof_env : "0")
              << "\n";
  }
  std::cout << "\n";

  std::string err;
  if (gcore::rt::GretaContext::instance().initialize() !=
      gcore::rt::GretaResult::SUCCESS) {
    std::cerr << "Failed to initialize GRETA context\n";
    return 1;
  }

  // Initialize model config
  auto config = gcore::inference::ModelConfig::llama2_7b();
  std::unique_ptr<gcore::inference::WeightLoader> loader;
  if (!model_path.empty()) {
    loader = gcore::inference::create_weight_loader(model_path, &err);
    if (!loader) {
      std::cerr << "Failed to open model: " << err << "\n";
      return 1;
    }
    config = loader->get_config();
    if (config.num_heads_kv == 0)
      config.num_heads_kv = config.num_heads;
    if (config.num_heads > 0)
      config.head_dim = config.dim / config.num_heads;
  }

  // =========================================================================
  // GUARD RAIL: Validar compatibilidad del modelo con kernels GRETA
  // =========================================================================
  // GRETA binaries están especializados para arquitecturas específicas.
  // Si el modelo no coincide, abortamos con error explícito en lugar de
  // crash en kernels (illegal memory access).
  //
  // Para debugging B3.62, usar GRETA_DISABLE_GUARD_RAIL=1 para saltar
  // validación
  const char *disable_guard = getenv("GRETA_DISABLE_GUARD_RAIL");
  bool guard_disabled = (disable_guard && strcmp(disable_guard, "1") == 0);

  if (guard_disabled) {
    std::cout << "\n[GUARD_RAIL] WARNING: Guard rail DISABLED via "
                 "GRETA_DISABLE_GUARD_RAIL\n";
    std::cout
        << "[GUARD_RAIL] Continuing with potentially incompatible model...\n";
  }

  if (!model_path.empty() && !guard_disabled) {
    std::cout << "\n[GUARD_RAIL] Validating model compatibility...\n";

    // Valores esperados para GRETA v1 (basado en Llama-2-7B)
    const uint32_t EXPECTED_DIM = 4096;
    const uint32_t EXPECTED_NUM_HEADS = 32;
    const uint32_t EXPECTED_NUM_LAYERS = 32;
    const uint32_t EXPECTED_HEAD_DIM = 128;
    const uint32_t EXPECTED_HIDDEN_DIM = 11008;

    bool mismatch = false;

    if (config.dim != EXPECTED_DIM) {
      std::cerr << "[GUARD_RAIL_ERROR] dim mismatch!\n";
      std::cerr << "  Expected: " << EXPECTED_DIM << "\n";
      std::cerr << "  Got:      " << config.dim << "\n";
      mismatch = true;
    }

    if (config.num_heads != EXPECTED_NUM_HEADS) {
      std::cerr << "[GUARD_RAIL_ERROR] num_heads mismatch!\n";
      std::cerr << "  Expected: " << EXPECTED_NUM_HEADS << "\n";
      std::cerr << "  Got:      " << config.num_heads << "\n";
      mismatch = true;
    }

    if (config.num_layers != EXPECTED_NUM_LAYERS) {
      std::cerr << "[GUARD_RAIL_ERROR] num_layers mismatch!\n";
      std::cerr << "  Expected: " << EXPECTED_NUM_LAYERS << "\n";
      std::cerr << "  Got:      " << config.num_layers << "\n";
      mismatch = true;
    }

    if (config.hidden_dim != EXPECTED_HIDDEN_DIM) {
      std::cerr << "[GUARD_RAIL_ERROR] hidden_dim mismatch!\n";
      std::cerr << "  Expected: " << EXPECTED_HIDDEN_DIM << "\n";
      std::cerr << "  Got:      " << config.hidden_dim << "\n";
      mismatch = true;
    }

    // Resolver path real (seguir symlinks)
    char real_path[PATH_MAX];
    if (realpath(model_path.c_str(), real_path) != nullptr) {
      std::cout << "[GUARD_RAIL] Model path (realpath): " << real_path << "\n";
    } else {
      std::cerr << "[GUARD_RAIL_WARNING] Could not resolve realpath: "
                << strerror(errno) << "\n";
    }

    if (mismatch) {
      std::cerr
          << "\n[GUARD_RAIL_FATAL] Model incompatible with GRETA kernels!\n";
      std::cerr
          << "GRETA binary was compiled with hardcoded tensor dimensions\n";
      std::cerr << "for Llama-2-7B (dim=4096, heads=32, layers=32).\n";
      std::cerr
          << "Running a different architecture will cause illegal memory\n";
      std::cerr << "access in kernels (RMSNorm, attention, etc.).\n\n";
      std::cerr << "Solutions:\n";
      std::cerr << "  1. Use greta-v1.gguf (Llama-2-7B compatible)\n";
      std::cerr << "  2. Recompile GRETA with dynamic shape support\n";
      std::cerr << "  3. Use a model matching GRETA's expected dimensions\n\n";
      return 1;
    }

    std::cout << "[GUARD_RAIL] Model passed compatibility check.\n";
  }

  std::cout << "Model config: layers=" << config.num_layers
            << ", dim=" << config.dim << ", heads=" << config.num_heads
            << ", hidden=" << config.hidden_dim
            << ", vocab=" << config.vocab_size
            << ", params=" << (config.param_count() / 1e9) << "B\n";

  // Initialize scheduler
  std::cout << "[GRETA_MAIN] Initializing scheduler..." << std::endl;
  gcore::inference::BlockScheduler scheduler;
  if (!scheduler.init(config, &err)) {
    std::cerr << "Scheduler init failed: " << err << "\n";
    return 1;
  }
  std::cout << "[GRETA_MAIN] Initialized scheduler for "
            << scheduler.num_layers() << " layers\n";

  // Allocate buffers
  std::cout << "Allocating buffers...\n";
  if (!scheduler.allocate_weights(&err)) {
    std::cerr << "Weight allocation failed: " << err << "\n";
    return 1;
  }
  size_t max_seq_len = 2048; // Default context for bench
  if (const char *max_seq_env = std::getenv("GRETA_MAX_SEQ_LEN")) {
    char *end = nullptr;
    long v = std::strtol(max_seq_env, &end, 10);
    if (end != max_seq_env && v > 0) {
      max_seq_len = static_cast<size_t>(v);
      std::cout << "[GRETA_MAIN] GRETA_MAX_SEQ_LEN=" << max_seq_len
                << std::endl;
    }
  }
  if (!scheduler.allocate_activations(batch_size, max_seq_len,
                                      &err)) { // Configurable max_seq_len
    std::cerr << "Activation allocation failed: " << err << "\n";
    return 1;
  }
  std::cout << "Buffers allocated\n";

  // Load weights from model file if provided
  double model_load_s = 0;
  if (!model_path.empty()) {
    std::cout << "\nLoading weights from: " << model_path << "\n";
    auto start_load = std::chrono::high_resolution_clock::now();
    if (!loader) {
      std::cerr << "Failed to open model: " << err << "\n";
      return 1;
    }
    if (!scheduler.load_weights(*loader, &err)) {
      std::cerr << "Weight loading failed: " << err << "\n";
      return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    model_load_s = std::chrono::duration<float>(end_load - start_load).count();
    std::cout << "Weights loaded (vocab size: " << config.vocab_size << ")\n";
  }

  // Initialize tokenizer
  gcore::inference::Tokenizer tokenizer;
  if (force_demo_tokenizer) {
    std::cout << "[TOKENIZER] Forced ASCII fallback (--demo-tokenizer)\n";
    tokenizer.use_ascii_fallback();
  } else if (!config.vocabulary.empty()) {
    tokenizer.set_vocabulary(config.vocabulary);
    std::cout << "[TOKENIZER] Loaded GGUF vocab: " << config.vocabulary.size()
              << "\n";
  } else {
    // Try to find .model file near the GGUF model
    std::string tokenizer_path = "tokenizer.model";
    if (!model_path.empty()) {
      size_t last_slash = model_path.find_last_of("/\\");
      if (last_slash != std::string::npos) {
        tokenizer_path =
            model_path.substr(0, last_slash + 1) + "tokenizer.model";
      }
    }
    if (!tokenizer.load(tokenizer_path, &err)) {
      std::cout << "[TOKENIZER] Info: Loading failed (" << err
                << "). Falling back to ASCII.\n";
    }
  }
  std::cout << "[TOKENIZER] Mode: "
            << (tokenizer.is_using_sentencepiece()
                    ? "SentencePiece"
                    : (tokenizer.vocab_size() > 0 ? "GGUF vocab"
                                                  : "ASCII Fallback"))
            << "\n";

  // Initialize generator
  gcore::inference::Generator generator;
  if (!generator.init(config, &scheduler, &err)) {
    std::cerr << "Generator init failed: " << err << "\n";
    return 1;
  }
  std::cout << "Generator initialized\n\n";

  // Generate
  std::cout << "═══════════════════════════════════════════════════════════\n";
  std::cout << "Generating...\n\n";

  struct CapturedLogit {
    uint32_t step;
    int32_t token_id;
    std::vector<float> logits;
  };
  struct CapturedToken {
    uint32_t token_idx;
    int32_t token_id;
  };
  std::vector<CapturedLogit> captured_logits;
  std::vector<CapturedToken> captured_tokens;

  gcore::inference::AlignmentCallback align_cb = nullptr;

  // B3.69: If dumping logits, capture them via alignment callback
  // B3.82: Only enable the callback if dump_logits_span > 0 to avoid
  // performance bottleneck in generator
  if (!dump_logits_dir.empty() && dump_logits_span > 0) {
    align_cb = [&captured_logits,
                dump_logits_span](const gcore::inference::AlignmentStep &step) {
      if (captured_logits.size() < (size_t)dump_logits_span) {
        CapturedLogit entry;
        entry.step = step.step;
        entry.token_id = step.token_id;
        entry.logits = step.full_logits;
        captured_logits.push_back(std::move(entry));
      }
    };
  } else if (enable_alignment) {
    align_cb = [](const gcore::inference::AlignmentStep &step) {
      std::cout << "[ALIGNMENT_STEP] {\"step\":" << step.step
                << ",\"token_id\":" << step.token_id
                << ",\"logit\":" << step.logit
                << ",\"stats\":{\"min\":" << step.logit_min
                << ",\"max\":" << step.logit_max
                << ",\"avg\":" << step.logit_mean
                << ",\"nan\":" << step.nan_count
                << ",\"inf\":" << step.inf_count << "},\"topk_ids\":[";
      for (size_t i = 0; i < step.topk_ids.size(); ++i) {
        std::cout << step.topk_ids[i] << (i == 9 ? "" : ",");
      }
      std::cout << "]}" << std::endl;
    };
  }

  gcore::inference::GenerationStats stats;
  std::string output = generator.generate(
      prompt, params, &stats,
      [&captured_tokens, &stats, &dump_logits_dir](int32_t id,
                                                   const std::string &text) {
        // Collect token stream ONLY if dump_logits_dir is set (for
        // verification)
        if (!dump_logits_dir.empty()) {
          CapturedToken t;
          t.token_idx =
              (uint32_t)(stats.prompt_tokens + captured_tokens.size());
          t.token_id = id;
          captured_tokens.push_back(t);
        }
      },
      align_cb);

  // Avoid printing massive prompts/outputs to stdout during long context
  // benchmarks
  if (prompt.length() < 1000) {
    std::cout << "Prompt: " << prompt << "\n";
  } else {
    std::cout << "Prompt: <" << prompt.length() << " chars>\n";
  }

  if (output.length() < 1000) {
    std::cout << "Generated: " << output << "\n\n";
  } else {
    std::cout << "Generated: <" << output.length() << " chars>\n\n";
  }
  std::cout << "═══════════════════════════════════════════════════════════\n";

  // Print stats
  std::cout << "Statistics:\n";
  std::cout << "  Prompt tokens: " << stats.prompt_tokens << "\n";
  std::cout << "  Generated tokens: " << stats.generated_tokens << "\n";
  std::cout << "  Total time: " << stats.total_time_ms << " ms\n";
  std::cout << "  Time to first token: " << stats.time_to_first_token_ms
            << " ms\n";
  std::cout << "  Tokens/second: " << stats.tokens_per_second << "\n";

  // B3.85: Machine-readable timings for RCA
  std::cout << "[PERF_TIMING] {"
            << "\"model_load_s\":" << model_load_s << ","
            << "\"tokenize_s\":" << (stats.tokenize_time_ms / 1000.0) << ","
            << "\"prefill_s\":" << (stats.prefill_time_ms / 1000.0) << ","
            << "\"decode_s\":" << (stats.decode_time_ms / 1000.0) << ","
            << "\"attn_impl\":\"flash_v2_naive\"}\n";

  std::cout << "\nSTATUS=OK\n";

  // B3.68: Write metadata.json if dump_logits_dir is set
  if (!dump_logits_dir.empty()) {
    // Create directory if needed
    mkdir(dump_logits_dir.c_str(), 0755);

    // Get timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    gmtime_r(&time_t_now, &tm_now);
    std::ostringstream ts_stream;
    ts_stream << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%SZ");

    // Write metadata.json
    std::string metadata_path = dump_logits_dir + "/metadata.json";
    std::ofstream meta_out(metadata_path);
    if (meta_out.is_open()) {
      meta_out << "{\n";
      meta_out << "  \"dtype\": \"bf16\",\n";
      meta_out << "  \"prompt_len\": " << stats.prompt_tokens << ",\n";
      meta_out << "  \"gen_len\": " << stats.generated_tokens << ",\n";
      meta_out << "  \"seed\": " << (seed >= 0 ? seed : 0) << ",\n";
      meta_out << "  \"kv_aligned\": " << (kv_aligned >= 0 ? kv_aligned : 0)
               << ",\n";
      meta_out << "  \"mode\": \"" << (exec_mode.empty() ? "decode" : exec_mode)
               << "\",\n";
      // token_span: which tokens are dumped (for B3.67/B3.69 comparison)
      // B3.69: Use dump_logits_span for count instead of hardcoded 1
      meta_out << "  \"token_span\": {\"start\": " << stats.prompt_tokens
               << ", \"count\": " << dump_logits_span << "},\n";
      meta_out << "  \"timestamp\": \"" << ts_stream.str() << "\",\n";
      meta_out << "  \"repo_branch\": \"main\"\n";
      meta_out << "}\n";
      meta_out.close();
      std::cout << "[B3.69] Wrote metadata to: " << metadata_path << "\n";
    } else {
      std::cerr << "[B3.69] ERROR: Could not write " << metadata_path << "\n";
    }

    // B3.69: Write real logits.jsonl.gz using captured_logits (zlib)
    std::string logits_path = dump_logits_dir + "/logits.jsonl.gz";
    gzFile gz = gzopen(logits_path.c_str(), "wb");
    if (gz) {
      size_t prompt_len = stats.prompt_tokens;
      for (size_t i = 0; i < captured_logits.size(); ++i) {
        const auto &entry = captured_logits[i];
        // Format: {"token_idx": <absolute>, "token_id": <int>, "logits":
        // [<floats>]}
        std::ostringstream line;
        line << "{\"token_idx\":" << (prompt_len + i)
             << ",\"token_id\":" << entry.token_id << ",\"logits\":[";
        for (size_t j = 0; j < entry.logits.size(); ++j) {
          if (j > 0)
            line << ",";
          line << std::setprecision(8) << entry.logits[j];
        }
        line << "]}\n";
        std::string s = line.str();
        gzwrite(gz, s.c_str(), s.size());
      }
      gzclose(gz);
      std::cout << "[B3.69] Wrote logits (" << captured_logits.size()
                << " entries) to: " << logits_path << "\n";
    } else if (dump_logits_span > 0) {
      std::cerr << "[B3.69] ERROR: Could not write " << logits_path << "\n";
    }

    // B3.82: Write tokens.jsonl.gz (full sequence token IDs)
    std::string tokens_path = dump_logits_dir + "/tokens.jsonl.gz";
    gzFile gzt = gzopen(tokens_path.c_str(), "wb");
    if (gzt) {
      for (const auto &t : captured_tokens) {
        std::ostringstream line;
        line << "{\"token_idx\":" << t.token_idx
             << ",\"token_id\":" << t.token_id << "}\n";
        std::string s = line.str();
        gzwrite(gzt, s.c_str(), (unsigned int)s.size());
      }
      gzclose(gzt);
      std::cout << "[B3.82] Wrote tokens (" << captured_tokens.size()
                << " entries) to: " << tokens_path << "\n";
    } else {
      std::cerr << "[B3.82] ERROR: Could not write " << tokens_path << "\n";
    }
  }

  return 0;
}
