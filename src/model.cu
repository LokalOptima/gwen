#include "gwen/model.h"
#include "gwen/ggml_quants.h"
#include <fstream>

namespace gwen {

// Helper: fill WeightRef from a GGUF tensor
static WeightRef weight_from_tensor(const GGUFTensor& t) {
    WeightRef w;
    w.host_data = t.data;
    w.type = t.type;
    w.n_elements = t.n_elements;
    w.size_bytes = t.size_bytes;
    w.shape = t.shape;
    return w;
}

static WeightRef weight_from_tensor(const GGUFFile& gguf, const std::string& name) {
    return weight_from_tensor(gguf.get_tensor(name));
}

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto model = std::make_unique<Model>();

    // Open GGUF
    model->gguf = GGUFFile::open(gguf_path);
    model->config = model->gguf->build_config();
    const auto& cfg = model->config;
    const auto& gguf = *model->gguf;

    // Global weights
    model->token_embd = weight_from_tensor(gguf, "token_embd.weight");
    model->output_norm = weight_from_tensor(gguf, "output_norm.weight");

    // Per-layer weights
    model->layers.resize(cfg.n_layers);
    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        auto& layer = model->layers[i];
        std::string prefix = "blk." + std::to_string(i) + ".";

        layer.is_full_attention = cfg.is_full_attention_layer(i);

        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            w.attn_norm      = weight_from_tensor(gguf, prefix + "attn_norm.weight");
            w.attn_q         = weight_from_tensor(gguf, prefix + "attn_q.weight");
            w.attn_k         = weight_from_tensor(gguf, prefix + "attn_k.weight");
            w.attn_v         = weight_from_tensor(gguf, prefix + "attn_v.weight");
            w.attn_q_norm    = weight_from_tensor(gguf, prefix + "attn_q_norm.weight");
            w.attn_k_norm    = weight_from_tensor(gguf, prefix + "attn_k_norm.weight");
            w.attn_output    = weight_from_tensor(gguf, prefix + "attn_output.weight");
            w.post_attn_norm = weight_from_tensor(gguf, prefix + "post_attention_norm.weight");
            w.ffn_gate       = weight_from_tensor(gguf, prefix + "ffn_gate.weight");
            w.ffn_up         = weight_from_tensor(gguf, prefix + "ffn_up.weight");
            w.ffn_down       = weight_from_tensor(gguf, prefix + "ffn_down.weight");
        } else {
            auto& w = layer.deltanet;
            w.attn_norm      = weight_from_tensor(gguf, prefix + "attn_norm.weight");
            w.attn_qkv       = weight_from_tensor(gguf, prefix + "attn_qkv.weight");
            w.attn_gate      = weight_from_tensor(gguf, prefix + "attn_gate.weight");
            w.ssm_conv1d     = weight_from_tensor(gguf, prefix + "ssm_conv1d.weight");
            w.ssm_a          = weight_from_tensor(gguf, prefix + "ssm_a");
            w.ssm_dt_bias    = weight_from_tensor(gguf, prefix + "ssm_dt.bias");
            w.ssm_alpha      = weight_from_tensor(gguf, prefix + "ssm_alpha.weight");
            w.ssm_beta       = weight_from_tensor(gguf, prefix + "ssm_beta.weight");
            w.ssm_norm       = weight_from_tensor(gguf, prefix + "ssm_norm.weight");
            w.ssm_out        = weight_from_tensor(gguf, prefix + "ssm_out.weight");
            w.post_attn_norm = weight_from_tensor(gguf, prefix + "post_attention_norm.weight");
            w.ffn_gate       = weight_from_tensor(gguf, prefix + "ffn_gate.weight");
            w.ffn_up         = weight_from_tensor(gguf, prefix + "ffn_up.weight");
            w.ffn_down       = weight_from_tensor(gguf, prefix + "ffn_down.weight");
        }
    }

    return model;
}

// ============================================================
// MTP weight loading (GWMT binary format)
// ============================================================

void Model::load_mtp(const std::string& mtp_path) {
    std::ifstream f(mtp_path, std::ios::binary);
    GWEN_CHECK(f.is_open(), ("Failed to open MTP file: " + mtp_path).c_str());

    // Read header
    char magic[4];
    f.read(magic, 4);
    GWEN_CHECK(memcmp(magic, "GWMT", 4) == 0, "Invalid MTP file magic (expected GWMT)");

    uint32_t version, n_tensors;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&n_tensors), 4);
    GWEN_CHECK(version >= 1 && version <= 4, "Unsupported MTP file version (expected 1-4)");

    printf("Loading MTP weights: %u tensors from %s\n", n_tensors, mtp_path.c_str());

    // Read tensors
    mtp_host_buffers.resize(n_tensors);
    size_t total_bytes = 0;

    for (uint32_t i = 0; i < n_tensors; i++) {
        // Read name
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);

        // Read dtype, ndims, shape
        uint32_t dtype, ndims;
        f.read(reinterpret_cast<char*>(&dtype), 4);
        f.read(reinterpret_cast<char*>(&ndims), 4);
        std::vector<uint64_t> shape(ndims);
        f.read(reinterpret_cast<char*>(shape.data()), ndims * 8);

        // Read data
        uint64_t data_size;
        f.read(reinterpret_cast<char*>(&data_size), 8);
        mtp_host_buffers[i].resize(data_size);
        f.read(reinterpret_cast<char*>(mtp_host_buffers[i].data()), data_size);

        // Compute n_elements
        size_t n_elements = 1;
        for (auto s : shape) n_elements *= s;

        // Build WeightRef
        WeightRef w;
        w.host_data = mtp_host_buffers[i].data();
        if (dtype == 0)      w.type = GGMLType::F32;
        else if (dtype == 1) w.type = GGMLType::F16;
        else if (dtype == 8) w.type = GGMLType::Q8_0;
        else GWEN_CHECK(false, "Unsupported MTP weight dtype");
        w.n_elements = n_elements;
        w.size_bytes = data_size;
        w.shape = shape;

        total_bytes += data_size;

        // Map to MTP weight fields by name
        if (name == "mtp.fc.weight") {
            mtp.fc = w;
        } else if (name == "mtp.pre_fc_norm_embedding.weight") {
            mtp.pre_fc_norm_embed = w;
        } else if (name == "mtp.pre_fc_norm_hidden.weight") {
            mtp.pre_fc_norm_hidden = w;
        } else if (name == "mtp.layers.0.self_attn.q_proj.weight") {
            mtp.layer.attn_q = w;
        } else if (name == "mtp.layers.0.self_attn.k_proj.weight") {
            mtp.layer.attn_k = w;
        } else if (name == "mtp.layers.0.self_attn.v_proj.weight") {
            mtp.layer.attn_v = w;
        } else if (name == "mtp.layers.0.self_attn.o_proj.weight") {
            mtp.layer.attn_output = w;
        } else if (name == "mtp.layers.0.self_attn.q_norm.weight") {
            mtp.layer.attn_q_norm = w;
        } else if (name == "mtp.layers.0.self_attn.k_norm.weight") {
            mtp.layer.attn_k_norm = w;
        } else if (name == "mtp.layers.0.input_layernorm.weight") {
            mtp.layer.attn_norm = w;
        } else if (name == "mtp.layers.0.post_attention_layernorm.weight") {
            mtp.layer.post_attn_norm = w;
        } else if (name == "mtp.layers.0.mlp.gate_proj.weight") {
            mtp.layer.ffn_gate = w;
        } else if (name == "mtp.layers.0.mlp.up_proj.weight") {
            mtp.layer.ffn_up = w;
        } else if (name == "mtp.layers.0.mlp.down_proj.weight") {
            mtp.layer.ffn_down = w;
        } else if (name == "mtp.norm.weight") {
            mtp.output_norm = w;
        } else if (name == "mtp.lm_head.weight") {
            // Reduced lm_head from fine-tuning — wire into reduced_lm_head
            reduced_lm_head.weights = w;
            reduced_lm_head.type = w.type;
            // For FP16: row_bytes = n_embed * 2
            if (w.type == GGMLType::F16) {
                reduced_lm_head.row_bytes = config.n_embed * 2;
            }
        } else {
            printf("  Warning: unknown MTP tensor: %s\n", name.c_str());
        }

        printf("  %-50s [", name.c_str());
        for (uint32_t d = 0; d < ndims; d++) {
            if (d > 0) printf(", ");
            printf("%lu", shape[d]);
        }
        const char* dtype_str = "???";
        if (dtype == 0) dtype_str = "F32";
        else if (dtype == 1) dtype_str = "F16";
        else if (dtype == 8) dtype_str = "Q8_0";
        printf("] %s  %.1f KB\n", dtype_str, data_size / 1024.0f);
    }

    has_mtp = true;
    printf("MTP weights loaded: %.1f MB total\n", total_bytes / 1024.0 / 1024.0);

    // v3+ footer: restricted vocab mapping [K, restricted_ids[K]]
    if (version >= 3 && f.peek() != EOF) {
        uint32_t K;
        f.read(reinterpret_cast<char*>(&K), 4);
        if (K > 0) {
            reduced_lm_head.token_ids.resize(K);
            f.read(reinterpret_cast<char*>(reduced_lm_head.token_ids.data()), K * sizeof(int32_t));
            reduced_lm_head.K = K;
            // lm_head weights are in MTP tensors (mtp.lm_head.weight), FP16
            // The reduced_lm_head.weights will be set during upload from the MTP lm_head tensor
            has_reduced_lm_head = true;
            printf("GWMT v%u: restricted vocab K=%u embedded in MTP file\n", version, K);
        }

        // v4 footer: has_idk flag
        if (version >= 4 && f.peek() != EOF) {
            uint8_t idk_flag;
            f.read(reinterpret_cast<char*>(&idk_flag), 1);
            reduced_lm_head.has_idk = (idk_flag != 0);
            if (reduced_lm_head.has_idk) {
                printf("GWMT v4: IDK token enabled (index %u maps to -1)\n", K);
            }
        }
    }
}

// ============================================================
// Reduced LM head loading (GWRL binary format)
// ============================================================

void Model::load_reduced_lm_head(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GWEN_CHECK(f.is_open(), ("Failed to open reduced LM head file: " + path).c_str());

    // Read header
    char magic[4];
    f.read(magic, 4);
    GWEN_CHECK(memcmp(magic, "GWRL", 4) == 0, "Invalid reduced LM head file magic (expected GWRL)");

    uint32_t version, K, n_embed, ggml_type, row_bytes;
    f.read(reinterpret_cast<char*>(&version), 4);
    GWEN_CHECK(version == 1, "Unsupported GWRL version");

    f.read(reinterpret_cast<char*>(&K), 4);
    f.read(reinterpret_cast<char*>(&n_embed), 4);
    f.read(reinterpret_cast<char*>(&ggml_type), 4);
    f.read(reinterpret_cast<char*>(&row_bytes), 4);

    printf("Loading reduced LM head: %u tokens, %u embed, type=%u, %u bytes/row\n",
           K, n_embed, ggml_type, row_bytes);

    // Read token ID mapping
    reduced_lm_head.token_ids.resize(K);
    f.read(reinterpret_cast<char*>(reduced_lm_head.token_ids.data()), K * sizeof(int32_t));

    // Read weight data
    size_t weight_bytes = (size_t)K * row_bytes;
    reduced_lm_head.host_buffer.resize(weight_bytes);
    f.read(reinterpret_cast<char*>(reduced_lm_head.host_buffer.data()), weight_bytes);

    // Set up WeightRef
    reduced_lm_head.weights.host_data = reduced_lm_head.host_buffer.data();
    reduced_lm_head.weights.type = static_cast<GGMLType>(ggml_type);
    reduced_lm_head.weights.n_elements = (size_t)K * n_embed;
    reduced_lm_head.weights.size_bytes = weight_bytes;
    reduced_lm_head.weights.shape = {n_embed, K};  // GGML convention
    reduced_lm_head.K = K;
    reduced_lm_head.row_bytes = row_bytes;
    reduced_lm_head.type = static_cast<GGMLType>(ggml_type);

    has_reduced_lm_head = true;
    printf("Reduced LM head loaded: %u tokens, %.1f MB (%.1fx reduction)\n",
           K, weight_bytes / 1024.0 / 1024.0,
           (float)config.n_vocab / K);
}

// Upload all weight tensors to GPU
static void upload_weight(CudaAllocator& alloc, WeightRef& w) {
    if (w.host_data && w.size_bytes > 0 && !w.on_device()) {
        w.device_data = alloc.upload(w.host_data, w.size_bytes);
    }
}

// ============================================================
// Q4_K weight reshuffling for Marlin-style mma.sync GEMV
// ============================================================
//
// Marlin formulation: C[M,N] = A[M,K] * B[K,N]
//   A = activation (M=batch), B = weights (K×N), C = output
//
// Reshuffled layout: organized by column tiles of N_TILE=128 output features.
// For each column-tile × Q4_K-block (256 K-elements):
//
// Nibbles: 16 K-chunks × 256 threads × 4 bytes = 16384 bytes
//   Thread tid loads 4 bytes at offset tid*4 per chunk.
//   Byte packing (for thread tid in warp w, lane t):
//     byte[0]: nib_k0 | (nib_k8 << 4)  for column group 0
//     byte[1]: nib_k1 | (nib_k9 << 4)  for column group 0
//     byte[2]: nib_k0 | (nib_k8 << 4)  for column group 1
//     byte[3]: nib_k1 | (nib_k9 << 4)  for column group 1
//   where k0=(t%4)*2, k1=k0+1, k8=k0+8, k9=k1+8
//   column group 0 = warp_col + t/4, column group 1 = warp_col + 8 + t/4
//
// Scales:  8 sub-blocks × 128 FP16 values = 2048 bytes (combined d*sc)
// Offsets: 8 sub-blocks × 128 FP16 values = 2048 bytes (combined dmin*mn)
//
// Total: 16384 + 2048 + 2048 = 20480 bytes per column-tile per Q4_K block

static constexpr int N_TILE_RESHUFFLE = 64;
static constexpr int MARLIN_THREADS_R = 128;
static constexpr int MARLIN_NIB  = 8192;   // 16 chunks × 128 threads × 4 bytes
static constexpr int MARLIN_SC   = 1024;   // 8 sub-blocks × 64 × sizeof(half)
static constexpr int MARLIN_OFF  = 1024;
static constexpr int MARLIN_TILE = MARLIN_NIB + MARLIN_SC + MARLIN_OFF;  // 10240

// Extract 6-bit packed Q4_K scale/min values into separate 8-bit arrays
static void unpack_q4k_scales(const uint8_t scales_packed[12], uint8_t sc[8], uint8_t mn[8]) {
    for (int sb = 0; sb < 4; sb++) {
        sc[sb] = scales_packed[sb] & 0x3F;
        mn[sb] = scales_packed[sb + 4] & 0x3F;
    }
    for (int sb = 4; sb < 8; sb++) {
        sc[sb] = (scales_packed[sb + 4] & 0xF) | ((scales_packed[sb - 4] >> 6) << 4);
        mn[sb] = (scales_packed[sb + 4] >> 4) | ((scales_packed[sb] >> 6) << 4);
    }
}

static uint8_t q4k_get_nibble(const uint8_t qs[128], int abs_elem) {
    int qs_byte = (abs_elem / 64) * 32 + (abs_elem % 32);
    bool is_high = (abs_elem % 64) >= 32;
    return is_high ? (qs[qs_byte] >> 4) : (qs[qs_byte] & 0xF);
}

static std::vector<uint8_t> reshuffle_q4k_for_mma(const void* host_data,
                                                     int out_features, int in_features) {
    const auto* blocks = static_cast<const block_q4_k*>(host_data);
    int blocks_per_row = in_features / 256;
    int n_col_tiles = (out_features + N_TILE_RESHUFFLE - 1) / N_TILE_RESHUFFLE;

    size_t total_bytes = (size_t)n_col_tiles * blocks_per_row * MARLIN_TILE;
    std::vector<uint8_t> reshuffled(total_bytes, 0);

    for (int ct = 0; ct < n_col_tiles; ct++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            uint8_t* tile = reshuffled.data() + (size_t)(ct * blocks_per_row + blk) * MARLIN_TILE;
            uint8_t* nib_base = tile;
            half* sc_base = reinterpret_cast<half*>(tile + MARLIN_NIB);
            half* off_base = reinterpret_cast<half*>(tile + MARLIN_NIB + MARLIN_SC);

            // Pre-compute combined scales/offsets for each column × sub-block
            for (int col_local = 0; col_local < N_TILE_RESHUFFLE; col_local++) {
                int col_global = ct * N_TILE_RESHUFFLE + col_local;
                if (col_global >= out_features) continue;

                const block_q4_k& src = blocks[col_global * blocks_per_row + blk];
                uint8_t sc[8], mn[8];
                unpack_q4k_scales(src.scales, sc, mn);
                float d = __half2float(src.d);
                float dmin = __half2float(src.dmin);

                for (int sb = 0; sb < 8; sb++) {
                    sc_base[sb * N_TILE_RESHUFFLE + col_local] = __float2half(d * sc[sb]);
                    off_base[sb * N_TILE_RESHUFFLE + col_local] = __float2half(dmin * mn[sb]);
                }
            }

            // Pack nibbles for each K-chunk
            for (int chunk = 0; chunk < 16; chunk++) {
                int sb = chunk / 2;
                int half_sb = chunk % 2;

                // For each thread tid (0..255)
                for (int tid = 0; tid < MARLIN_THREADS_R; tid++) {
                    int w = tid / 32;   // warp id
                    int t = tid % 32;   // lane

                    int b_k_base = (t % 4) * 2;
                    int b_n_in_group = t / 4;

                    int col_g0 = w * 16 + b_n_in_group;
                    int col_g1 = w * 16 + 8 + b_n_in_group;
                    int col_g0_global = ct * N_TILE_RESHUFFLE + col_g0;
                    int col_g1_global = ct * N_TILE_RESHUFFLE + col_g1;

                    int k0 = b_k_base;
                    int k1 = b_k_base + 1;
                    int k8 = b_k_base + 8;
                    int k9 = b_k_base + 9;

                    // Absolute K-element indices within the Q4_K block
                    int abs_k0 = sb * 32 + half_sb * 16 + k0;
                    int abs_k1 = sb * 32 + half_sb * 16 + k1;
                    int abs_k8 = sb * 32 + half_sb * 16 + k8;
                    int abs_k9 = sb * 32 + half_sb * 16 + k9;

                    // Pack nibbles for dequant_u4:
                    // dequant_u4 expects: bits[3:0]=nib_a, [7:4]=nib_c, [19:16]=nib_b, [23:20]=nib_d
                    // Produces: out0 = {nib_a, nib_b}, out1 = {nib_c, nib_d}
                    // We want: b_frag[0] = {B[k0,col], B[k1,col]}, b_frag[1] = {B[k8,col], B[k9,col]}
                    // So: out0 = {B[k0,col], B[k1,col]} → nib_a=B[k0], nib_b=B[k1]
                    //     out1 = {B[k8,col], B[k9,col]} → nib_c=B[k8], nib_d=B[k9]
                    // Pack: nib_k0 | (nib_k8 << 4) | (nib_k1 << 16) | (nib_k9 << 20)
                    // As bytes: byte[0] = nib_k0|(nib_k8<<4), byte[1] = 0,
                    //           byte[2] = nib_k1|(nib_k9<<4), byte[3] = 0
                    // But we want 4 bytes for 2 groups. Repack:
                    //   For group 0: 2 bytes packing 4 nibbles
                    //   For group 1: 2 bytes packing 4 nibbles
                    // Store as: byte[0..1] = group 0, byte[2..3] = group 1
                    // byte[0] = nib_k0 | (nib_k8 << 4)
                    // byte[1] = nib_k1 | (nib_k9 << 4)
                    // byte[2] = nib_k0' | (nib_k8' << 4)  (group 1)
                    // byte[3] = nib_k1' | (nib_k9' << 4)

                    uint8_t* dst = nib_base + chunk * (MARLIN_THREADS_R * 4) + tid * 4;

                    auto get_nib = [&](int col_global, int abs_k) -> uint8_t {
                        if (col_global >= out_features) return 0;
                        const block_q4_k& src = blocks[col_global * blocks_per_row + blk];
                        return q4k_get_nibble(src.qs, abs_k);
                    };

                    // Byte packing for uint32 load → dequant_u4:
                    // uint32 = byte[0] | (byte[1]<<8) | (byte[2]<<16) | (byte[3]<<24)
                    // dequant_u4 reads: bits[3:0]=nib_a, [7:4]=nib_c, [19:16]=nib_b, [23:20]=nib_d
                    // → out0={nib_a, nib_b}, out1={nib_c, nib_d}
                    // For group 0: want out0={B[k0,col], B[k1,col]}, out1={B[k8,col], B[k9,col]}
                    //   byte[0] = B[k0]|(B[k8]<<4),  byte[2] = B[k1]|(B[k9]<<4)
                    // For group 1: want same pattern
                    //   byte[1] = B[k0']|(B[k8']<<4), byte[3] = B[k1']|(B[k9']<<4)
                    dst[0] = get_nib(col_g0_global, abs_k0) | (get_nib(col_g0_global, abs_k8) << 4);
                    dst[1] = get_nib(col_g1_global, abs_k0) | (get_nib(col_g1_global, abs_k8) << 4);
                    dst[2] = get_nib(col_g0_global, abs_k1) | (get_nib(col_g0_global, abs_k9) << 4);
                    dst[3] = get_nib(col_g1_global, abs_k1) | (get_nib(col_g1_global, abs_k9) << 4);
                }
            }
        }
    }

    return reshuffled;
}

// Reshuffle and upload mma-format weight data for a Q4_K weight tensor
static void upload_weight_mma_q4k(CudaAllocator& alloc, WeightRef& w) {
    if (!w.host_data || w.size_bytes == 0 || w.type != GGMLType::Q4_K) return;
    if (w.shape.size() < 2) return;

    int in_features = (int)w.shape[0];
    int out_features = (int)w.shape[1];

    auto reshuffled = reshuffle_q4k_for_mma(w.host_data, out_features, in_features);
    w.device_data_mma = alloc.upload(reshuffled.data(), reshuffled.size());
}

void Model::upload_weights(CudaAllocator& allocator) {
    upload_weight(allocator, token_embd);
    upload_weight(allocator, output_norm);

    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            upload_weight(allocator, w.attn_norm);
            upload_weight(allocator, w.attn_q);
            upload_weight(allocator, w.attn_k);
            upload_weight(allocator, w.attn_v);
            upload_weight(allocator, w.attn_q_norm);
            upload_weight(allocator, w.attn_k_norm);
            upload_weight(allocator, w.attn_output);
            upload_weight(allocator, w.post_attn_norm);
            upload_weight(allocator, w.ffn_gate);
            upload_weight(allocator, w.ffn_up);
            upload_weight(allocator, w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            upload_weight(allocator, w.attn_norm);
            upload_weight(allocator, w.attn_qkv);
            upload_weight(allocator, w.attn_gate);
            upload_weight(allocator, w.ssm_conv1d);
            upload_weight(allocator, w.ssm_a);
            upload_weight(allocator, w.ssm_dt_bias);
            upload_weight(allocator, w.ssm_alpha);
            upload_weight(allocator, w.ssm_beta);
            upload_weight(allocator, w.ssm_norm);
            upload_weight(allocator, w.ssm_out);
            upload_weight(allocator, w.post_attn_norm);
            upload_weight(allocator, w.ffn_gate);
            upload_weight(allocator, w.ffn_up);
            upload_weight(allocator, w.ffn_down);
        }
    }

    // Reshuffle Q4_K weights for mma.sync GEMV
    printf("Reshuffling Q4_K weights for mma.sync...\n");
    size_t mma_bytes = 0;
    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            upload_weight_mma_q4k(allocator, w.attn_q);
            upload_weight_mma_q4k(allocator, w.attn_k);
            upload_weight_mma_q4k(allocator, w.attn_output);
            upload_weight_mma_q4k(allocator, w.ffn_gate);
            upload_weight_mma_q4k(allocator, w.ffn_up);
            upload_weight_mma_q4k(allocator, w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            upload_weight_mma_q4k(allocator, w.attn_gate);
            upload_weight_mma_q4k(allocator, w.ffn_gate);
            upload_weight_mma_q4k(allocator, w.ffn_up);
        }
    }
    // Count mma reshuffled bytes
    for (auto& layer : layers) {
        auto count_mma = [&](WeightRef& w) { if (w.device_data_mma) mma_bytes += w.size_bytes; };
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            count_mma(w.attn_q); count_mma(w.attn_k); count_mma(w.attn_output);
            count_mma(w.ffn_gate); count_mma(w.ffn_up); count_mma(w.ffn_down);
        } else {
            auto& w = layer.deltanet;
            count_mma(w.attn_gate); count_mma(w.ffn_gate); count_mma(w.ffn_up);
        }
    }
    if (mma_bytes > 0)
        printf("MMA reshuffled: %zu Q4_K tensors, original %.1f MB\n",
               mma_bytes > 0 ? 1UL : 0UL, mma_bytes / 1024.0 / 1024.0);

    // Upload reduced LM head if loaded
    if (has_reduced_lm_head) {
        upload_weight(allocator, reduced_lm_head.weights);
        // Upload token ID mapping to device
        size_t ids_bytes = reduced_lm_head.K * sizeof(int32_t);
        reduced_lm_head.d_token_ids = static_cast<int*>(allocator.upload(
            reduced_lm_head.token_ids.data(), ids_bytes));
        printf("Reduced LM head uploaded: %.1f MB weights + %.1f KB token map\n",
               reduced_lm_head.weights.size_bytes / 1024.0 / 1024.0,
               ids_bytes / 1024.0);
    }

    // Upload MTP weights if loaded
    if (has_mtp) {
        upload_weight(allocator, mtp.fc);
        upload_weight(allocator, mtp.pre_fc_norm_embed);
        upload_weight(allocator, mtp.pre_fc_norm_hidden);
        upload_weight(allocator, mtp.layer.attn_norm);
        upload_weight(allocator, mtp.layer.attn_q);
        upload_weight(allocator, mtp.layer.attn_k);
        upload_weight(allocator, mtp.layer.attn_v);
        upload_weight(allocator, mtp.layer.attn_q_norm);
        upload_weight(allocator, mtp.layer.attn_k_norm);
        upload_weight(allocator, mtp.layer.attn_output);
        upload_weight(allocator, mtp.layer.post_attn_norm);
        upload_weight(allocator, mtp.layer.ffn_gate);
        upload_weight(allocator, mtp.layer.ffn_up);
        upload_weight(allocator, mtp.layer.ffn_down);
        upload_weight(allocator, mtp.output_norm);
    }
}

void Model::print_info() const {
    printf("=== GWEN Model Info ===\n");
    printf("Model: %s\n", gguf->path().c_str());
    printf("Layers: %u (%u DeltaNet + %u FullAttn)\n",
           config.n_layers,
           config.n_layers - config.n_layers / config.full_attn_interval,
           config.n_layers / config.full_attn_interval);
    printf("Embed dim: %u\n", config.n_embed);
    printf("FFN dim: %u\n", config.n_ff);
    printf("Vocab: %u\n", config.n_vocab);
    printf("Full Attn: %u heads (%u KV), head_dim=%u\n",
           config.n_head, config.n_head_kv, config.head_dim);
    printf("DeltaNet: %u heads, state=%ux%u, inner=%u\n",
           config.ssm_n_heads, config.ssm_state_size, config.ssm_state_size,
           config.ssm_inner_size);
    printf("RoPE: theta=%.0f, dim=%u, sections=[%d,%d,%d,%d]\n",
           config.rope_theta, config.rope_dim,
           config.rope_sections[0], config.rope_sections[1],
           config.rope_sections[2], config.rope_sections[3]);
    printf("Context length: %u\n", config.context_length);
    printf("RMSNorm eps: %e\n", config.rms_norm_eps);

    // Print layer pattern
    printf("\nLayer pattern: ");
    for (uint32_t i = 0; i < config.n_layers; i++) {
        printf("%c", config.is_full_attention_layer(i) ? 'A' : 'D');
    }
    printf("\n");

    // Weight summary
    size_t total_bytes = token_embd.size_bytes + output_norm.size_bytes;
    for (auto& layer : layers) {
        if (layer.is_full_attention) {
            auto& w = layer.full_attn;
            total_bytes += w.attn_norm.size_bytes + w.attn_q.size_bytes +
                          w.attn_k.size_bytes + w.attn_v.size_bytes +
                          w.attn_q_norm.size_bytes + w.attn_k_norm.size_bytes +
                          w.attn_output.size_bytes + w.post_attn_norm.size_bytes +
                          w.ffn_gate.size_bytes + w.ffn_up.size_bytes +
                          w.ffn_down.size_bytes;
        } else {
            auto& w = layer.deltanet;
            total_bytes += w.attn_norm.size_bytes + w.attn_qkv.size_bytes +
                          w.attn_gate.size_bytes + w.ssm_conv1d.size_bytes +
                          w.ssm_a.size_bytes + w.ssm_dt_bias.size_bytes +
                          w.ssm_alpha.size_bytes + w.ssm_beta.size_bytes +
                          w.ssm_norm.size_bytes + w.ssm_out.size_bytes +
                          w.post_attn_norm.size_bytes +
                          w.ffn_gate.size_bytes + w.ffn_up.size_bytes +
                          w.ffn_down.size_bytes;
        }
    }
    printf("Total weight size: %.1f MB\n", total_bytes / 1024.0 / 1024.0);
    printf("Tensors: %zu\n", gguf->n_tensors());
}

} // namespace gwen
