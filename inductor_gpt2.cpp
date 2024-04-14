#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/BinaryOps.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/types.h>
#include <ATen/ops/bernoulli_native.h>

#define reinterpret_tensor torch::inductor::_reinterpret_tensor
#define alloc_from_pool torch::inductor::_alloc_from_pool
#include <c10/util/generic_math.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace torch::aot_inductor;

class RAIIPyObject {
public:
    RAIIPyObject(PyObject* obj) : obj_(obj) {}
    ~RAIIPyObject() {
        Py_XDECREF(obj_);
    }
    operator PyObject*() {
        return obj_;
    }
    PyObject* get() {
        return obj_;
    }
private:
    PyObject* obj_;
};

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
#include <filesystem>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/EmptyTensor.h>

#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    cuGetErrorString(code, &msg);                  \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

namespace {

struct Grid {
    Grid(uint32_t x, uint32_t y, uint32_t z)
      : grid_x(x), grid_y(y), grid_z(z) {}
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;

    bool is_non_zero() {
        return grid_x > 0 && grid_y > 0 && grid_z > 0;
    }
};

}  // anonymous namespace

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}

static CUfunction triton_per_fused_add_embedding_native_layer_norm_2 = nullptr;
static CUfunction triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3 = nullptr;
static CUfunction triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = nullptr;
static CUfunction triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 = nullptr;
static CUfunction triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 = nullptr;
static CUfunction triton_poi_fused_8 = nullptr;
static CUfunction triton_poi_fused_add_mul_pow_tanh_4 = nullptr;
static CUfunction triton_poi_fused_arange_0 = nullptr;
static CUfunction triton_poi_fused_embedding_1 = nullptr;
static CUfunction triton_red_fused__log_softmax_9 = nullptr;
static CUfunction triton_red_fused_nll_loss_forward_10 = nullptr;


void inductor_entry_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles  // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed)
) {

    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, 151);
    auto primals_1 = std::move(inputs[0]);
    auto primals_2 = std::move(inputs[1]);
    auto primals_3 = std::move(inputs[2]);
    auto primals_4 = std::move(inputs[3]);
    auto primals_5 = std::move(inputs[4]);
    auto primals_6 = std::move(inputs[5]);
    auto primals_7 = std::move(inputs[6]);
    auto primals_8 = std::move(inputs[7]);
    auto primals_9 = std::move(inputs[8]);
    auto primals_10 = std::move(inputs[9]);
    auto primals_11 = std::move(inputs[10]);
    auto primals_12 = std::move(inputs[11]);
    auto primals_13 = std::move(inputs[12]);
    auto primals_14 = std::move(inputs[13]);
    auto primals_15 = std::move(inputs[14]);
    auto primals_16 = std::move(inputs[15]);
    auto primals_17 = std::move(inputs[16]);
    auto primals_18 = std::move(inputs[17]);
    auto primals_19 = std::move(inputs[18]);
    auto primals_20 = std::move(inputs[19]);
    auto primals_21 = std::move(inputs[20]);
    auto primals_22 = std::move(inputs[21]);
    auto primals_23 = std::move(inputs[22]);
    auto primals_24 = std::move(inputs[23]);
    auto primals_25 = std::move(inputs[24]);
    auto primals_26 = std::move(inputs[25]);
    auto primals_27 = std::move(inputs[26]);
    auto primals_28 = std::move(inputs[27]);
    auto primals_29 = std::move(inputs[28]);
    auto primals_30 = std::move(inputs[29]);
    auto primals_31 = std::move(inputs[30]);
    auto primals_32 = std::move(inputs[31]);
    auto primals_33 = std::move(inputs[32]);
    auto primals_34 = std::move(inputs[33]);
    auto primals_35 = std::move(inputs[34]);
    auto primals_36 = std::move(inputs[35]);
    auto primals_37 = std::move(inputs[36]);
    auto primals_38 = std::move(inputs[37]);
    auto primals_39 = std::move(inputs[38]);
    auto primals_40 = std::move(inputs[39]);
    auto primals_41 = std::move(inputs[40]);
    auto primals_42 = std::move(inputs[41]);
    auto primals_43 = std::move(inputs[42]);
    auto primals_44 = std::move(inputs[43]);
    auto primals_45 = std::move(inputs[44]);
    auto primals_46 = std::move(inputs[45]);
    auto primals_47 = std::move(inputs[46]);
    auto primals_48 = std::move(inputs[47]);
    auto primals_49 = std::move(inputs[48]);
    auto primals_50 = std::move(inputs[49]);
    auto primals_51 = std::move(inputs[50]);
    auto primals_52 = std::move(inputs[51]);
    auto primals_53 = std::move(inputs[52]);
    auto primals_54 = std::move(inputs[53]);
    auto primals_55 = std::move(inputs[54]);
    auto primals_56 = std::move(inputs[55]);
    auto primals_57 = std::move(inputs[56]);
    auto primals_58 = std::move(inputs[57]);
    auto primals_59 = std::move(inputs[58]);
    auto primals_60 = std::move(inputs[59]);
    auto primals_61 = std::move(inputs[60]);
    auto primals_62 = std::move(inputs[61]);
    auto primals_63 = std::move(inputs[62]);
    auto primals_64 = std::move(inputs[63]);
    auto primals_65 = std::move(inputs[64]);
    auto primals_66 = std::move(inputs[65]);
    auto primals_67 = std::move(inputs[66]);
    auto primals_68 = std::move(inputs[67]);
    auto primals_69 = std::move(inputs[68]);
    auto primals_70 = std::move(inputs[69]);
    auto primals_71 = std::move(inputs[70]);
    auto primals_72 = std::move(inputs[71]);
    auto primals_73 = std::move(inputs[72]);
    auto primals_74 = std::move(inputs[73]);
    auto primals_75 = std::move(inputs[74]);
    auto primals_76 = std::move(inputs[75]);
    auto primals_77 = std::move(inputs[76]);
    auto primals_78 = std::move(inputs[77]);
    auto primals_79 = std::move(inputs[78]);
    auto primals_80 = std::move(inputs[79]);
    auto primals_81 = std::move(inputs[80]);
    auto primals_82 = std::move(inputs[81]);
    auto primals_83 = std::move(inputs[82]);
    auto primals_84 = std::move(inputs[83]);
    auto primals_85 = std::move(inputs[84]);
    auto primals_86 = std::move(inputs[85]);
    auto primals_87 = std::move(inputs[86]);
    auto primals_88 = std::move(inputs[87]);
    auto primals_89 = std::move(inputs[88]);
    auto primals_90 = std::move(inputs[89]);
    auto primals_91 = std::move(inputs[90]);
    auto primals_92 = std::move(inputs[91]);
    auto primals_93 = std::move(inputs[92]);
    auto primals_94 = std::move(inputs[93]);
    auto primals_95 = std::move(inputs[94]);
    auto primals_96 = std::move(inputs[95]);
    auto primals_97 = std::move(inputs[96]);
    auto primals_98 = std::move(inputs[97]);
    auto primals_99 = std::move(inputs[98]);
    auto primals_100 = std::move(inputs[99]);
    auto primals_101 = std::move(inputs[100]);
    auto primals_102 = std::move(inputs[101]);
    auto primals_103 = std::move(inputs[102]);
    auto primals_104 = std::move(inputs[103]);
    auto primals_105 = std::move(inputs[104]);
    auto primals_106 = std::move(inputs[105]);
    auto primals_107 = std::move(inputs[106]);
    auto primals_108 = std::move(inputs[107]);
    auto primals_109 = std::move(inputs[108]);
    auto primals_110 = std::move(inputs[109]);
    auto primals_111 = std::move(inputs[110]);
    auto primals_112 = std::move(inputs[111]);
    auto primals_113 = std::move(inputs[112]);
    auto primals_114 = std::move(inputs[113]);
    auto primals_115 = std::move(inputs[114]);
    auto primals_116 = std::move(inputs[115]);
    auto primals_117 = std::move(inputs[116]);
    auto primals_118 = std::move(inputs[117]);
    auto primals_119 = std::move(inputs[118]);
    auto primals_120 = std::move(inputs[119]);
    auto primals_121 = std::move(inputs[120]);
    auto primals_122 = std::move(inputs[121]);
    auto primals_123 = std::move(inputs[122]);
    auto primals_124 = std::move(inputs[123]);
    auto primals_125 = std::move(inputs[124]);
    auto primals_126 = std::move(inputs[125]);
    auto primals_127 = std::move(inputs[126]);
    auto primals_128 = std::move(inputs[127]);
    auto primals_129 = std::move(inputs[128]);
    auto primals_130 = std::move(inputs[129]);
    auto primals_131 = std::move(inputs[130]);
    auto primals_132 = std::move(inputs[131]);
    auto primals_133 = std::move(inputs[132]);
    auto primals_134 = std::move(inputs[133]);
    auto primals_135 = std::move(inputs[134]);
    auto primals_136 = std::move(inputs[135]);
    auto primals_137 = std::move(inputs[136]);
    auto primals_138 = std::move(inputs[137]);
    auto primals_139 = std::move(inputs[138]);
    auto primals_140 = std::move(inputs[139]);
    auto primals_141 = std::move(inputs[140]);
    auto primals_142 = std::move(inputs[141]);
    auto primals_143 = std::move(inputs[142]);
    auto primals_144 = std::move(inputs[143]);
    auto primals_145 = std::move(inputs[144]);
    auto primals_146 = std::move(inputs[145]);
    auto primals_147 = std::move(inputs[146]);
    auto primals_148 = std::move(inputs[147]);
    auto primals_149 = std::move(inputs[148]);
    auto primals_150 = std::move(inputs[149]);
    auto primals_151 = std::move(inputs[150]);

    at::cuda::CUDAGuard device_guard(0);
    at::Tensor buf0 = at::detail::empty_strided_cuda({1024L, }, {1L, }, at::kLong, c10::DeviceType::CUDA);
    // Source Nodes: [pos], Original ATen: [aten.arange]
    if (triton_poi_fused_arange_0 == nullptr) {
        triton_poi_fused_arange_0 = loadKernel("/tmp/torchinductor_chilli/6d/c6dyug3iwtdqjp5n4xsiq6j5syow7xkrbygaysezuwu2ywyyamoh.cubin", "triton__0d1d", 0);
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(buf0.data_ptr());
    auto var_1 = 1024;
    void* kernel_args_var_0[] = {&var_0, &var_1};
    cudaStream_t stream0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(0, (void**)&stream0));
    Grid triton_poi_fused_arange_0_grid_0 = Grid(4L, 1L, 1L);
    launchKernel(triton_poi_fused_arange_0, triton_poi_fused_arange_0_grid_0.grid_x, triton_poi_fused_arange_0_grid_0.grid_y, triton_poi_fused_arange_0_grid_0.grid_z, 4, 0, kernel_args_var_0, stream0);
    at::Tensor buf2 = at::detail::empty_strided_cuda({1024L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [pos_emb], Original ATen: [aten.embedding]
    if (triton_poi_fused_embedding_1 == nullptr) {
        triton_poi_fused_embedding_1 = loadKernel("/tmp/torchinductor_chilli/l7/cl75222q64zrk54siviv7pcqyz5qnpws3t7j5vb22src4ciykyjr.cubin", "triton__0d1d2d", 0);
    }
    CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(primals_2.data_ptr());
    CUdeviceptr var_3 = reinterpret_cast<CUdeviceptr>(buf2.data_ptr());
    auto var_4 = 786432;
    void* kernel_args_var_1[] = {&var_2, &var_3, &var_4};
    Grid triton_poi_fused_embedding_1_grid_1 = Grid(3072L, 1L, 1L);
    launchKernel(triton_poi_fused_embedding_1, triton_poi_fused_embedding_1_grid_1.grid_x, triton_poi_fused_embedding_1_grid_1.grid_y, triton_poi_fused_embedding_1_grid_1.grid_z, 4, 0, kernel_args_var_1, stream0);
    primals_2.reset();
    at::Tensor buf1 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf3 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf4 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 4096L}, at::kFloat, c10::DeviceType::CUDA);
    decltype(auto) buf6 = reinterpret_tensor(buf4, {4L, 1024L, 1L}, {1024L, 1L, 1L}, 0L); buf4.reset();  // reuse
    at::Tensor buf7 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_0_ln_1, tok_emb, x], Original ATen: [aten.add, aten.embedding, aten.native_layer_norm]
    if (triton_per_fused_add_embedding_native_layer_norm_2 == nullptr) {
        triton_per_fused_add_embedding_native_layer_norm_2 = loadKernel("/tmp/torchinductor_chilli/rm/crmio3iwgepqm3hw67lpaqpnwfz52453hibrpcofhxgtrrzjdzsc.cubin", "triton__0d1d2d3d4d5d6d7d8d9d10d", 32);
    }
    CUdeviceptr var_5 = reinterpret_cast<CUdeviceptr>(buf6.data_ptr());
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(primals_150.data_ptr());
    CUdeviceptr var_7 = reinterpret_cast<CUdeviceptr>(primals_1.data_ptr());
    CUdeviceptr var_8 = reinterpret_cast<CUdeviceptr>(buf2.data_ptr());
    CUdeviceptr var_9 = reinterpret_cast<CUdeviceptr>(primals_3.data_ptr());
    CUdeviceptr var_10 = reinterpret_cast<CUdeviceptr>(primals_4.data_ptr());
    CUdeviceptr var_11 = reinterpret_cast<CUdeviceptr>(buf1.data_ptr());
    CUdeviceptr var_12 = reinterpret_cast<CUdeviceptr>(buf3.data_ptr());
    CUdeviceptr var_13 = reinterpret_cast<CUdeviceptr>(buf7.data_ptr());
    auto var_14 = 4096;
    auto var_15 = 768;
    void* kernel_args_var_2[] = {&var_5, &var_6, &var_7, &var_8, &var_9, &var_10, &var_11, &var_12, &var_13, &var_14, &var_15};
    Grid triton_per_fused_add_embedding_native_layer_norm_2_grid_2 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_embedding_native_layer_norm_2, triton_per_fused_add_embedding_native_layer_norm_2_grid_2.grid_x, triton_per_fused_add_embedding_native_layer_norm_2_grid_2.grid_y, triton_per_fused_add_embedding_native_layer_norm_2_grid_2.grid_z, 8, 32, kernel_args_var_2, stream0);
    primals_1.reset();
    primals_4.reset();
    at::Tensor buf8 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv], Original ATen: [aten.addmm]
    at::addmm_out(buf8, primals_6, reinterpret_tensor(buf7, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_5, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_6.reset();
    // Source Nodes: [y], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf9 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf10 = std::get<0>(buf9);
    auto buf11 = std::get<1>(buf9);
    auto buf12 = std::get<2>(buf9);
    auto buf13 = std::get<3>(buf9);

    at::Tensor buf14 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf14, reinterpret_tensor(buf10, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_7, {768L, 768L}, {1L, 768L}, 0L));
    at::Tensor buf18 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf19 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf291 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_0_ln_2, x, x_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    if (triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3 == nullptr) {
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3 = loadKernel("/tmp/torchinductor_chilli/e2/ce2m7fb6rhmmsx6tt2wdx35624cke7npltiztykfmqguyjeos3y6.cubin", "triton__0d1d2d3d4d5d6d7d8d9d10d", 32);
    }
    CUdeviceptr var_16 = reinterpret_cast<CUdeviceptr>(buf1.data_ptr());
    CUdeviceptr var_17 = reinterpret_cast<CUdeviceptr>(buf2.data_ptr());
    CUdeviceptr var_18 = reinterpret_cast<CUdeviceptr>(buf14.data_ptr());
    CUdeviceptr var_19 = reinterpret_cast<CUdeviceptr>(primals_8.data_ptr());
    CUdeviceptr var_20 = reinterpret_cast<CUdeviceptr>(primals_9.data_ptr());
    CUdeviceptr var_21 = reinterpret_cast<CUdeviceptr>(primals_10.data_ptr());
    CUdeviceptr var_22 = reinterpret_cast<CUdeviceptr>(buf18.data_ptr());
    CUdeviceptr var_23 = reinterpret_cast<CUdeviceptr>(buf19.data_ptr());
    CUdeviceptr var_24 = reinterpret_cast<CUdeviceptr>(buf291.data_ptr());
    auto var_25 = 4096;
    auto var_26 = 768;
    void* kernel_args_var_3[] = {&var_16, &var_17, &var_18, &var_19, &var_20, &var_21, &var_22, &var_23, &var_24, &var_25, &var_26};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3_grid_3 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3_grid_3.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3_grid_3.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_3_grid_3.grid_z, 8, 32, kernel_args_var_3, stream0);
    primals_10.reset();
    at::Tensor buf20 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_2], Original ATen: [aten.addmm]
    at::addmm_out(buf20, primals_12, reinterpret_tensor(buf19, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_11, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_12.reset();
    at::Tensor buf21 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_2, add_3, mul, mul_1, mul_2, pow_1, tanh, x_3], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    if (triton_poi_fused_add_mul_pow_tanh_4 == nullptr) {
        triton_poi_fused_add_mul_pow_tanh_4 = loadKernel("/tmp/torchinductor_chilli/uv/cuvf33rg2jd2xraqr3bik5fqw5qpplv47cjrrgtu3pcipswauemw.cubin", "triton__0d1d2d", 0);
    }
    CUdeviceptr var_27 = reinterpret_cast<CUdeviceptr>(buf20.data_ptr());
    CUdeviceptr var_28 = reinterpret_cast<CUdeviceptr>(buf21.data_ptr());
    auto var_29 = 12582912;
    void* kernel_args_var_4[] = {&var_27, &var_28, &var_29};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_4 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_4.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_4.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_4.grid_z, 4, 0, kernel_args_var_4, stream0);
    at::Tensor buf22 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf22, reinterpret_tensor(buf21, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_13, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf23 = reinterpret_tensor(buf22, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf22.reset();  // reuse
    at::Tensor buf27 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf28 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf290 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_1_ln_1, x, x_1, x_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    if (triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 == nullptr) {
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = loadKernel("/tmp/torchinductor_chilli/ue/cueswmvhw5jpyapmt3mc3gxbrz5d65zthtzcrsnp3wxnqlgtjico.cubin", "triton__0d1d2d3d4d5d6d7d8d9d10d11d12d", 8);
    }
    CUdeviceptr var_30 = reinterpret_cast<CUdeviceptr>(buf23.data_ptr());
    CUdeviceptr var_31 = reinterpret_cast<CUdeviceptr>(buf1.data_ptr());
    CUdeviceptr var_32 = reinterpret_cast<CUdeviceptr>(buf2.data_ptr());
    CUdeviceptr var_33 = reinterpret_cast<CUdeviceptr>(buf14.data_ptr());
    CUdeviceptr var_34 = reinterpret_cast<CUdeviceptr>(primals_8.data_ptr());
    CUdeviceptr var_35 = reinterpret_cast<CUdeviceptr>(primals_14.data_ptr());
    CUdeviceptr var_36 = reinterpret_cast<CUdeviceptr>(primals_15.data_ptr());
    CUdeviceptr var_37 = reinterpret_cast<CUdeviceptr>(primals_16.data_ptr());
    CUdeviceptr var_38 = reinterpret_cast<CUdeviceptr>(buf27.data_ptr());
    CUdeviceptr var_39 = reinterpret_cast<CUdeviceptr>(buf28.data_ptr());
    CUdeviceptr var_40 = reinterpret_cast<CUdeviceptr>(buf290.data_ptr());
    auto var_41 = 4096;
    auto var_42 = 768;
    void* kernel_args_var_5[] = {&var_30, &var_31, &var_32, &var_33, &var_34, &var_35, &var_36, &var_37, &var_38, &var_39, &var_40, &var_41, &var_42};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5_grid_5 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5_grid_5.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5_grid_5.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5_grid_5.grid_z, 2, 8, kernel_args_var_5, stream0);
    primals_14.reset();
    primals_16.reset();
    primals_8.reset();
    at::Tensor buf29 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_1], Original ATen: [aten.addmm]
    at::addmm_out(buf29, primals_18, reinterpret_tensor(buf28, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_17, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_18.reset();
    // Source Nodes: [y_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf30 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf31 = std::get<0>(buf30);
    auto buf32 = std::get<1>(buf30);
    auto buf33 = std::get<2>(buf30);
    auto buf34 = std::get<3>(buf30);

    decltype(auto) buf35 = buf14; buf14.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf35, reinterpret_tensor(buf31, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_19, {768L, 768L}, {1L, 768L}, 0L));
    at::Tensor buf39 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf40 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf289 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_1_ln_2, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    if (triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 == nullptr) {
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 = loadKernel("/tmp/torchinductor_chilli/ke/ckeqy6cp5erhq6giyduoteye24k5m5axaitqkxp2mp2hghghf5g2.cubin", "triton__0d1d2d3d4d5d6d7d8d9d", 16);
    }
    CUdeviceptr var_43 = reinterpret_cast<CUdeviceptr>(buf23.data_ptr());
    CUdeviceptr var_44 = reinterpret_cast<CUdeviceptr>(buf35.data_ptr());
    CUdeviceptr var_45 = reinterpret_cast<CUdeviceptr>(primals_20.data_ptr());
    CUdeviceptr var_46 = reinterpret_cast<CUdeviceptr>(primals_21.data_ptr());
    CUdeviceptr var_47 = reinterpret_cast<CUdeviceptr>(primals_22.data_ptr());
    CUdeviceptr var_48 = reinterpret_cast<CUdeviceptr>(buf39.data_ptr());
    CUdeviceptr var_49 = reinterpret_cast<CUdeviceptr>(buf40.data_ptr());
    CUdeviceptr var_50 = reinterpret_cast<CUdeviceptr>(buf289.data_ptr());
    auto var_51 = 4096;
    auto var_52 = 768;
    void* kernel_args_var_6[] = {&var_43, &var_44, &var_45, &var_46, &var_47, &var_48, &var_49, &var_50, &var_51, &var_52};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_6 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_6.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_6.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_6.grid_z, 4, 16, kernel_args_var_6, stream0);
    primals_22.reset();
    at::Tensor buf41 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_7], Original ATen: [aten.addmm]
    at::addmm_out(buf41, primals_24, reinterpret_tensor(buf40, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_23, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_24.reset();
    at::Tensor buf42 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_6, add_7, mul_4, mul_5, mul_6, pow_2, tanh_1, x_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_53 = reinterpret_cast<CUdeviceptr>(buf41.data_ptr());
    CUdeviceptr var_54 = reinterpret_cast<CUdeviceptr>(buf42.data_ptr());
    auto var_55 = 12582912;
    void* kernel_args_var_7[] = {&var_53, &var_54, &var_55};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_7 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_7.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_7.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_7.grid_z, 4, 0, kernel_args_var_7, stream0);
    at::Tensor buf43 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf43, reinterpret_tensor(buf42, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_25, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf44 = reinterpret_tensor(buf43, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf43.reset();  // reuse
    at::Tensor buf48 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf49 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf288 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_2_ln_1, x_10, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    if (triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 == nullptr) {
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7 = loadKernel("/tmp/torchinductor_chilli/7a/c7a2qsc5zdr5bdjyzfz6n7mp7uwbp5tcfqi5ee6mg4f3g7taz7l5.cubin", "triton__0d1d2d3d4d5d6d7d8d9d10d11d", 16);
    }
    CUdeviceptr var_56 = reinterpret_cast<CUdeviceptr>(buf44.data_ptr());
    CUdeviceptr var_57 = reinterpret_cast<CUdeviceptr>(buf23.data_ptr());
    CUdeviceptr var_58 = reinterpret_cast<CUdeviceptr>(buf35.data_ptr());
    CUdeviceptr var_59 = reinterpret_cast<CUdeviceptr>(primals_20.data_ptr());
    CUdeviceptr var_60 = reinterpret_cast<CUdeviceptr>(primals_26.data_ptr());
    CUdeviceptr var_61 = reinterpret_cast<CUdeviceptr>(primals_27.data_ptr());
    CUdeviceptr var_62 = reinterpret_cast<CUdeviceptr>(primals_28.data_ptr());
    CUdeviceptr var_63 = reinterpret_cast<CUdeviceptr>(buf48.data_ptr());
    CUdeviceptr var_64 = reinterpret_cast<CUdeviceptr>(buf49.data_ptr());
    CUdeviceptr var_65 = reinterpret_cast<CUdeviceptr>(buf288.data_ptr());
    auto var_66 = 4096;
    auto var_67 = 768;
    void* kernel_args_var_8[] = {&var_56, &var_57, &var_58, &var_59, &var_60, &var_61, &var_62, &var_63, &var_64, &var_65, &var_66, &var_67};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_8 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_8.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_8.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_8.grid_z, 4, 16, kernel_args_var_8, stream0);
    primals_20.reset();
    primals_26.reset();
    primals_28.reset();
    at::Tensor buf50 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_2], Original ATen: [aten.addmm]
    at::addmm_out(buf50, primals_30, reinterpret_tensor(buf49, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_29, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_30.reset();
    // Source Nodes: [y_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf51 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf52 = std::get<0>(buf51);
    auto buf53 = std::get<1>(buf51);
    auto buf54 = std::get<2>(buf51);
    auto buf55 = std::get<3>(buf51);

    decltype(auto) buf56 = buf35; buf35.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf56, reinterpret_tensor(buf52, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_31, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf60 = buf23; buf23.reset();;  // reuse
    at::Tensor buf61 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf287 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_2_ln_2, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_68 = reinterpret_cast<CUdeviceptr>(buf44.data_ptr());
    CUdeviceptr var_69 = reinterpret_cast<CUdeviceptr>(buf56.data_ptr());
    CUdeviceptr var_70 = reinterpret_cast<CUdeviceptr>(primals_32.data_ptr());
    CUdeviceptr var_71 = reinterpret_cast<CUdeviceptr>(primals_33.data_ptr());
    CUdeviceptr var_72 = reinterpret_cast<CUdeviceptr>(primals_34.data_ptr());
    CUdeviceptr var_73 = reinterpret_cast<CUdeviceptr>(buf60.data_ptr());
    CUdeviceptr var_74 = reinterpret_cast<CUdeviceptr>(buf61.data_ptr());
    CUdeviceptr var_75 = reinterpret_cast<CUdeviceptr>(buf287.data_ptr());
    auto var_76 = 4096;
    auto var_77 = 768;
    void* kernel_args_var_9[] = {&var_68, &var_69, &var_70, &var_71, &var_72, &var_73, &var_74, &var_75, &var_76, &var_77};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_9 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_9.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_9.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_9.grid_z, 4, 16, kernel_args_var_9, stream0);
    primals_34.reset();
    at::Tensor buf62 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_12], Original ATen: [aten.addmm]
    at::addmm_out(buf62, primals_36, reinterpret_tensor(buf61, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_35, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_36.reset();
    at::Tensor buf63 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_10, add_11, mul_10, mul_8, mul_9, pow_3, tanh_2, x_13], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_78 = reinterpret_cast<CUdeviceptr>(buf62.data_ptr());
    CUdeviceptr var_79 = reinterpret_cast<CUdeviceptr>(buf63.data_ptr());
    auto var_80 = 12582912;
    void* kernel_args_var_10[] = {&var_78, &var_79, &var_80};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_10 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_10.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_10.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_10.grid_z, 4, 0, kernel_args_var_10, stream0);
    at::Tensor buf64 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf64, reinterpret_tensor(buf63, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_37, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf65 = reinterpret_tensor(buf64, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf64.reset();  // reuse
    at::Tensor buf69 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf70 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf286 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_3_ln_1, x_11, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_81 = reinterpret_cast<CUdeviceptr>(buf65.data_ptr());
    CUdeviceptr var_82 = reinterpret_cast<CUdeviceptr>(buf44.data_ptr());
    CUdeviceptr var_83 = reinterpret_cast<CUdeviceptr>(buf56.data_ptr());
    CUdeviceptr var_84 = reinterpret_cast<CUdeviceptr>(primals_32.data_ptr());
    CUdeviceptr var_85 = reinterpret_cast<CUdeviceptr>(primals_38.data_ptr());
    CUdeviceptr var_86 = reinterpret_cast<CUdeviceptr>(primals_39.data_ptr());
    CUdeviceptr var_87 = reinterpret_cast<CUdeviceptr>(primals_40.data_ptr());
    CUdeviceptr var_88 = reinterpret_cast<CUdeviceptr>(buf69.data_ptr());
    CUdeviceptr var_89 = reinterpret_cast<CUdeviceptr>(buf70.data_ptr());
    CUdeviceptr var_90 = reinterpret_cast<CUdeviceptr>(buf286.data_ptr());
    auto var_91 = 4096;
    auto var_92 = 768;
    void* kernel_args_var_11[] = {&var_81, &var_82, &var_83, &var_84, &var_85, &var_86, &var_87, &var_88, &var_89, &var_90, &var_91, &var_92};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_11 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_11.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_11.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_11.grid_z, 4, 16, kernel_args_var_11, stream0);
    primals_32.reset();
    primals_38.reset();
    primals_40.reset();
    at::Tensor buf71 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_3], Original ATen: [aten.addmm]
    at::addmm_out(buf71, primals_42, reinterpret_tensor(buf70, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_41, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_42.reset();
    // Source Nodes: [y_9], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf72 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf73 = std::get<0>(buf72);
    auto buf74 = std::get<1>(buf72);
    auto buf75 = std::get<2>(buf72);
    auto buf76 = std::get<3>(buf72);

    decltype(auto) buf77 = buf56; buf56.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf77, reinterpret_tensor(buf73, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_43, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf81 = buf44; buf44.reset();;  // reuse
    at::Tensor buf82 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf285 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_3_ln_2, x_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_93 = reinterpret_cast<CUdeviceptr>(buf65.data_ptr());
    CUdeviceptr var_94 = reinterpret_cast<CUdeviceptr>(buf77.data_ptr());
    CUdeviceptr var_95 = reinterpret_cast<CUdeviceptr>(primals_44.data_ptr());
    CUdeviceptr var_96 = reinterpret_cast<CUdeviceptr>(primals_45.data_ptr());
    CUdeviceptr var_97 = reinterpret_cast<CUdeviceptr>(primals_46.data_ptr());
    CUdeviceptr var_98 = reinterpret_cast<CUdeviceptr>(buf81.data_ptr());
    CUdeviceptr var_99 = reinterpret_cast<CUdeviceptr>(buf82.data_ptr());
    CUdeviceptr var_100 = reinterpret_cast<CUdeviceptr>(buf285.data_ptr());
    auto var_101 = 4096;
    auto var_102 = 768;
    void* kernel_args_var_12[] = {&var_93, &var_94, &var_95, &var_96, &var_97, &var_98, &var_99, &var_100, &var_101, &var_102};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_12 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_12.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_12.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_12.grid_z, 4, 16, kernel_args_var_12, stream0);
    primals_46.reset();
    at::Tensor buf83 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_17], Original ATen: [aten.addmm]
    at::addmm_out(buf83, primals_48, reinterpret_tensor(buf82, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_47, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_48.reset();
    at::Tensor buf84 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_14, add_15, mul_12, mul_13, mul_14, pow_4, tanh_3, x_18], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_103 = reinterpret_cast<CUdeviceptr>(buf83.data_ptr());
    CUdeviceptr var_104 = reinterpret_cast<CUdeviceptr>(buf84.data_ptr());
    auto var_105 = 12582912;
    void* kernel_args_var_13[] = {&var_103, &var_104, &var_105};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_13 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_13.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_13.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_13.grid_z, 4, 0, kernel_args_var_13, stream0);
    at::Tensor buf85 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf85, reinterpret_tensor(buf84, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_49, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf86 = reinterpret_tensor(buf85, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf85.reset();  // reuse
    at::Tensor buf90 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf91 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf284 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_4_ln_1, x_16, x_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_106 = reinterpret_cast<CUdeviceptr>(buf86.data_ptr());
    CUdeviceptr var_107 = reinterpret_cast<CUdeviceptr>(buf65.data_ptr());
    CUdeviceptr var_108 = reinterpret_cast<CUdeviceptr>(buf77.data_ptr());
    CUdeviceptr var_109 = reinterpret_cast<CUdeviceptr>(primals_44.data_ptr());
    CUdeviceptr var_110 = reinterpret_cast<CUdeviceptr>(primals_50.data_ptr());
    CUdeviceptr var_111 = reinterpret_cast<CUdeviceptr>(primals_51.data_ptr());
    CUdeviceptr var_112 = reinterpret_cast<CUdeviceptr>(primals_52.data_ptr());
    CUdeviceptr var_113 = reinterpret_cast<CUdeviceptr>(buf90.data_ptr());
    CUdeviceptr var_114 = reinterpret_cast<CUdeviceptr>(buf91.data_ptr());
    CUdeviceptr var_115 = reinterpret_cast<CUdeviceptr>(buf284.data_ptr());
    auto var_116 = 4096;
    auto var_117 = 768;
    void* kernel_args_var_14[] = {&var_106, &var_107, &var_108, &var_109, &var_110, &var_111, &var_112, &var_113, &var_114, &var_115, &var_116, &var_117};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_14 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_14.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_14.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_14.grid_z, 4, 16, kernel_args_var_14, stream0);
    primals_44.reset();
    primals_50.reset();
    primals_52.reset();
    at::Tensor buf92 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_4], Original ATen: [aten.addmm]
    at::addmm_out(buf92, primals_54, reinterpret_tensor(buf91, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_53, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_54.reset();
    // Source Nodes: [y_12], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf93 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf94 = std::get<0>(buf93);
    auto buf95 = std::get<1>(buf93);
    auto buf96 = std::get<2>(buf93);
    auto buf97 = std::get<3>(buf93);

    decltype(auto) buf98 = buf77; buf77.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf98, reinterpret_tensor(buf94, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_55, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf102 = buf65; buf65.reset();;  // reuse
    at::Tensor buf103 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf283 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_4_ln_2, x_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_118 = reinterpret_cast<CUdeviceptr>(buf86.data_ptr());
    CUdeviceptr var_119 = reinterpret_cast<CUdeviceptr>(buf98.data_ptr());
    CUdeviceptr var_120 = reinterpret_cast<CUdeviceptr>(primals_56.data_ptr());
    CUdeviceptr var_121 = reinterpret_cast<CUdeviceptr>(primals_57.data_ptr());
    CUdeviceptr var_122 = reinterpret_cast<CUdeviceptr>(primals_58.data_ptr());
    CUdeviceptr var_123 = reinterpret_cast<CUdeviceptr>(buf102.data_ptr());
    CUdeviceptr var_124 = reinterpret_cast<CUdeviceptr>(buf103.data_ptr());
    CUdeviceptr var_125 = reinterpret_cast<CUdeviceptr>(buf283.data_ptr());
    auto var_126 = 4096;
    auto var_127 = 768;
    void* kernel_args_var_15[] = {&var_118, &var_119, &var_120, &var_121, &var_122, &var_123, &var_124, &var_125, &var_126, &var_127};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_15 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_15.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_15.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_15.grid_z, 4, 16, kernel_args_var_15, stream0);
    primals_58.reset();
    at::Tensor buf104 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_22], Original ATen: [aten.addmm]
    at::addmm_out(buf104, primals_60, reinterpret_tensor(buf103, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_59, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_60.reset();
    at::Tensor buf105 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_18, add_19, mul_16, mul_17, mul_18, pow_5, tanh_4, x_23], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_128 = reinterpret_cast<CUdeviceptr>(buf104.data_ptr());
    CUdeviceptr var_129 = reinterpret_cast<CUdeviceptr>(buf105.data_ptr());
    auto var_130 = 12582912;
    void* kernel_args_var_16[] = {&var_128, &var_129, &var_130};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_16 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_16.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_16.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_16.grid_z, 4, 0, kernel_args_var_16, stream0);
    at::Tensor buf106 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf106, reinterpret_tensor(buf105, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_61, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf107 = reinterpret_tensor(buf106, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf106.reset();  // reuse
    at::Tensor buf111 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf112 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf282 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_5_ln_1, x_21, x_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_131 = reinterpret_cast<CUdeviceptr>(buf107.data_ptr());
    CUdeviceptr var_132 = reinterpret_cast<CUdeviceptr>(buf86.data_ptr());
    CUdeviceptr var_133 = reinterpret_cast<CUdeviceptr>(buf98.data_ptr());
    CUdeviceptr var_134 = reinterpret_cast<CUdeviceptr>(primals_56.data_ptr());
    CUdeviceptr var_135 = reinterpret_cast<CUdeviceptr>(primals_62.data_ptr());
    CUdeviceptr var_136 = reinterpret_cast<CUdeviceptr>(primals_63.data_ptr());
    CUdeviceptr var_137 = reinterpret_cast<CUdeviceptr>(primals_64.data_ptr());
    CUdeviceptr var_138 = reinterpret_cast<CUdeviceptr>(buf111.data_ptr());
    CUdeviceptr var_139 = reinterpret_cast<CUdeviceptr>(buf112.data_ptr());
    CUdeviceptr var_140 = reinterpret_cast<CUdeviceptr>(buf282.data_ptr());
    auto var_141 = 4096;
    auto var_142 = 768;
    void* kernel_args_var_17[] = {&var_131, &var_132, &var_133, &var_134, &var_135, &var_136, &var_137, &var_138, &var_139, &var_140, &var_141, &var_142};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_17 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_17.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_17.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_17.grid_z, 4, 16, kernel_args_var_17, stream0);
    primals_56.reset();
    primals_62.reset();
    primals_64.reset();
    at::Tensor buf113 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_5], Original ATen: [aten.addmm]
    at::addmm_out(buf113, primals_66, reinterpret_tensor(buf112, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_65, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_66.reset();
    // Source Nodes: [y_15], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf114 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf115 = std::get<0>(buf114);
    auto buf116 = std::get<1>(buf114);
    auto buf117 = std::get<2>(buf114);
    auto buf118 = std::get<3>(buf114);

    decltype(auto) buf119 = buf98; buf98.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf119, reinterpret_tensor(buf115, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_67, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf123 = buf86; buf86.reset();;  // reuse
    at::Tensor buf124 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf281 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_5_ln_2, x_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_143 = reinterpret_cast<CUdeviceptr>(buf107.data_ptr());
    CUdeviceptr var_144 = reinterpret_cast<CUdeviceptr>(buf119.data_ptr());
    CUdeviceptr var_145 = reinterpret_cast<CUdeviceptr>(primals_68.data_ptr());
    CUdeviceptr var_146 = reinterpret_cast<CUdeviceptr>(primals_69.data_ptr());
    CUdeviceptr var_147 = reinterpret_cast<CUdeviceptr>(primals_70.data_ptr());
    CUdeviceptr var_148 = reinterpret_cast<CUdeviceptr>(buf123.data_ptr());
    CUdeviceptr var_149 = reinterpret_cast<CUdeviceptr>(buf124.data_ptr());
    CUdeviceptr var_150 = reinterpret_cast<CUdeviceptr>(buf281.data_ptr());
    auto var_151 = 4096;
    auto var_152 = 768;
    void* kernel_args_var_18[] = {&var_143, &var_144, &var_145, &var_146, &var_147, &var_148, &var_149, &var_150, &var_151, &var_152};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_18 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_18.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_18.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_18.grid_z, 4, 16, kernel_args_var_18, stream0);
    primals_70.reset();
    at::Tensor buf125 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_27], Original ATen: [aten.addmm]
    at::addmm_out(buf125, primals_72, reinterpret_tensor(buf124, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_71, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_72.reset();
    at::Tensor buf126 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_22, add_23, mul_20, mul_21, mul_22, pow_6, tanh_5, x_28], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_153 = reinterpret_cast<CUdeviceptr>(buf125.data_ptr());
    CUdeviceptr var_154 = reinterpret_cast<CUdeviceptr>(buf126.data_ptr());
    auto var_155 = 12582912;
    void* kernel_args_var_19[] = {&var_153, &var_154, &var_155};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_19 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_19.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_19.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_19.grid_z, 4, 0, kernel_args_var_19, stream0);
    at::Tensor buf127 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf127, reinterpret_tensor(buf126, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_73, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf128 = reinterpret_tensor(buf127, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf127.reset();  // reuse
    at::Tensor buf132 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf133 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf280 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_6_ln_1, x_26, x_30], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_156 = reinterpret_cast<CUdeviceptr>(buf128.data_ptr());
    CUdeviceptr var_157 = reinterpret_cast<CUdeviceptr>(buf107.data_ptr());
    CUdeviceptr var_158 = reinterpret_cast<CUdeviceptr>(buf119.data_ptr());
    CUdeviceptr var_159 = reinterpret_cast<CUdeviceptr>(primals_68.data_ptr());
    CUdeviceptr var_160 = reinterpret_cast<CUdeviceptr>(primals_74.data_ptr());
    CUdeviceptr var_161 = reinterpret_cast<CUdeviceptr>(primals_75.data_ptr());
    CUdeviceptr var_162 = reinterpret_cast<CUdeviceptr>(primals_76.data_ptr());
    CUdeviceptr var_163 = reinterpret_cast<CUdeviceptr>(buf132.data_ptr());
    CUdeviceptr var_164 = reinterpret_cast<CUdeviceptr>(buf133.data_ptr());
    CUdeviceptr var_165 = reinterpret_cast<CUdeviceptr>(buf280.data_ptr());
    auto var_166 = 4096;
    auto var_167 = 768;
    void* kernel_args_var_20[] = {&var_156, &var_157, &var_158, &var_159, &var_160, &var_161, &var_162, &var_163, &var_164, &var_165, &var_166, &var_167};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_20 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_20.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_20.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_20.grid_z, 4, 16, kernel_args_var_20, stream0);
    primals_68.reset();
    primals_74.reset();
    primals_76.reset();
    at::Tensor buf134 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_6], Original ATen: [aten.addmm]
    at::addmm_out(buf134, primals_78, reinterpret_tensor(buf133, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_77, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_78.reset();
    // Source Nodes: [y_18], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf135 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf136 = std::get<0>(buf135);
    auto buf137 = std::get<1>(buf135);
    auto buf138 = std::get<2>(buf135);
    auto buf139 = std::get<3>(buf135);

    decltype(auto) buf140 = buf119; buf119.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf140, reinterpret_tensor(buf136, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_79, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf144 = buf107; buf107.reset();;  // reuse
    at::Tensor buf145 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf279 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_6_ln_2, x_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_168 = reinterpret_cast<CUdeviceptr>(buf128.data_ptr());
    CUdeviceptr var_169 = reinterpret_cast<CUdeviceptr>(buf140.data_ptr());
    CUdeviceptr var_170 = reinterpret_cast<CUdeviceptr>(primals_80.data_ptr());
    CUdeviceptr var_171 = reinterpret_cast<CUdeviceptr>(primals_81.data_ptr());
    CUdeviceptr var_172 = reinterpret_cast<CUdeviceptr>(primals_82.data_ptr());
    CUdeviceptr var_173 = reinterpret_cast<CUdeviceptr>(buf144.data_ptr());
    CUdeviceptr var_174 = reinterpret_cast<CUdeviceptr>(buf145.data_ptr());
    CUdeviceptr var_175 = reinterpret_cast<CUdeviceptr>(buf279.data_ptr());
    auto var_176 = 4096;
    auto var_177 = 768;
    void* kernel_args_var_21[] = {&var_168, &var_169, &var_170, &var_171, &var_172, &var_173, &var_174, &var_175, &var_176, &var_177};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_21 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_21.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_21.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_21.grid_z, 4, 16, kernel_args_var_21, stream0);
    primals_82.reset();
    at::Tensor buf146 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_32], Original ATen: [aten.addmm]
    at::addmm_out(buf146, primals_84, reinterpret_tensor(buf145, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_83, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_84.reset();
    at::Tensor buf147 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_26, add_27, mul_24, mul_25, mul_26, pow_7, tanh_6, x_33], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_178 = reinterpret_cast<CUdeviceptr>(buf146.data_ptr());
    CUdeviceptr var_179 = reinterpret_cast<CUdeviceptr>(buf147.data_ptr());
    auto var_180 = 12582912;
    void* kernel_args_var_22[] = {&var_178, &var_179, &var_180};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_22 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_22.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_22.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_22.grid_z, 4, 0, kernel_args_var_22, stream0);
    at::Tensor buf148 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf148, reinterpret_tensor(buf147, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_85, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf149 = reinterpret_tensor(buf148, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf148.reset();  // reuse
    at::Tensor buf153 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf154 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf278 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_7_ln_1, x_31, x_35], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_181 = reinterpret_cast<CUdeviceptr>(buf149.data_ptr());
    CUdeviceptr var_182 = reinterpret_cast<CUdeviceptr>(buf128.data_ptr());
    CUdeviceptr var_183 = reinterpret_cast<CUdeviceptr>(buf140.data_ptr());
    CUdeviceptr var_184 = reinterpret_cast<CUdeviceptr>(primals_80.data_ptr());
    CUdeviceptr var_185 = reinterpret_cast<CUdeviceptr>(primals_86.data_ptr());
    CUdeviceptr var_186 = reinterpret_cast<CUdeviceptr>(primals_87.data_ptr());
    CUdeviceptr var_187 = reinterpret_cast<CUdeviceptr>(primals_88.data_ptr());
    CUdeviceptr var_188 = reinterpret_cast<CUdeviceptr>(buf153.data_ptr());
    CUdeviceptr var_189 = reinterpret_cast<CUdeviceptr>(buf154.data_ptr());
    CUdeviceptr var_190 = reinterpret_cast<CUdeviceptr>(buf278.data_ptr());
    auto var_191 = 4096;
    auto var_192 = 768;
    void* kernel_args_var_23[] = {&var_181, &var_182, &var_183, &var_184, &var_185, &var_186, &var_187, &var_188, &var_189, &var_190, &var_191, &var_192};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_23 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_23.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_23.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_23.grid_z, 4, 16, kernel_args_var_23, stream0);
    primals_80.reset();
    primals_86.reset();
    primals_88.reset();
    at::Tensor buf155 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_7], Original ATen: [aten.addmm]
    at::addmm_out(buf155, primals_90, reinterpret_tensor(buf154, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_89, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_90.reset();
    // Source Nodes: [y_21], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf156 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf157 = std::get<0>(buf156);
    auto buf158 = std::get<1>(buf156);
    auto buf159 = std::get<2>(buf156);
    auto buf160 = std::get<3>(buf156);

    decltype(auto) buf161 = buf140; buf140.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf161, reinterpret_tensor(buf157, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_91, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf165 = buf128; buf128.reset();;  // reuse
    at::Tensor buf166 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf277 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_7_ln_2, x_36], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_193 = reinterpret_cast<CUdeviceptr>(buf149.data_ptr());
    CUdeviceptr var_194 = reinterpret_cast<CUdeviceptr>(buf161.data_ptr());
    CUdeviceptr var_195 = reinterpret_cast<CUdeviceptr>(primals_92.data_ptr());
    CUdeviceptr var_196 = reinterpret_cast<CUdeviceptr>(primals_93.data_ptr());
    CUdeviceptr var_197 = reinterpret_cast<CUdeviceptr>(primals_94.data_ptr());
    CUdeviceptr var_198 = reinterpret_cast<CUdeviceptr>(buf165.data_ptr());
    CUdeviceptr var_199 = reinterpret_cast<CUdeviceptr>(buf166.data_ptr());
    CUdeviceptr var_200 = reinterpret_cast<CUdeviceptr>(buf277.data_ptr());
    auto var_201 = 4096;
    auto var_202 = 768;
    void* kernel_args_var_24[] = {&var_193, &var_194, &var_195, &var_196, &var_197, &var_198, &var_199, &var_200, &var_201, &var_202};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_24 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_24.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_24.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_24.grid_z, 4, 16, kernel_args_var_24, stream0);
    primals_94.reset();
    at::Tensor buf167 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_37], Original ATen: [aten.addmm]
    at::addmm_out(buf167, primals_96, reinterpret_tensor(buf166, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_95, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_96.reset();
    at::Tensor buf168 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_30, add_31, mul_28, mul_29, mul_30, pow_8, tanh_7, x_38], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_203 = reinterpret_cast<CUdeviceptr>(buf167.data_ptr());
    CUdeviceptr var_204 = reinterpret_cast<CUdeviceptr>(buf168.data_ptr());
    auto var_205 = 12582912;
    void* kernel_args_var_25[] = {&var_203, &var_204, &var_205};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_25 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_25.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_25.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_25.grid_z, 4, 0, kernel_args_var_25, stream0);
    at::Tensor buf169 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf169, reinterpret_tensor(buf168, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_97, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf170 = reinterpret_tensor(buf169, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf169.reset();  // reuse
    at::Tensor buf174 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf175 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf276 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_8_ln_1, x_36, x_40], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_206 = reinterpret_cast<CUdeviceptr>(buf170.data_ptr());
    CUdeviceptr var_207 = reinterpret_cast<CUdeviceptr>(buf149.data_ptr());
    CUdeviceptr var_208 = reinterpret_cast<CUdeviceptr>(buf161.data_ptr());
    CUdeviceptr var_209 = reinterpret_cast<CUdeviceptr>(primals_92.data_ptr());
    CUdeviceptr var_210 = reinterpret_cast<CUdeviceptr>(primals_98.data_ptr());
    CUdeviceptr var_211 = reinterpret_cast<CUdeviceptr>(primals_99.data_ptr());
    CUdeviceptr var_212 = reinterpret_cast<CUdeviceptr>(primals_100.data_ptr());
    CUdeviceptr var_213 = reinterpret_cast<CUdeviceptr>(buf174.data_ptr());
    CUdeviceptr var_214 = reinterpret_cast<CUdeviceptr>(buf175.data_ptr());
    CUdeviceptr var_215 = reinterpret_cast<CUdeviceptr>(buf276.data_ptr());
    auto var_216 = 4096;
    auto var_217 = 768;
    void* kernel_args_var_26[] = {&var_206, &var_207, &var_208, &var_209, &var_210, &var_211, &var_212, &var_213, &var_214, &var_215, &var_216, &var_217};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_26 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_26.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_26.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_26.grid_z, 4, 16, kernel_args_var_26, stream0);
    primals_100.reset();
    primals_92.reset();
    primals_98.reset();
    at::Tensor buf176 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_8], Original ATen: [aten.addmm]
    at::addmm_out(buf176, primals_102, reinterpret_tensor(buf175, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_101, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_102.reset();
    // Source Nodes: [y_24], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf177 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf178 = std::get<0>(buf177);
    auto buf179 = std::get<1>(buf177);
    auto buf180 = std::get<2>(buf177);
    auto buf181 = std::get<3>(buf177);

    decltype(auto) buf182 = buf161; buf161.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf182, reinterpret_tensor(buf178, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_103, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf186 = buf149; buf149.reset();;  // reuse
    at::Tensor buf187 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf275 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_8_ln_2, x_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_218 = reinterpret_cast<CUdeviceptr>(buf170.data_ptr());
    CUdeviceptr var_219 = reinterpret_cast<CUdeviceptr>(buf182.data_ptr());
    CUdeviceptr var_220 = reinterpret_cast<CUdeviceptr>(primals_104.data_ptr());
    CUdeviceptr var_221 = reinterpret_cast<CUdeviceptr>(primals_105.data_ptr());
    CUdeviceptr var_222 = reinterpret_cast<CUdeviceptr>(primals_106.data_ptr());
    CUdeviceptr var_223 = reinterpret_cast<CUdeviceptr>(buf186.data_ptr());
    CUdeviceptr var_224 = reinterpret_cast<CUdeviceptr>(buf187.data_ptr());
    CUdeviceptr var_225 = reinterpret_cast<CUdeviceptr>(buf275.data_ptr());
    auto var_226 = 4096;
    auto var_227 = 768;
    void* kernel_args_var_27[] = {&var_218, &var_219, &var_220, &var_221, &var_222, &var_223, &var_224, &var_225, &var_226, &var_227};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_27 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_27.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_27.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_27.grid_z, 4, 16, kernel_args_var_27, stream0);
    primals_106.reset();
    at::Tensor buf188 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_42], Original ATen: [aten.addmm]
    at::addmm_out(buf188, primals_108, reinterpret_tensor(buf187, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_107, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_108.reset();
    at::Tensor buf189 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_34, add_35, mul_32, mul_33, mul_34, pow_9, tanh_8, x_43], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_228 = reinterpret_cast<CUdeviceptr>(buf188.data_ptr());
    CUdeviceptr var_229 = reinterpret_cast<CUdeviceptr>(buf189.data_ptr());
    auto var_230 = 12582912;
    void* kernel_args_var_28[] = {&var_228, &var_229, &var_230};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_28 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_28.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_28.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_28.grid_z, 4, 0, kernel_args_var_28, stream0);
    at::Tensor buf190 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf190, reinterpret_tensor(buf189, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_109, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf191 = reinterpret_tensor(buf190, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf190.reset();  // reuse
    at::Tensor buf195 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf196 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf274 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_9_ln_1, x_41, x_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_231 = reinterpret_cast<CUdeviceptr>(buf191.data_ptr());
    CUdeviceptr var_232 = reinterpret_cast<CUdeviceptr>(buf170.data_ptr());
    CUdeviceptr var_233 = reinterpret_cast<CUdeviceptr>(buf182.data_ptr());
    CUdeviceptr var_234 = reinterpret_cast<CUdeviceptr>(primals_104.data_ptr());
    CUdeviceptr var_235 = reinterpret_cast<CUdeviceptr>(primals_110.data_ptr());
    CUdeviceptr var_236 = reinterpret_cast<CUdeviceptr>(primals_111.data_ptr());
    CUdeviceptr var_237 = reinterpret_cast<CUdeviceptr>(primals_112.data_ptr());
    CUdeviceptr var_238 = reinterpret_cast<CUdeviceptr>(buf195.data_ptr());
    CUdeviceptr var_239 = reinterpret_cast<CUdeviceptr>(buf196.data_ptr());
    CUdeviceptr var_240 = reinterpret_cast<CUdeviceptr>(buf274.data_ptr());
    auto var_241 = 4096;
    auto var_242 = 768;
    void* kernel_args_var_29[] = {&var_231, &var_232, &var_233, &var_234, &var_235, &var_236, &var_237, &var_238, &var_239, &var_240, &var_241, &var_242};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_29 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_29.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_29.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_29.grid_z, 4, 16, kernel_args_var_29, stream0);
    primals_104.reset();
    primals_110.reset();
    primals_112.reset();
    at::Tensor buf197 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_9], Original ATen: [aten.addmm]
    at::addmm_out(buf197, primals_114, reinterpret_tensor(buf196, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_113, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_114.reset();
    // Source Nodes: [y_27], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf198 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf199 = std::get<0>(buf198);
    auto buf200 = std::get<1>(buf198);
    auto buf201 = std::get<2>(buf198);
    auto buf202 = std::get<3>(buf198);

    decltype(auto) buf203 = buf182; buf182.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf203, reinterpret_tensor(buf199, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_115, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf207 = buf170; buf170.reset();;  // reuse
    at::Tensor buf208 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf273 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_9_ln_2, x_46], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_243 = reinterpret_cast<CUdeviceptr>(buf191.data_ptr());
    CUdeviceptr var_244 = reinterpret_cast<CUdeviceptr>(buf203.data_ptr());
    CUdeviceptr var_245 = reinterpret_cast<CUdeviceptr>(primals_116.data_ptr());
    CUdeviceptr var_246 = reinterpret_cast<CUdeviceptr>(primals_117.data_ptr());
    CUdeviceptr var_247 = reinterpret_cast<CUdeviceptr>(primals_118.data_ptr());
    CUdeviceptr var_248 = reinterpret_cast<CUdeviceptr>(buf207.data_ptr());
    CUdeviceptr var_249 = reinterpret_cast<CUdeviceptr>(buf208.data_ptr());
    CUdeviceptr var_250 = reinterpret_cast<CUdeviceptr>(buf273.data_ptr());
    auto var_251 = 4096;
    auto var_252 = 768;
    void* kernel_args_var_30[] = {&var_243, &var_244, &var_245, &var_246, &var_247, &var_248, &var_249, &var_250, &var_251, &var_252};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_30 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_30.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_30.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_30.grid_z, 4, 16, kernel_args_var_30, stream0);
    primals_118.reset();
    at::Tensor buf209 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_47], Original ATen: [aten.addmm]
    at::addmm_out(buf209, primals_120, reinterpret_tensor(buf208, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_119, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_120.reset();
    at::Tensor buf210 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_38, add_39, mul_36, mul_37, mul_38, pow_10, tanh_9, x_48], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_253 = reinterpret_cast<CUdeviceptr>(buf209.data_ptr());
    CUdeviceptr var_254 = reinterpret_cast<CUdeviceptr>(buf210.data_ptr());
    auto var_255 = 12582912;
    void* kernel_args_var_31[] = {&var_253, &var_254, &var_255};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_31 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_31.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_31.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_31.grid_z, 4, 0, kernel_args_var_31, stream0);
    at::Tensor buf211 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf211, reinterpret_tensor(buf210, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_121, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf212 = reinterpret_tensor(buf211, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf211.reset();  // reuse
    at::Tensor buf216 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf217 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf272 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_10_ln_1, x_46, x_50], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_256 = reinterpret_cast<CUdeviceptr>(buf212.data_ptr());
    CUdeviceptr var_257 = reinterpret_cast<CUdeviceptr>(buf191.data_ptr());
    CUdeviceptr var_258 = reinterpret_cast<CUdeviceptr>(buf203.data_ptr());
    CUdeviceptr var_259 = reinterpret_cast<CUdeviceptr>(primals_116.data_ptr());
    CUdeviceptr var_260 = reinterpret_cast<CUdeviceptr>(primals_122.data_ptr());
    CUdeviceptr var_261 = reinterpret_cast<CUdeviceptr>(primals_123.data_ptr());
    CUdeviceptr var_262 = reinterpret_cast<CUdeviceptr>(primals_124.data_ptr());
    CUdeviceptr var_263 = reinterpret_cast<CUdeviceptr>(buf216.data_ptr());
    CUdeviceptr var_264 = reinterpret_cast<CUdeviceptr>(buf217.data_ptr());
    CUdeviceptr var_265 = reinterpret_cast<CUdeviceptr>(buf272.data_ptr());
    auto var_266 = 4096;
    auto var_267 = 768;
    void* kernel_args_var_32[] = {&var_256, &var_257, &var_258, &var_259, &var_260, &var_261, &var_262, &var_263, &var_264, &var_265, &var_266, &var_267};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_32 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_32.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_32.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_32.grid_z, 4, 16, kernel_args_var_32, stream0);
    primals_116.reset();
    primals_122.reset();
    primals_124.reset();
    at::Tensor buf218 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_10], Original ATen: [aten.addmm]
    at::addmm_out(buf218, primals_126, reinterpret_tensor(buf217, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_125, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_126.reset();
    // Source Nodes: [y_30], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf219 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf220 = std::get<0>(buf219);
    auto buf221 = std::get<1>(buf219);
    auto buf222 = std::get<2>(buf219);
    auto buf223 = std::get<3>(buf219);

    decltype(auto) buf224 = buf203; buf203.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf224, reinterpret_tensor(buf220, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_127, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf228 = buf191; buf191.reset();;  // reuse
    at::Tensor buf229 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf271 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_10_ln_2, x_51], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_268 = reinterpret_cast<CUdeviceptr>(buf212.data_ptr());
    CUdeviceptr var_269 = reinterpret_cast<CUdeviceptr>(buf224.data_ptr());
    CUdeviceptr var_270 = reinterpret_cast<CUdeviceptr>(primals_128.data_ptr());
    CUdeviceptr var_271 = reinterpret_cast<CUdeviceptr>(primals_129.data_ptr());
    CUdeviceptr var_272 = reinterpret_cast<CUdeviceptr>(primals_130.data_ptr());
    CUdeviceptr var_273 = reinterpret_cast<CUdeviceptr>(buf228.data_ptr());
    CUdeviceptr var_274 = reinterpret_cast<CUdeviceptr>(buf229.data_ptr());
    CUdeviceptr var_275 = reinterpret_cast<CUdeviceptr>(buf271.data_ptr());
    auto var_276 = 4096;
    auto var_277 = 768;
    void* kernel_args_var_33[] = {&var_268, &var_269, &var_270, &var_271, &var_272, &var_273, &var_274, &var_275, &var_276, &var_277};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_33 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_33.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_33.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_33.grid_z, 4, 16, kernel_args_var_33, stream0);
    primals_130.reset();
    at::Tensor buf230 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_52], Original ATen: [aten.addmm]
    at::addmm_out(buf230, primals_132, reinterpret_tensor(buf229, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_131, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_132.reset();
    at::Tensor buf231 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_42, add_43, mul_40, mul_41, mul_42, pow_11, tanh_10, x_53], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_278 = reinterpret_cast<CUdeviceptr>(buf230.data_ptr());
    CUdeviceptr var_279 = reinterpret_cast<CUdeviceptr>(buf231.data_ptr());
    auto var_280 = 12582912;
    void* kernel_args_var_34[] = {&var_278, &var_279, &var_280};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_34 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_34.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_34.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_34.grid_z, 4, 0, kernel_args_var_34, stream0);
    at::Tensor buf232 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf232, reinterpret_tensor(buf231, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_133, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf233 = reinterpret_tensor(buf232, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf232.reset();  // reuse
    at::Tensor buf237 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf238 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf270 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_11_ln_1, x_51, x_55], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_281 = reinterpret_cast<CUdeviceptr>(buf233.data_ptr());
    CUdeviceptr var_282 = reinterpret_cast<CUdeviceptr>(buf212.data_ptr());
    CUdeviceptr var_283 = reinterpret_cast<CUdeviceptr>(buf224.data_ptr());
    CUdeviceptr var_284 = reinterpret_cast<CUdeviceptr>(primals_128.data_ptr());
    CUdeviceptr var_285 = reinterpret_cast<CUdeviceptr>(primals_134.data_ptr());
    CUdeviceptr var_286 = reinterpret_cast<CUdeviceptr>(primals_135.data_ptr());
    CUdeviceptr var_287 = reinterpret_cast<CUdeviceptr>(primals_136.data_ptr());
    CUdeviceptr var_288 = reinterpret_cast<CUdeviceptr>(buf237.data_ptr());
    CUdeviceptr var_289 = reinterpret_cast<CUdeviceptr>(buf238.data_ptr());
    CUdeviceptr var_290 = reinterpret_cast<CUdeviceptr>(buf270.data_ptr());
    auto var_291 = 4096;
    auto var_292 = 768;
    void* kernel_args_var_35[] = {&var_281, &var_282, &var_283, &var_284, &var_285, &var_286, &var_287, &var_288, &var_289, &var_290, &var_291, &var_292};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_35 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_35.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_35.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_35.grid_z, 4, 16, kernel_args_var_35, stream0);
    primals_128.reset();
    primals_134.reset();
    primals_136.reset();
    at::Tensor buf239 = at::detail::empty_strided_cuda({4096L, 2304L}, {2304L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [qkv_11], Original ATen: [aten.addmm]
    at::addmm_out(buf239, primals_138, reinterpret_tensor(buf238, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_137, {768L, 2304L}, {1L, 768L}, 0L), 1L, 1L);
    primals_138.reset();
    // Source Nodes: [y_33], Original ATen: [aten._scaled_dot_product_efficient_attention]
    auto buf240 = at::_ops::_scaled_dot_product_efficient_attention::call(reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L), reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L), reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L), c10::nullopt, true, 0.0, true, c10::nullopt);
    auto buf241 = std::get<0>(buf240);
    auto buf242 = std::get<1>(buf240);
    auto buf243 = std::get<2>(buf240);
    auto buf244 = std::get<3>(buf240);

    decltype(auto) buf245 = buf224; buf224.reset();;  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf245, reinterpret_tensor(buf241, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_139, {768L, 768L}, {1L, 768L}, 0L));
    decltype(auto) buf249 = buf212; buf212.reset();;  // reuse
    at::Tensor buf250 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf269 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [l__self___transformer_h_11_ln_2, x_56], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_293 = reinterpret_cast<CUdeviceptr>(buf233.data_ptr());
    CUdeviceptr var_294 = reinterpret_cast<CUdeviceptr>(buf245.data_ptr());
    CUdeviceptr var_295 = reinterpret_cast<CUdeviceptr>(primals_140.data_ptr());
    CUdeviceptr var_296 = reinterpret_cast<CUdeviceptr>(primals_141.data_ptr());
    CUdeviceptr var_297 = reinterpret_cast<CUdeviceptr>(primals_142.data_ptr());
    CUdeviceptr var_298 = reinterpret_cast<CUdeviceptr>(buf249.data_ptr());
    CUdeviceptr var_299 = reinterpret_cast<CUdeviceptr>(buf250.data_ptr());
    CUdeviceptr var_300 = reinterpret_cast<CUdeviceptr>(buf269.data_ptr());
    auto var_301 = 4096;
    auto var_302 = 768;
    void* kernel_args_var_36[] = {&var_293, &var_294, &var_295, &var_296, &var_297, &var_298, &var_299, &var_300, &var_301, &var_302};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_36 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_36.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_36.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6_grid_36.grid_z, 4, 16, kernel_args_var_36, stream0);
    primals_142.reset();
    at::Tensor buf251 = at::detail::empty_strided_cuda({4096L, 3072L}, {3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_57], Original ATen: [aten.addmm]
    at::addmm_out(buf251, primals_144, reinterpret_tensor(buf250, {4096L, 768L}, {768L, 1L}, 0L), reinterpret_tensor(primals_143, {768L, 3072L}, {1L, 768L}, 0L), 1L, 1L);
    primals_144.reset();
    at::Tensor buf252 = at::detail::empty_strided_cuda({4L, 1024L, 3072L}, {3145728L, 3072L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_46, add_47, mul_44, mul_45, mul_46, pow_12, tanh_11, x_58], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh]
    CUdeviceptr var_303 = reinterpret_cast<CUdeviceptr>(buf251.data_ptr());
    CUdeviceptr var_304 = reinterpret_cast<CUdeviceptr>(buf252.data_ptr());
    auto var_305 = 12582912;
    void* kernel_args_var_37[] = {&var_303, &var_304, &var_305};
    Grid triton_poi_fused_add_mul_pow_tanh_4_grid_37 = Grid(24576L, 1L, 1L);
    launchKernel(triton_poi_fused_add_mul_pow_tanh_4, triton_poi_fused_add_mul_pow_tanh_4_grid_37.grid_x, triton_poi_fused_add_mul_pow_tanh_4_grid_37.grid_y, triton_poi_fused_add_mul_pow_tanh_4_grid_37.grid_z, 4, 0, kernel_args_var_37, stream0);
    at::Tensor buf253 = at::detail::empty_strided_cuda({4096L, 768L}, {768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf253, reinterpret_tensor(buf252, {4096L, 3072L}, {3072L, 1L}, 0L), reinterpret_tensor(primals_145, {3072L, 768L}, {1L, 3072L}, 0L));
    decltype(auto) buf254 = reinterpret_tensor(buf253, {4L, 1024L, 768L}, {786432L, 768L, 1L}, 0L); buf253.reset();  // reuse
    at::Tensor buf258 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf259 = at::detail::empty_strided_cuda({4L, 1024L, 768L}, {786432L, 768L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf268 = at::detail::empty_strided_cuda({4L, 1024L, 1L}, {1024L, 1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [x_56, x_60, x_61], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
    CUdeviceptr var_306 = reinterpret_cast<CUdeviceptr>(buf254.data_ptr());
    CUdeviceptr var_307 = reinterpret_cast<CUdeviceptr>(buf233.data_ptr());
    CUdeviceptr var_308 = reinterpret_cast<CUdeviceptr>(buf245.data_ptr());
    CUdeviceptr var_309 = reinterpret_cast<CUdeviceptr>(primals_140.data_ptr());
    CUdeviceptr var_310 = reinterpret_cast<CUdeviceptr>(primals_146.data_ptr());
    CUdeviceptr var_311 = reinterpret_cast<CUdeviceptr>(primals_147.data_ptr());
    CUdeviceptr var_312 = reinterpret_cast<CUdeviceptr>(primals_148.data_ptr());
    CUdeviceptr var_313 = reinterpret_cast<CUdeviceptr>(buf258.data_ptr());
    CUdeviceptr var_314 = reinterpret_cast<CUdeviceptr>(buf259.data_ptr());
    CUdeviceptr var_315 = reinterpret_cast<CUdeviceptr>(buf268.data_ptr());
    auto var_316 = 4096;
    auto var_317 = 768;
    void* kernel_args_var_38[] = {&var_306, &var_307, &var_308, &var_309, &var_310, &var_311, &var_312, &var_313, &var_314, &var_315, &var_316, &var_317};
    Grid triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_38 = Grid(4096L, 1L, 1L);
    launchKernel(triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_38.grid_x, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_38.grid_y, triton_per_fused_add_native_layer_norm_native_layer_norm_backward_7_grid_38.grid_z, 4, 16, kernel_args_var_38, stream0);
    buf233.reset();
    buf245.reset();
    buf254.reset();
    primals_140.reset();
    primals_146.reset();
    primals_148.reset();
    at::Tensor buf260 = at::detail::empty_strided_cuda({768L, 50260L}, {1L, 768L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    if (triton_poi_fused_8 == nullptr) {
        triton_poi_fused_8 = loadKernel("/tmp/torchinductor_chilli/4v/c4vobpn6gacslrjvkhnbu5ezoxch2ch5svfyspa4nzebkrqom4hj.cubin", "triton__0d1d2d", 0);
    }
    CUdeviceptr var_318 = reinterpret_cast<CUdeviceptr>(primals_149.data_ptr());
    CUdeviceptr var_319 = reinterpret_cast<CUdeviceptr>(buf260.data_ptr());
    auto var_320 = 38599680;
    void* kernel_args_var_39[] = {&var_318, &var_319, &var_320};
    Grid triton_poi_fused_8_grid_39 = Grid(75390L, 1L, 1L);
    launchKernel(triton_poi_fused_8, triton_poi_fused_8_grid_39.grid_x, triton_poi_fused_8_grid_39.grid_y, triton_poi_fused_8_grid_39.grid_z, 4, 0, kernel_args_var_39, stream0);
    at::Tensor buf261 = at::detail::empty_strided_cuda({4096L, 50260L}, {50260L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf261, reinterpret_tensor(buf259, {4096L, 768L}, {768L, 1L}, 0L), buf260);
    buf260.reset();
    at::Tensor buf262 = at::detail::empty_strided_cuda({4096L, 1L}, {1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf263 = at::detail::empty_strided_cuda({4096L, 1L}, {1L, 4096L}, at::kFloat, c10::DeviceType::CUDA);
    decltype(auto) buf264 = reinterpret_tensor(buf263, {4096L, 1L}, {1L, 1L}, 0L); buf263.reset();  // reuse
    // Source Nodes: [loss], Original ATen: [aten._log_softmax]
    if (triton_red_fused__log_softmax_9 == nullptr) {
        triton_red_fused__log_softmax_9 = loadKernel("/tmp/torchinductor_chilli/am/cam2gqsqiub2dzkwyughnqkzbclhy6yif3bwc3sjiawii3pk5ywf.cubin", "triton__0d1d2d3d4", 32);
    }
    CUdeviceptr var_321 = reinterpret_cast<CUdeviceptr>(buf264.data_ptr());
    CUdeviceptr var_322 = reinterpret_cast<CUdeviceptr>(buf261.data_ptr());
    CUdeviceptr var_323 = reinterpret_cast<CUdeviceptr>(buf262.data_ptr());
    auto var_324 = 4096;
    auto var_325 = 50257;
    void* kernel_args_var_40[] = {&var_321, &var_322, &var_323, &var_324, &var_325};
    Grid triton_red_fused__log_softmax_9_grid_40 = Grid(2048L, 1L, 1L);
    launchKernel(triton_red_fused__log_softmax_9, triton_red_fused__log_softmax_9_grid_40.grid_x, triton_red_fused__log_softmax_9_grid_40.grid_y, triton_red_fused__log_softmax_9_grid_40.grid_z, 4, 32, kernel_args_var_40, stream0);
    at::Tensor buf267 = at::detail::empty_strided_cuda({}, {}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf266 = at::detail::empty_strided_cuda({}, {}, at::kFloat, c10::DeviceType::CUDA);
    decltype(auto) buf292 = buf267; buf267.reset();;  // reuse
    // Source Nodes: [loss], Original ATen: [aten.nll_loss_forward]
    if (triton_red_fused_nll_loss_forward_10 == nullptr) {
        triton_red_fused_nll_loss_forward_10 = loadKernel("/tmp/torchinductor_chilli/s6/cs67lnnrlmhnfgzvcwj7fsdxuvqjpbri6bjxqquut4z2rowwawq5.cubin", "triton__0d1d2d3d4d5d6c7d", 32768);
    }
    CUdeviceptr var_326 = reinterpret_cast<CUdeviceptr>(buf292.data_ptr());
    CUdeviceptr var_327 = reinterpret_cast<CUdeviceptr>(primals_151.data_ptr());
    CUdeviceptr var_328 = reinterpret_cast<CUdeviceptr>(buf261.data_ptr());
    CUdeviceptr var_329 = reinterpret_cast<CUdeviceptr>(buf262.data_ptr());
    CUdeviceptr var_330 = reinterpret_cast<CUdeviceptr>(buf264.data_ptr());
    CUdeviceptr var_331 = reinterpret_cast<CUdeviceptr>(buf266.data_ptr());
    auto var_332 = 4096;
    void* kernel_args_var_41[] = {&var_326, &var_327, &var_328, &var_329, &var_330, &var_331, &var_332};
    Grid triton_red_fused_nll_loss_forward_10_grid_41 = Grid(1L, 1L, 1L);
    launchKernel(triton_red_fused_nll_loss_forward_10, triton_red_fused_nll_loss_forward_10_grid_41.grid_x, triton_red_fused_nll_loss_forward_10_grid_41.grid_y, triton_red_fused_nll_loss_forward_10_grid_41.grid_z, 16, 32768, kernel_args_var_41, stream0);
    output_handles[0] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf261, {4L, 1024L, 50257L}, {51466240L, 50260L, 1L}, 0L)));
    output_handles[1] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf292));
    output_handles[2] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_3));
    output_handles[3] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_9));
    output_handles[4] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_15));
    output_handles[5] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_21));
    output_handles[6] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_27));
    output_handles[7] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_33));
    output_handles[8] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_39));
    output_handles[9] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_45));
    output_handles[10] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_51));
    output_handles[11] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_57));
    output_handles[12] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_63));
    output_handles[13] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_69));
    output_handles[14] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_75));
    output_handles[15] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_81));
    output_handles[16] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_87));
    output_handles[17] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_93));
    output_handles[18] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_99));
    output_handles[19] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_105));
    output_handles[20] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_111));
    output_handles[21] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_117));
    output_handles[22] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_123));
    output_handles[23] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_129));
    output_handles[24] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_135));
    output_handles[25] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_141));
    output_handles[26] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_147));
    output_handles[27] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_150));
    output_handles[28] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(primals_151));
    output_handles[29] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf0));
    output_handles[30] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf1));
    output_handles[31] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf2));
    output_handles[32] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf3));
    output_handles[33] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf6));
    output_handles[34] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf7, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[35] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[36] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[37] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf8, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[38] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf11));
    output_handles[39] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf12));
    output_handles[40] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf13));
    output_handles[41] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf10, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[42] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf18));
    output_handles[43] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf19, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[44] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf20));
    output_handles[45] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf21, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[46] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf27));
    output_handles[47] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf28, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[48] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[49] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[50] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf29, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[51] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf32));
    output_handles[52] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf33));
    output_handles[53] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf34));
    output_handles[54] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf31, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[55] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf39));
    output_handles[56] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf40, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[57] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf41));
    output_handles[58] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf42, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[59] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf48));
    output_handles[60] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf49, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[61] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[62] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[63] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf50, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[64] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf53));
    output_handles[65] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf54));
    output_handles[66] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf55));
    output_handles[67] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf52, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[68] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf60));
    output_handles[69] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf61, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[70] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf62));
    output_handles[71] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf63, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[72] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf69));
    output_handles[73] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf70, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[74] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[75] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[76] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf71, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[77] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf74));
    output_handles[78] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf75));
    output_handles[79] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf76));
    output_handles[80] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf73, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[81] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf81));
    output_handles[82] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf82, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[83] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf83));
    output_handles[84] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf84, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[85] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf90));
    output_handles[86] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf91, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[87] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[88] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[89] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf92, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[90] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf95));
    output_handles[91] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf96));
    output_handles[92] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf97));
    output_handles[93] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf94, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[94] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf102));
    output_handles[95] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf103, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[96] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf104));
    output_handles[97] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf105, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[98] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf111));
    output_handles[99] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf112, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[100] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[101] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[102] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf113, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[103] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf116));
    output_handles[104] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf117));
    output_handles[105] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf118));
    output_handles[106] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf115, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[107] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf123));
    output_handles[108] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf124, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[109] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf125));
    output_handles[110] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf126, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[111] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf132));
    output_handles[112] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf133, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[113] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[114] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[115] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf134, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[116] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf137));
    output_handles[117] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf138));
    output_handles[118] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf139));
    output_handles[119] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf136, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[120] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf144));
    output_handles[121] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf145, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[122] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf146));
    output_handles[123] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf147, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[124] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf153));
    output_handles[125] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf154, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[126] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[127] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[128] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf155, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[129] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf158));
    output_handles[130] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf159));
    output_handles[131] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf160));
    output_handles[132] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf157, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[133] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf165));
    output_handles[134] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf166, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[135] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf167));
    output_handles[136] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf168, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[137] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf174));
    output_handles[138] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf175, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[139] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[140] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[141] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf176, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[142] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf179));
    output_handles[143] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf180));
    output_handles[144] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf181));
    output_handles[145] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf178, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[146] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf186));
    output_handles[147] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf187, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[148] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf188));
    output_handles[149] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf189, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[150] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf195));
    output_handles[151] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf196, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[152] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[153] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[154] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf197, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[155] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf200));
    output_handles[156] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf201));
    output_handles[157] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf202));
    output_handles[158] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf199, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[159] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf207));
    output_handles[160] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf208, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[161] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf209));
    output_handles[162] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf210, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[163] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf216));
    output_handles[164] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf217, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[165] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[166] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[167] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf218, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[168] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf221));
    output_handles[169] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf222));
    output_handles[170] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf223));
    output_handles[171] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf220, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[172] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf228));
    output_handles[173] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf229, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[174] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf230));
    output_handles[175] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf231, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[176] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf237));
    output_handles[177] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf238, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[178] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 768L)));
    output_handles[179] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 0L)));
    output_handles[180] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf239, {4L, 12L, 1024L, 64L}, {2359296L, 64L, 2304L, 1L}, 1536L)));
    output_handles[181] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf242));
    output_handles[182] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf243));
    output_handles[183] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf244));
    output_handles[184] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf241, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[185] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf249));
    output_handles[186] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf250, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[187] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf251));
    output_handles[188] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf252, {4096L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[189] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf258));
    output_handles[190] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf259, {4096L, 768L}, {768L, 1L}, 0L)));
    output_handles[191] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf261, {4L, 1024L, 50257L}, {51466240L, 50260L, 1L}, 0L)));
    output_handles[192] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf262));
    output_handles[193] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf264));
    output_handles[194] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf266));
    output_handles[195] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_149, {50257L, 768L}, {768L, 1L}, 0L)));
    output_handles[196] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf268));
    output_handles[197] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_145, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[198] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_143, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[199] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf269));
    output_handles[200] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_139, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[201] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf241));
    output_handles[202] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_137, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[203] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf270));
    output_handles[204] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_133, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[205] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_131, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[206] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf271));
    output_handles[207] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_127, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[208] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf220));
    output_handles[209] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_125, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[210] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf272));
    output_handles[211] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_121, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[212] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_119, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[213] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf273));
    output_handles[214] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_115, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[215] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf199));
    output_handles[216] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_113, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[217] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf274));
    output_handles[218] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_109, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[219] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_107, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[220] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf275));
    output_handles[221] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_103, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[222] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf178));
    output_handles[223] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_101, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[224] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf276));
    output_handles[225] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_97, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[226] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_95, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[227] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf277));
    output_handles[228] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_91, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[229] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf157));
    output_handles[230] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_89, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[231] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf278));
    output_handles[232] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_85, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[233] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_83, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[234] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf279));
    output_handles[235] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_79, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[236] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf136));
    output_handles[237] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_77, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[238] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf280));
    output_handles[239] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_73, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[240] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_71, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[241] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf281));
    output_handles[242] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_67, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[243] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf115));
    output_handles[244] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_65, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[245] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf282));
    output_handles[246] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_61, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[247] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_59, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[248] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf283));
    output_handles[249] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_55, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[250] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf94));
    output_handles[251] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_53, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[252] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf284));
    output_handles[253] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_49, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[254] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_47, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[255] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf285));
    output_handles[256] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_43, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[257] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf73));
    output_handles[258] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_41, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[259] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf286));
    output_handles[260] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_37, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[261] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_35, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[262] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf287));
    output_handles[263] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_31, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[264] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf52));
    output_handles[265] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_29, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[266] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf288));
    output_handles[267] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_25, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[268] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_23, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[269] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf289));
    output_handles[270] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_19, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[271] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf31));
    output_handles[272] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_17, {2304L, 768L}, {768L, 1L}, 0L)));
    output_handles[273] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf290));
    output_handles[274] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_13, {768L, 3072L}, {3072L, 1L}, 0L)));
    output_handles[275] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_11, {3072L, 768L}, {768L, 1L}, 0L)));
    output_handles[276] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf291));
    output_handles[277] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_7, {768L, 768L}, {768L, 1L}, 0L)));
    output_handles[278] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf10));
    output_handles[279] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(primals_5, {2304L, 768L}, {768L, 1L}, 0L)));
} // inductor_entry_impl
