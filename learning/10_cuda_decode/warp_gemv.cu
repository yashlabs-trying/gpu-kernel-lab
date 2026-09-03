#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Independent warps own independent output rows. All 32 lanes participate,
// including lanes with no valid K elements; therefore FULL_MASK is valid.
__device__ float warp_sum(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
  return v;
}

template<bool QUANT, bool SWIGLU>
__global__ void projection(const c10::BFloat16* x, const void* weight,
    const float* scales, const float* inverse, c10::BFloat16* y,
    int n, int k, int group) {
  int lane = threadIdx.x & 31;
  int warps = blockDim.x / 32;
  int outputs = SWIGLU ? n / 2 : n;
  for (int row = blockIdx.x * warps + threadIdx.x / 32;
       row < outputs; row += gridDim.x * warps) {
    float gate = 0.f, up = 0.f;
    for (int col = lane; col < k; col += 32) {
      float activation = float(x[col]);
      float w, u = 0.f;
      if constexpr (QUANT) {
        activation *= inverse[col];
        w = float(static_cast<const int8_t*>(weight)[int64_t(row)*k+col])
          * scales[int64_t(row)*(k/group)+col/group];
        if constexpr (SWIGLU)
          u = float(static_cast<const int8_t*>(weight)[int64_t(row+outputs)*k+col])
            * scales[int64_t(row+outputs)*(k/group)+col/group];
      } else {
        w = float(static_cast<const c10::BFloat16*>(weight)[int64_t(row)*k+col]);
        if constexpr (SWIGLU)
          u = float(static_cast<const c10::BFloat16*>(weight)[int64_t(row+outputs)*k+col]);
      }
      gate = fmaf(w, activation, gate);
      if constexpr (SWIGLU) up = fmaf(u, activation, up);
    }
    gate = warp_sum(gate);
    if constexpr (SWIGLU) up = warp_sum(up);
    if (lane == 0) {
      // Preserve BF16 projection and SiLU output boundaries of eager Qwen MLP.
      if constexpr (SWIGLU) {
        gate = float(c10::BFloat16(gate));
        up = float(c10::BFloat16(up));
        float activated = float(c10::BFloat16(gate / (1.f + expf(-gate))));
        y[row] = c10::BFloat16(activated * up);
      } else y[row] = c10::BFloat16(gate);
    }
  }
}

torch::Tensor project(torch::Tensor x, torch::Tensor w, torch::Tensor scales,
    torch::Tensor inverse, int64_t group, int64_t warps, bool persistent,
    bool swiglu) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device()==w.device(), "same CUDA device required");
  TORCH_CHECK(x.scalar_type()==torch::kBFloat16 && x.dim()==1 && x.is_contiguous(), "x: contiguous BF16 vector");
  TORCH_CHECK(w.dim()==2 && w.is_contiguous() && w.size(0)>0 && w.size(1)>0 && w.size(1)==x.numel(), "w: contiguous nonempty [N,K]");
  TORCH_CHECK(w.size(0)<INT_MAX && w.size(1)<INT_MAX, "dimensions exceed kernel indexing limits");
  TORCH_CHECK(warps==1 || warps==4 || warps==8, "warps must be 1,4,8");
  TORCH_CHECK(!swiglu || w.size(0)%2==0, "gate/up requires two equal row sets");
  bool quant = w.scalar_type()==torch::kInt8;
  TORCH_CHECK(quant || w.scalar_type()==torch::kBFloat16, "BF16 or INT8 weights required");
  if (quant) {
    TORCH_CHECK(group>0 && w.size(1)%group==0, "group must divide K");
    TORCH_CHECK(scales.device()==x.device() && inverse.device()==x.device(), "scales must share device");
    TORCH_CHECK(scales.scalar_type()==torch::kFloat32 && inverse.scalar_type()==torch::kFloat32 && scales.is_contiguous() && inverse.is_contiguous(), "FP32 contiguous scaling buffers required");
    TORCH_CHECK(scales.dim()==2 && scales.size(0)==w.size(0) && scales.size(1)==w.size(1)/group && inverse.dim()==1 && inverse.numel()==x.numel(), "scaling shapes incorrect");
  }
  c10::cuda::CUDAGuard guard(x.device());
  int n=w.size(0), k=w.size(1), outputs=swiglu?n/2:n;
  auto y=torch::empty({outputs}, x.options());
  int blocks=(outputs+warps-1)/warps;
  if (persistent) {
    int sms;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
    blocks=std::min(blocks, sms*2);
  }
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
  const float* s=quant?scales.data_ptr<float>():nullptr;
  const float* inv=quant?inverse.data_ptr<float>():nullptr;
  #define LAUNCH(Q,S) projection<Q,S><<<blocks,warps*32,0,stream>>>(x.data_ptr<c10::BFloat16>(),w.data_ptr(),s,inv,y.data_ptr<c10::BFloat16>(),n,k,group)
  if (quant) { if (swiglu) { LAUNCH(true,true); } else { LAUNCH(true,false); } }
  else { if (swiglu) { LAUNCH(false,true); } else { LAUNCH(false,false); } }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("project", &project); }
