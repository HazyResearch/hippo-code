#include <vector>
#include <utility>
#include <cmath>
#include <torch/extension.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCPU, #x " must be on CPU")

namespace legs {

at::Tensor euler_forward(const torch::Tensor& mem, const torch::Tensor& input, const float dt) {
  /* newmem = (I + dt A) mem + dt B input
    Parameters:
        mem: (batch_size, memsize, memorder)
        input: (batch_size, memsize)
        dt: float
    Returns:
        newmem: (batch_size, memsize, memorder)
  */
  const auto batch_size = mem.size(0);
  const auto memsize = mem.size(1);
  const auto N = mem.size(2);
  TORCH_CHECK(mem.dim() == 3, "legs::euler_forward: mem must have dimension 3");
  TORCH_CHECK(input.dim() == 2, "legs::euler_forward: input must have dimension 2");
  CHECK_DEVICE(mem);
  CHECK_DEVICE(input);
  auto newmem = torch::empty_like(mem);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mem.scalar_type(), "legs::euler_forward", [&] {
    const auto mem_a = mem.accessor<scalar_t, 3>();
    const auto input_a = input.accessor<scalar_t, 2>();
    const scalar_t dt_a = dt;
    auto newmem_a = newmem.accessor<scalar_t, 3>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t msz = 0; msz < memsize; ++msz) {
        scalar_t input_val_dt = input_a[b][msz] * dt_a;
        scalar_t cumsum = 0;
        for (int64_t n = 0; n < N; ++n) {
          scalar_t x = mem_a[b][msz][n];
          scalar_t sqrt_scale = std::sqrt(2 * n + 1);
          // cumsum += x / sqrt_scale * (2 * n + 1);
          // newmem_a[b][msz][n] = x - dt_a * (cumsum - x / sqrt_scale * n) * sqrt_scale;
          newmem_a[b][msz][n] = x - dt_a * (cumsum * sqrt_scale + x * (n + 1)) + input_val_dt * sqrt_scale;
          cumsum += x * sqrt_scale;
        }
      }
    }
  });
  return newmem;
}

at::Tensor euler_backward(const torch::Tensor& mem, const torch::Tensor& input, const float dt) {
  /* newmem = (I - dt A)^{-1} (mem + dt B input)
    Parameters:
        mem: (batch_size, memsize, memorder)
        input: (batch_size, memsize)
        dt: float
    Returns:
        newmem: (batch_size, memsize, memorder)
  */
  const auto batch_size = mem.size(0);
  const auto memsize = mem.size(1);
  const auto N = mem.size(2);
  TORCH_CHECK(mem.dim() == 3, "legs::euler_backward: mem must have dimension 3");
  TORCH_CHECK(input.dim() == 2, "legs::euler_backward: input must have dimension 2");
  CHECK_DEVICE(mem);
  CHECK_DEVICE(input);
  auto newmem = torch::empty_like(mem);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mem.scalar_type(), "legs::euler_backward", [&] {
    const auto mem_a = mem.accessor<scalar_t, 3>();
    const auto input_a = input.accessor<scalar_t, 2>();
    const scalar_t dt_a = dt;
    auto newmem_a = newmem.accessor<scalar_t, 3>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t msz = 0; msz < memsize; ++msz) {
        scalar_t input_val_dt = input_a[b][msz] * dt_a;
        scalar_t cumsum = 0;
        for (int64_t n = 0; n < N; ++n) {
          scalar_t sqrt_scale = std::sqrt(2 * n + 1);
          scalar_t x = mem_a[b][msz][n] + input_val_dt * sqrt_scale;
          scalar_t y = (x - dt_a * cumsum * sqrt_scale) / (1 + (n + 1) * dt_a);
          newmem_a[b][msz][n] = y;
          cumsum += y * sqrt_scale;
        }
      }
    }
  });
  return newmem;
}

at::Tensor trapezoidal(const torch::Tensor& mem, const torch::Tensor& input, const float dt) {
  /* newmem = (I - dt/2 A)^{-1} ((I + dt/2 A) mem + dt B input)
    Parameters:
        mem: (batch_size, memsize, memorder)
        input: (batch_size, memsize)
        dt: float
    Returns:
        newmem: (batch_size, memsize, memorder)
  */
  const auto batch_size = mem.size(0);
  const auto memsize = mem.size(1);
  const auto N = mem.size(2);
  TORCH_CHECK(mem.dim() == 3, "legs::trapezoidal: mem must have dimension 3");
  TORCH_CHECK(input.dim() == 2, "legs::trapezoidal: input must have dimension 2");
  CHECK_DEVICE(mem);
  CHECK_DEVICE(input);
  auto newmem = torch::empty_like(mem);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mem.scalar_type(), "legs::trapezoidal", [&] {
    const auto mem_a = mem.accessor<scalar_t, 3>();
    const auto input_a = input.accessor<scalar_t, 2>();
    const scalar_t dt_a = dt;
    auto newmem_a = newmem.accessor<scalar_t, 3>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t msz = 0; msz < memsize; ++msz) {
        scalar_t input_val_dt = input_a[b][msz] * dt_a;
        scalar_t cumsum_fwd = 0;
        scalar_t cumsum_bwd = 0;
        for (int64_t n = 0; n < N; ++n) {
          scalar_t x = mem_a[b][msz][n];
          scalar_t sqrt_scale = std::sqrt(2 * n + 1);
          scalar_t out_fwd = x - dt_a / 2 * (cumsum_fwd * sqrt_scale + x * (n + 1)) + input_val_dt * sqrt_scale;
          cumsum_fwd += x * sqrt_scale;
          scalar_t y = (out_fwd - dt_a / 2 * cumsum_bwd * sqrt_scale) / (1 + (n + 1) * dt_a / 2);
          newmem_a[b][msz][n] = y;
          cumsum_bwd += y * sqrt_scale;
        }
      }
    }
  });
  return newmem;
}

at::Tensor function_approx_trapezoidal(const torch::Tensor& input, const int memorder) {
  /*
    Parameters:
        input: (length, )
        memorder: int
    Returns:
        mem: (memorder, )
  */
  const auto length = input.size(0);
  const auto N = memorder;
  TORCH_CHECK(input.dim() == 1, "legs::function_approx_trapezoidal: input must have dimension 1");
  CHECK_DEVICE(input);
  auto mem = torch::zeros({N}, torch::dtype(input.dtype()).device(input.device()));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "legs::function_approx_trapezoidal", [&] {
    auto mem_a = mem.accessor<scalar_t, 1>();
    const auto input_a = input.accessor<scalar_t, 1>();
    mem_a[0] = input_a[0];
    for (int64_t t = 1; t < length; ++t) {
      const scalar_t dt = 1.0 / t;
      scalar_t input_val_dt = input_a[t] * dt;
      scalar_t cumsum_fwd = 0;
      scalar_t cumsum_bwd = 0;
      for (int64_t n = 0; n < N; ++n) {
        scalar_t x = mem_a[n];
        scalar_t sqrt_scale = std::sqrt(2 * n + 1);
        scalar_t out_fwd = x - dt / 2 * (cumsum_fwd * sqrt_scale + x * (n + 1)) + input_val_dt * sqrt_scale;
        cumsum_fwd += x * sqrt_scale;
        scalar_t y = (out_fwd - dt / 2 * cumsum_bwd * sqrt_scale) / (1 + (n + 1) * dt / 2);
        mem_a[n] = y;
        cumsum_bwd += y * sqrt_scale;
      }
    }
  });
  return mem;
}

}  // legs
