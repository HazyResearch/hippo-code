#include <torch/extension.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCPU, #x " must be on CPU")

namespace legt {

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
  TORCH_CHECK(mem.dim() == 3, "legt::euler_forward: mem must have dimension 3");
  TORCH_CHECK(input.dim() == 2, "legt::euler_forward: input must have dimension 2");
  CHECK_DEVICE(mem);
  CHECK_DEVICE(input);
  auto newmem = torch::empty_like(mem);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mem.scalar_type(), "hippolegt::euler_forward", [&] {
    const auto mem_a = mem.accessor<scalar_t, 3>();
    const auto input_a = input.accessor<scalar_t, 2>();
    const scalar_t dt_a = dt;
    auto newmem_a = newmem.accessor<scalar_t, 3>();
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t msz = 0; msz < memsize; ++msz) {
        scalar_t sum = 0;
        for (int64_t n = 0; n < N; ++n) {
          sum += mem_a[b][msz][n];
        }
        scalar_t input_val_dt = input_a[b][msz] * dt_a;
        scalar_t cumsum_even = 0, cumsum_odd = 0;
        for (int64_t i = 0; i < N / 2; ++i) {
          int64_t n_even = 2 * i;
          scalar_t x_even = mem_a[b][msz][n_even];
          newmem_a[b][msz][n_even] = x_even + (dt_a * (-sum + 2 * cumsum_odd) + input_val_dt) * (2 * n_even + 1);
          cumsum_even += x_even;
          int64_t n_odd = 2 * i + 1;
          scalar_t x_odd = mem_a[b][msz][n_odd];
          newmem_a[b][msz][n_odd] = x_odd + (dt_a * (-sum + 2 * cumsum_even) - input_val_dt) * (2 * n_odd + 1);
          cumsum_odd += x_odd;
        }
        if (N % 2 == 1) {  // Last element if there's an extra one
          int64_t n_even = N - 1;
          scalar_t x_even = mem_a[b][msz][n_even];
          newmem_a[b][msz][n_even] = x_even + (dt_a * (-sum + 2 * cumsum_odd) + input_val_dt) * (2 * n_even + 1);
        }
      }
    }
  });
  return newmem;
}

}  // legt

