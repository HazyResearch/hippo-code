#include <torch/extension.h>

namespace legs {
  at::Tensor euler_forward(const torch::Tensor& mem, const torch::Tensor& input, const float dt);
  at::Tensor euler_backward(const torch::Tensor& mem, const torch::Tensor& input, const float dt);
  at::Tensor trapezoidal(const torch::Tensor& mem, const torch::Tensor& input, const float dt);
  at::Tensor function_approx_trapezoidal(const torch::Tensor& input, const int memorder);
}

namespace legt {
  at::Tensor euler_forward(const torch::Tensor& mem, const torch::Tensor& input, const float dt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("legs_euler_forward", &legs::euler_forward, "Euler forward for Hippo-LegS");
  m.def("legs_euler_backward", &legs::euler_backward, "Euler backward for Hippo-LegS");
  m.def("legs_trapezoidal", &legs::trapezoidal, "Trapezoidal for Hippo-LegS");
  m.def("legs_function_approx_trapezoidal", &legs::function_approx_trapezoidal, "Function approx trapezoidal for Hippo-LegS");

  m.def("legt_euler_forward", &legt::euler_forward, "Euler forward for Hippo-LegT");
}
