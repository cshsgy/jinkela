// kintera
#include "check_resize.hpp"

namespace kintera {

torch::Tensor check_resize(torch::Tensor tensor, at::IntArrayRef desired_shape,
                           const torch::TensorOptions& desired_options) {
  // Check shape and options
  bool shape_matches = tensor.sizes().equals(desired_shape);
  bool options_match = tensor.options() == desired_options;

  if (shape_matches && options_match) {
    return tensor;  // No-op
  }

  // Resize (create new tensor with correct shape and options)
  return torch::empty(desired_shape, desired_options);
}

}  // namespace kintera
