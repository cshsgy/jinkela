#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "thermo.hpp"

namespace kintera {

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op);

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        ThermoOptions const& op);

torch::Tensor eval_compress_z(torch::Tensor temp, torch::Tensor conc,
                              ThermoOptions const& op);

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            ThermoOptions const& op);

}  // namespace kintera
