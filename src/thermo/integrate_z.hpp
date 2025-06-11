#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "thermo.hpp"

namespace kintera {

void integrate_z_(torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
                  double dz, std::string method, double grav, double adTdz,
                  ThermoX &thermo);

}  // namespace kintera
