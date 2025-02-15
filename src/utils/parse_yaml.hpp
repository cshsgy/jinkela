#pragma once

// C/C++
#include <string>
#include <vector>

// torch
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/reaction.hpp>

namespace kintera {

std::map<Reaction, torch::nn::AnyModule> parse_reactions_yaml(
    const std::string& filename);

}  // namespace kintera
