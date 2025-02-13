#pragma once

#include <string>
#include <vector>
#include "kintera/reaction.hpp"

namespace kintera {

std::vector<Reaction> parse_reactions_yaml(const std::string& filename);

} // namespace kintera

