#pragma once

#include <torch/torch.h>
#include <string>

namespace kintera
{


class Reaction;
class ReactionRate
{
public:
    ReactionRate() {}
    virtual ~ReactionRate() = default;
    // Copy assignment operator and copy constructor are deleted

    virtual const std::string type() const = 0;
    virtual std::string rateSummary() const = 0;
    virtual torch::Tensor evalRate(torch::Tensor T, torch::Tensor P) const = 0;
};

}

