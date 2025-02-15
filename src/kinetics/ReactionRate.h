#pragma once

#include <torch/torch.h>
#include <string>
#include <memory>

namespace kintera
{


class Reaction;
class ReactionRate
{
public:
    ReactionRate() {}
    virtual ~ReactionRate() = default;
    // Copy assignment operator and copy constructor are deleted

    virtual std::unique_ptr<ReactionRate> clone() const = 0;
    virtual const std::string type() const = 0;
    virtual std::string rateSummary() const = 0;
    virtual torch::Tensor evalRate(torch::Tensor T, torch::Tensor P) const = 0;
};

}

