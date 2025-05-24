#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "kinetic_rate.hpp"
#include "rate_constant.hpp"
#include "species_rate.hpp"

namespace kintera {

class ReactionSystem{
public:
    ReactionSystem(const kintera::KineticRate& kinetics, 
                      const kintera::RateConstant& rate_const)
        : kinetics_(kinetics), rate_const_(rate_const) {}

    torch::Tensor calculate_rates(const torch::Tensor& C, 
                                const torch::Tensor& P,
                                const torch::Tensor& Temp) {
        std::map<std::string, torch::Tensor> other;
        other["pres"] = P;
        auto log_rc = rate_const_->forward(Temp, other);
        return kinetics_->forward(C, log_rc);
    }

    torch::Tensor get_stoichiometry_matrix() const {
        return kinetics_->stoich;
    }

    torch::Tensor calculate_jacobian(const torch::Tensor& C,
                                   const torch::Tensor& P,
                                   const torch::Tensor& Temp) {
        std::map<std::string, torch::Tensor> other;
        other["pres"] = P;
        auto log_rc = rate_const_->forward(Temp, other);
        auto rates = kinetics_->forward(C, log_rc);
        return kintera::species_jacobian(C, rates, kinetics_->stoich, kinetics_->order);
    }

private:
    kintera::KineticRate kinetics_;
    kintera::RateConstant rate_const_;
};

} // namespace kintera 