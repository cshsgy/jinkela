#pragma once

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "ReactionRate.h"
#include "../utils/constants.h"

namespace kintera
{

class ArrheniusRate : public ReactionRate
{
public:
    ArrheniusRate() {}

    //! Constructor.
    /*!
     *  @param A  Pre-exponential factor. The unit system is (kmol, m, s); actual units
     *      depend on the reaction order and the dimensionality (surface or bulk).
     *  @param b  Temperature exponent (non-dimensional)
     *  @param Ea  Activation energy in energy units [J/kmol]
     */
    ArrheniusRate(double A, double b, double Ea);
    ArrheniusRate(const YAML::Node& node);

    virtual const std::string type() const override {
        return "Arrhenius";
    }
    
    double order() const {
        return m_order;
    }
    torch::Tensor evalRate(torch::Tensor T, torch::Tensor P) const override;
    std::string rateSummary() const override;
    torch::Tensor ddTRate(torch::Tensor T, torch::Tensor P) const;

protected:
    double m_A = NAN; //!< Pre-exponential factor
    double m_b = NAN; //!< Temperature exponent
    double m_Ea_R = 0.; //!< Activation energy (in temperature units)
    double m_E4_R = 0.; //!< Optional 4th energy parameter (in temperature units)
    double m_logA = NAN; //!< Logarithm of pre-exponential factor
    double m_order = NAN; //!< Reaction order
    string m_A_str = "A"; //!< The string for the pre-exponential factor
    string m_b_str = "b"; //!< The string for temperature exponent
    string m_Ea_str = "Ea"; //!< The string for activation energy
    string m_E4_str = ""; //!< The string for an optional 4th parameter
};

}