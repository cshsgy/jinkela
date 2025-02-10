#include <torch/torch.h>
#include "utils/units.h"
#include "kinetics/ReactionRate.h"
namespace kintera
{

class ArrheniusBase : public ReactionRate
{
public:
    ArrheniusBase() {}

    //! Constructor.
    /*!
     *  @param A  Pre-exponential factor. The unit system is (kmol, m, s); actual units
     *      depend on the reaction order and the dimensionality (surface or bulk).
     *  @param b  Temperature exponent (non-dimensional)
     *  @param Ea  Activation energy in energy units [J/kmol]
     */
    ArrheniusBase(double A, double b, double Ea);

    ArrheniusBase(const AnyValue& rate);
    explicit ArrheniusBase(const AnyMap& node);
    void setRateParameters(const AnyValue& rate);
    void setParameters(const AnyMap& node) override;

    virtual double preExponentialFactor() const {
        return m_A;
    }
    virtual double temperatureExponent() const {
        return m_b;
    }
    virtual double activationEnergy() const {
        return m_Ea_R * GasConstant;
    }
    double order() const {
        return m_order;
    }

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

class ArrheniusRate : public ArrheniusBase
{
public:
    using ArrheniusBase::ArrheniusBase; // inherit constructors

    const string type() const override {
        return "Arrhenius";
    }

    //! Evaluate reaction rate, I will leave the temperature evaluation to the calling function
    torch::Tensor evalRate(torch::Tensor T, torch::Tensor P) const {
        return m_A * torch::exp(m_b * torch::log(T) - m_Ea_R * 1.0 / T);
    }

    torch::Tensor ddTRate(torch::Tensor T, torch::Tensor P) const {
        return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
    }
};

}

#endif