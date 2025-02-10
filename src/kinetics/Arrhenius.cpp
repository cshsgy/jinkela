#include "kintera/kinetics/Arrhenius.h"
#include "kintera/utils/constants.h"
#include "kintera/kinetics/ReactionRate.h"

#include <string>

namespace kintera
{

ArrheniusBase::ArrheniusBase(double A, double b, double Ea)
    : m_A(A)
    , m_b(b)
    , m_Ea_R(Ea / GasConstant)
{
    if (m_A > 0.0) {
        m_logA = std::log(m_A);
    }
    m_valid = true;
}

ArrheniusBase::ArrheniusBase(const AnyValue& rate)
{
    setRateParameters(rate);
}

ArrheniusBase::ArrheniusBase(const AnyMap& node)
{
    setParameters(node);
}

void ArrheniusBase::setRateParameters(const AnyValue& rate)
{
    m_Ea_R = 0.; // assume zero if not provided
    m_E4_R = 0.; // assume zero if not provided
    if (rate.empty()) {
        m_A = NAN;
        m_b = NAN;
        m_logA = NAN;
        setRateUnits(Units(0.));
        return;
    }

    if (rate.is<AnyMap>()) {

        auto& rate_map = rate.as<AnyMap>();
        m_A = rate_map[m_A_str].asDouble();
        m_b = rate_map[m_b_str].asDouble();
        if (rate_map.hasKey(m_Ea_str)) {
            m_Ea_R = rate_map[m_Ea_str].asDouble();
        }
        if (rate_map.hasKey(m_E4_str)) {
            m_E4_R = rate_map[m_E4_str].asDouble();
        }
    } else {
        auto& rate_vec = rate.asVector<AnyValue>(2, 4);
        m_A = rate_vec[0].asDouble();
        m_b = rate_vec[1].asDouble();
        if (rate_vec.size() > 2) {
            m_Ea_R = rate_vec[2].asDouble();
        }
        if (rate_vec.size() > 3) {
            m_E4_R = rate_vec[3].asDouble();
        }
    }
    if (m_A > 0.0) {
        m_logA = std::log(m_A);
    }
}

void ArrheniusBase::setParameters(const AnyMap& node)
{
    ReactionRate::setParameters(node);
    if (!node.hasKey("rate-constant")) {
        setRateParameters(AnyValue());
        return;
    }
    setRateParameters(node["rate-constant"]);
}

}