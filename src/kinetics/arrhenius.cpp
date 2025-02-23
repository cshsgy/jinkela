// kintera
#include "arrhenius.hpp"

#include <kintera/utils/constants.hpp>

namespace kintera {

ArrheniusImpl::ArrheniusImpl(RateOptions const& options_) : options(options_) {
  reset();
}

void ArrheniusImpl::pretty_print(std::ostream& os) const {
  os << "Arrhenius Rate: A = " << options.A() << ", b = " << options.b()
     << ", Ea = " << options.Ea_R() * constants::GasConstant << " J/kmol";
}

torch::Tensor ArrheniusImpl::forward(
    torch::Tensor T, std::map<std::string, torch::Tensor> const& other) {
  return log(options.A()) + options.b() * T.log() - options.Ea_R() * 1.0 / T;
}

/*torch::Tensor ArrheniusRate::ddTRate(torch::Tensor T, torch::Tensor P) const {
    return (m_Ea_R * 1.0 / T + m_b) * 1.0 / T;
}*/

}  // namespace kintera
