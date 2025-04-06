// torch
#include <torch/torch.h>

// kintera
#include <kintera/eos/equation_of_state.hpp>

int main(int argc, char* argv[]) {
  auto peos = std::make_shared<kintera::EquationOfStateImpl>();
  auto rho = 0.1 * torch::randn({2, 3});
  auto pres = torch::rand({2, 3}) * 1.e5;
  auto temp = peos->get_temp(rho, pres);
  auto intEng = peos->get_intEng(rho, pres);

  auto gamma = peos->get_gamma(rho, intEng);

  std::cout << "temp = " << temp << std::endl;
  std::cout << "intEng = " << intEng << std::endl;
  std::cout << "gamma = " << gamma << std::endl;
}
