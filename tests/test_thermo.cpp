// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/constants.h>

#include <kintera/eos/equation_of_state.hpp>
#include <kintera/thermo/eval_uh.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/thermo/thermo_formatter.hpp>

// tests
#include "device_testing.hpp"

using namespace kintera;
using namespace torch::indexing;

TEST_P(DeviceTest, thermo_y) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny =
      thermo->options.vapor_ids().size() + thermo->options.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  ////////// Testing Y->X conversion //////////
  auto xfrac = thermo->compute("Y->X", {yfrac});
  EXPECT_EQ(torch::allclose(
                xfrac.sum(-1),
                torch::ones({1, 2, 3}, torch::device(device).dtype(dtype)),
                /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  ////////// Testing DY->V conversion //////////
  auto rho = torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto dens = thermo->compute("DY->V", {rho, yfrac});
  EXPECT_EQ(torch::allclose(rho, dens.sum(-1),
                            /*rtol=*/1e-4, /*atol=*/1e-4),
            true);

  ////////// Testing V->Y conversion //////////
  auto yfrac2 = thermo->compute("V->Y", {dens});
  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);

  ////////// Testing VT->P conversion //////////
  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = thermo->compute("VT->P", {dens, temp});
  std::cout << "pres = " << pres << std::endl;

  ////////// Testing PV->T conversion //////////
  auto temp2 = thermo->compute("PV->T", {pres, dens});
  EXPECT_EQ(torch::allclose(temp, temp2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_x) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  ThermoX thermo(op_thermo);
  thermo->to(device, dtype);

  int ny =
      thermo->options.vapor_ids().size() + thermo->options.cloud_ids().size();
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo->compute("X->Y", {xfrac});
  std::cout << "yfrac = " << yfrac << std::endl;
}

TEST_P(DeviceTest, thermo_xy) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();
  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto yfrac = thermo_x->compute("X->Y", {xfrac});
  auto xfrac2 = thermo_y->compute("Y->X", {yfrac});

  EXPECT_EQ(torch::allclose(xfrac, xfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, thermo_yx) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  auto yfrac2 = thermo_x->compute("X->Y", {xfrac});

  EXPECT_EQ(torch::allclose(yfrac, yfrac2, /*rtol=*/1e-4, /*atol=*/1e-4), true);
}

TEST_P(DeviceTest, evals) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");

  ThermoY thermo(op_thermo);
  thermo->to(device, dtype);

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho = torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto conc = thermo->compute("DY->C", {rho, yfrac});
  auto temp = 300. * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  auto cv_R = eval_cv_R(temp, conc, thermo->options);
  std::cout << "cv_R = " << cv_R << std::endl;

  throw std::runtime_error("Test not implemented yet");
}

/*TEST_P(DeviceTest, eng_pres) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml");
  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() +
           thermo_y->options.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto temp = 200.0 * torch::ones({1}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1}, torch::device(device).dtype(dtype));

  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  auto rho = thermo_x->compute("TPX->D", {temp, pres, xfrac});

  auto intEng = thermo_y->compute("DPY->U", {rho, pres, yfrac});
  auto pres2 = thermo_y->compute("DUY->P", {rho, intEng, yfrac});
  EXPECT_EQ(torch::allclose(pres, pres2, 1e-4, 1e-4), true);

  auto temp2 = thermo_y->compute("DPY->T", {rho, pres, yfrac});
  EXPECT_EQ(torch::allclose(temp, temp2, 1e-4, 1e-4), true);
}

TEST_P(DeviceTest, equilibrate_tp) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  int ny = op_thermo.vapor_ids().size() + op_thermo.cloud_ids().size();

  ThermoX thermo_x(op_thermo);
  thermo_x->to(device, dtype);

  auto xfrac =
      torch::zeros({1, 2, 3, 1 + ny}, torch::device(device).dtype(dtype));

  for (int i = 0; i < ny; ++i) xfrac.select(-1, i + 1) = 0.01 * (i + 1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto temp =
      200.0 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));

  std::cout << "xfrac before = " << xfrac[0][0][0] << std::endl;
  thermo_x->forward(temp, pres, xfrac);
  std::cout << "xfrac after = " << xfrac[0][0][0] << std::endl;
}

TEST_P(DeviceTest, equilibrate_uv) {
  auto op_thermo = ThermoOptions::from_yaml("jupiter.yaml").max_iter(10);
  std::cout << fmt::format("{}", op_thermo) << std::endl;

  ThermoY thermo_y(op_thermo);
  thermo_y->to(device, dtype);

  int ny = thermo_y->options.vapor_ids().size() +
           thermo_y->options.cloud_ids().size();
  auto yfrac = torch::zeros({ny, 1, 2, 3}, torch::device(device).dtype(dtype));
  for (int i = 0; i < ny; ++i) yfrac[i] = 0.01 * (i + 1);

  auto rho = 0.1 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto pres = 1.e5 * torch::ones({1, 2, 3}, torch::device(device).dtype(dtype));
  auto intEng = thermo_y->compute("DPY->U", {rho, pres, yfrac});
  std::cout << "intEng = " << intEng << std::endl;
  std::cout << "pres before = " << pres << std::endl;
  std::cout << "yfrac before = " << yfrac.index({Slice(), 0, 0, 0})
            << std::endl;
  std::cout << "temp before = "
            << thermo_y->compute("DPY->T", {rho, pres, yfrac}) << std::endl;

  thermo_y->forward(rho, intEng, yfrac);
  std::cout << "yfrac after = " << yfrac.index({Slice(), 0, 0, 0}) << std::endl;
  auto pres2 = thermo_y->compute("DUY->P", {rho, intEng, yfrac});
  std::cout << "pres after = " << pres2 << std::endl;
  auto intEng2 = thermo_y->compute("DPY->U", {rho, pres2, yfrac});
  std::cout << "intEng after = " << intEng2 << std::endl;
  std::cout << "temp after = "
            << thermo_y->compute("DPY->T", {rho, pres2, yfrac}) << std::endl;

  EXPECT_EQ(torch::allclose(intEng, intEng2, 1e-4, 1e-4), true);
}*/

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
