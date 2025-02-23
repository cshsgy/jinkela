// kintera
#include "photolysis.hpp"
#include "reaction.hpp"
#include "utils/parse_comp_string.hpp"

// torch
#include <torch/torch.h>

// C++
#include <string>
#include <vector>
#include <map>

// yaml-cpp
#include <yaml-cpp/yaml.h>

namespace kintera {

// bool PhotolysisData::update(const ThermoPhase& thermo, const Kinetics& kin) {
//   bool changed = false;
//   double T = thermo.temperature();
//   if (T != temperature) {
//     update(T);
//     changed = true;
//   }

//   if (wavelength.empty()) {
//     size_t nwave = kin.nWavelengths();

//     wavelength.resize(nwave);
//     actinicFlux.resize(nwave);

//     kin.getWavelength(wavelength.data());
//     kin.getActinicFlux(actinicFlux.data());
//     changed = true;
//   } else if (kin.hasNewActinicFlux()) {
//     kin.getActinicFlux(actinicFlux.data());
//     changed = true;
//   }

//   return changed;
// }

// PhotolysisBase::PhotolysisBase(torch::Tensor temp,
//                                torch::Tensor wavelength,
//                                const std::vector<std::string>& branches,
//                                torch::Tensor xsection)
//     : m_crossSection(xsection) {
//   m_ntemp = temp.size();
//   m_nwave = wavelength.size();

//   m_temp_wave_grid.resize(m_ntemp + m_nwave);
//   for (size_t i = 0; i < m_ntemp; i++) {
//     m_temp_wave_grid[i] = temp[i];
//   }

//   for (size_t i = 0; i < m_nwave; i++) {
//     m_temp_wave_grid[m_ntemp + i] = wavelength[i];
//   }

//   for (auto const& branch : branches) {
//     m_branch.push_back(parseCompString(branch));
//   }

//   if (m_ntemp * m_nwave * branches.size() != m_crossSection.size()) {
//     throw CanteraError(
//         "PhotolysisBase::PhotolysisBase",
//         "Cross-section data size does not match the temperature, "
//         "wavelength, and branch grid sizes.");
//   }

//   m_valid = true;
// }

void PhotolysisBase::setRateParameters(const YAML::Node& rate,
                                       map<string, int> const& branch_map) {
  if (rate["resolution"]) {
    double resolution = rate["resolution"].as<double>();
    if (resolution <= 0.0) {
      throw std::runtime_error("In PhotolysisBase::setRateParameters, Resolution must be positive.");
    }
  }

  if (rate["scale"]) {
    vector<double> scales;
    if (rate["scale"].is<double>()) {
      for (size_t i = 0; i < m_branch.size(); i++) {
        scales[i] = rate["scale"].as<double>();
      }
    } else {
      for (auto const& [branch, scale] : rate["scale"]()) {
        auto it = branch_map.find(branch);
        if (it == branch_map.end()) {
          throw std::runtime_error("In PhotolysisBase::setRateParameters, Branch '" + branch + "' not found");
        }

        scales[it->second] = scale.as<double>();
      }
    }
  }
}

void PhotolysisBase::setParameters(const YAML::Node& node) {
  std::map<std::string, int> branch_map;
  std::pair<torch::Tensor, torch::Tensor> result;
  torch::Tensor temperature;

  ReactionRate::setParameters(node, rate_units);

  // set up a dummy reaction to parse the reaction equation
  Reaction rtmp = Reaction(node["equation"].asString());

  if (rtmp.reactants.size() != 1 || rtmp.reactants.begin()->second != 1) {
    throw std::runtime_error(
        "In PhotolysisBase::setParameters, Photolysis reaction must have one reactant with stoichiometry 1.");
  }

  // b0 is reserved for the photoabsorption cross section
  branch_map["b0"] = 0;
  m_branch.push_back(rtmp.reactants);

  if (node["branches"]) {
    for (auto const& branch : node["branches"]) {
      std::string branch_name = branch["name"].as<std::string>();

      // check duplicated branch name
      if (branch_map.find(branch_name) != branch_map.end()) {
        throw std::runtime_error(
            "In PhotolysisBase::setParameters, Duplicated branch name '" +
            branch_name + "'.");
      }

      branch_map[branch_name] = m_branch.size();
      m_branch.push_back(parse_comp_string(branch["product"].as<std::string>()));
    }
  } else if (rtmp.products != rtmp.reactants) {  // this is not photoabsorption
    m_branch.push_back(rtmp.products);
  }

  if (node["cross-section"]) {
    for (auto const& data : node["cross-section"]) {
      auto format = data["format"].as<std::string>();
      if (format != "YAML" && format != "VULCAN" && format != "KINETICS7") {
        throw std::runtime_error(
            "In PhotolysisBase::setParameters, unsupported cross-section format '" +
            format + "'.");
      }
      
      auto temp = data["temperature-range"].as<std::vector<double>>(2, 2);
      if (temp[0] >= temp[1]) {
        throw std::runtime_error(
            "In PhotolysisBase::setParameters, Temperature range must be "
            "strictly increasing.");
      }

      if (temperature.empty()) {
        temperature = temp;
      } else {
        if (temperature.back() < temp.front()) {
          throw std::runtime_error(
              "In PhotolysisBase::setParameters, Temperature ranges has gap "
              "in between.");
        }

        temperature.pop_back();
        temperature.insert(temperature.end(), temp.begin(), temp.end());
      }

      if (format == "YAML") {
        for (auto const& entry : data["data"]) {
          result.first.push_back(entry[0]);
          result.second.push_back(entry[1]);
        }
      } else if (format == "VULCAN") {
        auto files = data["filenames"].as<std::vector<std::string>>();
        result = load_xsection_vulcan(files, m_branch);
      } else if (format == "KINETICS7") {
        auto files = data["filenames"].as<std::vector<std::string>>();
        result = load_xsection_kinetics7(files, m_branch);
      } else {
        throw std::runtime_error(
            "In PhotolysisBase::setParameters, unsupported cross-section format '" +
            format + "'.");
      }
    }
  }

  auto wavelength = result.first;
  auto xsection = result.second;

  std::cout << "temperature shape: " << temperature.sizes() << std::endl;
  std::cout << "wavelength shape: " << wavelength.sizes() << std::endl;
  std::cout << "xsection shape: " << xsection.sizes() << std::endl;

  m_ntemp = temperature.size(0);
  m_nwave = wavelength.size(0);
  m_temp_wave_grid = torch::zeros({m_ntemp + m_nwave});

  for (size_t i = 0; i < m_ntemp; i++) {
    m_temp_wave_grid[i] = temperature[i];
  }

  for (size_t i = 0; i < m_nwave; i++) {
    m_temp_wave_grid[m_ntemp + i] = wavelength[i];
  }

  // TODO: test this support for multiple temperature ranges
  if (m_crossSection.empty()) {
    m_crossSection = xsection;
  } else {
    m_crossSection = torch::cat({m_crossSection, xsection}, 0);
  }

  if (node["rate-constant"]) {
    setRateParameters(node["rate-constant"], branch_map);
  }

  /* debug
  std::cout << "number of temperature: " << m_ntemp << std::endl;
  std::cout << "number of wavelength: " << m_nwave << std::endl;
  std::cout << "number of branches: " << m_branch.size() << std::endl;
  for (auto const& branch : branch_map) {
    std::cout << "branch: " << branch.first << std::endl;
    for (auto const& [name, stoich] : m_branch[branch.second]) {
      std::cout << name << " " << stoich << std::endl;
    }
  }
  std::cout << "number of cross-section: " << m_crossSection.size() <<
  std::endl;
  */

  if (m_ntemp * m_nwave * m_branch.size() != m_crossSection.size()) {
    throw CanteraError(
        "PhotolysisBase::PhotolysisBase",
        "Cross-section data size does not match the temperature, "
        "wavelength, and branch grid sizes.");
  }

  m_valid = true;
}


void PhotolysisBase::validate(string const& equation, Kinetics const& kin) {
  if (!valid()) {
    throw InputFileError("PhotolysisBase::validate", m_input,
                         "Rate object for reaction '{}' is not configured.",
                         equation);
  }

  std::vector<std::string> tokens;
  tokenizeString(equation, tokens);
  auto reactor_comp = parseCompString(tokens[0] + ":1");

  // set up a dummy reaction to parse the reaction equation
  Reaction rtmp;
  parseReactionEquation(rtmp, equation, m_input, nullptr);

  std::set<std::string> species_from_equation, species_from_branches;

  species_from_equation.insert(rtmp.reactants.begin()->first);
  for (auto const& [name, stoich] : rtmp.products) {
    species_from_equation.insert(name);
  }

  species_from_branches.insert(rtmp.reactants.begin()->first);
  for (auto const& branch : m_branch) {
    // create a Arrhenius reaction placeholder to check balance
    Reaction rtmp(reactor_comp, branch, newReactionRate("Arrhenius"));
    rtmp.reversible = false;
    rtmp.checkSpecies(kin);
    rtmp.checkBalance(kin);
    for (auto const& [name, stoich] : branch) {
      species_from_branches.insert(name);
    }
  }

  if (species_from_equation != species_from_branches) {
    throw InputFileError(
        "PhotolysisBase::validate", m_input,
        "Reaction '{}' has different products than the photolysis branches.",
        equation);
  }
}

vector<double> PhotolysisBase::getCrossSection(double temp,
                                               double wavelength) const {
  if (m_crossSection.empty()) {
    return {0.};
  }

  std::vector<double> cross(m_branch.size());

  double coord[2] = {temp, wavelength};
  size_t len[2] = {m_ntemp, m_nwave};

  interpn(cross.data(), coord, m_crossSection.data(), m_temp_wave_grid.data(),
          len, 2, m_branch.size());

  return cross;
}

double PhotolysisRate::evalFromStruct(PhotolysisData const& data) {
  double wmin = m_temp_wave_grid[m_ntemp];
  double wmax = m_temp_wave_grid.back();

  if (m_crossSection.empty() || wmin > data.wavelength.back() ||
      wmax < data.wavelength.front()) {
    for (size_t n = 1; n < m_branch.size(); n++)
      for (auto const& [name, stoich] : m_branch[n]) m_net_products[name] = 0.;

    return 0.;
  }

  /* debug
  std::cout << "wavelength data range: " << wmin << " " << wmax << std::endl;
  std::cout << "temperature = " << data.temperature << std::endl;
  std::cout << "wavelength = " << std::endl;
  for (auto w : data.wavelength) {
    std::cout << w << std::endl;
  }
  std::cout << "nbranch = " << m_branch.size() << std::endl;*/

  double* cross1 = new double[m_branch.size()];
  double* cross2 = new double[m_branch.size()];

  double coord[2] = {data.temperature, data.wavelength[0]};
  size_t len[2] = {m_ntemp, m_nwave};

  // debug
  // std::cout << "coord = " << coord[0] << " " << coord[1] << std::endl;

  interpn(cross1, coord, m_crossSection.data(), m_temp_wave_grid.data(), len, 2,
          m_branch.size());

  double total_rate = 0.0;
  // prevent division by zero
  double eps = 1.e-30;
  double total_rate_eps = 0.;

  // first branch is photoabsorption
  for (size_t n = 1; n < m_branch.size(); n++)
    for (auto const& [name, stoich] : m_branch[n]) m_net_products[name] = 0.;

  /* debug
  std::cout << m_crossSection[0] << std::endl;
  std::cout << m_crossSection[1] << std::endl;
  std::cout << m_crossSection[2] << std::endl;
  std::cout << "grid = " << std::endl;
  for (size_t n = 0; n < m_temp_wave_grid.size(); n++) {
    std::cout << m_temp_wave_grid[n] << std::endl;
  }*/

  for (size_t i = 0; i < data.wavelength.size() - 1; ++i) {
    // debug
    // std::cout << "wavelength = " << data.wavelength[i] << " " <<
    // data.wavelength[i+1] << std::endl;
    coord[1] = data.wavelength[i + 1];
    interpn(cross2, coord, m_crossSection.data(), m_temp_wave_grid.data(), len,
            2, m_branch.size());

    // photodissociation only
    for (size_t n = 1; n < m_branch.size(); n++) {
      double rate = 0.5 * (data.wavelength[i + 1] - data.wavelength[i]) *
                    (cross1[n] * data.actinicFlux[i] +
                     cross2[n] * data.actinicFlux[i + 1]);

      // debug
      // std::cout << "actinic flux [ " << i << "] = " << data.actinicFlux[i] <<
      // " " << data.actinicFlux[i+1] << std::endl; std::cout << "cross section
      // [ " << n << "] = " << cross1[n] << " " << cross2[n] << std::endl;

      for (auto const& [name, stoich] : m_branch[n]) {
        m_net_products.at(name) += (rate + eps) * stoich;
      }
      total_rate += rate;
      total_rate_eps += rate + eps;

      cross1[n] = cross2[n];
    }
  }

  for (auto& [name, stoich] : m_net_products) stoich /= total_rate_eps;

  /* debug
  for (auto const& [name, stoich] : m_net_products)
    std::cout << name << " " << stoich << std::endl;
  std::cout << "photodissociation rate: " << total_rate << std::endl;*/

  delete[] cross1;
  delete[] cross2;

  return total_rate;
}

}  // namespace Cantera
