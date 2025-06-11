#pragma once

// fmt
#include <fmt/format.h>

// kintera
#include <kintera/kintera_formatter.hpp>

#include "thermo.hpp"

template <>
struct fmt::formatter<kintera::Nucleation> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Nucleation& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "({}; minT = {:.2f}; maxT = {:.2f})",
                          p.reaction(), p.minT(), p.maxT());
  }
};

template <>
struct fmt::formatter<kintera::ThermoOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::ThermoOptions& p, FormatContext& ctx) const {
    std::ostringstream vapors;
    for (size_t i = 0; i < p.vapor_ids().size(); ++i) {
      vapors << p.vapor_ids()[i];
      if (i != p.vapor_ids().size() - 1) {
        vapors << ", ";
      }
    }

    std::ostringstream clouds;
    for (size_t i = 0; i < p.cloud_ids().size(); ++i) {
      clouds << p.cloud_ids()[i];
      if (i != p.cloud_ids().size() - 1) {
        clouds << ", ";
      }
    }

    std::ostringstream reactions;
    for (size_t i = 0; i < p.react().size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, p.react()[i]);
      if (i != p.react().size() - 1) {
        reactions << "; ";
      }
    }

    std::ostringstream species;
    for (size_t i = 0; i < p.species().size(); ++i) {
      species << p.species()[i];
      if (i != p.species().size() - 1) {
        species << ", ";
      }
    }

    std::ostringstream cref;
    for (size_t i = 0; i < p.cref_R().size(); ++i) {
      cref << p.cref_R()[i];
      if (i != p.cref_R().size() - 1) {
        cref << ", ";
      }
    }

    std::ostringstream uref;
    for (size_t i = 0; i < p.uref_R().size(); ++i) {
      uref << p.uref_R()[i];
      if (i != p.uref_R().size() - 1) {
        uref << ", ";
      }
    }

    return fmt::format_to(
        ctx.out(),
        "(Rd = {:.2f}; cv_R = {}; u0_R = {}; vapors = ({}); clouds = "
        "({}); Tref = {}; Pref = {}; react = ({})); species = ({})",
        p.Rd(), cref.str(), uref.str(), vapors.str(), clouds.str(), p.Tref(),
        p.Pref(), reactions.str(), species.str());
  }
};
