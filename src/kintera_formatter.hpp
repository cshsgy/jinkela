#pragma once

// fvm
#include "reaction.hpp"

template <>
struct fmt::formatter<kintera::Composition> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Composition& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", kintera::to_string(p));
  }
};

template <>
struct fmt::formatter<kintera::Reaction> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const kintera::Reaction& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", p.equation());
  }
};
