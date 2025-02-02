#pragma once

// kintera
#include <kinetics/reaction.hpp>

//! Parse a composition string into a map consisting of individual
//! key:composition pairs (std::map<std::string, double>).
//! Adapted from Cantera's `parseCompString` function.
/*!
 * Elements present in *names* but not in the composition string will have
 * a value of 0. Elements present in the composition string but not in *names*
 * will generate an exception. The composition is a double. Example:
 *
 * Input is
 *
 *    "ice:1   snow:2"
 *    names = ["fire", "ice", "snow"]
 *
 * Output is
 *             x["fire"] = 0
 *             x["ice"]  = 1
 *             x["snow"] = 2
 *
 * \param ss    original string consisting of multiple key:composition
 *              pairs on multiple lines
 *
 * \param names (optional) valid names for elements in the composition map. 
 *              If empty or unspecified, all values are allowed.
 *
 * \return      map of names to values
 */
namespace kintera {
  
Composition parse_comp_string(const string& ss,
                              const vector<string>& names=vector<string>());

} // namespace kintera
