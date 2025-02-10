/**
 * @file Units.h
 * Header for unit conversion utilities, which are used to translate
 * user input from input files (See @ref inputGroup and
 * class @link Cantera::Units Units@endlink).
 */

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#ifndef CT_UNITS_H
#define CT_UNITS_H

#include <map>
#include <string>
#include <vector>
#include <any>
#include <pair>

namespace kintera
{

class AnyValue;
class AnyMap;

class Units
{
public:
    //! Create a Units object with the specified dimensions.
    explicit Units(double factor=1.0, double mass=0, double length=0,
                   double time=0, double temperature=0, double current=0,
                   double quantity=0);

    explicit Units(const std::string& units, bool force_unity=false);
    bool convertible(const Units& other) const;
    double factor() const { return m_factor; }
    Units& operator*=(const Units& other);
    std::string str(bool skip_unity=true) const;
    Units pow(double exponent) const;
    bool operator==(const Units& other) const;
    double dimension(const std::string& primary) const;

private:
    void scale(double k) { m_factor *= k; }

    double m_factor = 1.0; //!< conversion factor to %Cantera base units
    double m_mass_dim = 0.0;
    double m_length_dim = 0.0;
    double m_time_dim = 0.0;
    double m_temperature_dim = 0.0;
    double m_current_dim = 0.0;
    double m_quantity_dim = 0.0;
    double m_pressure_dim = 0.0; //!< pseudo-dimension to track explicit pressure units
    double m_energy_dim = 0.0; //!< pseudo-dimension to track explicit energy units

    friend class UnitSystem;
};

struct UnitStack
{
    UnitStack(const Units& standardUnits) {
        stack.reserve(2); // covers memory requirements for most applications
        stack.emplace_back(standardUnits, 0.);
    }

    UnitStack(std::initializer_list<std::pair<Units, double>> units)
        : stack(units) {}

    UnitStack() = default;
    size_t size() const { return stack.size(); }
    Units standardUnits() const;
    void setStandardUnits(Units& standardUnits);
    double standardExponent() const;
    void join(double exponent);
    void update(const Units& units, double exponent);
    Units product() const;
    std::vector<std::pair<Units, double>> stack;
};

/*!
 * String representations of units can be written using multiplication,
 * division, and exponentiation. Spaces are ignored. Positive, negative, and
 * decimal exponents are permitted. Examples:
 *
 *     kg*m/s^2
 *     J/kmol
 *     m*s^-2
 *     J/kg/K
 *
 * Metric prefixes are recognized for all units, such as nm, hPa, mg, EJ, mL, kcal.
 *
 * Special functions for converting activation energies allow these values to be
 * expressed as either energy per quantity, energy (for example, eV), or temperature by
 * applying a factor of the Avogadro number or the gas constant where needed.
 */
class UnitSystem
{
public:
    UnitSystem(std::initializer_list<std::string> units);

    UnitSystem() : UnitSystem({}) {}

    std::map<std::string, std::string> defaults() const;
    //! * To use SI+kmol: `setDefaults({"kg", "m", "s", "Pa", "J", "kmol"});`
    //! * To use CGS+mol: `setDefaults({"cm", "g", "dyn/cm^2", "erg", "mol"});`
    void setDefaults(std::initializer_list<string> units);

    //! map<string, string> defaults{
    //!     {"length", "m"}, {"mass", "kg"}, {"time", "s"},
    //!     {"quantity", "kmol"}, {"pressure", "Pa"}, {"energy", "J"},
    //!     {"activation-energy", "J/kmol"}
    //! };
    //! setDefaults(defaults);
    //! ```
    void setDefaults(const std::map<std::string, std::string>& units);
    void setDefaultActivationEnergy(const string& e_units);

    double convert(double value, const std::string& src, const std::string& dest) const;
    double convert(double value, const Units& src, const Units& dest) const;

    double convertTo(double value, const std::string& dest) const;
    double convertTo(double value, const Units& dest) const;

    double convertFrom(double value, const std::string& src) const;
    double convertFrom(double value, const Units& src) const;

    double convert(const AnyValue& val, const std::string& dest) const;
    double convert(const AnyValue& val, const Units& dest) const;

    double convertRateCoeff(const AnyValue& val, const Units& dest) const;

    std::vector<double> convert(const std::vector<AnyValue>& vals, const std::string& dest) const;
    std::vector<double> convert(const std::vector<AnyValue>& vals, const Units& dest) const;
    double convertActivationEnergy(double value, const std::string& src,
                                   const std::string& dest) const;

    double convertActivationEnergyTo(double value, const std::string& dest) const;
    double convertActivationEnergyTo(double value, const Units& dest) const;
    double convertActivationEnergyFrom(double value, const std::string& src) const;
    double convertActivationEnergy(const AnyValue& val, const std::string& dest) const;

    AnyMap getDelta(const UnitSystem& other) const;

private: // Factors to convert from this unit system to base units (kg, m, s, Pa, J, kmol)
    double m_mass_factor = 1.0;
    double m_length_factor = 1.0;
    double m_time_factor = 1.0;
    double m_pressure_factor = 1.0;
    double m_energy_factor = 1.0;
    double m_activation_energy_factor = 1.0;
    double m_quantity_factor = 1.0;
    bool m_explicit_activation_energy = false;
    map<string, string> m_defaults;
};

}

#endif