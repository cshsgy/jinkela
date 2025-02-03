#ifndef CT_ELEMENTS_H
#define CT_ELEMENTS_H

#include <map>
#include <string>
#include <vector>

namespace kintera
{

//! An element constraint that is current turned off
#define CT_ELEM_TYPE_TURNEDOFF -1

//! Normal element constraint consisting of positive coefficients for the
//! formula matrix.
/*!
 * All species have positive coefficients within the formula matrix. With this
 * constraint, we may employ various strategies to handle small values of the
 * element number successfully.
 */
#define CT_ELEM_TYPE_ABSPOS 0

//! This refers to conservation of electrons
/*!
 * Electrons may have positive or negative values in the Formula matrix.
 */
#define CT_ELEM_TYPE_ELECTRONCHARGE 1

//! This refers to a charge neutrality of a single phase
/*!
 * Charge neutrality may have positive or negative values in the Formula matrix.
 */
#define CT_ELEM_TYPE_CHARGENEUTRALITY 2

//! Constraint associated with maintaining a fixed lattice stoichiometry in a solid
/*!
 * The constraint may have positive or negative values. The lattice 0 species
 * will have negative values while higher lattices will have positive values
 */
#define CT_ELEM_TYPE_LATTICERATIO 3

//! Constraint associated with maintaining frozen kinetic equilibria in
//! some functional groups within molecules
/*!
 * We seek here to say that some functional groups or ionic states should be
 * treated as if they are separate elements given the time scale of the problem.
 * This will be abs positive constraint. We have not implemented any examples
 * yet. A requirement will be that we must be able to add and subtract these
 * constraints.
 */
#define CT_ELEM_TYPE_KINETICFROZEN 4

//! Constraint associated with the maintenance of a surface phase
/*!
 * We don't have any examples of this yet either. However, surfaces only exist
 * because they are interfaces between bulk layers. If we want to treat surfaces
 * within thermodynamic systems we must come up with a way to constrain their
 * total number.
 */
#define CT_ELEM_TYPE_SURFACECONSTRAINT 5

//! Other constraint equations
/*!
 * currently there are none
 */
#define CT_ELEM_TYPE_OTHERCONSTRAINT 6
//! @}

//! Number indicating we don't know the entropy of the element in its most
//! stable state at 298.15 K and 1 bar.
#define ENTROPY298_UNKNOWN -123456789.

const std::vector<std::string>& elementSymbols();
const std::vector<std::string>& elementNames();
const std::map<std::string, double>& elementWeights();
double getElementWeight(const std::string& ename);
double getElementWeight(int atomicNumber);
std::string getElementSymbol(const std::string& ename);
std::string getElementSymbol(int atomicNumber);
std::string getElementName(const std::string& ename);
std::string getElementName(int atomicNumber);
int getAtomicNumber(const std::string& ename);
size_t numElementsDefined();
size_t numIsotopesDefined();

} // namespace

#endif
