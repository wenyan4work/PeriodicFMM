#ifndef REGULARIZED_STOKESLET_HPP_
#define REGULARIZED_STOKESLET_HPP_

#include <cmath>
#include <cstdlib>
#include <vector>

/// Function to compute the velocity and pressure in Stokes flow given a set of
/// source points, forces at the source points, and target evaluation points
/// using the method of regularized Stokeslets. The formulas used correspond to
/// equations (10a) and (10b) in
///  The method of regularized Stokeslets in three dimensions: Analysis, validation,
///  and application to helical swimming.
///  Ricardo Cortez, Lisa Fauci, and Alexei Medovikov. Phyiscs of Fluids. 2005
///
///
/// \param[in] src_points  Location of the sources and corresponding regularization
///                        parameter, epsilon. The structure is
///                        [x1, y1, z1, eps1, x2, y2, z2, eps2, ...]
///
/// \param[in] forces      Forces at the source points. The structure is
///                        [Fx1, Fy1, Fz1, Fx2, Fy2, Fz2, ...]
///
/// \param[in] eval_pnts   Location to evaluate the flow and pressure at. The
///                        structure is
///                        [x1, y1, z1, x2, y2, z2, ...]
///
/// \param[out] values     The pressure and velocity values at the requested locations
///                        [p1, ux1, uy1, uz1, p1, ux2, uy2, uz2, ...]
///                        (note, a factor of 1/(8 \pi \ mu) has been omited)
///
void reg_stokelset(const std::vector<double> &src_points, const std::vector<double> &forces,
                   const std::vector<double> &eval_pnts, std::vector<double> &values);
void combined_flow(const std::vector<double> &src_points, const std::vector<double> &src_values,
                   const std::vector<double> &eval_pnts, std::vector<double> &values);
#endif
