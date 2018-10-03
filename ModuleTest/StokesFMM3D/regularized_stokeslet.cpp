//
//  regularized_stokeslet.cpp
//

#include "regularized_stokeslet.hpp"
#include <iostream>

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
///                        [x1, y1, z1, x2, y2, z2,  ...]
///
/// \param[in] forces      Forces at the source points. The structure is
///                        [Fx1, Fy1, Fz1, eps1, Fx2, Fy2, Fz2, eps2, ...]
///
/// \param[in] eval_pnts   Location to evaluate the flow and pressure at. The
///                        structure is
///                        [x1, y1, z1, x2, y2, z2, ...]
///
/// \param[out] values     The pressure and velocity values at the requested locations
///                        [p1, ux1, uy1, uz1, p1, ux2, uy2, uz2, ...]
///                        (note, a factor of 1/(8 \pi \mu) has been omited)
///
void reg_stokelset(const std::vector<double> &src_points, const std::vector<double> &forces,
                   const std::vector<double> &eval_pnts, std::vector<double> &values) {

    // compute the number source locations and evaluation locations
    std::size_t n_src = src_points.size() / 3;
    std::size_t n_eval = eval_pnts.size() / 3;

    // zero out the return values in case there is something in them
    values.resize(n_eval * 4);
    std::fill(values.begin(), values.end(), 0);

#pragma omp parallel for
    // loop over the evaluation points
    for (std::size_t i = 0; i < n_eval; ++i) {

        // loop over the source points
        for (std::size_t j = 0; j < n_src; ++j) {

            double dx = eval_pnts[3 * i] - src_points[3 * j];
            double dy = eval_pnts[3 * i + 1] - src_points[3 * j + 1];
            double dz = eval_pnts[3 * i + 2] - src_points[3 * j + 2];
            double fx = forces[4 * j + 0];
            double fy = forces[4 * j + 1];
            double fz = forces[4 * j + 2];
            double eps = forces[4 * j + 3];

            // length squared of r
            double r2 = dx * dx + dy * dy + dz * dz;
            // regularization parameter squared
            double eps2 = eps * eps;

            double denom = std::sqrt(eps2 + r2);
            double velocity_denom = denom * (eps2 + r2);
            double velocity_numer = r2 + 2 * eps2;
            double pressure_denom = velocity_denom * (eps2 + r2);
            double pressure_numer = 2 * r2 + 5 * eps2;
            double fdotr = dx * fx + dy * fy + dz * fz;

            // pressure
            values[4 * i] += fdotr * pressure_numer / pressure_denom;

            // x component of velocity
            values[4 * i + 1] += (velocity_numer * fx + fdotr * dx) / velocity_denom;

            // y component of velocity
            values[4 * i + 2] += (velocity_numer * fy + fdotr * dy) / velocity_denom;

            // z component of velocity
            values[4 * i + 3] += (velocity_numer * fz + fdotr * dz) / velocity_denom;
        }
        values[4 * i] *= (1 / (8 * 3.14159265358979323846));
        values[4 * i + 1] *= (1 / (8 * 3.14159265358979323846));
        values[4 * i + 2] *= (1 / (8 * 3.14159265358979323846));
        values[4 * i + 3] *= (1 / (8 * 3.14159265358979323846));
    }
}
