#ifndef STOKESREGSINGLELAYER_HPP
#define STOKESREGSINGLELAYER_HPP

#include <cmath>
#include <cstdlib>
#include <vector>

// pvfmm headers
#include <pvfmm.hpp>

namespace pvfmm {
// TODO: Stokes Reg Force Torque Vel kernel, 7 -> 3
// TODO: Stokes Reg Force Torque Vel Omega kernel, 7 -> 6
// TODO: Stokes Force Vel Omega kernel, 3 -> 6

/*********************************************************
 *                                                        *
 *     Stokes Reg Vel kernel, source: 4, target: 3        *
 *              fx,fy,fz,eps -> ux,uy,uz                  *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_regvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                           Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facp = set_intrin<Vec_t, Real_t>(2 * FACV);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                const Vec_t reg = bcast_intrin<Vec_t>(&src_value[3][s]); // reg parameter

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));
                r2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2+eps^2

                Vec_t r2reg2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2 + 2 eps^2

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2reg2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2reg2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2reg2, fz), mul_intrin(dz, commonCoeff)), rinv3));
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
#undef SRC_BLK
}

// '##' is the token parsing operator
template <class T, int newton_iter = 0>
void stokes_regvel(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                   mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 4, 3, stokes_regvel_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(             \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

/**********************************************************
 *                                                        *
 *     Stokes Reg Vel kernel, source: 7, target: 3        *
 *       fx,fy,fz,tx,ty,tz,eps -> ux,uy,uz                *
 **********************************************************/
template <class Real_t, class Vec_t = Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t) = rsqrt_intrin0<Vec_t>>
void stokes_regftvel_uKernel(Matrix<Real_t> &src_coord, Matrix<Real_t> &src_value, Matrix<Real_t> &trg_coord,
                             Matrix<Real_t> &trg_value) {

#define SRC_BLK 500
    size_t VecLen = sizeof(Vec_t) / sizeof(Real_t);

    //// Number of newton iterations
    size_t NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin0<Vec_t, Real_t>)
        NWTN_ITER = 0;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin1<Vec_t, Real_t>)
        NWTN_ITER = 1;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin2<Vec_t, Real_t>)
        NWTN_ITER = 2;
    if (RSQRT_INTRIN == (Vec_t(*)(Vec_t))rsqrt_intrin3<Vec_t, Real_t>)
        NWTN_ITER = 3;

    Real_t nwtn_scal = 1; // scaling factor for newton iterations
    for (int i = 0; i < NWTN_ITER; i++) {
        nwtn_scal = 2 * nwtn_scal * nwtn_scal * nwtn_scal;
    }
    const Real_t FACV = 1.0 / (8 * nwtn_scal * nwtn_scal * nwtn_scal * const_pi<Real_t>());
    const Vec_t facv = set_intrin<Vec_t, Real_t>(FACV);
    const Vec_t facp = set_intrin<Vec_t, Real_t>(2 * FACV);

    size_t src_cnt_ = src_coord.Dim(1);
    size_t trg_cnt_ = trg_coord.Dim(1);

    for (size_t sblk = 0; sblk < src_cnt_; sblk += SRC_BLK) {
        size_t src_cnt = src_cnt_ - sblk;
        if (src_cnt > SRC_BLK)
            src_cnt = SRC_BLK;
        for (size_t t = 0; t < trg_cnt_; t += VecLen) {
            const Vec_t tx = load_intrin<Vec_t>(&trg_coord[0][t]);
            const Vec_t ty = load_intrin<Vec_t>(&trg_coord[1][t]);
            const Vec_t tz = load_intrin<Vec_t>(&trg_coord[2][t]);

            Vec_t vx = zero_intrin<Vec_t>(); // vx
            Vec_t vy = zero_intrin<Vec_t>(); // vy
            Vec_t vz = zero_intrin<Vec_t>(); // vz

            for (size_t s = sblk; s < sblk + src_cnt; s++) {
                const Vec_t dx = sub_intrin(tx, bcast_intrin<Vec_t>(&src_coord[0][s]));
                const Vec_t dy = sub_intrin(ty, bcast_intrin<Vec_t>(&src_coord[1][s]));
                const Vec_t dz = sub_intrin(tz, bcast_intrin<Vec_t>(&src_coord[2][s]));

                // force
                const Vec_t fx = bcast_intrin<Vec_t>(&src_value[0][s]);
                const Vec_t fy = bcast_intrin<Vec_t>(&src_value[1][s]);
                const Vec_t fz = bcast_intrin<Vec_t>(&src_value[2][s]);
                // torque
                const Vec_t lx = bcast_intrin<Vec_t>(&src_value[3][s]);
                const Vec_t ly = bcast_intrin<Vec_t>(&src_value[4][s]);
                const Vec_t lz = bcast_intrin<Vec_t>(&src_value[5][s]);
                // reg
                const Vec_t reg = bcast_intrin<Vec_t>(&src_value[6][s]);

                Vec_t r2 = mul_intrin(dx, dx);
                r2 = add_intrin(r2, mul_intrin(dy, dy));
                r2 = add_intrin(r2, mul_intrin(dz, dz));
                r2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2+eps^2

                Vec_t r2reg2 = add_intrin(r2, mul_intrin(reg, reg)); // r^2 + 2 eps^2

                Vec_t rinv = RSQRT_INTRIN(r2);
                Vec_t rinv3 = mul_intrin(mul_intrin(rinv, rinv), rinv);

                Vec_t commonCoeff = mul_intrin(fx, dx);
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fy, dy));
                commonCoeff = add_intrin(commonCoeff, mul_intrin(fz, dz));

                // force contribution to velocity
                vx = add_intrin(vx, mul_intrin(add_intrin(mul_intrin(r2reg2, fx), mul_intrin(dx, commonCoeff)), rinv3));
                vy = add_intrin(vy, mul_intrin(add_intrin(mul_intrin(r2reg2, fy), mul_intrin(dy, commonCoeff)), rinv3));
                vz = add_intrin(vz, mul_intrin(add_intrin(mul_intrin(r2reg2, fz), mul_intrin(dz, commonCoeff)), rinv3));
                // torque contribution to velocity
            }

            vx = add_intrin(mul_intrin(vx, facv), load_intrin<Vec_t>(&trg_value[0][t]));
            vy = add_intrin(mul_intrin(vy, facv), load_intrin<Vec_t>(&trg_value[1][t]));
            vz = add_intrin(mul_intrin(vz, facv), load_intrin<Vec_t>(&trg_value[2][t]));

            store_intrin(&trg_value[0][t], vx);
            store_intrin(&trg_value[1][t], vy);
            store_intrin(&trg_value[2][t], vz);
        }
    }
#undef SRC_BLK
}

// '##' is the token parsing operator
template <class T, int newton_iter = 0>
void stokes_regftvel(T *r_src, int src_cnt, T *v_src, int dof, T *r_trg, int trg_cnt, T *v_trg,
                     mem::MemoryManager *mem_mgr) {
#define STK_KER_NWTN(nwtn)                                                                                             \
    if (newton_iter == nwtn)                                                                                           \
    generic_kernel<Real_t, 7, 3, stokes_regftvel_uKernel<Real_t, Vec_t, rsqrt_intrin##nwtn<Vec_t, Real_t>>>(           \
        (Real_t *)r_src, src_cnt, (Real_t *)v_src, dof, (Real_t *)r_trg, trg_cnt, (Real_t *)v_trg, mem_mgr)
#define STOKES_KERNEL                                                                                                  \
    STK_KER_NWTN(0);                                                                                                   \
    STK_KER_NWTN(1);                                                                                                   \
    STK_KER_NWTN(2);                                                                                                   \
    STK_KER_NWTN(3);

    if (mem::TypeTraits<T>::ID() == mem::TypeTraits<float>::ID()) {
        typedef float Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256
#elif defined __SSE3__
#define Vec_t __m128
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else if (mem::TypeTraits<T>::ID() == mem::TypeTraits<double>::ID()) {
        typedef double Real_t;
#if defined __MIC__
#define Vec_t Real_t
#elif defined __AVX__
#define Vec_t __m256d
#elif defined __SSE3__
#define Vec_t __m128d
#else
#define Vec_t Real_t
#endif
        STOKES_KERNEL;
#undef Vec_t
    } else {
        typedef T Real_t;
#define Vec_t Real_t
        STOKES_KERNEL;
#undef Vec_t
    }

#undef STK_KER_NWTN
#undef STOKES_KERNEL
}

template <class T>
struct StokesRegKernel {
    inline static const Kernel<T> &Vel(); //   3+1->3
                                          // inline static const Kernel<T> &FTVel(); //   3+3+1->3
  private:
    static constexpr int NEWTON_ITE = sizeof(T) / 4;
};

// 1 newton for float, 2 newton for double
// the string for stk_ker must be exactly the same as in kernel.txx of pvfmm
template <class T>
inline const Kernel<T> &StokesRegKernel<T>::Vel() {
    static Kernel<T> stk_ker = StokesKernel<T>::velocity();
    static Kernel<T> s2t_ker =
        BuildKernel<T, stokes_regvel<T, NEWTON_ITE>>("stokes_regvel", 3, std::pair<int, int>(4, 3), NULL, NULL, NULL,
                                                     &stk_ker, &stk_ker, &stk_ker, &stk_ker, &stk_ker, NULL, true);

    // static Kernel<T> tmp_ker = BuildKernel<T, stokes_regvel<T, NEWTON_ITE>>("tmp_ker", 3, std::pair<int, int>(4, 3));
    // static Kernel<T> s2t_ker = BuildKernel<T, stokes_vel<T, NEWTON_ITE>>("stokes_regvel", 3, std::pair<int, int>(4,
    // 3),
    //                                                                      &tmp_ker, &tmp_ker, &tmp_ker);
    return s2t_ker;
}

// template <class T>
// inline const Kernel<T> &StokesRegKernel<T>::FTVel() {
//     static Kernel<T> stk_ker = StokesKernel<T>::velocity();
//     static Kernel<T> s2t_ker =
//         BuildKernel<T, stokes_regftvel<T, NEWTON_ITE>>("stokes_regftvel", 3, std::pair<int, int>(7, 3), NULL, NULL,
//                                                        NULL, &stk_ker, &stk_ker, &stk_ker, &stk_ker, &stk_ker);
//     return s2t_ker;
// }

} // namespace pvfmm
#endif // STOKESSINGLELAYERKERNEL_HPP
