/*
 * Ewald.hpp
 *
 *  Created on: Mar 2, 2017
 *      Author: wyan
 */

/*
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * !!!   For unit cubic box only !!!
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * */

#ifndef STOKES3D3D_EWALD_HPP_
#define STOKES3D3D_EWALD_HPP_

#include <cmath>

#include <Eigen/Dense>
#include <boost/math/special_functions/erf.hpp>

constexpr int DIRECTLAYER = 2;
constexpr double PI314 = 3.1415926535897932384626433;

inline double ERFC(double x) { return boost::math::erfc(x); }

inline double ERF(double x) { return boost::math::erf(x); }

inline double boxperiodic(double x, double xlow, double xhigh) {
    double temp = (x - xlow) / (xhigh - xlow);
    return (x - xlow) - floor(temp) * (xhigh - xlow);
}

inline void Gkernel(const Eigen::Vector3d &target, const Eigen::Vector3d &source, Eigen::Matrix3d &answer) {
    auto rst = target - source;
    double rnorm = rst.norm();
    double rnormReg = rnorm;
    if (rnorm < 1e-13) {
        answer = Eigen::Matrix3d::Zero();
        return;
    }
    auto part2 = rst * rst.transpose() / (rnormReg * rnormReg * rnormReg);
    auto part1 = Eigen::Matrix3d::Identity() / rnormReg;
    answer = part1 + part2;
}

inline void Gkernel1D(const Eigen::Vector3d &target, Eigen::Matrix3d &G) {
    const int Nsum = 10000;
    G.setZero();
    Eigen::Matrix3d Gtemp1;
    Eigen::Matrix3d Gtemp2;
    Gkernel(target, Eigen::Vector3d(0, 0, 0), G);
    for (int i = 1; i < Nsum + 1; i++) {
        Gkernel(target, Eigen::Vector3d(0, 0, i), Gtemp1);
        Gkernel(target, Eigen::Vector3d(0, 0, -i), Gtemp2);
        G += (Gtemp1 + Gtemp2);
    }
}

/*
 * def AEW(xi,rvec):
 r=np.sqrt(rvec.dot(rvec))
 A = 2*(xi*np.exp(-(xi**2)*(r**2))/(np.sqrt(np.pi)*r**2)+ss.erfc(xi*r)/(2*r**3)) \
    *(r*r*np.identity(3)+np.outer(rvec,rvec)) - 4*xi/np.sqrt(np.pi)*np.exp(-(xi**2)*(r**2))*np.identity(3)
 return A
 *
 * */
inline Eigen::Matrix3d AEW(const double xi, const Eigen::Vector3d &rvec) {
    const double r = rvec.norm();
    Eigen::Matrix3d A = 2 * (xi * exp(-(xi * xi) * (r * r)) / (sqrt(PI314) * r * r) + erfc(xi * r) / (2 * r * r * r)) *
                            (r * r * Eigen::Matrix3d::Identity() + (rvec * rvec.transpose())) -
                        4 * xi / sqrt(PI314) * exp(-(xi * xi) * (r * r)) * Eigen::Matrix3d::Identity();
    return A;
}

/*
 *
 def BEW(xi,kvec):
 k=np.sqrt(kvec.dot(kvec))
 B = 8*np.pi*(1+k*k/(4*(xi**2)))*((k**2)*np.identity(3)-np.outer(kvec,kvec))/(k**4)
 return B*np.exp(-k**2/(4*xi**2))
 *
 * */
inline Eigen::Matrix3d BEW(const double xi, const Eigen::Vector3d &kvec) {
    const double k = kvec.norm();
    Eigen::Matrix3d B = 8 * PI314 * (1 + k * k / (4 * (xi * xi))) *
                        ((k * k) * Eigen::Matrix3d::Identity() - (kvec * kvec.transpose())) / (k * k * k * k);
    B *= exp(-k * k / (4 * xi * xi));
    return B;
}

/*
 * def stokes3DEwald(rvec,force):
 xi = 2
 r=np.sqrt(rvec.dot(rvec))
 real = 0
 N=4
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 real = real + AEW(xi,rvec+1.0*np.array([i,j,k])).dot(force)
 wave = 0
 N=4
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 kvec=2*np.pi*np.array([i,j,k]) # L = 1
 if(i==0 and j==0 and k==0):
 continue
 else:
 wave = wave + BEW(xi,kvec).dot(force)*np.exp(-complex(0,1)*kvec.dot(rvec))

 return (np.real(wave)+real)

 * */
inline void GkernelEwald3D(const Eigen::Vector3d &rvecIn, Eigen::Matrix3d &Gsum, double box) {
    const double xi = 2;
    Eigen::Vector3d rvec = rvecIn;
    rvec[0] = rvec[0] - floor(rvec[0]);
    rvec[1] = rvec[1] - floor(rvec[1]);
    rvec[2] = rvec[2] - floor(rvec[2]);
    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 10;
    if (r < 1e-11) {
        auto Gself = -4 * xi / sqrt(PI314) * Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i * box, j * box, k * box));
                }
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i * box, j * box, k * box));
                }
            }
        }
    }
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Eigen::Vector3d kvec(2 * PI314 * i / box, 2 * PI314 * j / box, 2 * PI314 * k / box);
                if (i == 0 and j == 0 and k == 0) {
                    continue;
                } else {
                    wave = wave + BEW(xi, kvec) * cos(kvec.dot(rvec));
                }
            }
        }
    }
    Gsum = real + wave * (1 / (box * box * box));
}

/*
 *
 def stokes3DM2L(rvec,force):
 uEwald=stokes3DEwald(rvec,force)
 uNB=0
 N=3
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 uNB=uNB+Gkernel(rvec-np.array([i,j,k])).dot(force)
 return uEwald-uNB
 * */
// Out of Layer 1
inline void GkernelEwald3DFF(const Eigen::Vector3d &rvec, Eigen::Matrix3d &GsumO1) {
    Eigen::Matrix3d Gfree = Eigen::Matrix3d::Zero();
    GkernelEwald3D(rvec, GsumO1, 1.0);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Gkernel(rvec, Eigen::Vector3d(i, j, k), Gfree);
                GsumO1 -= Gfree;
            }
        }
    }
}

inline double lbda(double k, double xi, double z) { return exp(-k * k / (4 * xi * xi) - (xi * xi) * (z * z)); }

inline double thetaplus(double k, double xi, double z) { return exp(k * z) * ERFC(k / (2 * xi) + xi * z); }

inline double thetaminus(double k, double xi, double z) { return exp(-k * z) * ERFC(k / (2 * xi) - xi * z); }

inline double J00(double k, double xi, double z) { return sqrt(PI314) * lbda(k, xi, z) * xi; }

inline double J10(double k, double xi, double z) {
    return PI314 * (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (4 * k);
}

inline double J20(double k, double xi, double z) {
    return sqrt(PI314) * lbda(k, xi, z) / (4 * k * k * xi) +
           PI314 * ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k * k * k) +
                    (thetaminus(k, xi, z) - thetaplus(k, xi, z)) * z / (8 * k * k) -
                    (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (16 * k * (xi * xi)));
}

inline double J12(double k, double xi, double z) {
    return PI314 * (-thetaplus(k, xi, z) - thetaminus(k, xi, z)) * k / 4 + sqrt(PI314) * lbda(k, xi, z) * xi;
}

inline double J22(double k, double xi, double z) {
    return PI314 * ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) * k / (16 * xi * xi) +
                    (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k) +
                    (thetaplus(k, xi, z) - thetaminus(k, xi, z)) * z / 8) -
           sqrt(PI314) * lbda(k, xi, z) / (4 * xi);
}

inline double K11(double k, double xi, double z) { return PI314 * ((thetaminus(k, xi, z) - thetaplus(k, xi, z))) / 4; }

inline double K12(double k, double xi, double z) {
    return PI314 * ((thetaplus(k, xi, z) - thetaminus(k, xi, z)) / (16 * xi * xi) +
                    (thetaminus(k, xi, z) + thetaplus(k, xi, z)) * z / (8 * k));
}

inline void QI(const Eigen::Vector3d &kvec, double xi, double z, Eigen::Matrix3d &QI) {
    // 3*3 tensor
    // kvec: np.array([k1,k2,0])
    double knorm = sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1]);
    QI = 2 * (J00(knorm, xi, z) / (4 * xi * xi) + J10(knorm, xi, z)) * Eigen::Matrix3d::Identity();
}

inline void Qkk(const Eigen::Vector3d &kvec, double xi, double z, Eigen::Matrix3d &Qreal, Eigen::Matrix3d &Qimg) {
    double k1 = kvec[0];
    double k2 = kvec[1];
    double knorm = sqrt(k1 * k1 + k2 * k2);
    auto j10 = J10(knorm, xi, z);
    auto j20 = J20(knorm, xi, z);
    auto j12 = J12(knorm, xi, z);
    auto j22 = J22(knorm, xi, z);

    auto k11 = K11(knorm, xi, z);
    auto k12 = K12(knorm, xi, z);
    Qreal.setZero();
    Qreal(0, 0) = k1 * k1;
    Qreal(1, 1) = k2 * k2;
    Qreal(0, 1) = k1 * k2;
    Qreal(1, 0) = k1 * k2;

    Qreal *= (j10 / (4 * (xi * xi)) + j20);
    Qreal(2, 2) = (j12 / (4 * xi * xi) + j22);
    Qreal *= -2;

    Qimg.setZero();
    Qimg(0, 2) = k1;
    Qimg(1, 2) = k2;
    Qimg(2, 0) = k1;
    Qimg(2, 1) = k2;
    // Qimg=np.array([[0,0,k1],[0,0,k2],[k1,k2,0]])*( k11/(4*xi**2) + k12 )
    Qimg *= (k11 / (4 * xi * xi) + k12);
    Qimg *= -2;
}

// inline Eigen::Matrix3d uFk0(double xi, double zmn) {
//	Eigen::Matrix3d wavek0;
//	wavek0 = -(4.0 / 1) * (PI314 * (zmn) * ERF(zmn * xi) + sqrt(PI314) / (2 * xi) * exp(-zmn * zmn * xi * xi));
//	return wavek0;
//
//}

inline void GkernelEwald2D(const Eigen::Vector3d &rvecIn, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    Eigen::Vector3d rvec = rvecIn;
    rvec[0] = rvec[0] - floor(rvec[0]);
    rvec[1] = rvec[1] - floor(rvec[1]); // reset to a periodic cell

    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < 1e-14) {
        auto Gself = -4 * xi / sqrt(PI314) * Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
    }

    // k
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    double zmn = rvec[2];
    Eigen::Vector3d rhomn = rvec;
    rhomn[2] = 0;
    Eigen::Matrix3d Qreal;
    Eigen::Matrix3d Qimg;
    Eigen::Matrix3d QImat;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            Eigen::Vector3d kvec(2 * PI314 * i, 2 * PI314 * j, 0);
            if (i == 0 and j == 0) {
                continue;
            }
            Qkk(kvec, xi, zmn, Qreal, Qimg);
            QI(kvec, xi, zmn, QImat);
            wave = wave + (QImat + Qreal) * cos(kvec.dot(rhomn)) - (Qimg)*sin(kvec.dot(rhomn));
        }
    }
    wave *= 4;

    // k=0
    Eigen::Matrix3d waveK0;
    waveK0.setZero();
    /*
     *   I2fn=force
     I2fn[2]=0
     wavek0=-(4/1)*(np.pi*(zmn)*ss.erf(zmn*xi)+np.sqrt(np.pi)/(2*xi)*np.exp(-zmn**2*xi**2))*I2fn
     *
     * */
    waveK0 = -(4 / 1.0) * (PI314 * (zmn)*ERF(zmn * xi) + sqrt(PI314) / (2 * xi) * exp(-zmn * zmn * xi * xi)) *
             Eigen::Matrix3d::Identity();
    waveK0(2, 2) = 0;

    Gsum = real + wave + waveK0;
}

#endif /* STOKES3D3D_EWALD_HPP_ */
