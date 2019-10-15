#include <FMMWrapper-c.h>
#include <FMMWrapper.hpp>

extern "C" {

FMM_Wrapper *create_fmm_wrapper(int mult_order, int max_pts, int init_depth,
                                int pbc, int regularize) {
  return new FMM_Wrapper(mult_order, max_pts, init_depth,
                         static_cast<FMM_Wrapper::PAXIS>(pbc), regularize);
}

void delete_fmm_wrapper(FMM_Wrapper *fmm_wrapper) { delete fmm_wrapper; }

void FMM_SetBox(FMM_Wrapper *fmm, double xlow, double xhigh, double ylow,
                double yhigh, double zlow, double zhigh) {
  fmm->FMM_SetBox(xlow, xhigh, ylow, yhigh, zlow, zhigh);
}

void FMM_TreeClear(FMM_Wrapper *fmm) { fmm->FMM_TreeClear(); }

void FMM_DataClear(FMM_Wrapper *fmm) { fmm->FMM_DataClear(); }

void FMM_UpdateTree(FMM_Wrapper *fmm, const double *trg_coord,
                    const double *src_coord, const int num_trg,
                    const int num_src) {
  // Copy arrays to vectors
  std::vector<double> trg_coord_vec(trg_coord, trg_coord + 3 * num_trg);
  std::vector<double> src_coord_vec(src_coord, src_coord + 3 * num_src);

  // Call method to update Tree
  fmm->FMM_UpdateTree(src_coord_vec, trg_coord_vec);
}

void FMM_Evaluate(FMM_Wrapper *fmm, double *trg_value, const double *src_value,
                  const int num_trg, const int num_src) {
  std::vector<double> trg_value_vec(num_trg * 3);

  // Copy array to vector
  std::vector<double> src_value_vec(src_value, src_value + 3 * num_src);

  // Call method to evaluate FMM
  fmm->FMM_Evaluate(trg_value_vec, num_trg, &src_value_vec);

  // Copy vector to array
  std::copy(trg_value_vec.begin(), trg_value_vec.end(), trg_value);
}
}
