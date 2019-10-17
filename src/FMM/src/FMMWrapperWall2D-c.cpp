#include <FMM/FMMWrapperWall2D-c.h>
#include <FMM/FMMWrapperWall2D.hpp>

extern "C" {

FMM_WrapperWall2D* create_fmm_wrapperwall2d(int mult_order, int max_pts, int init_depth, int pbc){
  return new FMM_WrapperWall2D(mult_order, max_pts, init_depth, static_cast<FMM_WrapperWall2D::PAXIS>(pbc));
}

void delete_fmm_wrapperwall2d(FMM_WrapperWall2D* fmm){
  delete fmm;
}
    
void FMMWall2D_SetBox(FMM_WrapperWall2D* fmm, double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh){
  fmm->FMM_SetBox(xlow, xhigh, ylow, yhigh, zlow, zhigh);
}

void FMMWall2D_TreeClear(FMM_WrapperWall2D* fmm){
  fmm->FMM_TreeClear();
}

void FMMWall2D_DataClear(FMM_WrapperWall2D* fmm){
  fmm->FMM_DataClear();
}

void FMMWall2D_UpdateTree(FMM_WrapperWall2D* fmm, const double* trg_coord, const double* src_coord, const int num_trg, const int num_src){
  // Copy arrays to vectors
  std::vector<double> trg_coord_vec(trg_coord, trg_coord + 3*num_trg);
  std::vector<double> src_coord_vec(src_coord, src_coord + 3*num_src);

  // Call method to update Tree
  fmm->FMM_UpdateTree(src_coord_vec, trg_coord_vec);
}

void FMMWall2D_Evaluate(FMM_WrapperWall2D* fmm, double *trg_value, const double *src_value, const int num_trg, const int num_src){
  std::vector<double> trg_value_vec(num_trg * 3);

  // Copy array to vector
  std::vector<double> src_value_vec(src_value, src_value + 3*num_src);

  // Call method to evaluate FMM
  fmm->FMM_Evaluate(trg_value_vec, num_trg, &src_value_vec);
  
  // Copy vector to array
  std::copy(trg_value_vec.begin(), trg_value_vec.end(), trg_value);
}

}
