
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FMM_WrapperWall2D FMM_WrapperWall2D;

FMM_WrapperWall2D *create_fmm_wrappperwall2d(int mult_order, int max_pts, int init_depth, int pbc);

void delete_fmm_wrapperwall2d(FMM_WrapperWall2D *fmm);

void FMMWall2D_SetBox(FMM_WrapperWall2D *fmm, double xlow, double xhigh, double ylow, double yhigh, double zlow,
                      double zhigh);

void FMMWall2D_TreeClear(FMM_WrapperWall2D *fmm);

void FMMWall2D_DataClear(FMM_WrapperWall2D *fmm);

void FMMWall2D_UpdateTree(FMM_WrapperWall2D *fmm, const double *trg_coor, const double *src_coord, const int num_trg,
                          const int num_src);

void FMMWall2D_Evaluate(FMM_WrapperWall2D *fmm, double *trg_value, const double *src_value, const int num_trg,
                        const int num_src);
#ifdef __cplusplus
}
#endif