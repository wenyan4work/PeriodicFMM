#ifdef __cplusplus
extern "C" {
#endif

typedef struct FMM_Wrapper FMM_Wrapper;

FMM_Wrapper *create_fmm_wrapper(int mult_order, int max_pts, int init_depth,
                                int pbc, int regularize);

void delete_fmm_wrapper(FMM_Wrapper *fmm);

void FMM_SetBox(FMM_Wrapper *fmm, double xlow, double xhigh, double ylow,
                double yhigh, double zlow, double zhigh);

void FMM_TreeClear(FMM_Wrapper *fmm);

void FMM_DataClear(FMM_Wrapper *fmm);

void FMM_UpdateTree(FMM_Wrapper *fmm, const double *trg_coor,
                    const double *src_coord, const int num_trg,
                    const int num_src);

void FMM_Evaluate(FMM_Wrapper *fmm, double *trg_value, const double *src_value,
                  const int num_trg, const int num_src);

#ifdef __cplusplus
}
#endif