! C functions declaration
! Modified by Wen Yan from the code by Florencio Balboa Usabiaga
! Fortran 2003 is required for the iso_c_binding

module fmmwrapper
   use iso_c_binding
   implicit none

   interface

      function create_fmm_wrapper(mult_order, max_pts, init_depth, pbc) result(fmm_wrapper) bind(C, name="create_fmm_wrapper")
         use iso_c_binding
         implicit none
         type(c_ptr):: fmm_wrapper
         integer(c_int), value :: mult_order
         integer(c_int), value :: max_pts
         integer(c_int), value :: init_depth
         integer(c_int), value :: pbc
      end function create_fmm_wrapper

      subroutine delete_fmm_wrapper(fmm_wrapper) bind(C, name="delete_fmm_wrapper")
         use iso_c_binding
         implicit none
         type(c_ptr), value :: fmm_wrapper
      end subroutine delete_fmm_wrapper

      subroutine FMM_SetBox(fmm, xlow, xhigh, ylow, yhigh, zlow, zhigh) bind(C, name="FMM_SetBox")
         use iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: fmm
         real(c_double), value :: xlow, xhigh, ylow, yhigh, zlow, zhigh
      end subroutine FMM_SetBox

      subroutine FMM_TreeClear(fmm) bind(C, name="FMM_TreeClear")
         use iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: fmm
      end subroutine FMM_TreeClear

      subroutine FMM_DataClear(fmm) bind(C, name="FMM_DataClear")
         use iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: fmm
      end subroutine FMM_DataClear

      subroutine FMM_UpdateTree(fmm, trg_coord, src_coord, num_trg, num_src) bind(C, name="FMM_UpdateTree")
         use iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: fmm
          real(c_double), intent(in) :: src_coord(*)
          real(c_double), intent(in):: trg_coord(*)
         integer(c_int), intent(in), value :: num_src, num_trg
      end subroutine FMM_UpdateTree

      subroutine FMM_Evaluate(fmm, trg_value, src_value, num_trg, num_src) bind(C, name="FMM_Evaluate")
         use iso_c_binding
         implicit none
         type(c_ptr), intent(in), value :: fmm
         real(c_double), intent(in):: src_value(*)
         real(c_double), intent(inout):: trg_value(*)
         integer(c_int), intent(in), value :: num_src, num_trg
      end subroutine FMM_Evaluate

   end interface
end module fmmwrapper
