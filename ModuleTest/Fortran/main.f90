program main
   use fmmwrapper
   use mpi
   use iso_c_binding
   implicit none

   integer(kind=4) :: rank, size, ierror, tag, status(MPI_STATUS_SIZE)
   integer(kind=4) :: nsrc, ntrg
   integer(kind=4) :: i
   real(kind=8) :: seed
   real(kind=8), dimension(:), allocatable :: srcCoord
   real(kind=8), dimension(:), allocatable :: srcValue
   real(kind=8), dimension(:), allocatable :: trgCoord
   real(kind=8), dimension(:), allocatable :: trgValue
   type(c_ptr) :: fmm
   type(c_ptr) :: fmmwall2d

   call MPI_INIT(ierror)
   call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
   call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

   nsrc = 16384
   ntrg = 16384

   allocate (srcCoord(nsrc*3))
   allocate (srcValue(nsrc*3))
   allocate (trgCoord(ntrg*3))
   allocate (trgValue(ntrg*3))

   do i = 1, nsrc
      seed = rank*nsrc + i; 
      srcCoord(3*(i - 1) + 1) = sin(seed); 
      srcCoord(3*(i - 1) + 2) = cos(seed); 
      srcCoord(3*(i - 1) + 3) = sin(exp(seed)); 
      srcValue(3*(i - 1) + 1) = sin(seed); 
      srcValue(3*(i - 1) + 2) = sin(sin(seed)); 
      srcValue(3*(i - 1) + 3) = cos(sin(seed)); 
   end do
   do i = 1, ntrg
      seed = rank*nsrc + i; 
      trgCoord(3*(i - 1) + 1) = cos(seed); 
      trgCoord(3*(i - 1) + 2) = sin(seed); 
      trgCoord(3*(i - 1) + 3) = cos(exp(seed)); 
      trgValue(3*(i - 1) + 1) = 0; 
      trgValue(3*(i - 1) + 2) = 0; 
      trgValue(3*(i - 1) + 3) = 0; 
   end do

   call MPI_BARRIER(MPI_COMM_WORLD, ierror); 

   ! Test FMM
   fmm = create_fmm_wrapper(8, 2000, 0, 7)
   call FMM_SetBox(fmm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
   call FMM_UpdateTree(fmm, trgCoord, srcCoord, ntrg, nsrc); 
!    call FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc); 
!    call FMM_DataClear(fmm); 
!    call FMM_Evaluate(fmm, trgValue, srcValue, ntrg, nsrc); 
   call delete_fmm_wrapper(fmm); 
   ! Test FMMWall2D
   call MPI_BARRIER(MPI_COMM_WORLD, ierror); 
   call MPI_FINALIZE(ierror)

   deallocate (srcCoord)
   deallocate (srcValue)
   deallocate (trgCoord)
   deallocate (trgValue)

end program main
