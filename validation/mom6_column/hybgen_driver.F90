!> HYCOM hybgen regridding (MOM6 port) on single columns written by compare_hybgen.py.
!! The column routines hybgen_column_init / cushn / hybgen_column_regrid are private in
!! MOM_hybgen_regrid.F90, so build_reference.sh extracts them verbatim into the module
!! hybgen_column_mom6 (with a public mirror of the 7 hybgen_regrid_CS fields they read).
!! This driver repeats the set-up of init_hybgen_regrid (MOM_hybgen_regrid.F90:224-239)
!! and the z* dilation of hybgen_regrid (:505-512).  Columns surface first, H = m, R = kg m-3.
!! Usage: hybgen_driver in.txt out.txt   (out: new interface depths, positive down, nk+1)
program hybgen_driver
  use hybgen_column_mom6, only : hybgen_regrid_CS, hybgen_column_init, hybgen_column_regrid
  implicit none
  type(hybgen_regrid_CS) :: CS
  integer :: ncase, c, nk, nsigma, fixlay, k
  real :: dp00i, topiso, qhybrlx, thkbot, nom_depth, h_tot, dilate, dpns, dsns
  real, parameter :: min_dilate = 0.5, max_dilate = 2.0     ! MOM6 defaults
  real, allocatable :: dp0k(:), ds0k(:), h(:), Rcv(:), Rcv_tgt(:)
  real, allocatable :: qhrlx(:), dp0ij(:), dp0cum(:), dp_int(:), p(:)
  character(len=512) :: fin, fout
  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    read(10,*) nk, nsigma
    allocate(dp0k(nk), ds0k(nk), h(nk), Rcv(nk), Rcv_tgt(nk))
    allocate(qhrlx(nk+1), dp0ij(nk), dp0cum(nk+1), dp_int(nk+1), p(nk+1))
    read(10,*) dp0k ; read(10,*) ds0k
    read(10,*) dp00i, topiso, qhybrlx, thkbot, nom_depth, CS%min_thickness
    read(10,*) h ; read(10,*) Rcv ; read(10,*) Rcv_tgt
    CS%nk = nk ; CS%onem = 1.0 ; CS%rho_eps = 1.0e-11          ! init_hybgen_regrid defaults
    CS%dp_far_from_sfc = 10.0 ; CS%dp_far_from_bot = 1.0 ; CS%h_thin = 1.0e-3
    dp0k(:) = max(dp0k(:), CS%min_thickness) ; dp00i = max(dp00i, CS%min_thickness)
    if (nsigma == 0) then
      dpns = dp0k(1) ; dsns = 0.0
    else
      dpns = sum(dp0k(1:nsigma)) ; dsns = sum(ds0k(1:nsigma))
    endif
    h_tot = sum(h)
    dilate = min(max(h_tot / nom_depth, min_dilate), max_dilate)
    call hybgen_column_init(nk, nsigma, dp0k, ds0k, dp00i, topiso, qhybrlx, dpns, dsns, &
                            h_tot, dilate, h, fixlay, qhrlx, dp0ij, dp0cum)
    call hybgen_column_regrid(CS, nk, thkbot, Rcv_tgt, fixlay, qhrlx, dp0ij, dp0cum, Rcv, h, dp_int)
    p(1) = 0.0
    do k = 1, nk ; p(k+1) = p(k) + h(k) ; enddo
    write(11,'(*(es25.17))') p(:) + dp_int(:)
    deallocate(dp0k, ds0k, h, Rcv, Rcv_tgt, qhrlx, dp0ij, dp0cum, dp_int, p)
  enddo
  close(10) ; close(11)
end program hybgen_driver
