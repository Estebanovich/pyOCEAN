!> MOM6 column builders on cases written by columns.py (surface first, depths positive down
!! for build_and_interpolate_grid; interface heights for z* / sigma as MOM6 returns them).
program regrid_driver
  use coord_zlike,   only : zlike_CS, init_coord_zlike, end_coord_zlike, set_zlike_params, build_zstar_column
  use coord_sigma,   only : sigma_CS, init_coord_sigma, end_coord_sigma, set_sigma_params, build_sigma_column
  use regrid_interp, only : interp_CS_type, set_interp_scheme, set_interp_extrap, &
                            set_interp_answer_date, build_and_interpolate_grid
  implicit none
  type(zlike_CS), pointer :: zCS => null()
  type(sigma_CS), pointer :: sCS => null()
  type(interp_CS_type) :: iCS
  integer :: ncase, c, n
  character(len=16) :: kind
  character(len=512) :: fin, fout
  real :: depth, eta
  real, allocatable :: dz(:), zi(:), h(:), rho(:), tgt(:), x0(:), h1(:), x1(:)
  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  call set_interp_scheme(iCS, "P1M_H2") ; call set_interp_extrap(iCS, .false.)
  call set_interp_answer_date(iCS, 99991231)
  do c = 1, ncase
    read(10,*) kind
    if (trim(kind) == 'zstar' .or. trim(kind) == 'sigma') then
      read(10,*) n, depth, eta ; allocate(dz(n), zi(n+1)) ; read(10,*) dz
      if (trim(kind) == 'zstar') then
        call init_coord_zlike(zCS, n, dz) ; call set_zlike_params(zCS, min_thickness=1.0e-3)
        call build_zstar_column(zCS, depth, depth+eta, zi)
        call end_coord_zlike(zCS) ; nullify(zCS)
      else
        call init_coord_sigma(sCS, n, dz) ; call set_sigma_params(sCS, min_thickness=1.0e-3)
        call build_sigma_column(sCS, depth, depth+eta, zi)        ! heights, surface first (as z*)
        call end_coord_sigma(sCS) ; nullify(sCS)
      endif
      write(11,'(*(es25.17))') zi
      deallocate(dz, zi)
    else
      read(10,*) n ; allocate(h(n), rho(n), tgt(n+1), x0(n+1), h1(n), x1(n+1))
      read(10,*) h ; read(10,*) rho ; read(10,*) tgt
      x0(1) = 0.0
      block ; integer :: k ; do k = 1, n ; x0(k+1) = x0(k) + h(k) ; enddo ; end block
      call build_and_interpolate_grid(iCS, rho, n, h, x0, tgt, n, h1, x1, 1.0e-30, 1.0e-10)
      write(11,'(*(es25.17))') x1
      deallocate(h, rho, tgt, x0, h1, x1)
    endif
  enddo
  close(10) ; close(11)
end program regrid_driver
