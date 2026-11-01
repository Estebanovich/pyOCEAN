!> Nonlinear equations of state of MOM6 (WRIGHT_FULL and ROQUET_RHO) on a cloud of points written
!! by eos_columns.py (TEOS-10 in the 3-D core, task 1).  The EOS is selected with
!! EOS_manual_init(form_of_EOS = EOS_WRIGHT_FULL / EOS_ROQUET_RHO) (MOM_EOS.F90:1735-1801), which
!! allocates the same EOS class as EOS_init with EQN_OF_STATE = "WRIGHT_FULL" / "ROQUET_RHO"
!! (MOM_EOS.F90:1563-1594) but needs no parameter file; unit scaling stays 1 (MOM_EOS.F90:139-152),
!! so the public front ends call the EOS directly (MOM_EOS.F90:336-338).  MOM6 git 0e9a561.
!!
!! Usage: eos_driver eval in.txt out.txt
!!          in:  n, then "T S p" per line (T [degC], S [PSU or g/kg], p [Pa])
!!          out: for each EOS (WRIGHT_FULL, then ROQUET_RHO) n lines
!!               "rho  rho-rho_ref  drho_dT  drho_dS" with rho_ref = 1035 kg m-3, from
!!               calculate_density (1-D, without and with rho_ref; MOM_EOS.F90:310-350) and
!!               calculate_density_derivs (1-D; MOM_EOS.F90:782-834)
!!        eos_driver coef out.txt
!!          out: "name value" for every coefficient of MOM_EOS_Wright_full.F90:23-37 and
!!               MOM_EOS_Roquet_rho.F90:14-156, as the compiler folds the parameter expressions
!!               (the parameter lines are copied verbatim by build_reference.sh into *.inc files)
program eos_driver
  use MOM_EOS, only : EOS_type, EOS_manual_init, calculate_density, calculate_density_derivs
  use MOM_EOS, only : EOS_WRIGHT_FULL, EOS_ROQUET_RHO
  implicit none
  type(EOS_type) :: EOS
  real, allocatable :: T(:), S(:), p(:), rho(:), rho_a(:), dT(:), dS(:)
  real, parameter :: rho_ref = 1035.0
  integer :: n, i, f, forms(2)
  character(len=8) :: mode
  character(len=512) :: fin, fout

  call get_command_argument(1, mode)
  if (trim(mode) == 'coef') then
    call get_command_argument(2, fout)
    open(11, file=trim(fout), status='replace')
    call dump_wright(11)
    call dump_roquet(11)
    close(11)
    stop
  endif
  call get_command_argument(2, fin) ; call get_command_argument(3, fout)
  open(10, file=trim(fin), status='old')
  read(10,*) n
  allocate(T(n), S(n), p(n), rho(n), rho_a(n), dT(n), dS(n))
  do i = 1, n
    read(10,*) T(i), S(i), p(i)
  enddo
  close(10)
  forms = (/ EOS_WRIGHT_FULL, EOS_ROQUET_RHO /)
  open(11, file=trim(fout), status='replace')
  write(11,'(i8)') n
  do f = 1, 2
    call EOS_manual_init(EOS, form_of_EOS=forms(f))
    call calculate_density(T, S, p, rho, EOS)
    call calculate_density(T, S, p, rho_a, EOS, rho_ref=rho_ref)
    call calculate_density_derivs(T, S, p, dT, dS, EOS)
    do i = 1, n
      write(11,'(4es25.17)') rho(i), rho_a(i), dT(i), dS(i)
    enddo
  enddo
  close(11)

contains

  subroutine dump_wright(u)
    integer, intent(in) :: u
    include 'wright_full_params.inc'
    include 'wright_full_write.inc'
  end subroutine dump_wright

  subroutine dump_roquet(u)
    integer, intent(in) :: u
    include 'roquet_rho_params.inc'
    include 'roquet_rho_write.inc'
  end subroutine dump_roquet

end program eos_driver
