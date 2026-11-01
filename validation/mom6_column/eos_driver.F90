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
!!        eos_driver n2 in.txt out.txt   (TEOS-10 in the 3-D core, task 6)
!!          in:  ncol, then per column "nz rho0 g" and nz lines "h T S T_sl S_sl", surface first
!!               (h [m]; T_sl, S_sl: the surface-layer means KPP pairs with layer k)
!!          out: for each EOS (WRIGHT_FULL, then ROQUET_RHO) and column,
!!               nz-1 lines K = 2..nz "N2 dbuoy_dT dbuoy_dS p_K": the locally referenced N2 of
!!                 MOM_kappa_shear.F90:1128-1136 (pressure(K) = pressure(K-1) + gR0*h(k-1),
!!                 T_int, S_int, calculate_density_derivs(..., scale=-g_R0)) and :1496-1500
!!                 (I_dz_int = 2/(h(k-1)+h(k)), :1033), written without the max(0, .) so its sign shows;
!!               nz lines k = 1..nz "deltaRho rho_k-rho_km1 pRef": the KPP density differences of
!!                 MOM_CVMix_KPP.F90:1396-1436 (rho of layer k, of its surface layer and of layer
!!                 km1 = max(1,k-1) at the pressure pRef of the top interface of layer k, one
!!                 calculate_density call on the 3*nz points, MOM_CVMix_KPP.F90:1423)
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
  if (trim(mode) == 'n2') then
    call n2_columns(fin, fout)
    stop
  endif
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

  subroutine n2_columns(fin, fout)
    character(len=*), intent(in) :: fin, fout
    real, allocatable :: h(:,:), Tc(:,:), Sc(:,:), Tsl(:,:), Ssl(:,:), r0(:), gg(:)
    real, allocatable :: pres(:), T_int(:), S_int(:), dbdT(:), dbdS(:)
    real, allocatable :: T1(:), S1(:), p1(:), rho1(:)
    integer, allocatable :: nzs(:)
    real :: gR0, g_R0, I_dz, N2, pRef
    integer :: ncol, c, nz, k, K2, kk, km1, ff, nzmax, frm(2)
    open(10, file=trim(fin), status='old')
    read(10,*) ncol
    allocate(nzs(ncol), r0(ncol), gg(ncol))
    nzmax = 400
    allocate(h(nzmax,ncol), Tc(nzmax,ncol), Sc(nzmax,ncol), Tsl(nzmax,ncol), Ssl(nzmax,ncol))
    do c = 1, ncol
      read(10,*) nzs(c), r0(c), gg(c)
      do k = 1, nzs(c)
        read(10,*) h(k,c), Tc(k,c), Sc(k,c), Tsl(k,c), Ssl(k,c)
      enddo
    enddo
    close(10)
    frm = (/ EOS_WRIGHT_FULL, EOS_ROQUET_RHO /)
    open(11, file=trim(fout), status='replace')
    write(11,'(i8)') ncol
    do ff = 1, 2
      call EOS_manual_init(EOS, form_of_EOS=frm(ff))
      do c = 1, ncol
        nz = nzs(c)
        allocate(pres(nz+1), T_int(nz+1), S_int(nz+1), dbdT(nz+1), dbdS(nz+1))
        allocate(T1(3*nz), S1(3*nz), p1(3*nz), rho1(3*nz))
        gR0 = r0(c) * gg(c)            ! GV%H_to_RZ * GV%g_Earth (MOM_kappa_shear.F90:1008)
        g_R0 = gg(c) / r0(c)           ! GV%g_Earth_Z_T2 / GV%Rho0 (:1009)
        ! MOM_kappa_shear.F90:1128-1136
        pres(1) = 0.0
        T_int(1) = 0.0 ; S_int(1) = 0.0 ; dbdT(1) = 0.0 ; dbdS(1) = 0.0
        do K2 = 2, nz
          pres(K2) = pres(K2-1) + gR0*h(K2-1,c)
          T_int(K2) = 0.5*(Tc(K2-1,c) + Tc(K2,c))
          S_int(K2) = 0.5*(Sc(K2-1,c) + Sc(K2,c))
        enddo
        call calculate_density_derivs(T_int, S_int, pres, dbdT, dbdS, EOS, (/2,nz/), scale=-g_R0)
        ! MOM_kappa_shear.F90:1496-1500 (argument of max(0, .)), I_dz_int of :1033
        do K2 = 2, nz
          I_dz = 2.0 / (h(K2-1,c) + h(K2,c))
          N2 = I_dz * (dbdT(K2) * (Tc(K2-1,c)-Tc(K2,c)) + dbdS(K2) * (Sc(K2-1,c)-Sc(K2,c)))
          write(11,'(4es25.17)') N2, dbdT(K2), dbdS(K2), pres(K2)
        enddo
        ! MOM_CVMix_KPP.F90:1396-1423
        pRef = 0.0
        do k = 1, nz
          km1 = max(1, k-1)
          kk = 3*(k-1)
          p1(kk+1) = pRef ; p1(kk+2) = pRef ; p1(kk+3) = pRef
          T1(kk+1) = Tsl(k,c) ; T1(kk+2) = Tc(k,c) ; T1(kk+3) = Tc(km1,c)
          S1(kk+1) = Ssl(k,c) ; S1(kk+2) = Sc(k,c) ; S1(kk+3) = Sc(km1,c)
          pRef = pRef + (gg(c) * r0(c)) * h(k,c)
        enddo
        call calculate_density(T1, S1, p1, rho1, EOS)
        do k = 1, nz
          kk = 3*(k-1)
          write(11,'(3es25.17)') rho1(kk+2) - rho1(kk+1), rho1(kk+2) - rho1(kk+3), p1(kk+1)
        enddo
        deallocate(pres, T_int, S_int, dbdT, dbdS, T1, S1, p1, rho1)
      enddo
    enddo
    close(11)
  end subroutine n2_columns

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
