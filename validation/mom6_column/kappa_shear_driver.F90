!> Jackson et al. (2008) shear mixing (MOM6 kappa_shear_column) on single columns written by
!! columns.write_kappa_shear_cases.  kappa_shear_column / calculate_projected_state /
!! find_kappa_tke are private in MOM_kappa_shear.F90, so build_reference.sh extracts them
!! verbatim into the module kappa_shear_column_mom6 (with a public Kappa_shear_CS and four
!! integer counters).  This driver repeats the column set-up of Calculate_kappa_shear
!! (MOM_kappa_shear.F90:299-327, no massless layers so nzc = nz, kappa seeded with
!! KD_SEED_KAPPA_SHEAR at :325) in the Boussinesq form with unit scaling 1; without temperature
!! MOM6 itself passes the density as "T" (:307-310).
!!
!! Buoyancy: tv%T is not associated, so MOM6 uses dbuoy_dT = -g/Rho0 and dbuoy_dS = 0
!! (:1148-1149).  With g_Earth = Rho0 = 1 and T = -b this is a linear EOS in T only and
!! dbuoy_dT*(T(k-1)-T(k)) = b(k-1)-b(k) exactly, the buoyancy difference of pyOCEAN's oracle.
!! Columns surface first (k = 1 at the surface), dz in m, u, v in m s-1, b in m s-2.
!! Usage: kappa_shear_driver in.txt out.txt
!! out per case: kappa_avg (nz+1), tke_avg (nz+1), then "n_newton n_ramp n_abort n_iter".
!!
!! Nonlinear EOS (Jackson two-tracer projection, 2026-11-01):
!!        kappa_shear_driver eos in.txt out.txt
!! Per case "nz dt f2 kappa0 form rho0 g", the parameter lines, then dz, u, v, T, S (surface first);
!! form 4 WRIGHT_FULL or 7 ROQUET_RHO (EOS_manual_init, as eos_driver).  tv%T is associated (a
!! dummy target: kappa_shear_column only tests associated(tv%T), :1023) and tv%eqn_of_state points
!! to the EOS, so MOM6 itself carries T and Sal through calculate_projected_state and builds
!! dbuoy_dT, dbuoy_dS once per column from the column after the kappa0 step (Boussinesq path
!! :1126-1136: pressure(1) = surface_pres = 0, pressure(K) = pressure(K-1) + gR0*hlay(k-1),
!! calculate_density_derivs(T_int, Sal_int, pressure, ..., scale=-g_R0)).  Boussinesq, unit
!! scaling 1: GV%H_to_RZ = Rho0, GV%g_Earth = GV%g_Earth_Z_T2 = g, so gR0 = Rho0 g and
!! g_R0 = g / Rho0 (:1008-1009).  The 1-D calculate_density_derivs needs no grid (no EOS_domain).
!! Same output as the linear mode.
program kappa_shear_driver
  use kappa_shear_column_mom6, only : Kappa_shear_CS, kappa_shear_column, &
                                      ks_n_newton, ks_n_ramp, ks_n_abort, ks_n_iter
  use MOM_verticalGrid, only : verticalGrid_type
  use MOM_unit_scaling, only : unit_scale_type
  use MOM_variables, only : thermo_var_ptrs
  use MOM_EOS, only : EOS_type, EOS_manual_init
  implicit none
  type(Kappa_shear_CS), pointer :: CS
  type(verticalGrid_type) :: GV
  type(unit_scale_type) :: US
  type(thermo_var_ptrs) :: tv
  type(EOS_type), target :: EOS
  real, target :: T_dummy(1,1,1), S_dummy(1,1,1)
  logical :: nonlinear
  integer :: form, nargs
  real :: rho0, grav
  real, allocatable :: Tc(:), Sc(:)
  integer :: ncase, c, nz, max_rino_it, max_ks_it
  real :: dt, f2, kappa0, ri_crit, shearmix_rate, fri_curvature, lambda, c_n, c_s, kd_seed
  real :: kd_trunc_frac, tol_err, max_kap_src_chg, tke_bg, lz_rescale
  real, allocatable :: dz(:), u(:), v(:), b(:), kappa(:), tke(:), kappa_avg(:), tke_avg(:)
  real, allocatable :: N2_init(:), S2_init(:), N2_mean(:), S2_mean(:)
  character(len=512) :: fin, fout, mode
  nargs = command_argument_count()
  nonlinear = .false.
  if (nargs == 3) then
    call get_command_argument(1, mode) ; nonlinear = (trim(mode) == 'eos')
    call get_command_argument(2, fin) ; call get_command_argument(3, fout)
  else
    call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  endif
  allocate(CS)
  US%L_to_Z = 1.0 ; US%m_to_Z = 1.0 ; US%T_to_s = 1.0 ; US%Z_to_m = 1.0 ; US%s_to_T = 1.0
  GV%Boussinesq = .true. ; GV%semi_Boussinesq = .false.
  GV%H_to_Z = 1.0 ; GV%H_to_RZ = 1.0 ; GV%g_Earth = 1.0 ; GV%g_Earth_Z_T2 = 1.0 ; GV%Rho0 = 1.0
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    if (nonlinear) then
      read(10,*) nz, dt, f2, kappa0, form, rho0, grav
      GV%Rho0 = rho0 ; GV%H_to_RZ = rho0 ; GV%g_Earth = grav ; GV%g_Earth_Z_T2 = grav
      call EOS_manual_init(EOS, form_of_EOS=form)
      tv%eqn_of_state => EOS ; tv%T => T_dummy ; tv%S => S_dummy
    else
      read(10,*) nz, dt, f2, kappa0
    endif
    read(10,*) ri_crit, shearmix_rate, fri_curvature, lambda, c_n, c_s, kd_seed, kd_trunc_frac, &
               tol_err, max_kap_src_chg, tke_bg, lz_rescale
    read(10,*) max_rino_it, max_ks_it
    allocate(dz(nz), u(nz), v(nz), b(nz), kappa(nz+1), tke(nz+1), kappa_avg(nz+1), tke_avg(nz+1))
    allocate(N2_init(nz+1), S2_init(nz+1), N2_mean(nz+1), S2_mean(nz+1))
    read(10,*) dz ; read(10,*) u ; read(10,*) v
    if (nonlinear) then
      allocate(Tc(nz), Sc(nz)) ; read(10,*) Tc ; read(10,*) Sc
    else
      read(10,*) b
    endif
    GV%ke = nz
    ! Kappa_shear_CS as kappa_shear_init sets it (MOM_kappa_shear.F90:2159-2248)
    CS%RiNo_crit = ri_crit ; CS%Shearmix_rate = shearmix_rate ; CS%FRi_curvature = fri_curvature
    CS%lambda = lambda ; CS%C_N = c_n ; CS%C_S = c_s ; CS%lambda2_N_S = 0.0
    CS%lz_rescale = lz_rescale ; CS%TKE_bg = tke_bg ; CS%kappa_0 = kappa0
    CS%kappa_seed = kd_seed ; CS%kappa_trunc = kd_trunc_frac * kappa0
    CS%kappa_tol_err = tol_err ; CS%Prandtl_turb = 1.0 ; CS%nkml = 1
    CS%max_RiNo_it = max_rino_it ; CS%max_KS_it = max_ks_it
    CS%dKdQ_iteration_bug = .false. ; CS%KS_at_vertex = .false. ; CS%eliminate_massless = .true.
    CS%vel_underflow = 0.0 ; CS%kappa_src_max_chg = max_kap_src_chg
    CS%restrictive_tolerance_check = .false. ; CS%all_layer_TKE_bug = .false.
    ks_n_newton = 0 ; ks_n_ramp = 0 ; ks_n_abort = 0 ; ks_n_iter = 0
    kappa(:) = CS%kappa_seed                                   ! :325
    if (nonlinear) then
      call kappa_shear_column(kappa, tke, dt, nz, f2, 0.0, dz, dz, u*dz, v*dz, Tc*dz, Sc*dz, &
                              kappa_avg, tke_avg, N2_init, S2_init, N2_mean, S2_mean, tv, CS, GV, US)
      deallocate(Tc, Sc)
    else
      call kappa_shear_column(kappa, tke, dt, nz, f2, 0.0, dz, dz, u*dz, v*dz, -b*dz, 0.0*dz, &
                              kappa_avg, tke_avg, N2_init, S2_init, N2_mean, S2_mean, tv, CS, GV, US)
    endif
    write(11,'(*(es25.17))') kappa_avg
    write(11,'(*(es25.17))') tke_avg
    write(11,'(4i10)') ks_n_newton, ks_n_ramp, ks_n_abort, ks_n_iter
    deallocate(dz, u, v, b, kappa, tke, kappa_avg, tke_avg, N2_init, S2_init, N2_mean, S2_mean)
  enddo
  close(10) ; close(11)
end program kappa_shear_driver
