!> Reference driver for N2b criterion 1: the NCAR bulk formulae of the FMS coupler
!! (FMScoupler 2025.02, git 5b700a4, shared/surface_flux.F90) on the points that
!! validation/bulk_points/points.py writes.
!!
!! surface_flux.F90 is compiled unchanged.  The driver calls the public surface_flux
!! (surface_flux_1d, surface_flux.F90:216-500) on an all-seawater exchange grid, so
!! ncar_ocean_fluxes (:774-897) or ncar_ocean_fluxes_multilevel (:905-1052) overrides the
!! Monin-Obukhov coefficients (:402-415) exactly as in OM4.  The namelist (input.nml in the
!! working directory, written by build_reference.sh) selects the routine and the heights;
!! raoult_sat_vap = .true. as OM4_025/input.nml:114-115.  gust = 0 (OM4's data_table uses
!! 1e-4 m/s; bulk_ncar_fms has no gust argument), land = .false., seawater = avail = .true.
!!
!! Usage: fms_ncar_driver in.txt out.txt z_atm
!!   in.txt : first line n, then n lines "u_atm v_atm t_atm q_atm t_surf p u_surf v_surf"
!!            (m/s, K, kg/kg, K, Pa, m/s); p is both p_atm and p_surf (OM4 data_table).
!!   out.txt: n lines "flux_u flux_v flux_t flux_q cd_m cd_t cd_q u_star w_atm" in FMS signs
!!            (stress on the atmosphere, sensible and vapour flux positive upward).
program fms_ncar_driver
  use fms_mod, only : fms_init, fms_end
  use sat_vapor_pres_mod, only : sat_vapor_pres_init
  use surface_flux_mod, only : surface_flux, surface_flux_init
  implicit none
  integer :: n, i, u
  real :: z_in
  character(len=512) :: fin, fout, zarg
  real, allocatable, dimension(:) :: u_atm, v_atm, t_atm, q_atm, t_surf, p, u_surf, v_surf
  real, allocatable, dimension(:) :: z_atm, t_ca, q_surf, rough_mom, rough_heat, rough_moist
  real, allocatable, dimension(:) :: rough_scale, gust
  real, allocatable, dimension(:) :: flux_t, flux_q, flux_r, flux_u, flux_v, cd_m, cd_t, cd_q
  real, allocatable, dimension(:) :: w_atm, u_star, b_star, q_star, thv_atm, thv_surf
  real, allocatable, dimension(:) :: dhdt_surf, dedt_surf, dedq_surf, drdt_surf
  real, allocatable, dimension(:) :: dhdt_atm, dedq_atm, dtaudu_atm, dtaudv_atm
  logical, allocatable, dimension(:) :: land, seawater, avail

  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  call get_command_argument(3, zarg) ; read(zarg, *) z_in
  call fms_init()
  call sat_vapor_pres_init()
  call surface_flux_init()

  open(newunit=u, file=trim(fin), status='old', action='read')
  read(u, *) n
  allocate(u_atm(n), v_atm(n), t_atm(n), q_atm(n), t_surf(n), p(n), u_surf(n), v_surf(n))
  do i = 1, n
    read(u, *) u_atm(i), v_atm(i), t_atm(i), q_atm(i), t_surf(i), p(i), u_surf(i), v_surf(i)
  enddo
  close(u)

  allocate(z_atm(n), t_ca(n), q_surf(n), rough_mom(n), rough_heat(n), rough_moist(n))
  allocate(rough_scale(n), gust(n))
  allocate(flux_t(n), flux_q(n), flux_r(n), flux_u(n), flux_v(n), cd_m(n), cd_t(n), cd_q(n))
  allocate(w_atm(n), u_star(n), b_star(n), q_star(n), thv_atm(n), thv_surf(n))
  allocate(dhdt_surf(n), dedt_surf(n), dedq_surf(n), drdt_surf(n))
  allocate(dhdt_atm(n), dedq_atm(n), dtaudu_atm(n), dtaudv_atm(n))
  allocate(land(n), seawater(n), avail(n))
  z_atm = z_in ; t_ca = t_surf ; q_surf = 0.0
  ! roughness only feeds mo_drag, whose coefficients ncar_ocean_fluxes overwrites over seawater
  ! (surface_flux.F90:390-404); 5.8e-5 m = ocean_rough.F90 roughness_mom default
  rough_mom = 5.8e-5 ; rough_heat = 5.8e-5 ; rough_moist = 5.8e-5 ; rough_scale = 5.8e-5
  gust = 0.0
  land = .false. ; seawater = .true. ; avail = .true.

  call surface_flux(t_atm, q_atm, u_atm, v_atm, p, z_atm, &
       p, t_surf, t_ca, q_surf, u_surf, v_surf, &
       rough_mom, rough_heat, rough_moist, rough_scale, gust, &
       flux_t, flux_q, flux_r, flux_u, flux_v, cd_m, cd_t, cd_q, &
       w_atm, u_star, b_star, q_star, thv_atm, thv_surf, &
       dhdt_surf, dedt_surf, dedq_surf, drdt_surf, &
       dhdt_atm, dedq_atm, dtaudu_atm, dtaudv_atm, &
       0.0, land, seawater, avail)

  open(newunit=u, file=trim(fout), status='replace', action='write')
  do i = 1, n
    write(u, '(9es26.17e3)') flux_u(i), flux_v(i), flux_t(i), flux_q(i), cd_m(i), cd_t(i), &
                             cd_q(i), u_star(i), w_atm(i)
  enddo
  close(u)
  call fms_end()
end program fms_ncar_driver
