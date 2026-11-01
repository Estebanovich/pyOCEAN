!> Drag-driven BBL mixing of MOM6 on single columns written by columns.write_bbl_cases and
!! columns.write_bbl_tke_cases.  add_LOTW_BBL_diffusivity and set_BBL_TKE are module procedures
!! of MOM_set_diffusivity.F90 that read the private set_diffusivity_CS, so build_reference.sh
!! copies both verbatim into the module bbl_column_mom6 with that type made public
!! (MOM_set_diffusivity.F90:62-203, :1606-1789, :1947-2152; MOM6 git 0e9a561).  The MOM6
!! derived types they take (ocean_grid_type, verticalGrid_type, unit_scale_type,
!! thermo_var_ptrs, forcing, vertvisc_type) are public with public fields, so this driver builds
!! the minimal ones by hand rather than through MOM_grid_init and the parameter files: a 3 × 3
!! cell grid (computational cell i = j = 1, one halo cell, symmetric memory), Boussinesq,
!! unit scaling 1 (H = Z = L = m, T = s).  Only the fields the two routines read are set.
!!
!! Usage: bbl_driver lotw in.txt out.txt   (add_LOTW_BBL_diffusivity, out: Kd_BBL(nz+1) per case)
!!        bbl_driver tke  in.txt out.txt   (set_BBL_TKE, out: "ustar_BBL BBL_meanKE_loss" per case)
!!        bbl_driver n2   in.txt out.txt   (find_N2, out: N2_int(nz+1) per case; BBL N2_bot fix,
!!                                          2026-11-01).  Per case: "nz form rho0 g r0 drdt drds",
!!                                          then h, T, S (nz each); form 1 LINEAR (r0, drdt, drds),
!!                                          4 WRIGHT_FULL, 7 ROQUET_RHO (EOS_manual_init, as eos_driver).
!!                                          T_f = T, S_f = S (no massless layers), p_surf absent,
!!                                          no tidal mixing (h_amp = 0).
!! Columns surface first (k = 1 at the surface), thicknesses in m, N2 in s-2.
program bbl_driver
  use bbl_column_mom6, only : set_diffusivity_CS, add_LOTW_BBL_diffusivity, set_BBL_TKE, find_N2
  use MOM_EOS, only : EOS_type, EOS_manual_init
  use MOM_grid, only : ocean_grid_type
  use MOM_verticalGrid, only : verticalGrid_type
  use MOM_unit_scaling, only : unit_scale_type
  use MOM_variables, only : thermo_var_ptrs, vertvisc_type
  use MOM_forcing_type, only : forcing
  use MOM_open_boundary, only : ocean_OBC_type
  implicit none
  type(set_diffusivity_CS), pointer :: CS
  type(ocean_grid_type) :: G
  type(verticalGrid_type) :: GV
  type(unit_scale_type) :: US
  type(thermo_var_ptrs) :: tv
  type(forcing) :: fluxes
  type(vertvisc_type) :: visc
  type(ocean_OBC_type), pointer :: OBC => NULL()
  real, dimension(:,:,:), pointer :: Kd_BBL
  real, allocatable :: h(:,:,:), u(:,:,:), v(:,:,:), N2_int(:,:), rho_bot(:), Kd_int(:,:)
  real, allocatable :: dz(:), n2(:)
  real, allocatable, target :: Tf(:,:,:), Sf(:,:,:)
  real, allocatable :: dRho_int(:,:), N2_lay(:,:), N2_bot(:), h_bot(:)
  integer, allocatable :: k_bot(:)
  type(EOS_type), target :: EOS
  integer :: form
  real :: rho0, grav, r0, drdt, drds
  real :: ustar, absf, meanKE, effic, max_decay, von_karm, kd_max, omega
  real :: cdrag, uvel, ubg, bbl_thick, kv_min, ustar_f, kv_bbl, cdrag_sqrt
  integer :: ncase, c, nz, use_omega, k
  character(len=8) :: mode
  character(len=512) :: fin, fout

  call get_command_argument(1, mode)
  call get_command_argument(2, fin) ; call get_command_argument(3, fout)
  allocate(CS)
  US%L_to_Z = 1.0 ; US%m_to_Z = 1.0 ; US%Z_to_m = 1.0 ; US%T_to_s = 1.0 ; US%s_to_T = 1.0
  US%L_to_m = 1.0 ; US%m_to_L = 1.0
  GV%Boussinesq = .true. ; GV%semi_Boussinesq = .false. ; GV%H_to_Z = 1.0 ; GV%Z_to_H = 1.0
  GV%m_to_H = 1.0 ; GV%H_to_m = 1.0 ; GV%Angstrom_Z = 1.0e-10
  GV%dZ_subroundoff = 1e-20 * max(GV%Angstrom_Z, US%m_to_Z*1e-17)   ! MOM_verticalGrid.F90:220
  ! 3 x 3 cells, computational cell (1,1), symmetric memory (MOM_grid.F90:303-308)
  G%isc = 1 ; G%iec = 1 ; G%jsc = 1 ; G%jec = 1
  G%isd = 0 ; G%ied = 2 ; G%jsd = 0 ; G%jed = 2
  G%IscB = 0 ; G%IecB = 1 ; G%JscB = 0 ; G%JecB = 1
  G%IsdB = -1 ; G%IedB = 2 ; G%JsdB = -1 ; G%JedB = 2
  G%HI%isc = G%isc ; G%HI%iec = G%iec ; G%HI%isd = G%isd ; G%HI%ied = G%ied   ! EOS_domain (find_N2)
  allocate(G%CoriolisBu(G%IsdB:G%IedB, G%JsdB:G%JedB))
  allocate(G%mask2dCu(G%IsdB:G%IedB, G%jsd:G%jed), G%areaCu(G%IsdB:G%IedB, G%jsd:G%jed))
  allocate(G%mask2dCv(G%isd:G%ied, G%JsdB:G%JedB), G%areaCv(G%isd:G%ied, G%JsdB:G%JedB))
  allocate(G%IareaT(G%isd:G%ied, G%jsd:G%jed), G%mask2dT(G%isd:G%ied, G%jsd:G%jed))
  G%mask2dT(:,:) = 1.0
  ! equal 1 km x 1 km cells
  G%mask2dCu(:,:) = 1.0 ; G%mask2dCv(:,:) = 1.0
  G%areaCu(:,:) = 1.0e6 ; G%areaCv(:,:) = 1.0e6 ; G%IareaT(:,:) = 1.0 / 1.0e6
  allocate(visc%ustar_BBL(G%isd:G%ied, G%jsd:G%jed), visc%BBL_meanKE_loss(G%isd:G%ied, G%jsd:G%jed))
  allocate(visc%BBL_meanKE_loss_sqrtCd(G%isd:G%ied, G%jsd:G%jed))
  ! visc%Ray_u/Ray_v not allocated (no Rayleigh drag), fluxes%ustar_tidal and
  ! fluxes%BBL_tidal_dis not associated, tv%SpV_avg not allocated (Boussinesq)
  CS%initialized = .true. ; CS%bottomdraglaw = .true.
  CS%ePBL_BBL_effic = 0.0 ; CS%ePBL_BBL_mstar = .false.
  CS%LOTW_BBL_answer_date = 99991231

  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    if (trim(mode) == 'n2') then
      read(10,*) nz, form, rho0, grav, r0, drdt, drds
      GV%ke = nz
      ! Boussinesq, unit scaling 1: H_to_RZ = Rho0 (MOM_verticalGrid.F90:224), G_Rho0 = g/Rho0
      GV%Rho0 = rho0 ; GV%H_to_RZ = rho0 ; GV%H_to_kg_m2 = rho0 ; GV%g_Earth = grav
      GV%g_Earth_Z_T2 = grav ; GV%Angstrom_H = GV%Angstrom_Z
      GV%H_subroundoff = 1e-20 * max(GV%Angstrom_H, GV%m_to_H*1e-17)   ! MOM_verticalGrid.F90:219
      CS%use_tidal_mixing = .false. ; CS%dz_BBL_avg_min = 0.0
      if (form == 1) then
        call EOS_manual_init(EOS, form_of_EOS=form, Rho_T0_S0=r0, drho_dT=drdt, dRho_dS=drds)
      else
        call EOS_manual_init(EOS, form_of_EOS=form)
      endif
      tv%eqn_of_state => EOS
      allocate(h(G%isd:G%ied, G%jsd:G%jed, nz), Tf(G%isd:G%ied, G%jsd:G%jed, nz))
      allocate(Sf(G%isd:G%ied, G%jsd:G%jed, nz), dz(nz))
      allocate(dRho_int(G%isd:G%ied, nz+1), N2_int(G%isd:G%ied, nz+1), N2_lay(G%isd:G%ied, nz))
      allocate(N2_bot(G%isd:G%ied), rho_bot(G%isd:G%ied), h_bot(G%isd:G%ied), k_bot(G%isd:G%ied))
      read(10,*) dz ; do k = 1, nz ; h(:,:,k) = dz(k) ; enddo
      read(10,*) dz ; do k = 1, nz ; Tf(:,:,k) = dz(k) ; enddo
      read(10,*) dz ; do k = 1, nz ; Sf(:,:,k) = dz(k) ; enddo
      tv%T => Tf ; tv%S => Sf
      call find_N2(h, tv, Tf, Sf, fluxes, 1, G, GV, US, CS, dRho_int, N2_lay, N2_int, N2_bot, rho_bot, &
                   h_bot, k_bot)   ! find_rho_bottom writes both (MOM_interface_heights.F90:503-504)
      write(11,'(*(es25.17))') N2_int(1,:)
      nullify(tv%T, tv%S, tv%eqn_of_state)
      deallocate(h, Tf, Sf, dz, dRho_int, N2_int, N2_lay, N2_bot, rho_bot, h_bot, k_bot)
    elseif (trim(mode) == 'lotw') then
      read(10,*) nz, ustar, absf, meanKE
      read(10,*) effic, max_decay, von_karm, kd_max, omega, use_omega
      allocate(dz(nz), n2(nz+1))
      read(10,*) dz ; read(10,*) n2
      GV%ke = nz
      ! set_diffusivity_CS as set_diffusivity_init sets it (SD:2417-2524)
      CS%BBL_effic = effic ; CS%von_karm = von_karm ; CS%Kd_max = kd_max ; CS%omega = omega
      CS%LOTW_BBL_use_omega = (use_omega /= 0) ; CS%cdrag = 0.003
      CS%IMax_decay = 0.0 ; if (max_decay > 0.0) CS%IMax_decay = 1.0 / max_decay   ! SD:2441-2442
      G%CoriolisBu(:,:) = absf
      visc%ustar_BBL(:,:) = ustar ; visc%BBL_meanKE_loss(:,:) = meanKE
      allocate(h(G%isd:G%ied, G%jsd:G%jed, nz), u(G%IsdB:G%IedB, G%jsd:G%jed, nz))
      allocate(v(G%isd:G%ied, G%JsdB:G%JedB, nz), N2_int(G%isd:G%ied, nz+1), rho_bot(G%isd:G%ied))
      allocate(Kd_int(G%isd:G%ied, nz+1), Kd_BBL(G%isd:G%ied, G%jsd:G%jed, nz+1))
      do k = 1, nz ; h(:,:,k) = dz(k) ; enddo
      u(:,:,:) = 0.0 ; v(:,:,:) = 0.0 ; rho_bot(:) = 1035.0
      do k = 1, nz+1 ; N2_int(:,k) = n2(k) ; enddo
      Kd_int(:,:) = 0.0 ; Kd_BBL(:,:,:) = 0.0
      call add_LOTW_BBL_diffusivity(h, u, v, tv, fluxes, visc, 1, N2_int, rho_bot, Kd_int, &
                                    G, GV, US, CS, Kd_BBL)
      write(11,'(*(es25.17))') Kd_BBL(1,1,:)
      deallocate(dz, n2, h, u, v, N2_int, rho_bot, Kd_int, Kd_BBL)
    else
      read(10,*) nz, cdrag, uvel, ubg, bbl_thick, kv_min
      allocate(dz(nz)) ; read(10,*) dz
      GV%ke = nz
      CS%cdrag = cdrag ; CS%BBL_effic = 0.2
      cdrag_sqrt = sqrt(cdrag)
      ! Inputs from set_viscous_BBL (MOM_set_viscosity.F90) for u = U, v = 0 in every layer:
      ! u* = sqrt(Cd) * (sum hweight*sqrt(u^2 + v_at_u^2 + U_bg^2)) / hwtot (:687-689, :711); at the
      ! v faces sqrt(v^2 + u_at_v^2 + U_bg^2) is the same; kv_bbl = (sqrt(Cd) u*) bbl_thick
      ! (:1115), max(KV_BBL_MIN, kv_bbl) (:1142).
      ustar_f = cdrag_sqrt * sqrt(uvel*uvel + ubg*ubg)
      kv_bbl = max(kv_min, (cdrag_sqrt*ustar_f) * bbl_thick)
      allocate(visc%Kv_bbl_u(G%IsdB:G%IedB, G%jsd:G%jed), visc%bbl_thick_u(G%IsdB:G%IedB, G%jsd:G%jed))
      allocate(visc%Kv_bbl_v(G%isd:G%ied, G%JsdB:G%JedB), visc%bbl_thick_v(G%isd:G%ied, G%JsdB:G%JedB))
      visc%Kv_bbl_u(:,:) = kv_bbl ; visc%Kv_bbl_v(:,:) = kv_bbl
      visc%bbl_thick_u(:,:) = bbl_thick ; visc%bbl_thick_v(:,:) = bbl_thick
      allocate(h(G%isd:G%ied, G%jsd:G%jed, nz), u(G%IsdB:G%IedB, G%jsd:G%jed, nz))
      allocate(v(G%isd:G%ied, G%JsdB:G%JedB, nz))
      do k = 1, nz ; h(:,:,k) = dz(k) ; enddo
      u(:,:,:) = uvel ; v(:,:,:) = 0.0
      call set_BBL_TKE(u, v, h, tv, fluxes, visc, G, GV, US, CS, OBC)
      write(11,'(2es25.17)') visc%ustar_BBL(1,1), visc%BBL_meanKE_loss(1,1)
      deallocate(dz, h, u, v, visc%Kv_bbl_u, visc%bbl_thick_u, visc%Kv_bbl_v, visc%bbl_thick_v)
    endif
  enddo
  close(10) ; close(11)
end program bbl_driver
