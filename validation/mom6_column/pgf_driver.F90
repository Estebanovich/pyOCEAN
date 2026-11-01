!> Nonlinear finite-volume pressure-gradient force of MOM6 on rows of columns written by
!! eos_columns.write_pgf_cases (TEOS-10 in the 3-D core, task 3).  MOM6 git 0e9a561.
!!
!! For each case and each EOS (WRIGHT_FULL, then ROQUET_RHO, selected with EOS_manual_init as in
!! eos_driver.F90) the driver
!!   1. builds the interface heights from the bed up, e(nz+1) = -bathyT, e(K) = e(K+1) + h(k)
!!      (MOM_PressureForce_FV.F90:1146-1148, 1195-1198);
!!   2. reconstructs T and S separately with ALE_PLM_edge_values (MOM_ALE.F90:1537-1596; copied
!!      verbatim by build_reference.sh into ale_plm_column_mom6 because ALE_CS is private) with
!!      BOUNDARY_EXTRAPOLATION_PRESSURE = True (default, MOM_PressureForce_FV.F90:2186-2190) and
!!      answer_date >= 20190101 (h_neglect = GV%H_subroundoff), as TS_PLM_edge_values does
!!      (MOM_ALE.F90:1512-1531);
!!   3. calls the public int_density_dz_generic_plm (MOM_density_integrals.F90:420-751) for every
!!      layer with the arguments of PressureForce_FV_Bouss (MOM_PressureForce_FV.F90:1285-1293):
!!      rho_0 = Rho0, dz_subroundoff = GV%dZ_subroundoff, Z_0p = meanSL = 0, accurate rho_ref form,
!!      and three mass-weighting settings (MOM_PressureForce_FV.F90:2124-2149):
!!        mw = 1 "vanished": MassWghtInterp = 1 (bit 0), MassWghtInterpVanOnly = .true.,  h_nv
!!        mw = 2 "full"    : MassWghtInterp = 1 (bit 0), MassWghtInterpVanOnly = .false.
!!        mw = 3 "none"    : MassWghtInterp = 0
!!      (MASS_WEIGHT_IN_PRESSURE_GRADIENT_TOP, bit 1, is off in all three, as in NWA12);
!!   4. assembles the force as PressureForce_FV_Bouss (MOM_PressureForce_FV.F90:1250-1260 pa(1),
!!      1338-1343 pa, 1539-1556 intx_pa, 1796-1801 PFu), twice:
!!        pfu_mom6: pa(1) = g*rho_ref*(e(1) - Z_ref), Z_ref = 0, verbatim;
!!        pfu_bc  : pa(1) = (g*e(1))*(rho_ref - rho_0), the only changed line: pyOCEAN returns the
!!                  baroclinic force and its barotropic solver supplies -g grad(eta)
!!                  (core/vertical/pgf.py), so pfu_bc = pfu_mom6 + g (e_R(1) - e_L(1))/dx.
!!
!! Grid: one row of nx columns, computational columns i = 1..nx-2 and one halo column on each
!! side (symmetric memory), so the nx-1 u-faces I = 0..nx-2 lie between columns I and I+1; the row
!! is copied to j = 0, 1, 2 (the routine reads the column integrals of j = JscB..JecB+1).
!! Boussinesq, unit scaling 1.
!!
!! Usage: pgf_driver in.txt out.txt
!!   in : ncase; per case "nz nx", "rho_0 rho_ref g h_nv", bathyT(nx), then h, T, S as nz lines of
!!        nx values (surface first, k = 1 at the surface), then dx(nx-1) [m]
!!   out: per case the edge values T_t, T_b, S_t, S_b (nz lines of nx values each), then for each
!!        EOS and mw = 1..3: dpa, intz_dpa (nz x nx), intx_dpa, pfu_mom6, pfu_bc (nz x nx-1)
program pgf_driver
  use MOM_density_integrals, only : int_density_dz_generic_plm
  use MOM_EOS, only : EOS_type, EOS_manual_init, EOS_WRIGHT_FULL, EOS_ROQUET_RHO
  use MOM_hor_index, only : hor_index_type
  use MOM_grid, only : ocean_grid_type
  use MOM_verticalGrid, only : verticalGrid_type
  use MOM_unit_scaling, only : unit_scale_type
  use MOM_variables, only : thermo_var_ptrs
  use ale_plm_column_mom6, only : ALE_CS, ALE_PLM_edge_values
  implicit none
  type(EOS_type) :: EOS
  type(hor_index_type) :: HI
  type(ocean_grid_type) :: G
  type(verticalGrid_type) :: GV
  type(unit_scale_type) :: US
  type(thermo_var_ptrs) :: tv
  type(ALE_CS) :: ACS
  real, allocatable :: h(:,:,:), T(:,:,:), S(:,:,:), e(:,:,:), T_t(:,:,:), T_b(:,:,:), S_t(:,:,:), S_b(:,:,:)
  real, allocatable :: bathyT(:,:), Z_0p(:,:), dpa(:,:,:), intz_dpa(:,:,:), intx_dpa(:,:,:)
  real, allocatable :: pa(:,:,:), intx_pa(:,:,:), PFu(:,:,:), IdxCu(:,:), dx(:), row(:)
  real :: rho_0, rho_ref, g_e, h_nv, I_Rho0, GxRho_ref, h_neglect, dz_neglect
  integer :: ncase, c, nz, nx, f, mw, k, i, j, pass, forms(2), mwi(3)
  logical :: van_only(3)
  character(len=512) :: fin, fout

  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  US%L_to_Z = 1.0 ; US%m_to_Z = 1.0 ; US%Z_to_m = 1.0 ; US%T_to_s = 1.0 ; US%s_to_T = 1.0
  US%L_to_m = 1.0 ; US%m_to_L = 1.0
  GV%Boussinesq = .true. ; GV%semi_Boussinesq = .false. ; GV%H_to_Z = 1.0 ; GV%Z_to_H = 1.0
  GV%m_to_H = 1.0 ; GV%H_to_m = 1.0 ; GV%Angstrom_Z = 1.0e-10 ; GV%Angstrom_H = 1.0e-10
  GV%H_subroundoff = 1e-20 * max(GV%Angstrom_H, GV%m_to_H*1e-17)    ! MOM_verticalGrid.F90:219
  GV%dZ_subroundoff = 1e-20 * max(GV%Angstrom_Z, US%m_to_Z*1e-17)   ! MOM_verticalGrid.F90:220
  h_neglect = GV%H_subroundoff ; dz_neglect = GV%dZ_subroundoff     ! MOM_PressureForce_FV.F90:1121-1122
  ACS%answer_date = 99991231
  forms = (/ EOS_WRIGHT_FULL, EOS_ROQUET_RHO /)
  mwi = (/ 1, 1, 0 /) ; van_only = (/ .true., .false., .false. /)

  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    read(10,*) nz, nx
    read(10,*) rho_0, rho_ref, g_e, h_nv
    GV%ke = nz ; GV%g_Earth = g_e ; GV%Rho0 = rho_0
    HI%isc = 1 ; HI%iec = nx - 2 ; HI%jsc = 1 ; HI%jec = 1
    HI%isd = 0 ; HI%ied = nx - 1 ; HI%jsd = 0 ; HI%jed = 2
    HI%IscB = 0 ; HI%IecB = nx - 2 ; HI%JscB = 0 ; HI%JecB = 1
    HI%IsdB = -1 ; HI%IedB = nx - 1 ; HI%JsdB = -1 ; HI%JedB = 2
    G%isc = HI%isc ; G%iec = HI%iec ; G%jsc = HI%jsc ; G%jec = HI%jec
    G%isd = HI%isd ; G%ied = HI%ied ; G%jsd = HI%jsd ; G%jed = HI%jed
    G%IscB = HI%IscB ; G%IecB = HI%IecB ; G%JscB = HI%JscB ; G%JecB = HI%JecB
    G%IsdB = HI%IsdB ; G%IedB = HI%IedB ; G%JsdB = HI%JsdB ; G%JedB = HI%JedB
    allocate(h(0:nx-1,0:2,nz), T(0:nx-1,0:2,nz), S(0:nx-1,0:2,nz), e(0:nx-1,0:2,nz+1))
    allocate(T_t(0:nx-1,0:2,nz), T_b(0:nx-1,0:2,nz), S_t(0:nx-1,0:2,nz), S_b(0:nx-1,0:2,nz))
    allocate(bathyT(0:nx-1,0:2), Z_0p(0:nx-1,0:2), dpa(0:nx-1,0:2,nz), intz_dpa(0:nx-1,0:2,nz))
    allocate(intx_dpa(-1:nx-1,0:2,nz), pa(0:nx-1,0:2,nz+1), intx_pa(-1:nx-1,0:2,nz+1))
    allocate(PFu(-1:nx-1,0:2,nz), IdxCu(-1:nx-1,0:2), dx(nx-1), row(nx))
    read(10,*) row
    do j = 0, 2 ; bathyT(:,j) = row ; enddo
    do k = 1, nz ; read(10,*) row ; do j = 0, 2 ; h(:,j,k) = row ; enddo ; enddo
    do k = 1, nz ; read(10,*) row ; do j = 0, 2 ; T(:,j,k) = row ; enddo ; enddo
    do k = 1, nz ; read(10,*) row ; do j = 0, 2 ; S(:,j,k) = row ; enddo ; enddo
    read(10,*) dx
    IdxCu(:,:) = 0.0
    do j = 0, 2 ; do i = 0, nx - 2 ; IdxCu(i,j) = 1.0 / dx(i+1) ; enddo ; enddo
    Z_0p(:,:) = 0.0                                          ! G%meanSL (MOM_PressureForce_FV.F90:1270-1274)
    do j = 0, 2 ; do i = 0, nx - 1
      e(i,j,nz+1) = -bathyT(i,j)
    enddo ; enddo
    do j = 0, 2 ; do k = nz, 1, -1 ; do i = 0, nx - 1
      e(i,j,k) = e(i,j,k+1) + h(i,j,k)*GV%H_to_Z
    enddo ; enddo ; enddo
    call ALE_PLM_edge_values(ACS, G, GV, h, S, .true., S_t, S_b)
    call ALE_PLM_edge_values(ACS, G, GV, h, T, .true., T_t, T_b)
    call write_layers(T_t, 0, nx - 1) ; call write_layers(T_b, 0, nx - 1)
    call write_layers(S_t, 0, nx - 1) ; call write_layers(S_b, 0, nx - 1)
    I_Rho0 = 1.0 / GV%Rho0
    GxRho_ref = GV%g_Earth * rho_ref
    do f = 1, 2
      call EOS_manual_init(EOS, form_of_EOS=forms(f))
      do mw = 1, 3
        intx_dpa(:,:,:) = 0.0
        do k = 1, nz
          call int_density_dz_generic_plm(k, tv, T_t, T_b, S_t, S_b, e, &
                    rho_ref, GV%Rho0, GV%g_Earth, dz_neglect, bathyT, &
                    HI, GV, EOS, US, .false., dpa(:,:,k), intz_dpa(:,:,k), &
                    intx_dpa(:,:,k), MassWghtInterp=mwi(mw), &
                    use_inaccurate_form=.false., Z_0p=Z_0p, &
                    MassWghtInterpVanOnly=van_only(mw), h_nv=h_nv)
        enddo
        call write_layers(dpa, 0, nx - 1) ; call write_layers(intz_dpa, 0, nx - 1)
        call write_layers(intx_dpa, 0, nx - 2)
        do pass = 1, 2
          if (pass == 1) then
            do j = 0, 2 ; do i = 0, nx - 1
              pa(i,j,1) = GxRho_ref * (e(i,j,1) - 0.0)       ! MOM_PressureForce_FV.F90:1257-1259, Z_ref = 0
            enddo ; enddo
          else
            do j = 0, 2 ; do i = 0, nx - 1
              pa(i,j,1) = (GV%g_Earth * e(i,j,1)) * (rho_ref - GV%Rho0)   ! pyOCEAN baroclinic convention
            enddo ; enddo
          endif
          do k = 1, nz ; do j = 0, 2 ; do i = 0, nx - 1
            pa(i,j,k+1) = pa(i,j,k) + dpa(i,j,k)               ! :1338-1343
          enddo ; enddo ; enddo
          do j = 1, 1 ; do i = 0, nx - 2
            intx_pa(i,j,1) = 0.5*(pa(i,j,1) + pa(i+1,j,1))     ! :1539-1542
          enddo ; enddo
          do k = 1, nz ; do j = 1, 1 ; do i = 0, nx - 2
            intx_pa(i,j,k+1) = intx_pa(i,j,k) + intx_dpa(i,j,k)   ! :1550-1556
          enddo ; enddo ; enddo
          PFu(:,:,:) = 0.0
          do k = 1, nz ; do j = 1, 1 ; do i = 0, nx - 2      ! :1796-1801
            PFu(i,j,k) = (((pa(i,j,k)*h(i,j,k) + intz_dpa(i,j,k)) - &
                           (pa(i+1,j,k)*h(i+1,j,k) + intz_dpa(i+1,j,k))) + &
                          ((h(i+1,j,k) - h(i,j,k)) * intx_pa(i,j,k) - &
                           (e(i+1,j,k+1) - e(i,j,k+1)) * intx_dpa(i,j,k) * GV%Z_to_H)) * &
                         ((2.0*I_Rho0*IdxCu(i,j)) / &
                          ((h(i,j,k) + h(i+1,j,k)) + h_neglect))
          enddo ; enddo ; enddo
          call write_layers(PFu, 0, nx - 2)
        enddo
      enddo
    enddo
    deallocate(h, T, S, e, T_t, T_b, S_t, S_b, bathyT, Z_0p, dpa, intz_dpa, intx_dpa, pa, intx_pa, PFu, IdxCu, dx, row)
  enddo
  close(10) ; close(11)

contains

  !> Write a(i0:i1, 1, k) for k = 1..nz, one line per layer (surface first).
  subroutine write_layers(a, i0, i1)
    real, intent(in) :: a(:,:,:)
    integer, intent(in) :: i0, i1
    integer :: kk, lb
    ! assumed-shape arrays start at 1: shift the halo offset of the actual argument
    lb = lbound_i(a)
    do kk = 1, size(a, 3)
      write(11,'(*(es25.17))') a(i0 - lb + 1:i1 - lb + 1, 2, kk)
    enddo
  end subroutine write_layers

  integer function lbound_i(a)
    real, intent(in) :: a(:,:,:)
    if (size(a, 1) == nx) then
      lbound_i = 0
    else
      lbound_i = -1
    endif
  end function lbound_i

end program pgf_driver
