# Mezcla diapicna del BBL por energía de arrastre — lectura de referencia (MOM6, Legg 2006, pyOCEAN)

Lectura del 2026-10-30. MOM6 en `~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/`; SD = `MOM_set_diffusivity.F90`,
SV = `MOM_set_viscosity.F90`. Es una implementación de referencia, no la verdad; lo no comprobado va **[verificar]**.

## 1. MOM6

**Orden en `set_diffusivity`, por fila j:**
1. `find_N2`, luego la mezcla de fondo (background).
2. `find_TKE_to_Kd` (SD:560-561), solo si BBL ON y LOTW OFF (SD:365-367).
3. Suma de `Kd_shear` de Jackson: `Kd_int = Kd_shear + ½(Kd_lay(k−1)+Kd_lay(k))` y `Kd_lay += ½(Kd_shear(K)+Kd_shear(K+1))` (SD:571-581).
4. Mezcla de marea y ML-rad.
5. BBL (SD:668-680): `if (BOTTOMDRAGLAW .and. BBL_EFFIC>0)` llama a `add_LOTW_BBL_diffusivity` o, si no, a `add_drag_diffusivity`. Son excluyentes.
6. `KD_ADD` y el piso de disipación.

`set_BBL_TKE` se ejecuta al inicio del paso diabático (`MOM_diabatic_driver.F90:374`). Ninguna de las dos rutinas BBL toca la viscosidad: solo producen Kd (`Kd_BBL` es diagnóstico; `Kv_slow` solo recibe el background, SD:374, 486-488). No hay número de Prandtl.

**u\* y energía.**
- SV:700-711: u\* = √C_d · ū, con ū = media ponderada por espesor de √(u² + v_at_u² + U_bg²) sobre HBBL (U_bg = `DRAG_BG_VEL`).
- Espesor del BBL de Killworth & Edwards (1999) (SV:886-918): h_bbl = h_N·u\*/(½u\* + √(¼u\*² + (2f h_N)²)), donde h_N cumple ∫(ρ − ρ_top) = 400 ρ₀ u\*²/g (SV:770).
- kv_bbl = √C_d u\* h_bbl (SV:1115), con `max(KV_BBL_MIN, ·)` (SV:1142).
- `set_BBL_TKE` (SD:1947-2152): por cara, u\*_cara = kv_bbl/(√C_d h_bbl) y u2_bbl = (media del componente normal sobre bbl_thick)².
  - `ustar_BBL = √(½ Σ_4caras A u*² / A_T)` (SD:2132).
  - `BBL_meanKE_loss = √C_d Σ_4caras A u*·u2_bbl / A_T` (SD:2137).
- **[verificar]** Hecho a mano, esto da ≈ √2 u\* y ≈ 2 C_d|U|³. Hay que comprobar el factor con una columna de MOM6 compilado antes de copiarlo. Si `KV_BBL_MIN` recorta kv_bbl, el u\* recuperado queda inflado.

**`add_drag_diffusivity`** (SD:1364-1601; esquema de Legg, por defecto)
- Decaimiento: `I2decay = |f|/u*`, con piso ½/`BBL_MIXING_MAX_DECAY` (SD:1461-1469).
- Energía inicial: `TKE₀ = BBL_EFFIC·exp(−I2decay h_nz)·BBL_meanKE_loss` (SD:1470-1474).
- Tope de la zona de mezcla `Rho_top`: ∫(ρ−ρ_top)dh = 400 ρ₀ u\*²/g, con las densidades objetivo de capa `GV%Rlay` (SD:1480-1506).
- Bucle k = nz−1…2 (SD:1509-1599); la capa del fondo nunca recibe Kd (SD:1051).
  - `TKE *= exp(−I2decay(h_k+h_{k+1}))`.
  - Reparto cuadrático en densidad (SD:1523-1529).
- Con `BBL_MIXING_AS_MAX` (SD:1540-1560): `Kd_lay = max(Kd_lay, TKE·TKE_to_Kd)`.
- `find_TKE_to_Kd` en modo de capas: 1/(g/ρ₀·Δρ_lay + Ω²dz) (SD:1066-1090).
  - Con `SIMPLE_TKE_TO_KD`: 1/(dz(N²+Ω²)) y maxTKE = dz(N²+Ω²) KD_MAX (SD:951-962).
  - Ese modo exige KD_MAX > 0 y MOM6 avisa que "no es fiable con USE_REGRIDDING" (SD:2518-2523).

**`add_LOTW_BBL_diffusivity`** (SD:1606-1789; OM4, independiente de la coordenada)
- `Idecay = max(IMax_decay, |f|/u*)`, es decir, escala min(u\*/|f|, 200 m) (SD:1699-1700).
- `TKE_col = BBL_EFFIC·BBL_meanKE_loss` (SD:1709).
- Bucle K = nz…2 desde el fondo:
  - `TKE *= exp(−Idecay h_k)` (SD:1741).
  - Kd_wall = κ u\*² z_b (D − z)/(u\* D + |f| h_b (D − z)), con κ = `VON_KARMAN_BBL` = 0.41 (SD:1756).
  - Consumo: `TKE_Kd_wall = Kd_wall·dz_int·max(N², N2_min)`, con N2_min = Ω² si `LOTW_BBL_USE_OMEGA` (SD:1666-1667, 1762).
  - Kd se reescala para no gastar más que la TKE restante (SD:1765-1767).
  - Si TKE_Kd_wall = 0, entonces `Kd_wall = KD_MAX` (SD:1771).
- Es aditivo: `Kd_int += Kd_wall` y `Kd_lay += ½(Kd_wall + Kd_lower)` (SD:1782-1783).

**Defectos** (SD:2417-2524): BOTTOMDRAGLAW T, CDRAG 0.003, BBL_EFFIC 0.20, BBL_MIXING_MAX_DECAY 200 m, BBL_MIXING_AS_MAX T, USE_LOTW_BBL_DIFFUSIVITY F, LOTW_BBL_USE_OMEGA T, SIMPLE_TKE_TO_KD F, KD_MAX −1. En SV: HBBL (SV:3129), DRAG_BG_VEL (SV:3179) y BBL_THICK_MIN (SV:3200).

## 2. Configuraciones

- **MOM6 DOME** (`ocean_only/DOME/MOM_input`:230-247, 366-379; `MOM_parameter_doc.all`:795-849, 1427-1506):
  - modo de capas, `add_drag_diffusivity` con EFFIC 0.2, DECAY 200 m, AS_MAX T, KD_MAX −1;
  - CDRAG 0.002, **DRAG_BG_VEL 0.05**, HBBL 10, KV_BBL_MIN 1e-4, KD 1e-4, Jackson T.
- **OM4_025 y OM4_05** (`ice_ocean_SIS2/OM4_025/MOM_input`:404-424, 661-689):
  - ALE, ePBL, **USE_LOTW_BBL_DIFFUSIVITY T**, AS_MAX F, SIMPLE_TKE_TO_KD T, LOTW_BBL_USE_OMEGA T, VON_KARMAN_BBL 0.41, BBL_EFFIC 0.2;
  - **KD_MAX 0.1**, CDRAG 0.003, DRAG_BG_VEL 0.1, HBBL 10, KV_BBL_MIN 0, KD 1.5e-5, Jackson T, EPBL_BBL_EFFIC 0.

## 3. Legg et al. (2006), Apéndice A

Archivo `Legg_2006 … Models.md`, líneas 1410-1459; parámetros en 152-165 y 1465.

- (A.1): E_BBL = ρ₀ c_D |u_BBL| u_BBL·u_BBL. El texto escribe "= ρ₀u\*³", lo que no es consistente con u\* = √c_D|u| **[verificar en el PDF]**.
- (A.2): κ_{B−1} = max(κ_bg, κ_Ri, γ E_BBL exp[−(h_B + h_{B−1}/2) f/(κu\*)] / (g Δρ_{B−1})), con κ = 0.4.
- Corridas isopicnas: γ = 0.2, C_d = 2e-3, U_bg = 5 cm/s, κ_bg = 1e-4 y mezcla por Ri de Hallberg (2000).
- Sin la mezcla BBL aparece una lámina densa, viscosa y adiabática (Fig. A.1).

**Diferencias entre Legg y MOM6:**
- Legg pone toda la energía en una sola capa, la segunda desde el fondo. MOM6 la reparte con un perfil y deja subir el remanente.
- Escala de decaimiento: Legg usa 0.4 u\*/f; MOM6 usa u\*/(2f) (`add_drag`) o u\*/f (LOTW), con tope de 200 m.
- MOM6 agrega Ω² y el tope de realizabilidad.
- LOTW es otro esquema: pared logarítmica limitada por la energía, y aditivo.

## 4. pyOCEAN hoy

- `core/components/bottom_drag.py`:
  - `QuadraticBedDrag.rates` (l. 54-68): r = C_d √(ū_bbl² + v̄² + U_bg²) y fracciones f_k sobre hbbl en las caras.
  - `bed_ustar2` (l. 77-85): |τ_b|/ρ₀ en los centros.
  - `LogLawBedDrag`: C_d(d_b).
- `core/model/driver.py`:
  - `r_u, r_v, f_u, f_v` existen solo en el dict local `visc` (l. 548-559); `ctx.ustar2_bot` (l. 571-572, 626-640).
  - En la fase 7, `_kpp_update` (l. 328-350) tiene u, v en centros, ρ = EOS(T, S), h húmedo y `ctx.ustar2_bot`.
  - **No están disponibles** |U_bbl| ni la energía. Hay que guardar r (o E) en `ctx`. u\*³/√C_d es incorrecto con U_bg ≠ 0 o con ley logarítmica.
- `core/components/closure.py`:
  - `ShearAddedClosure` (l. 877-1076) es aditivo (κ = κ_base + κ_shear), con fusión de capas delgadas y columnas secas resueltas.
  - `KPPClosure` ignora `ustar2_bot` (l. 707).
  - El GLS usa `ustar2_bot` como condición de fondo.
- DOME (`validation/dome/cases.py:77`, `run_pyocean.py:117`): cd = 2e-3 y hbbl = 10 m, pero **`drag_bg_vel` = 0** (defecto, `sigma_baroclinic.py:1109`). MOM6 DOME y Legg usan 0.05 m/s.

## 5. CROCO y MITgcm

- CROCO no tiene un equivalente por energía: la mezcla de fondo viene del GLS de fondo o de `LMD_BKPP` (`lmd_bkpp2005.F:132-264`, KPP de fondo con Ri_cr = 0.45).
- MITgcm: `pkg/bbl` es una subcapa advectiva y `pkg/down_slope` es Campin–Goosse; ninguno es mezcla por energía.
- Las corridas z de Legg no usaron esquema BBL (líneas 147-158 del texto).

## Riesgos

- Doble conteo con el GLS: se rechaza en el pre-flight.
- KPP: sin solape salvo en columnas someras donde la OBL llega al fondo.
- `add_drag` depende de capas y densidades objetivo: en z\*/σ solo funcionaría `SIMPLE` con KD_MAX > 0, que MOM6 desaconseja con regridding. **LOTW es lo portable a z\*/σ/híbrido.**
- Interfaces sin masa: LOTW da `Kd = KD_MAX` si dz_int = 0. Hay que fusionar las capas delgadas antes (como Jackson y el GLS) y exigir KD_MAX > 0.
- N² ≤ 0: LOTW usa el piso Ω².
- f ≈ 0: la escala queda limitada por los 200 m.
- Posible factor 2 en `BBL_meanKE_loss` **[verificar]**.
- Elegir y documentar con qué velocidades se calcula u\*: uⁿ (r) o u_new.
