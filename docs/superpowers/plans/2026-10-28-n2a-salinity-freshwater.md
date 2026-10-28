# N2a: salinidad, agua dulce real y balance exacto de varianza — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que el núcleo 3-D tenga salinidad activa, evaporación y precipitación como flujo real de volumen, ríos y un balance discreto exacto de la varianza de salinidad, validado contra la relación de mezcla exacta de Lorenz et al. (2021) en su estuario idealizado.

**Architecture:**
- **Estado.** S es un campo de `SigmaState` y viaja en el stack de trazadores con T, con el mismo esquema y el mismo FCT.
- **Densidad.** La EOS lineal gana un término β(S − S₀); todos los consumidores de la densidad reciben S.
- **Agua dulce.** Entra a la superficie libre como fuente de volumen. En los trazadores es un **subpaso separado sobre la capa superior** después de la advección: el volumen cambia y la sal no. Así su término de varianza queda aislado y el balance cierra a redondeo.
- **Diagnósticos opcionales.** El balance por operador y el TEF en un transecto se calculan dentro del modelo con los flujos efectivos del paso.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, SciPy, netCDF4, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md`.

**Fuentes:**
- **Artículo:** Lorenz et al. 2021, biblioteca local (`Lorenz_2021 - …md`): ec. (24)–(36), (43), Tablas 1–2.
- **Varianza discreta:** Klingbeil et al. 2014 (local).
- **Flujo real de agua dulce:** MITgcm `model/src/integr_continuity.F:93,142` y `external_forcing_surf.F:149,258-262,367`; MOM6 `src/parameterizations/vertical/MOM_diabatic_aux.F90:692-1459`, en particular `netMassIn`/`netMassOut`.

## Global Constraints

- **Sin sal, nada cambia.** Con `salinity=False` (por defecto) la matriz dorada de N1 (`tests/test_n1_golden.py`) queda **bit a bit** y la suite completa pasa. Las ramas nuevas no tocan la aritmética del camino T-solo; donde haga falta una fórmula nueva, va en una rama aparte.
- **Costo.** El costo por paso de los casos sin S no cambia más de 1 % (BayIW corte z\* y overflow de Ilicak, `scripts/n1_smoke.py`).
- **No dividir entre `t_coef`** en ningún camino con S. El caso de Lorenz usa T constante y α puede ser 0.
- **Unidades y signos.**
  - `fw` = P − E [m s⁻¹], positivo hacia el océano.
  - S en g kg⁻¹.
  - `s_coef` = β [kg m⁻³ (g kg⁻¹)⁻¹].
  - EOS: ρ = ρ₀ − α(T − T₀) + β(S − S₀) (Vallis 2006, ec. 1.156).
- **Agua dulce:** lleva S = 0. Su T es la T de la capa superior, que no cambia T.
- **Evaporación excesiva:** si vaciaría la capa superior (h < `DZ_FLOOR`), el resto se toma de la capa de abajo a sal fija y se cuenta en `n_limited`. Nunca en silencio.
- **Lenguaje y estilo:** solo Python, type hints, docstrings con la física y su cita, índices (k, j, i) con k = 0 en el fondo (CLAUDE.md).
- **Hilos y corridas:** ≤ 5 hilos y solo P-cores; las corridas largas, una a la vez.
- **Commits** con rutas explícitas. Terminan con:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```
- **Rama:** `n2a-salinity`, desde `main`.

## Review Focus

1. **`fw` sobre tierra o celdas secas** (H ≤ 0 o D ≤ d_min): debe ignorarse, sin volumen fantasma. Lo cubre Task 4, `test_fw_on_land_and_dry_cells_is_ignored`.
2. **`salinity=True` sin S en `initial_state`, o un río sin S en su `OBData`:** debe dar un error claro antes de correr, no NaN después. Lo cubren Task 1, `test_salinity_needs_initial_S`, y Task 2, `test_open_edge_without_S_is_an_error`.
3. **Restart con S:** ida y vuelta bit a bit. Lo cubre Task 1, `test_restart_with_salinity_bit_for_bit`.
4. **α = 0, T constante,** con `rho_ref="auto"` y `kh > 0`: no debe dividir entre cero. Lo cubre Task 3, `test_zero_t_coef_with_reference_profile_and_kh`.
5. **GLS no actúa sobre los trazadores.** En la configuración σ + GLS, la difusividad de S es la constante `kv`. Es física faltante conocida (b), no un error, y no se "corrige" en N2a. El README del benchmark lo declara. Lo cubre Task 8, `test_gls_config_uses_constant_tracer_kv`.

---

### Task 1: S en el estado, EOS (T, S), restart y pre-flight

**Files:**
- Modify:
  - `core/components/eos.py` (`LinearEOS`);
  - `core/dynamics/sigma_baroclinic.py`: `SigmaState`, firma de `__init__`, `initial_state`, `density`, `_PARAMS`/`_fixed_after_construction` para `s_coef`/`s_ref`;
  - `core/model/registry.py`, `core/model/preflight.py`, `core/restart.py`.
- Test: `tests/test_n2a_salinity.py` (nuevo).

**Interfaces:**
- Produces:
  - **`LinearEOS`**: campos nuevos `s_coef: float = 0.0`, `s_ref: float = 35.0`; método `density(T, S=None) -> ndarray`. Con `S is None` devuelve **exactamente** la expresión de hoy; con S, `self.rho0 - self.t_coef * (T - self.t_ref) + self.s_coef * (S - self.s_ref)`.
  - **`SigmaState.S: np.ndarray | None = None`**, de forma (N, ny, nx).
  - **`SigmaBaroclinic(..., salinity: bool = False, s_coef: float = 0.0, s_ref: float = 35.0)`**. `s_coef` y `s_ref` quedan fijos después de construir, igual que `t_coef`.
  - **`SigmaBaroclinic.density(T, S=None)`.**
  - **`initial_state(eta, T, h_src=None, tr=None, S=None)`.** Si `salinity=True` y `S is None` → `ConfigurationError("salinity=True needs S in initial_state")`. Con `h_src`, S se remapea desde la malla de origen como T.
  - **`REGISTRY["S"]`**: `VarSpec("S", ("k","y","x"), "t", "1e-3", "sea_water_salinity", "prognostic", "salinity")`, las mismas unidades que `diagnostics/output.py:FIELDS["S"]`.
  - **Restart:** escribe y lee `S` con `component="state"`. La huella de configuración agrega `salinity`, `s_coef` y `s_ref`.
  - **Pre-flight:** `fw`, `variance_budget` o `tef_face` sin `salinity=True` → error. `s_coef != 0` sin `salinity=True` → aviso.

- [ ] **Step 1: escribir los tests**
  - `test_linear_eos_without_S_is_bitwise_today`: `np.array_equal(LinearEOS(...).density(T), rho0 - t_coef*(T - t_ref))`.
  - `test_linear_eos_with_S`: `density(T, S) == rho0 - a(T-T0) + b(S-S0)` con valores concretos (ρ₀ = 1000, α = 0.2, β = 0.78, T₀ = 10, S₀ = 30).
  - `test_salinity_needs_initial_S`.
  - `test_uniform_S_at_s_ref_leaves_dynamics_bitwise`: el caso `zstar_vi_arakawa` de `tests/n1_cases.py`, reconstruido con `salinity=True, s_coef=0.78, s_ref=30` y S ≡ 30. Después de 10 pasos, η, u, v, T y h son bit a bit iguales a `GOLDEN["zstar_vi_arakawa"]["10"]`; se comparan solo esas claves de `state_hashes`. Este test depende de Task 2: se escribe aquí, se marca `xfail(strict=True, reason="task 2")` y se quita la marca al terminar Task 2.
  - `test_restart_with_salinity_bit_for_bit`: escribir en el paso 2, leer, dar un paso; igual a la corrida continua en el paso 3.
- [ ] **Step 2:** `pytest tests/test_n2a_salinity.py -q` → FAIL (salvo el xfail).
- [ ] **Step 3: implementar** las interfaces de arriba.
- [ ] **Step 4:** `pytest tests/test_n2a_salinity.py tests/test_n1_golden.py tests/test_restart.py tests/test_preflight.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 2: S en el paso de tiempo, σ y ALE

**Files:**
- Modify:
  - `core/model/driver.py`: fases `tracers`, `vertical_coordinate`, `interior_sponge`, `finalize` de los dos caminos;
  - `core/components/open_boundary.py`: `S_ext`, `inflow_stack` y exterior en S;
  - `core/dynamics/open_boundary.py`, `exterior_tracer` con `field="S"`;
  - `core/components/sponge.py`, `sponge_S` opcional;
  - `core/dynamics/sigma_baroclinic.py`: argumentos `sponge_S`, `S_ext`, CONFIG.
- Test: `tests/test_n2a_salinity.py`.

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - **Orden del stack de trazadores:** `[T, (T²), S, (S²), tr…]`.
    - T² va según `mix_diagnostic` en ALE y siempre en σ, como hoy.
    - S² va solo con `variance_budget=True` (Task 6).
    - Las posiciones salen de un helper `_stack_layout(model) -> dict[str, int]` en `core/model/driver.py`, que usan todas las fases y el inflow.
  - **`OpenBoundary`:**
    - gana `S_ext` (fijado en `initial_state` como `T_ext`, en `CONFIG`);
    - `inflow_stack(c, T_in, tr_in, S_in=None)`;
    - `exterior_S(h, ext) -> ndarray`.
  - **Proveedor de exterior:** usa `OBData.S`. Si el modelo tiene `salinity=True` y un borde con datos tiene `S=None` → `ValueError` en `validate_state`, con el borde nombrado.
  - **`InteriorSponge`:** gana `S: ndarray | None`, el perfil en z como `T`.

- [ ] **Step 1: escribir los tests**
  - `test_S_equals_T_when_both_passive_sigma`: σ con `t_coef=0`, `s_coef=0`, S₀ = T₀ (el T del caso `sigma_kh_kv`), K_h y K_v > 0, 10 pasos; `np.array_equal(st.S, st.T)`.
  - `test_S_equals_passive_tracer_ale`: ALE z\* con `s_coef=0`, `n_tracers=1` y `tr[0]` = S₀; `np.array_equal(st.S, st.tr[0])` en 10 pasos.
  - `test_uniform_S_stays_uniform_with_open_edges`: `zstar_obc_radiating` con exterior S = 30 y S₀ ≡ 30; `np.abs(st.S - 30).max() < 1e-12` y la sal total es exacta con frontera cerrada (rtol 1e-13).
  - `test_open_edge_without_S_is_an_error`.
  - Quitar el xfail de `test_uniform_S_at_s_ref_leaves_dynamics_bitwise` (Task 1).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.** S recorre el mismo camino que T: advección, K_h (sin anomalía, ver Task 3), K_v, esponja de borde, remapeo y esponja interior.
- [ ] **Step 4:** `pytest tests/test_n2a_salinity.py tests/test_n1_golden.py tests/test_sigma_ale.py tests/test_sigma_baroclinic.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 3: densidad con S en todos sus consumidores

**Files:**
- Modify:
  - `core/components/pgf.py`: jacobiano y dos términos con `density(T, S)`;
  - `core/vertical/pgf.py`: ruta nueva `fv_pgf_rho(h, H, eta, rho, du, dv, rho0, rho_ref, g, mass_weight=True, h_vanished=H_VANISHED)` que integra la anomalía ρ − ρ_ref con PLM y Boole como `fv_pgf`, pero en unidades de densidad. **`fv_pgf` no cambia.** El componente usa `fv_pgf_rho` solo si `S is not None`.
  - `core/components/vertical_coordinate.py`: densidad de la regla híbrida con S;
  - `core/dynamics/sigma_baroclinic.py`: `reference_profile` con `density(T, S)`;
  - `core/components/tracers.py`: `TracerDiffusion` sin anomalía cuando hay S;
  - `core/model/preflight.py`: aviso para `salinity` con `rho_ref` y `kh > 0`.
- Test: `tests/test_n2a_density.py`.

**Interfaces:**
- Consumes: Task 1.
- Produces: `PressureGradient.__call__(geom, eta, T, h=None, S=None)` y `TracerDiffusion.diffuse_h(T, V, eta, h=None, anomaly=True)`, que se llama con `anomaly=False` para T y S cuando `salinity=True`.

- [ ] **Step 1: escribir los tests**
  - `test_salt_only_linear_density_pgf_exact_flat_bed` (jacobiano y FV): T constante, `t_coef=0`, S lineal en x y z. Se compara con el PGF exacto como en `tests/test_sigma_baroclinic.py:176-189`, con rtol 1e-10.
  - `test_salt_only_stratification_seamount_rest` (jacobiano y FV): S lineal en z sobre el seamount de `tests/test_sigma_baroclinic.py:165`; max\|PGF\| < 1e-13.
  - `test_fv_pgf_rho_equals_fv_pgf_for_T_only`: con ρ = ρ₀ − α(T − T₀), `fv_pgf_rho` coincide con `fv_pgf` a rtol 1e-12. No se exige bit a bit: son caminos distintos.
  - `test_zero_t_coef_with_reference_profile_and_kh`: `salinity=True`, `t_coef=0`, `rho_ref="auto"`, `kh=5`; 5 pasos sin NaN ni excepción; avisa `PyOceanConfigWarning`.
  - `test_hybrid_regrid_uses_salt_density`: híbrida con T constante y S estratificada. Las capas siguen las isopicnas de S: las interfaces finales están a ≤ 1 cm de las profundidades objetivo.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.**
- [ ] **Step 4:** `pytest tests/test_n2a_density.py tests/test_vertical_pgf.py tests/test_n1_golden.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 4: agua dulce real en la superficie libre

**Files:**
- Modify:
  - `core/dynamics/wetting_drying.py` (`WetDryFreeSurface.step`: `rhs_full` y recálculo de η con conservación exacta);
  - `core/dynamics/sigma_wd.py` (`SigmaWetDry.step` pasa `fw`);
  - `core/components/free_surface.py` (`FreeSurfaceVerticalViscosity.step(..., fw=None)`);
  - `core/model/driver.py` (`step(st, tau_x, tau_y, fw=None)`);
  - `core/dynamics/sigma_baroclinic.py` (`step(st, tau_x=None, tau_y=None, fw=None)`).
- Test: `tests/test_n2a_freshwater.py`.

**Interfaces:**
- Produces:
  - **`WetDryFreeSurface.step(..., fw: np.ndarray | None = None)`:** `fw` (ny, nx) [m s⁻¹], enmascarado a `active & (H + η > d_min)`.
  - **Continuidad:** `rhs_full = area·η − Δt·flux + Δt·area·fw_eff`. El recálculo exacto de CG usa la misma fuente.
  - **`fw=None`:** ninguna operación nueva, para que siga bit a bit.

- [ ] **Step 1: escribir los tests**
  - `test_freshwater_changes_volume_exactly`: cuenca cerrada σ y ALE z\*, `fw` = +2e-6 en la mitad oeste y −2e-6 en la este, 20 pasos; `Σ A(H+η)` cambia en `Δt Σ A fw` por paso, a rtol 1e-13. En ALE, `Σ_k h = H + η` a 1e-12.
  - `test_fw_on_land_and_dry_cells_is_ignored`: `fw` ≠ 0 sobre tierra y sobre celdas secas no cambia el volumen.
  - `test_no_fw_is_bitwise`: la matriz dorada (`pytest tests/test_n1_golden.py`).
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.**
- [ ] **Step 4:** `pytest tests/test_n2a_freshwater.py tests/test_n1_golden.py tests/test_wetting_drying.py tests/test_sigma_wd.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 5: el agua dulce en los trazadores, como subpaso sobre la capa superior

**Files:**
- Modify:
  - `core/components/layer_transport.py`: `LayerFluxes` gana `V_fw` (volumen final después del agua dulce);
    - σ: `V1` de la advección = `layer_volumes(η_new)` con la capa superior menos `Δt·A·fw`, de modo que Fz_top = 0 como hoy;
    - ALE: h† sin agua dulce para la advección; `h_fw` = h† con la capa superior + `fw·Δt`.
  - `core/model/driver.py`: después de la advección y antes de la difusión vertical, el subpaso a sal fija.
- Create: `core/components/freshwater.py` con `apply_freshwater(stack, layout, V_adv, V_fw, fw_area_dt) -> (stack, n_limited, var_surf)`.
- Test: `tests/test_n2a_freshwater.py`.

**Interfaces:**
- Consumes: Task 4 (`fw` en el driver), Task 2 (`_stack_layout`).
- Produces: `apply_freshwater(...)`, que transforma cada miembro del stack:
  - S y S² → `c·V_adv/V_fw` en la capa superior (el contenido no cambia);
  - T y T² → sin cambio;
  - trazadores pasivos → como S (sin fuente en el agua dulce);
  - devuelve también `var_surf`, el término de varianza por celda `(V_fw·S_new² − V_adv·S_old²)/Δt` para S (la ec. 43 de Lorenz), y `n_limited` (capas superiores que se habrían vaciado, con el déficit tomado de la capa de abajo).

- [ ] **Step 1: escribir los tests**
  - `test_evaporation_conserves_salt_and_concentrates`: cuenca cerrada con E uniforme 1e-6 y S₀ = 30 uniforme; en 20 pasos `Σ V S` se conserva a rtol 1e-13, S sube como 30·V₀/V y T uniforme queda uniforme a 1e-12. Para σ y ALE z\*.
  - `test_precipitation_freshens_only_where_it_rains`: P en el tercio este; en el paso 1, S cambia solo en esas columnas.
  - `test_evaporation_deficit_moves_to_next_layer_and_is_counted`: E grande que vacía la capa superior (ALE, h_top = 1 mm); `st.n_limited > 0`, h ≥ 0 y la sal se conserva.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.**
- [ ] **Step 4:** `pytest tests/test_n2a_freshwater.py tests/test_n1_golden.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 6: balance discreto exacto de varianza (opcional)

**Files:**
- Create: `core/components/variance_budget.py`.
- Modify:
  - `core/model/driver.py`: S² en el stack y registro de cada operador;
  - `core/dynamics/sigma_baroclinic.py`: argumento `variance_budget: bool = False`;
  - `SigmaState.budget: dict | None = None`, de clase `diagnostic` en el registro.
- Test: `tests/test_n2a_budget.py`.

**Interfaces:**
- Consumes: Tasks 2, 5.
- Produces: `st.budget` con arreglos (N, ny, nx) por celda, en [(g kg⁻¹)² m³ s⁻¹], solo con `variance_budget=True`.
  - **Claves:** `"S_num_adv"`, `"S_phy_v"`, `"S_phy_h"`, `"S_num_remap"` (ALE), `"S_surf"` (= `var_surf` de Task 5), `"S_sponge"`, y las mismas para T.
  - **Mezcla numérica de advección:** `V1·(S²_adv − S_adv²)/Δt`, Burchard & Rennau 2008 con S² advectado en el mismo stack.
  - **Mezcla física:** `V·(S_antes² − S_después²)/Δt` de cada operador de difusión.
  - **Remapeo:** como el `mix_num` de ALE.
  - **Signos:** mezcla > 0 destruye varianza. Las fronteras abiertas, el río y las esponjas se cuentan aparte.

- [ ] **Step 1: escribir el test** `test_closed_basin_salinity_square_budget_closes`, para σ y ALE z\*, hybrid incluida.
  - **Montaje:** cuenca cerrada, S con gradiente, E en el oeste y P en el este, K_h y K_v > 0, sin esponjas.
  - **Aserción, en cada paso:**
    ```python
    lhs = (V1 * S1**2).sum() - (V0 * S0**2).sum()
    rhs = dt * (b["S_surf"].sum() - b["S_num_adv"].sum() - b["S_phy_v"].sum() - b["S_phy_h"].sum() - b.get("S_num_remap", 0).sum())
    assert abs(lhs - rhs) <= 1e-12 * (V0 * S0**2).sum()
    ```
  - `test_budget_off_is_bitwise`: la matriz dorada.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.**
- [ ] **Step 4:** `pytest tests/test_n2a_budget.py tests/test_n1_golden.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 7: TEF en línea y la relación exacta de Lorenz

**Files:**
- Create: `diagnostics/tef.py`.
- Modify:
  - `core/components/tracers.py`: `advect`, `advect_fct` y `advect_numpy` ganan `flux_face_i: int | None = None`. Cuando se da, devuelven también el flujo efectivo del paso en esa columna de caras u: `(C, fx_i)` con `fx_i` (nstack, N, ny).
    - SSP-RK3: `(f0 + f1)/6 + 2f2/3`.
    - FCT: `FL + C·A`.
    - 'pc': el flujo del corrector.
  - `core/model/driver.py`: con `tef_face` acumula volumen, s y s² por bin; con `tef_region` (máscara de celdas) integra las claves de `st.budget` en la región.
  - `core/dynamics/sigma_baroclinic.py`: argumento `tef_face: int | None = None`.
- Test: `tests/test_n2a_tef.py`.

**Interfaces:**
- Consumes: Task 6.
- Produces:
  - **`TEFTransect`:** `TEFTransect(face_i: int, sign: int = +1, n_bins: int = 250, s_range=(0.0, 40.0))`.
    - `add(Q, Fs, Fs2, dt)`: (N, ny) por paso; la salinidad del bin es `Fs/Q` de cada cara-capa.
    - `bulk() -> dict` con Q_in, Q_out, s_in, s_out, (s²)_in y (s²)_out.
    - La división entrada/salida va por el signo de q(s) en cada bin (MacCready 2011).
  - **`lorenz_mixing(bulk, Qr, Qsurf, dV, dS, dS2, Fs2_surf) -> dict`:** las ec. (32), (33), (34) y (35), con sus nombres `Me`, `Mc`, `Mp` y `Mcp`.

- [ ] **Step 1: escribir el test** `test_lorenz_identity_holds_to_round_off`.
  - **Montaje:** canal corto (40 × 1 × 20, fondo 15 → 5 m), río por el borde este con `'clamped'` (S = 0, Q = 5 m³ s⁻¹), E en la mitad este (−5 m³ s⁻¹ integrados), frontera oeste con S = 30, 200 pasos; transecto en la cara del medio; región = celdas al este.
  - **Aserción:**
    ```python
    Me = lorenz_mixing(tef.bulk(), Qr, Qsurf, dV, dS, dS2, F)["Me"]
    assert abs(Me - M_model_tot) <= 1e-10 * abs(M_model_tot)
    ```
    con `M_model_tot` = Σ región de (S_num_adv + S_phy_v + S_phy_h (+ S_num_remap)), promediada en el tiempo.
  - **Configuraciones:** σ + RK3, σ + 'pc' y ALE-σ + FCT.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: implementar.** Las fórmulas se transcriben de Lorenz 2021, ec. (24)–(35); los signos se comprueban contra el caso de su Tabla 2 en Task 8.
- [ ] **Step 4:** `pytest tests/test_n2a_tef.py -q` → PASS.
- [ ] **Step 5: commit.**

---

### Task 8: benchmark `validation/lorenz_estuary/`

**Files:**
- Create:
  - `validation/lorenz_estuary/__init__.py`;
  - `case.py`: geometría, forzamiento de la Tabla 1, transecto;
  - `run_pyocean.py`: `python validation/lorenz_estuary/run_pyocean.py {A,B,C,D} {sigma_gls,ale_sigma} [--cycles 600] [--avg 100]`;
  - `compare.py`: tabla como la Tabla 2 de Lorenz, con la columna GETM;
  - `README.md`.
- Test: `tests/test_lorenz_estuary.py`.

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: `case.build(exp: str, config: str) -> (model, state, forcing)`, con `forcing(n) -> dict(tau_x=None, tau_y=None, fw=...)`.
  - **Geometría:** 500 × 1 celdas, dx = 200 m, fondo lineal de 15 m (oeste, océano) a 5 m (este, río), 60 capas σ uniformes.
  - **Frontera oeste:** S = 30, marea M2 de 0.5 m en B–D, radiante + esponja.
  - **Tabla 1:**

    | Exp. | Q_r | Q_evap | Q_precip | M2 |
    |---|---|---|---|---|
    | A | 0 | −5 | 0 | 0 |
    | B | 5 | −5 | 0 | 0.5 m |
    | C | 0 | −5 | 5 | 0.5 m |
    | D | 5 | −5 | 5 | 0.5 m |

    Todos los flujos en m³ s⁻¹. En A, E va en la mitad interior; en B–D, E en el tercio medio y P en el tercio este, como en el pie de su Fig. 2.
  - **Configuraciones:**
    - `sigma_gls`: σ euleriano, `closure="gls"`, `z0=1e-3`;
    - `ale_sigma`: ALE `regrid="sigma"`, `closure="constant"`.
  - **Comunes:** `salinity=True`, `t_coef=0`, `kh=0`. β y ν/κ de fondo se toman del artículo, o se marcan **[no dado en el artículo]** con el valor por defecto de GETM o el más simple, sin ajustar.

- [ ] **Step 1: escribir los tests**
  - `test_case_matches_table_1`: los flujos integrados de la tabla, a 1e-12.
  - `test_short_run_closes_budget_and_identity`: el experimento A en las dos configuraciones, 50 pasos; cierre de Task 6 y relación de Task 7.
  - `test_gls_config_uses_constant_tracer_kv`: en `sigma_gls`, la difusividad de S es `kv`, aunque GLS cambie ν (Review Focus 5).
- [ ] **Step 2:** FAIL, luego implementar, luego PASS.
- [ ] **Step 3: medir el costo.** 2 h simuladas de A y B en cada configuración, 5 hilos, Δt estable. Se extrapola a 600 ciclos M2. **Si una corrida pasa de 24 h de reloj, se detiene el plan y se consulta a Esteban** (spec).
- [ ] **Step 4: correr las 8 corridas, una a la vez,** y escribir con `compare.py`:
  - la Tabla 2 de pyOCEAN contra GETM;
  - la clasificación (a)–(d) de cada diferencia;
  - la lista "Qué debe mejorar pyOCEAN" (CLAUDE.md, Validation Strategy).
- [ ] **Step 5: commit** de los archivos de `validation/lorenz_estuary/` con rutas explícitas y del test. Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary/`, nunca al repo.

---

### Task 9: documentación y cierre

**Files:**
- Modify:
  - `CLAUDE.md`: fila nueva y conteo de tests;
  - `docs/design/vision_roadmap.md`: §7 marcador, fila "Estuario de Lorenz"; §8 N2a ✅;
  - `docs/manual/architecture/time-step.md`: el subpaso de agua dulce y la fase de trazadores;
  - `docs/manual/theory/tracers.md`: S, agua dulce y balance de varianza;
  - `docs/literature/reading_list.md`: Lorenz et al. 2019 (TEF) si se usó; Klingbeil 2014 ★.

- [ ] **Step 1: verificar la suite completa.** `pytest tests/ -q` → todo pasa, con los mismos 2 xfail.
- [ ] **Step 2: medir el costo.** `python -I scripts/n1_smoke.py` contra `main`: los casos sin S quedan bit a bit, con costo ≤ 1.01×.
- [ ] **Step 3: documentación y commit.**
