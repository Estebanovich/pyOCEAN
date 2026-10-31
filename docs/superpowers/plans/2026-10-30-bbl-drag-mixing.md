# Diagnóstico teórico de la pluma y mezcla BBL por arrastre — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** agregar a pyOCEAN la mezcla diapicna de capa de fondo alimentada por el trabajo del arrastre: port del LOTW de OM4, con la energía física de pyOCEAN. Agregar también un diagnóstico teórico de la pluma de DOME, y comparar contra la corrida isopicna de 10 km de Legg (2006).

**Architecture:**
- **Kernel de columna** en `physics/mixing/bbl_mixing.py`: oráculo NumPy y kernel Numba por columnas, con la fusión de capas delgadas de Jackson.
- **Envoltura aditiva** `BBLAddedClosure` en `core/components/closure.py`. Va por fuera de `ShearAddedClosure` y del cierre base, y suma Kd_BBL solo a la difusividad de trazadores, sin viscosidad.
- **Energía:** el driver la calcula en la fase implícita con las tasas r·f_k del arrastre y la pasa a `update`.
- **Diagnóstico teórico:** es posproceso en `validation/dome/theory.py`.

**Tech Stack:** Python, NumPy 1.26.4, Numba. El Fortran de MOM6 compilado se usa solo para la prueba de referencia (`validation/mom6_column/`).

**Spec:** `docs/superpowers/specs/2026-10-30-bbl-drag-mixing-design.md` (aprobado por Esteban el 2026-10-30). Lecturas: `docs/design/bbl_refs/` (`mom6_bbl_mixing.md` con `SD:línea`; `entrainment_theory.md`).

## Global Constraints

- Solo Python (NumPy, Numba), sin dependencias nuevas, nunca `pip install`; numpy 1.26.4.
- Corridas: exportar `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, ≤ 5 hilos en P-cores, una a la vez, salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`. Si una corrida pasa de ~24 h, se consulta a Esteban.
- Con `bbl_mixing=None` (valor por defecto) el camino actual no cambia ni en el orden de las operaciones. El driver no calcula la energía si el esquema está apagado. Goldens (`tests/test_n1_golden.py`) y humo (`scripts/n1_smoke.py`) idénticos bit a bit.
- **Energía física** (decisión de Esteban): E = trabajo del arrastre de pyOCEAN, no `BBL_meanKE_loss` de MOM6. u\* = √(`ctx.ustar2_bot`). Si se confirma el factor 2 de MOM6, se clasifica (d) y se dice que γ = 0.2 aquí equivale a 0.1 en MOM6/OM4.
- Parámetros con los valores por defecto de OM4/MOM6, sin ajustar a ningún caso: `bbl_effic` 0.2, `max_decay` 200 m, `von_karman` 0.41, `use_omega` True, `kd_max` 0.1 m² s⁻¹.
- Solo ALE (σ, z\* e híbrido de ALE). El σ heredado se rechaza en el pre-flight. `closure='gls'` + BBL también se rechaza.
- Cada ecuación cita `MOM_set_diffusivity.F90:línea` (y Legg 2006 Ap. A donde aplique) en el docstring, con unidades.
- Índices (k, j, i) con k = 0 en el fondo; interfaces 0 = fondo, N = superficie. MOM6 va de arriba abajo y se invierte.
- Commits con rutas explícitas: nunca `git add validation/`, `-A` ni `.`, ni `logs/`, `pyOCEAN-literature-review-guide.docx` o salidas de corridas. Trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```

## Review Focus

1. **N² ≤ 0 junto al fondo** (vuelco en el borde de la pluma): con `use_omega`, N² se acota por abajo con Ω²; Kd finito y ≤ `kd_max`, sin NaN. Test en Task 1.
2. **Capas evanescentes en el fondo de la híbrida** (dz_int → 0, donde LOTW pone `kd_max`): se fusionan antes; Kd igual que sin ellas, sin `kd_max` espurio. Test en Task 3.
3. **`bbl_mixing` sin arrastre de fondo** (`bottom_drag` distinto de 'quadratic' y `z0=None`): error claro en el pre-flight, no una mezcla nula silenciosa. Test en Task 4.
4. **f = 0** (canal sin rotación, como Ilicak): la escala de decaimiento queda limitada por `max_decay`; Kd finito. Test en Task 1.
5. **Fronteras periódicas y abiertas:** la energía de las caras pasa a los centros respetando las caras de envoltura (`bed_ustar2` ya lo hace). Una celda junto a una frontera abierta no recibe energía de fuera. Test en Task 4.

---

### Task 1: oráculo de columna en NumPy

**Files:**
- Create: `physics/mixing/bbl_mixing.py`
- Test: `tests/test_bbl_mixing.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class BBLParams`: `bbl_effic=0.2` (BBL_EFFIC), `max_decay=200.0` (BBL_MIXING_MAX_DECAY, m), `von_karman=0.41` (VON_KARMAN_BBL), `use_omega=True` (LOTW_BBL_USE_OMEGA), `kd_max=0.1` (KD_MAX, m² s⁻¹; OM4).
  - `lotw_bbl_column(dz, n2, ustar: float, absf: float, tke: float, p: BBLParams) -> np.ndarray`:
    - dz [m] de forma (N,), con k = 0 en el fondo y sin capas delgadas;
    - n2 [s⁻²] de forma (N+1,) en las interfaces;
    - ustar [m s⁻¹];
    - absf = |f| [s⁻¹];
    - tke = energía de la columna ya multiplicada por `bbl_effic` [m³ s⁻³], es decir, `TKE_col` de `SD:1709`.

    Devuelve Kd [m² s⁻¹] de forma (N+1,), con Kd = 0 en el fondo y en la superficie.

Port de `add_LOTW_BBL_diffusivity` (`SD:1606-1789`):
- `Idecay = max(1/max_decay, absf/ustar)`;
- decaimiento `exp(−Idecay h_k)`;
- Kd_wall de la ley de pared (`SD:1756`);
- consumo `Kd·dz_int·max(N², Ω²)`, con Ω = 7.2921e-5 (`core/grid/horizontal.OMEGA`) si `use_omega`, y 0 en otro caso;
- reescalado para no gastar más TKE de la disponible;
- `Kd = kd_max` cuando el consumo es 0.

Con ustar = 0 o tke = 0, Kd = 0.

- [ ] **Step 1: Write the failing tests** in `tests/test_bbl_mixing.py`:
  - `test_no_drag_gives_zero`: ustar = 0 y tke = 0 → `np.all(kd == 0)`.
  - `test_energy_bound`: columna estratificada (N² = 1e-6) → `kd[0] == kd[-1] == 0`, `kd >= 0`, y Σ_K Kd_K·dz_int,K·max(N², Ω²) ≤ tke·(1 + 1e-12).
  - `test_decays_with_height`: columna con N² uniforme → Kd en las interfaces inferiores > Kd lejos del fondo, por encima de la escala ustar/absf.
  - `test_unstable_is_finite` (Review Focus 1): N² < 0 en dos interfaces → finito y ≤ `kd_max`.
  - `test_f_zero_uses_max_decay` (Review Focus 4): absf = 0 → finito; el decaimiento con absf = 0 coincide con el de absf = ustar/200.
  - `test_defaults_are_om4`: `BBLParams()` igual a los valores citados.
- [ ] **Step 2:** Run `pytest tests/test_bbl_mixing.py -q`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implementar `BBLParams` y `lotw_bbl_column`.
- [ ] **Step 4:** Run `pytest tests/test_bbl_mixing.py -q`. Expected: 6 passed.
- [ ] **Step 5: Commit** `physics/mixing/bbl_mixing.py tests/test_bbl_mixing.py`.

### Task 2: columna contra MOM6 compilado, y el factor de `set_BBL_TKE` (criterio 1)

**Files:**
- Create: `validation/mom6_column/bbl_driver.F90`
- Modify:
  - `validation/mom6_column/build_reference.sh`: extraer `add_LOTW_BBL_diffusivity` y lo que necesita a un módulo, como `kappa_shear`, y compilar el driver;
  - `validation/mom6_column/columns.py`: `bbl_cases() -> list[dict]`, `write_bbl_cases(path, cases)`, `read_bbl_results(path) -> list[dict]`;
  - `validation/mom6_column/README.md`.
- Test: `tests/test_vertical_vs_mom6.py`: `test_lotw_bbl_matches_mom6` con `@needs("bbl_driver")` y `test_set_bbl_tke_factor`.

**Interfaces:**
- Consumes: `lotw_bbl_column`, `BBLParams` (Task 1).

**Casos de `test_lotw_bbl_matches_mom6`:** MOM6 recibe `ustar_BBL` y `TKE_BBL` = la tke de pyOCEAN, de modo que se compara el kernel con la misma energía. Las columnas, todas sin fusión (spec, decisión 6):
- estratificada;
- N² ≤ 0 en dos interfaces;
- f = 0;
- con una capa de 1 mm;
- con u\* grande (escala de 200 m).

Tolerancia 1e-10 relativa, salvo que el implementador justifique otra en el test y en el README.

**`test_set_bbl_tke_factor`:** compara `BBL_meanKE_loss`/`ustar_BBL` de `set_BBL_TKE` (`SD:1947-2152`, sobre todo `:2132-2137`) con C_d|U|³ y √C_d|U| para un flujo uniforme en x.
- Si extraer `set_BBL_TKE` exige los tipos de grilla de MOM6 y no es razonable, la prueba transcribe `SD:2085-2145` en una función de prueba, citada línea por línea.
- El implementador registra esa elección como ruling en su reporte.
- Expected: el factor medido (≈ 2 y ≈ √2 según la lectura) queda escrito en el README como clase (d) **solo si se confirma**.

- [ ] **Step 1:** Escribir casos y tests (con skip si el binario no existe). Run `pytest tests/test_vertical_vs_mom6.py -q -k "bbl"`. Expected: SKIPPED / FAIL.
- [ ] **Step 2:** Extender `build_reference.sh`, escribir el driver y correr `bash validation/mom6_column/build_reference.sh`. Expected: `bbl_driver` en la lista "built:".
- [ ] **Step 3:** Run `pytest tests/test_vertical_vs_mom6.py -q`. Expected: todo pasa (remap, regrid, kappa_shear, bbl). Si LOTW no coincide, la causa se corrige en el oráculo de Task 1.
- [ ] **Step 4: Commit** las rutas de "Files" (y `physics/mixing/bbl_mixing.py` si se corrigió).

### Task 3: kernel Numba por columnas con fusión de capas delgadas

**Files:**
- Modify: `physics/mixing/bbl_mixing.py`
- Test: `tests/test_bbl_mixing.py`

**Interfaces:**
- Consumes:
  - `lotw_bbl_column` (oráculo);
  - `thin_layer_threshold`, `active_layers` (`core/components/closure.py:55-118`);
  - el patrón de fusión de `kappa_shear_columns` (`physics/mixing/kappa_shear.py`): unir a la de arriba, sumar h y volver a las interfaces;
  - `H_MASSLESS`.
- Produces:
  - `lotw_bbl_columns(dz, n2, ustar, absf, tke, h_thin, p: BBLParams) -> np.ndarray`:
    - dz de forma (N, M), en metros;
    - n2 de forma (N+1, M), en las interfaces del modelo;
    - ustar, absf, tke y h_thin de forma (M,).

    Devuelve Kd de forma (N+1, M). Es Numba `prange` sobre columnas. Las columnas secas (Σdz ≤ `H_MASSLESS`) dan 0. El N² de la columna fusionada se toma en las interfaces que sobreviven.
  - `lotw_bbl_columns_numpy(...)`: misma firma; es el oráculo, que recorre las columnas con la fusión más `lotw_bbl_column`.

- [ ] **Step 1: Write the failing tests:**
  - `test_numba_matches_numpy_oracle`: 40 columnas variadas, con capas de 1 mm–2 cm, columnas secas y N² < 0 → `np.array_equal`.
  - `test_threads_bit_for_bit`: 1 y 5 hilos → `np.array_equal`.
  - `test_thin_bed_layers_do_not_change_kd` (criterio 2 y Review Focus 2): columna con y sin capas de 1 mm en el fondo y en el interior → Kd en las interfaces comunes iguales a `rtol=1e-12`, y ninguna interfaz con Kd = `kd_max` por una capa evanescente.
  - `test_dry_column_zero`: Σdz = 0 → Kd = 0.
- [ ] **Step 2:** Run `pytest tests/test_bbl_mixing.py -q`. Expected: los 4 nuevos FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `pytest tests/test_bbl_mixing.py tests/test_vertical_vs_mom6.py -q`. Expected: todo pasa.
- [ ] **Step 5: Commit.**

### Task 4: la mezcla BBL en el modelo

**Files:**
- Modify:
  - `core/model/driver.py` (energía en `_ale_bed_ustar2` o junto a ella; paso a `update` en la fase de trazadores);
  - `core/components/closure.py` (`BBLAddedClosure`, `make_closure`);
  - `core/dynamics/sigma_baroclinic.py` (parámetros, docstring, propiedad `bbl_params` reenviada como `jackson_params`);
  - `core/model/preflight.py` (`bbl_params_errors`);
  - `core/model/registry.py` (`kd_bbl`);
  - `core/restart.py` (solo la comprobación de configuración: `bbl_mixing` y `bbl_params`).
- Test: `tests/test_bbl_model.py`

**Interfaces:**
- Consumes: `lotw_bbl_columns`, `BBLParams` (Task 3); `bed_ustar2`, las tasas `visc["r_u"]` y `visc["r_v"]` de la fase implícita; `ShearAddedClosure` (existente).
- Produces:
  - `SigmaBaroclinic(..., bbl_mixing: str | None = None, bbl_params: BBLParams | None = None)`, con `bbl_mixing` ∈ {None, 'lotw'}.
  - `ctx.bbl_energy`: (ny, nx) [m³ s⁻³], el trabajo del arrastre por unidad de área, solo si `bbl_mixing` está activo.
    - En cada cara, e = Σ_k r_k u_k², con r_k = `visc["r_u"]` (ya multiplicada por f_k en el arrastre cuadrático) y las velocidades nuevas, igual que `ustar2_bot`.
    - Con ley logarítmica se recalculan las tasas como en `_ale_bed_ustar2`.
    - A los centros: ½(e_W + e_E) + ½(e_S + e_N), con la misma regla de caras de envoltura que `bed_ustar2`.
  - `class BBLAddedClosure(Closure)`: envuelve al cierre (base o `ShearAddedClosure`).
    - `update_phase = "tracers"` y `stateful = True`, como `ShearAddedClosure`, para que el driver llame a `update` en la fase 7 aunque el envuelto sea el cierre constante. `state_fields`/`load_state` solo incluyen los del envuelto si este es `stateful`, igual que `ShearAddedClosure`.
    - `update(..., bbl_energy=None)` llama al envuelto sin `bbl_energy` y calcula Kd_BBL con:
      - tke = `bbl_effic`·E;
      - ustar = √`ustar2_bot`;
      - absf = |f| en los centros (media de los 4 f de las esquinas, como Jackson);
      - n2 de ρ con `physics/mixing/gls.buoyancy_frequency2`;
      - h_thin = `thin_layer_threshold(dz)`.
    - `tracer_diffusivity()` = κ_envuelto (o `kv` constante si es None) + Kd_BBL, y el valor del envuelto sin cambios cuando Kd_BBL ≡ 0.
    - `viscosity`, `state`, `load`, `state_fields`, `load_state` y `remap` delegan al envuelto. No hay estado propio, porque Kd_BBL se calcula y se usa en el mismo paso.
  - `make_closure`: con `bbl_mixing` envuelve el resultado, incluido el `ShearAddedClosure`.
  - Pre-flight, con todos los errores en un reporte:
    - `bbl_mixing` con `closure='gls'` o `'parabolic'` es error;
    - `bbl_mixing` fuera de ALE es error;
    - `bbl_mixing` sin arrastre de fondo (`bottom_drag != 'quadratic'` y `z0 is None`) es error;
    - un valor desconocido es error;
    - cada campo inválido de `BBLParams` es un error (`bbl_effic` ∈ [0, 1], `max_decay` > 0, `von_karman` > 0, `kd_max` > 0, todos finitos).
  - Registro: `kd_bbl` (interfaces, m² s⁻¹).

- [ ] **Step 1: Write the failing tests** in `tests/test_bbl_model.py`:
  - `test_off_is_bit_for_bit`: ALE híbrido y z\*, 10 pasos con `bbl_mixing=None` contra el mismo sin el parámetro → `np.array_equal` en η, u, v, T, h.
  - `test_no_flow_no_mixing`: columna estratificada en reposo con `drag_bg_vel=0` y `'lotw'` contra el base → `np.array_equal`.
  - `test_energy_is_drag_work`: canal con flujo uniforme U en x, arrastre cuadrático → `ctx.bbl_energy` = C_d|U|³ a 1e-12 relativo en el interior.
  - `test_bbl_mixes_dense_bottom_layer`: corriente de fondo densa sobre talud, 20 pasos → Kd_BBL > 0 en las interfaces inferiores, y la varianza de T de las capas de fondo baja más que con el base solo.
  - `test_adds_to_jackson`: base constante + Jackson + BBL → `tracer_diffusivity` = κ_base + κ_shear + Kd_BBL a 1e-15, y la viscosidad igual a la de base + Jackson bit a bit.
  - `test_preflight_refuses_all_at_once` (Review Focus 3): GLS, σ heredado, sin arrastre y `kd_max=0` en un solo reporte.
  - `test_restart_bit_for_bit_with_bbl`: 6 pasos = 3 + restart + 3 en ALE híbrido con Jackson + BBL.
  - `test_dry_column_stays_still`: como `tests/test_dry_column_closure.py`.
  - `test_periodic_and_open_energy` (Review Focus 5): canal periódico en x con borde abierto en y → la energía de las celdas 0 y n−1 usa la cara de envoltura compartida, y es finita y ≥ 0 junto al borde abierto.
  - `test_bbl_params_after_construction`: asignar `bbl_params` llega al cierre; uno inválido levanta `ConfigurationError`.
- [ ] **Step 2:** Run `pytest tests/test_bbl_model.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `pytest tests/test_bbl_model.py tests/test_shear_model.py tests/test_n1_golden.py tests/test_restart.py tests/test_preflight.py tests/test_closure_component.py -q`. Expected: todo pasa; goldens sin cambios.
- [ ] **Step 5:** Run la suite completa `pytest tests/ -q`. Expected: 1845 + los nuevos passed, 2 xfailed.
- [ ] **Step 6: Commit.**

### Task 5: humo y costo

**Files:**
- Create: `scripts/bbl_cost.py`. Como `scripts/shear_cost.py`: mediana de s/paso de 50 pasos tras 10 de calentamiento, 5 hilos, con y sin `'lotw'`. Casos:
  - BayIW cut z\* con `bottom_drag='quadratic'`;
  - DOME 10 km híbrido `kv_low` + Smagorinsky 0.15 + Jackson.

  Verifica que el término esté activo (Kd_BBL máximo > 0) e imprime cuánto se va en la actualización BBL.
- Modify: `validation/dome/README.md` (sección de costo).

- [ ] **Step 1:** `python scripts/n1_smoke.py` contra `main` en un worktree temporal del scratchpad. Expected: 9/9 hashes idénticos (pegarlos en el reporte).
- [ ] **Step 2:** `python -I scripts/bbl_cost.py`, tres rondas. Expected: razones ≤ 1.05 (spec, criterio 6). Si no se cumple, perfilar y documentar qué parte cuesta.
- [ ] **Step 3: Commit** `scripts/bbl_cost.py validation/dome/README.md`.

### Task 6: diagnóstico teórico de la pluma y opciones del runner

**Files:**
- Create: `validation/dome/theory.py`
- Modify:
  - `validation/dome/run_pyocean.py`: `--bbl lotw` y `--drag-bg-vel U`. Las etiquetas agregan `_ubg<U>` cuando U ≠ 0 y `_bbl` con `--bbl`, después de `_jackson`. Por ejemplo `kv_low_smag0.15_jackson_ubg0.05_bbl`;
  - `validation/dome/compare.py`: una tabla teórica con `--theory`;
  - `validation/dome/figures.py`: `plume_theory(...)`, una figura a lo largo de x′ con Ri_B, Fr, celdas con Ri < 0.25 y E medido contra E(Ri) de Ellison–Turner y de Xu;
  - `validation/dome/legg2006_nonhydro.json`: aclarar α_E.
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Produces, en `theory.py`, sobre las series diarias (h, T, s, u, v):
  - `plume_bulk(series, t0, t1, thr=0.01) -> dict[str, np.ndarray]`: por x′ a lo largo de Y(x), con:
    - h, la altura de la traza s > thr sobre el fondo;
    - ΔU, la velocidad media de la pluma menos la del ambiente justo encima;
    - Δb;
    - Ri_B = Δb h/ΔU²;
    - Fr = ΔU/√(Δb h).
  - `interface_richardson(series, t0, t1, thr=0.01) -> dict`: Ri de gradiente en las interfaces dentro de la banda 0.01 < s < 0.99, un histograma y el número de celdas con Ri < 0.25 a través de la interfaz, por x′. Este es el requisito de Jackson §6 de ≥ 2–4 celdas.
  - `legg_numbers(nu_v: float, cd: float) -> dict`: Ekman friccional ν_v/(h₀² f) y drag-Ekman C_d Δb₀ tanα/(f² h₀), con los parámetros de Legg Tabla 1–2 tomados de `validation/dome/cases.py`.
  - `bottom_ekman_transport(series, t0, t1) -> np.ndarray`: el transporte talud abajo dentro del hbbl inferior por x′. La definición operativa la escribe el implementador en el docstring; la sugerencia de Petersen 2015b §5 es una referencia, no un requisito.
  - `nof_speed(series, t0, t1) -> dict`: velocidad de fondo a lo largo del talud contra g′s/f (Nof 1983, por medio de Ezer ec. 2).
  - `entrainment_laws(ri) -> dict`: Ellison–Turner E = (0.08 − 0.1 Ri)/(1 + 5 Ri) para Ri < 0.8 (Xu ecs. 3–4) y Xu E = 0.20(1 − Ri/0.25) para Ri < 0.25 (Xu ec. 16).
- α_E: el implementador confirma en el PDF si 8.6e-4 (texto de Legg) es la ec. 14 (esquina) y si el α_E de la Fig. 3b (nuestro 1.03e-3 digitalizado) sigue la ec. 12. Corrige el JSON y la tabla si estaban mal etiquetados y lo reporta.

- [ ] **Step 1: Write the failing tests:**
  - `test_plume_bulk_synthetic`: pluma sintética de h = 100 m, ΔU = 0.5 m/s, Δb = 0.01 → Ri_B = 4.0 y Fr = 0.5 a 1e-12.
  - `test_interface_cells`: perfil sintético con 3 interfaces de Ri < 0.25 → cuenta 3.
  - `test_entrainment_laws`: Ri = 0 → E_ET = 0.08 y E_Xu = 0.20; Ri = 0.8 → E_ET = 0; Ri = 0.25 → E_Xu = 0.
  - `test_bbl_and_ubg_tags`: `--shear jackson --drag-bg-vel 0.05 --bbl lotw` → `..._jackson_ubg0.05_bbl`.
- [ ] **Step 2:** Run `pytest tests/test_dome_benchmark.py -q`. Expected: los nuevos FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `python validation/dome/compare.py 10 --theory` sobre las corridas existentes:
  - `mitgcm_z_legg`, `mitgcm_z_common`;
  - `pyocean_hybrid`, `pyocean_hybrid_gls`;
  - `pyocean_hybrid_kv_low_smag0.15` y sus `_jackson`, `_jacksonpr0`;
  - `pyocean_zstar_kv_low_smag0.15` y su `_jackson`;
  - `pyocean_sigma_kv_low_smag0.15` y su `_jackson`.

  Expected: imprime la tabla teórica sin error para cada una.
- [ ] **Step 5:** Run `pytest tests/test_dome_benchmark.py -q`. Expected: todo pasa.
- [ ] **Step 6: Commit** las rutas de "Files".

### Task 7: corridas (una a la vez; el controlador las lanza en cadena)

Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/10km/`; bitácoras en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bbl_logs/`. Antes de lanzar, verificar que no haya otra corrida en curso.

| # | Corrida | Comando |
|---|---|---|
| 1 | Control: híbrido `kv_low` + Smag 0.15 + Jackson, U_bg 0.05 | `python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05` |
| 2 | **Principal:** la misma + BBL | `... --bbl lotw` |
| 3 | z\* control | `python validation/dome/run_pyocean.py 10 zstar --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05` |
| 4 | z\* + BBL | `... --bbl lotw` |
| 5 | σ control | `python validation/dome/run_pyocean.py 10 sigma --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05` |
| 6 | σ + BBL | `... --bbl lotw` |

- [ ] **Step 1:** El controlador escribe la cadena en `bbl_logs/chain.sh` y la lanza en segundo plano.
- [ ] **Step 2:** Revisar rc y bitácoras. Si una corrida se detiene, guardar el post-mortem. No se reajusta nada.

### Task 8: análisis y documentación

**Files:**
- Modify:
  - `validation/dome/README.md`:
    - una sección BBL con los criterios 4, 5 y 7 con números contra la isopicna de 10 km de Legg (−11.07) y, en segundo lugar, contra la no hidrostática;
    - la comparación de la corrida principal contra el control a U_bg 0.05 y contra la corrida principal de Jackson (U_bg 0);
    - una sección "qué dice la teoría" con el diagnóstico de Task 6 sobre todas las corridas, cada afirmación citada a `archivo §` y lo inferido marcado;
    - cada diferencia clasificada de (a) a (d) y una lista de "qué debe mejorar pyOCEAN".
  - `validation/mom6_column/README.md` (criterio 1 y el factor de `set_BBL_TKE`, si falta).
  - `docs/manual/theory/turbulence-friction.md` (sección "Drag-driven bottom boundary layer mixing": ecuaciones, energía física frente a MOM6, suma, capas delgadas).
  - `docs/manual/user-guide/configuration-reference.md` (`bbl_mixing`, `bbl_params`).
  - `CLAUDE.md` (fila nueva y conteo de tests).
  - `docs/design/vision_roadmap.md` §7 y §8.
  - `docs/literature/reading_list.md`: los papers citados de `entrainment_theory.md` que se usen, con DOI, nombre de archivo y ⬜. Lo que esté de memoria va [verificar].
- Figuras y animación del trazador de fondo de la corrida principal contra su control, en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/figs/bbl/`, nunca en el repo.

- [ ] **Step 1:** Escribir. Cada número sale de una corrida de Task 7, de un test, de `legg2006_nonhydro.json` o de la teoría citada. La verdad digitalizada se marca como tal.
- [ ] **Step 2:** Run `pytest tests/ -q`. Expected: todo pasa; el conteo de CLAUDE.md coincide.
- [ ] **Step 3: Commit** las rutas de "Files".
