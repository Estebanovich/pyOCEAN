# N3: cierres turbulentos en σ y ALE (GLS k-ε y KPP) y flujo de calor de superficie — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que σ y ALE (σ, z\*, híbrido) tengan la misma GLS k-ε estratificada y una KPP estilo CVMix, con flujo de calor de superficie prescrito, ley de pared en ALE y fronteras periódicas, validados en columnas contra CROCO y MOM6 y en Lorenz y DOME.

**Architecture:**
- **El cierre es un componente propio** (`core/components/closure.py`) con la misma interfaz para `'constant'`, `'none'`, `'gls'` y `'kpp'`. Da ν en interfaces antes de la viscosidad implícita, se actualiza después con u, v nuevos, y da κ_T, κ_S y el término no local a la difusión vertical de trazadores. En ALE su estado se remapea tras el regrid.
- **La GLS es la de hoy** (`physics/mixing/gls.py`, port de CROCO): mismo código en σ y ALE; en ALE cambian solo la geometría (h), la regla de capas delgadas y el remapeo de k y ε.
- **KPP** en un módulo nuevo de columna (`physics/mixing/kpp_cvmix.py`) que sigue a CVMix como lo usa MOM6.
- **Infraestructura nueva que piden las columnas:** flujo de calor de superficie con onda corta penetrante, ley de pared en ALE, fronteras periódicas en x e y.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, SciPy, netCDF4, pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md` (aprobado por Esteban, 2026-10-29). Las decisiones de diseño 1–10 del spec son vinculantes.

**Fuentes** (leer antes de cada tarea la que le toque; todas con `archivo:línea`):
- `docs/design/n2c_refs/tke_in_ale.md`: HYCOM, MOM6, CROCO, Oceananigans.
- `docs/design/n2c_refs/croco_column_cases.md`: los siete casos de columna de CROCO, su KPP y su onda corta.
- `docs/design/n2c_refs/kpp_mom6_cvmix.md`: KPP de MOM6/CVMix, `CVmix_SCM_tests` y los huecos de `physics/mixing/kpp.py`.
- `docs/design/n2c_refs/closure_algorithms_literature.md`: literatura por decisión.
- Biblioteca local (`python scripts/litsearch.py`): Burchard 2002, Van Roekel 2018, Large 1994, Warner 2005, Umlauf & Burchard 2003 y 2005, Price 1979, Paulson & Simpson 1977, Klingbeil 2018.

## Global Constraints

- **Sin los cierres nuevos, nada cambia.** Con `closure` en `'constant'`, `'none'` o `'gls'` (σ, valores por defecto), sin `heat` y sin `periodic`, la matriz dorada de N1 (`tests/test_n1_golden.py`) y `scripts/n1_smoke.py` contra `main` quedan **bit a bit**, y la suite completa pasa.
- **Costo:** los casos de `n1_smoke.py` no suben más de 1 %.
- **Constantes sin ajustar:** GLS con las de CROCO (Canuto A, `tke_min` 1e-10, `eps_min` 1e-12, Galperin); KPP con las de CVMix tal como las usa MOM6. Cada constante cita su fuente.
- **KPP sin difusividad de fondo por defecto** (0), configurable.
- **Signos y unidades:**
  - `heat["Q_ns"]`, `heat["Q_sw"]` en W m⁻², positivos hacia el océano;
  - c_p = 3985 J kg⁻¹ K⁻¹ (CROCO `Cp`) como valor por defecto del constructor (`cp`);
  - flujo de flotabilidad de superficie B_f positivo = estabilizador (convención de MOM6);
  - índices (k, j, i) con k = 0 en el fondo; interfaces k = 0 … N.
- **Lenguaje y estilo:** solo Python, type hints, docstrings con la física y su cita, Numba en bucles calientes con oráculo NumPy en los tests (CLAUDE.md).
- **Hilos y corridas:** ≤ 5 hilos y solo P-cores; corridas largas una a la vez, con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`. Si una corrida pasa de ~24 h de reloj, se consulta a Esteban. Salidas a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`, nunca al repo.
- **Commits** con rutas explícitas (nunca `git add validation/` ni `-A`). Terminan con:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```
- **Rama:** `n3-closures`, desde `main` (`97cabe6` o posterior).

## Review Focus

1. **Capa casi vacía en una columna híbrida o z\*** (h ~ 1e-10 m entre capas gruesas): ni N², ni el corte, ni k, ν o κ deben dar picos, NaN o valores negativos. Lo cubre Task 5, `test_gls_vanished_layer_no_spike`.
2. **Columna entera seca o muy somera** (D ≤ d_min) con `closure='gls'` o `'kpp'` y flujo de calor: el cierre la deja quieta, sin NaN; el calor sobre tierra o celda seca se ignora, como `fw`. Lo cubre Task 2, `test_heat_on_land_and_dry_cells_is_ignored`, y Task 7, `test_kpp_dry_column_is_skipped`.
3. **Restart a mitad de una corrida con GLS en ALE o con KPP:** bit a bit; un restart escrito con otro cierre se rechaza con un error claro. Lo cubren Task 5, `test_gls_ale_restart_bit_for_bit`, y Task 7, `test_kpp_restart_bit_for_bit_and_wrong_closure_rejected`.
4. **`periodic` junto con una frontera abierta en el mismo eje:** error claro en pre-flight, no un dominio mal conectado. Lo cubre Task 3, `test_periodic_with_open_edge_same_axis_is_an_error`.
5. **u\* = 0 con enfriamiento** (convección libre, KPP): la profundidad de la capa límite y el término no local salen finitos, sin dividir entre cero. Lo cubre Task 6, `test_kpp_free_convection_ustar_zero`.

---

### Task 1: el cierre como componente propio (sin cambio de física)

**Files:**
- Create: `core/components/closure.py`
- Modify: `core/components/free_surface.py` (sacar `sync_closure_in`/`closure_out` y la llamada a la GLS), `core/dynamics/sigma_wd.py:181-207,321-363` (la GLS deja de vivir en `SigmaWetDry`; queda el solver de viscosidad que recibe ν), `core/model/driver.py` (fases 4 y 7 llaman al cierre), `core/dynamics/sigma_baroclinic.py` (construye el componente), `core/restart.py` (estado etiquetado `closure`).
- Test: `tests/test_closure_component.py`

**Interfaces:**
- Produces, en `core/components/closure.py`:
  - `class Closure` (base) con:
    - `viscosity(st) -> tuple[np.ndarray, np.ndarray] | None`: ν en las interfaces de las caras u y v, `(N+1, ny, nx+1)` y `(N+1, ny+1, nx)`, del estado de cierre actual; `None` = el solver usa su ν constante como hoy.
    - `update(u, v, h, rho, ustar2_sfc, ustar2_bot, bflux=None) -> None`: después de la viscosidad implícita, con u, v nuevos en centros; `h` (N, ny, nx) espesores de capa; `rho` (N, ny, nx) o `None`; `bflux` dict opcional (Task 6–7).
    - `tracer_diffusivity() -> np.ndarray | None`: κ (N+1, ny, nx) para T y S; `None` = usar `kv` constante como hoy.
    - `nonlocal_flux() -> np.ndarray | None`: término no local por interfaz (N+1, ny, nx) normalizado por el flujo de superficie, o `None`.
    - `remap(h_src, h_dst) -> None`: ALE, tras el regrid (no hace nada en `ConstantClosure`).
    - `state_fields() -> dict[str, np.ndarray]` y `load_state(d) -> None` (restart, prefijo `closure_`).
  - `ConstantClosure` (`'constant'`, `'none'`, `'parabolic'` tal como hoy) y `GLSClosure` (envuelve `GLSKEpsilon`).
  - `make_closure(model) -> Closure`.

- [ ] **Step 1: tests.**
  - `test_closure_extraction_is_bitwise`: corre `tests/n1_cases.py` en todas las configuraciones doradas con GLS y sin GLS y compara con `GOLDEN` como `tests/test_n1_golden.py` (puede reutilizar su parametrización).
  - `test_gls_stratified_bitwise_after_extraction`: una columna σ+GLS estratificada de 50 pasos da el mismo hash antes y después (hash fijado con el código de `main` antes de empezar; guardarlo en `tests/data/n3_gls_sigma_hash.json` con `scripts/n1_make_golden.py` o un script equivalente).
  - `test_closure_state_in_restart`: el restart de σ+GLS sigue bit a bit (`tests/test_restart.py` ya lo cubre; correrlo).
- [ ] **Step 2:** los tests de bit a bit pasan antes del cambio (son la red de seguridad); escribir el componente y mover las llamadas; volver a correr.
  Run: `pytest tests/test_closure_component.py tests/test_n1_golden.py tests/test_gls*.py tests/test_restart.py tests/test_sigma_wd.py -q`
  Expected: todo PASS, sin tocar las tolerancias.
- [ ] **Step 3:** suite completa (`pytest tests/ -q`) PASS.
- [ ] **Step 4: commit** con rutas explícitas.

---

### Task 2: flujo de calor de superficie con onda corta penetrante

**Files:**
- Create: `physics/surface/__init__.py`, `physics/surface/shortwave.py`
- Modify: `core/model/driver.py` (`step(st, tau_x=None, tau_y=None, fw=None, heat=None)`; subpaso en la fase 7), `core/dynamics/sigma_baroclinic.py` (`step` y constructor: `cp: float = 3985.0`, `jerlov: str = "I"`), `core/components/variance_budget.py` (término `T_surf`), `core/model/preflight.py`.
- Test: `tests/test_surface_heat.py`

**Interfaces:**
- Produces:
  - `swfrac(z: np.ndarray, jerlov: str = "I") -> np.ndarray`: fracción de onda corta que llega a la profundidad z ≤ 0 [m], Paulson & Simpson (1977) en dos bandas; coeficientes (R, ζ₁, ζ₂) por tipo de Jerlov I, IA, IB, II, III de Large et al. 1994 / CROCO `lmd_swfrac.F` (citar línea); tipo I: R = 0.58, ζ₁ = 0.35 m, ζ₂ = 23 m.
  - `heat = {"Q_ns": (ny, nx), "Q_sw": (ny, nx) opcional}` en `step`.
  - `ctx.bflux` para Task 7: `{"Q_ns": …, "Q_sw": …, "sw_frac_iface": (N+1, ny, nx)}` en el `StepContext`.
- Regla: Q_ns entra por la interfaz superior de T; Q_sw se reparte entre capas con la diferencia de `swfrac` en sus interfaces; la fracción que llega al fondo se deposita en la capa del fondo (confirmar con CROCO `step3d_t.F:578-587` y MOM6, citar). ΔT = Q Δt / (ρ₀ c_p h).

- [ ] **Step 1: tests.**
  - `test_swfrac_type_I_values`: `swfrac(0) == 1`, `swfrac(-1e4) < 1e-12`, `swfrac(-10)` igual a 0.58 e^(−10/0.35) + 0.42 e^(−10/23) a 1e-14.
  - `test_heat_content_change_is_exact`: columna cerrada con `Q_ns = -100`, `Q_sw = 200` durante n pasos: Σ ρ₀ c_p T V cambia en (Q_ns + Q_sw) A n Δt a rtol 1e-12, en σ y ALE z\*.
  - `test_heat_on_land_and_dry_cells_is_ignored` (Review Focus 2).
  - `test_no_heat_is_bitwise`: sin `heat` las configuraciones doradas no cambian (parte de `test_n1_golden.py`).
  - `test_variance_budget_closes_with_heat`: con `variance_budget=True` y `heat`, el balance de T cierra a ≤ 5e-10 relativo, con el término `T_surf`.
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_surface_heat.py tests/test_n1_golden.py tests/test_n2a_budget.py -q`
- [ ] **Step 3: commit.**

---

### Task 3: fronteras periódicas en x e y

**Files:**
- Modify: constructor y geometría (`core/dynamics/sigma_baroclinic.py`, `core/grid/horizontal.py` si hace falta), todos los operadores horizontales que leen vecinos (momento, Coriolis, PGF Jacobiano y FV, advección de trazadores con FCT, difusión K_h, continuidad σ y ALE, transporte de capas, superficie libre implícita: matriz y CG/LU en `core/dynamics/wetting_drying.py` y `free_surface.py`), `core/model/preflight.py`, `core/restart.py` (config check).
- Test: `tests/test_periodic.py`

**Interfaces:**
- Produces: argumento `periodic: tuple[str, ...] = ()` con `"x"`, `"y"` o ambos. Con periodicidad en x, la cara u de la columna 0 y la de la columna nx son la misma (se guarda una sola, o se mantienen iguales; el constructor elige y lo documenta); igual en y para v.

- [ ] **Step 1: tests.**
  - `test_periodic_column_stays_uniform`: dominio 2×2 doblemente periódico, viento uniforme y `heat` uniforme, 200 pasos, σ y ALE z\*: todas las columnas iguales a 1e-12 y u, v ≠ 0.
  - `test_periodic_x_tracer_returns`: canal periódico en x con u uniforme; un trazador da una vuelta completa y se conserva a rtol 1e-12.
  - `test_periodic_free_surface_wave_period`: η = a cos(2πx/L) en un canal periódico oscila con el periodo de la onda de gravedad discreta a 1 %.
  - `test_periodic_with_open_edge_same_axis_is_an_error` (Review Focus 4).
  - `test_periodic_restart_bit_for_bit` y `test_no_periodic_is_bitwise` (golden).
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_periodic.py tests/test_n1_golden.py -q` y la suite.
- [ ] **Step 3: commit.**

---

### Task 4: ley de pared en ALE

**Files:**
- Modify: `core/model/preflight.py:127-129` (quitar "ALE needs z0=None"), arrastre de fondo ALE (`core/components/bottom_drag.py` o una clase nueva `LogLawBedDrag` ahí), `core/model/driver.py` (`_ale_implicit_and_free_surface`), `physics/bottom_friction.py` (reutilizar C_d(D, z₀)).
- Test: `tests/test_ale_loglaw.py`

**Interfaces:**
- Produces: en ALE, `z0` > 0 activa C_d de ley de pared con la distancia al fondo `z_b`; u\*²_fondo por columna disponible para el cierre (`ctx.ustar2_bot`, (ny, nx)) en σ y ALE.
- Regla de `z_b` (decisión del constructor, citada en el docstring): la capa con masa más cercana al fondo o la suma de capas hasta `hbbl`, con referencia a MOM6 (`MOM_set_viscosity.F90`, BBL) o HYCOM.

- [ ] **Step 1: tests.**
  - `test_ale_loglaw_matches_sigma_column`: columna z\* con capas fijas (sin superficie libre que las mueva) y σ con los mismos espesores: u tras 100 pasos igual a 1e-12 relativo.
  - `test_ale_loglaw_vanished_bed_layer_finite`: capa de fondo de 1e-6 m: C_d finito y ≤ el C_d a la distancia `hbbl`; sin NaN.
  - `test_quadratic_drag_unchanged`: `bottom_drag='quadratic'` sigue bit a bit (golden DOME de N1).
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_ale_loglaw.py tests/test_n1_golden.py tests/test_bottom_friction.py -q`
- [ ] **Step 3: commit.**

---

### Task 5: GLS k-ε estratificada en ALE

**Files:**
- Modify: `core/components/closure.py` (`GLSClosure` con geometría ALE y `remap`), `physics/mixing/gls.py` (opción `production` y regla de capas delgadas, si van ahí), `core/model/driver.py` (fase 8: `closure.remap` tras el regrid), `core/model/preflight.py:130-132` (permitir `'gls'` en ALE), `core/restart.py`.
- Test: `tests/test_gls_ale.py`

**Interfaces:**
- Consumes: `Closure` (Task 1), `ctx.ustar2_bot` (Task 4).
- Produces:
  - `closure='gls'` válido en ALE (σ, z\*, híbrido).
  - Argumento `gls_production: str = "croco"` (`"croco"`: P = ν M², B = −κ N² como hoy; `"energy"`: forma de Burchard 2002 que conserva energía, leer el artículo local y citar ecuación). El valor por defecto queda `"croco"` en σ y ALE para respetar el bit a bit; la columna de Kato–Phillips (Task 8) corre las dos y el resultado se reporta a Esteban antes de cambiar el defecto.
  - `GLSClosure.remap(h_src, h_dst)`: k y ε de interfaces a capas, remapeo conservativo de h·k y h·ε con el remapeo existente (`core/vertical/remap.py`, PPM o el esquema positivo que elija el constructor), regreso a interfaces, pisos `tke_min`/`eps_min` y límite de Galperin.
  - Regla de capas delgadas: N² y M² entre centros con un espesor mínimo; las capas con h < h_min se juntan con la vecina con masa (MOM6 `eliminate_massless`, `MOM_kappa_shear.F90`, citar línea) o se excluyen.

- [ ] **Step 1: tests.**
  - `test_gls_ale_column_matches_sigma`: columna z\* con capas fijas contra σ, 200 pasos con viento y N² inicial: k, ε, ν a rtol 1e-10 (si no se alcanza, el implementador justifica la tolerancia en el reporte; el plan acepta hasta 1e-8).
  - `test_gls_remap_conserves_column_tke`: remapeo con interfaces aleatorias: ∫k dz conservado a rtol 1e-12; k ≥ 1e-10 y ε ≥ 1e-12 en todos lados.
  - `test_gls_vanished_layer_no_spike` (Review Focus 1): columna híbrida con capas de 1e-10 m entre capas de 5 m: max ν, κ, k finitos y ≤ los de la misma columna sin las capas vacías × 1.01.
  - `test_gls_ale_restart_bit_for_bit` (Review Focus 3).
  - `test_gls_production_energy_closes_energy`: con `gls_production="energy"`, en una columna sin fuentes externas, la energía cinética media perdida por la viscosidad en un paso es igual a la P integrada que entra a k a rtol 1e-10.
  - `test_gls_ale_lorenz_short`: experimento A de Lorenz en ALE-σ con `closure='gls'`, 50 pasos, sin NaN y balance de varianza cerrando a ≤ 5e-10.
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_gls_ale.py tests/test_gls*.py tests/test_n1_golden.py -q` y la suite.
- [ ] **Step 3: commit.**

---

### Task 6: KPP de columna estilo CVMix, y corrección del signo en `kpp.py`

**Files:**
- Create: `physics/mixing/kpp_cvmix.py`
- Modify: `physics/mixing/kpp.py` (solo el signo de `b_flux`), `tests/test_kpp.py` (solo `test_surface_heating_produces_nonlocal_warming`, que fija el error)
- Test: `tests/test_kpp_cvmix.py`

**Interfaces:**
- Produces: `kpp_column(h, z_iface, u, v, b, bflux_sfc, ustar, sw_frac_iface, f, params) -> KPPResult` sobre M columnas a la vez (arrays (N, M) con k = 0 en el fondo); `KPPResult` con `obl_depth (M,)`, `kappa_m (N+1, M)`, `kappa_s (N+1, M)`, `nonlocal (N+1, M)` (forma G_s = 6.327 σ(1−σ)², solo con B_f desestabilizador) y `k_obl (M,)`. `params` un dataclass `KPPParams` con los valores de CVMix por defecto que usa MOM6 (Ri_c, C_v, Vtc, ε = 0.1, sin empate con el interior, mezcla aumentada en la interfaz de la base, `kappa_bg = 0`), cada uno citado en `kpp_mom6_cvmix.md` o en `cvmix_kpp.F90:línea`.
- Búsqueda de la capa límite: Ri_b con promedio sobre la capa superficial ε·h, Vt², interpolación como MOM6 (`MOM_CVMix_KPP.F90`, citar), y la OBL al menos el espesor de la capa superior.

- [ ] **Step 1: tests.**
  - `test_old_kpp_heating_is_stabilising` en `tests/test_kpp.py`: con calentamiento de superficie no hay término no local ni rama convectiva (corrige el test que fijaba el error).
  - `test_kpp_matches_cvmix_reg_test`: reproduce la salida de `~/pyOCEAN_refs/MOM6/pkg/CVMix-src/reg_tests/kpp` (las funciones de forma, Vt² y la profundidad de la capa límite del test) a 1e-10; si el test de CVMix no se puede correr, el implementador reproduce sus valores de referencia escritos en el código o en los `.nl`, y lo dice.
  - `test_kpp_free_convection_ustar_zero` (Review Focus 5).
  - `test_kpp_neutral_wind_obl_depth`: perfil lineal, viento, sin calor: la OBL coincide con la que da el algoritmo de Large et al. 1994 ec. (21)–(23) calculado a mano en el test.
  - `test_kpp_resolution_large_appendix_c`: con Δz = 0.2, 2.0 y 3.33 m (Large 1994, Apéndice C) la OBL no difiere más de lo que reporta ese apéndice.
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_kpp_cvmix.py tests/test_kpp.py -q`
- [ ] **Step 3: commit.**

---

### Task 7: KPP en el modelo (σ y ALE)

**Files:**
- Modify: `core/components/closure.py` (`KPPClosure`), `core/model/driver.py` (término no local en la difusión vertical de T y S de la fase 7, explícito en el lado derecho), `core/model/preflight.py` (`closure='kpp'` en σ y ALE), `core/restart.py`.
- Test: `tests/test_kpp_model.py`

**Interfaces:**
- Consumes: `kpp_column` (Task 6), `ctx.bflux` (Task 2), `Closure` (Task 1), `ctx.ustar2_bot` (Task 4).
- Produces: `closure='kpp'`, con `kpp_params: KPPParams | None = None`. B_f de superficie con α, β del modelo (EOS lineal) e incluye solo la onda corta absorbida dentro de la capa límite (Large 1994, A3–A4). ν de KPP en u, v; κ_s a T y S.

- [ ] **Step 1: tests.**
  - `test_kpp_column_model_matches_kpp_column`: columna periódica 2×2 (Task 3) con viento y enfriamiento: κ_s del modelo tras un paso = `kpp_column` con las mismas entradas a 1e-12.
  - `test_kpp_nonlocal_conserves_heat`: con enfriamiento, el término no local redistribuye sin cambiar el contenido de calor de la columna (rtol 1e-12).
  - `test_kpp_dry_column_is_skipped` (Review Focus 2).
  - `test_kpp_restart_bit_for_bit_and_wrong_closure_rejected` (Review Focus 3).
  - `test_kpp_sigma_vs_ale_zstar_column`: columna fija, σ contra ALE z\* a 1e-10.
- [ ] **Step 2:** FAIL, implementar, PASS. Run: `pytest tests/test_kpp_model.py tests/test_n1_golden.py -q` y la suite.
- [ ] **Step 3: commit.**

---

### Task 8: columnas contra CROCO (GLS y KPP)

**Files:**
- Create: `validation/turbulence_columns/__init__.py`, `cases.py` (los siete casos de `croco_column_cases.md` con sus valores exactos), `run_pyocean.py`, `croco/` (`build_croco.sh` y `run_croco.sh` como `validation/croco_*`/`thacker`, con GLS y con `LMD_SKPP` para Kato–Phillips, Willis–Deardorff y el ciclo diurno), `compare.py`, `README.md`.
- Test: `tests/test_turbulence_columns.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: `cases.build(name: str, coord: str, closure: str) -> (model, state, forcing)`; `name` ∈ {`kato_phillips`, `willis_deardorff`, `diurnal_cycle`, `forced_ekbbl`, `forced_dbleek`, `forced_nonrotbbl`, `forced_oscnonrotbbl`}; `coord` ∈ {`sigma`, `zstar`, `hybrid`}.

- [ ] **Step 1: tests.**
  - `test_cases_match_croco_setup`: grid, forzamiento y N² inicial de cada caso iguales a los de `croco_column_cases.md`.
  - `test_kato_phillips_price_law_short`: 6 h de Kato–Phillips en σ con GLS: D_m dentro de ±10 % de 1.05 u\* N₀^(−1/2) t^(1/2) (Warner et al. 2005, ec. 60; la definición de D_m — donde k < 1e-5 — va en `compare.py`).
- [ ] **Step 2:** FAIL, implementar, PASS.
- [ ] **Step 3: correr** (corridas de minutos, una a la vez): todos los casos en σ y ALE z\* con GLS (híbrido donde haya estratificación), KPP en los tres casos de superficie, y Kato–Phillips con `gls_production` `"croco"` y `"energy"` y con y sin regrid. CROCO compilado en `~/pyOCEAN_refs/croco_runs/`.
- [ ] **Step 4: comparar y escribir el README** con los criterios 1 y 3 del spec:
  - Kato–Phillips: D_m a ±5 % de CROCO y de la ley de Price;
  - los demás: perfiles de T, u, k, ε contra CROCO;
  - σ contra ALE;
  - KPP contra CROCO-KPP, clasificando como (c) las diferencias por constantes;
  - el resultado de `gls_production="energy"` con una recomendación para Esteban;
  - clasificación (a)–(d) y "Qué debe mejorar pyOCEAN".
- [ ] **Step 5: commit** de `validation/turbulence_columns/` (rutas explícitas) y del test.

---

### Task 9: columnas contra MOM6/CVMix (KPP)

**Files:**
- Create: `validation/turbulence_columns/mom6_scm.py` (casos `wind_only`, `cooling_only`, `mech_only`, `skin_warming_wind` con la configuración exacta de `kpp_mom6_cvmix.md`: 400 capas de 1 m, 2×2 periódico, Δt 1200 s, 5 días, f = 1e-4, EOS lineal ρ₀ = 1000, dρ/dT = −0.255, S = 35, T = 15 + 0.01 z), `validation/turbulence_columns/mom6/run_mom6.sh` (como `validation/mom6_lock_exchange/run_mom6.sh`, con el MOM6 de `~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6`); sección nueva en el README.
- Test: `tests/test_turbulence_columns.py` (`test_mom6_scm_cases_match_mom_override`).

- [ ] **Step 1:** test de configuración (FAIL, implementar, PASS).
- [ ] **Step 2: correr** MOM6 y pyOCEAN (ALE z\* y σ, `closure='kpp'`) los cuatro casos.
- [ ] **Step 3: comparar** (criterio 2 del spec): OBL y profundidad de mezcla a ±5 % de MOM6 en `wind_only`, `cooling_only`, `mech_only`; `skin_warming_wind` se reporta. Clasificación (a)–(d), "Qué debe mejorar pyOCEAN".
- [ ] **Step 4: commit.**

---

### Task 10: Lorenz ALE-σ + GLS y DOME híbrido + GLS

Dos despachos, como en N2a (Ruling R3): (10a) el implementador prepara y mide el costo y se detiene; el controlador lanza las corridas largas una a la vez; (10b) análisis.

**Files:**
- Modify: `validation/lorenz_estuary/case.py` (configuración `ale_sigma_gls`: ALE `regrid="sigma"`, `closure="gls"`, ley de pared con `z0` = el de `sigma_gls`, Δt estable medido), `validation/lorenz_estuary/run_pyocean.py` y `compare.py` (columna nueva), `validation/dome/run_pyocean.py` (`--closure gls`; arrastre cuadrático igual a la línea base; piso de κ_t de la GLS, decisión 10 del spec), ambos README.
- Test: `tests/test_lorenz_estuary.py` (`test_ale_sigma_gls_short_run_closes_budget`), `tests/test_dome_benchmark.py` (`test_dome_gls_short_run`).

- [ ] **Step 1 (10a):** tests cortos FAIL, implementar, PASS; medir el costo (2 h simuladas de Lorenz A y B; 1 día de DOME 10 km híbrido) y extrapolar. Si una corrida pasa de ~24 h, se detiene y se consulta a Esteban.
- [ ] **Step 2 (controlador):** corridas largas, una a la vez: Lorenz A–D `ale_sigma_gls` (600 ciclos M2) y DOME 10 km híbrido + GLS (50 días); si se corre también con el `kv` viejo como piso, se reporta aparte.
- [ ] **Step 3 (10b):** tabla de Lorenz con la columna nueva contra GETM, σ+GLS y ALE-σ constante (criterio 4); DOME b_f, transporte de la pluma, T en rango y truncamientos (criterio 5); clasificación (a)–(d) y "Qué debe mejorar pyOCEAN" en ambos README.
- [ ] **Step 4: commit.**

---

### Task 11: documentación y cierre

**Files:**
- Modify: `CLAUDE.md` (filas nuevas: cierre como componente, flujo de calor, periódicas, ley de pared ALE, GLS en ALE, KPP; columnas; conteo de tests), `docs/design/vision_roadmap.md` (§7 scorecard: columnas, Lorenz ALE+GLS, DOME+GLS; §8 N3 ✅), `docs/manual/architecture/time-step.md` y `docs/manual/theory/` (cierres y flujo de calor), `docs/literature/reading_list.md` (estados).

- [ ] **Step 1:** `pytest tests/ -q` con las variables BLAS: todo pasa, con los mismos 2 xfail.
- [ ] **Step 2:** `python -I scripts/n1_smoke.py` contra un worktree de `main`: 9/9 idénticos, costo ≤ 1.01×.
- [ ] **Step 3:** documentación y commit.
