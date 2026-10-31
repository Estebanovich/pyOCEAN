# N2b núcleo: flujos bulk acoplados en el modelo 3D — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que el modelo 3D calcule en cada paso el esfuerzo del viento, el calor y el agua dulce con fórmulas bulk a partir de una atmósfera (ERA5 o WRF) y de su propia SST. Las fórmulas son NCAR en la forma de FMS/MOM6 y COARE portada de CROCO, y cada una se valida contra su modelo de referencia.

**Architecture:**
- **Algoritmos** en `forcing/bulk.py` (NCAR forma FMS) y `forcing/bulk_coare.py` (port de CROCO). Cada uno se compara a 1e-10 con un driver Fortran compilado de su referencia.
- **Proveedor de atmósfera** `forcing/atmosphere.py`: función del tiempo del modelo que devuelve los campos canónicos en los puntos T, con caché de dos registros.
- **Componente** `core/components/surface_fluxes.py`: lo llama el driver al inicio del paso; produce `tau_x`, `tau_y`, `heat` y `fw`, que entran por las vías existentes sin tocar las fases.

**Tech Stack:** Python, NumPy 1.26.4, xarray, Numba solo si el costo lo pide. gfortran/mpif90 de Homebrew para los drivers de referencia.

**Spec:** `docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md` (aprobado por Esteban el 2026-10-31). Lecturas: `docs/design/n2b_refs/mom6_fms_bulk.md`, `docs/design/n2b_refs/croco_bulk.md`.

## Global Constraints

- Solo Python, sin dependencias nuevas, nunca `pip install`; numpy 1.26.4. `cdsapi` ya está instalado y `~/.cdsapirc` existe.
- Corridas con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, ≤ 5 hilos en P-cores, una a la vez. Salidas y datos descargados en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/` (enlace al disco LaCie; ERA5 en `pyOCEAN_runs/forcing/era5/`). Si una corrida pasa de ~24 h, se consulta a Esteban.
- **Nunca tocar** `/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan*` (datos de tesis).
- Código de referencia descargado en `~/pyOCEAN_refs/`, fuera del repo.
- Sin `surface_fluxes`, el camino actual no cambia ni en el orden de las operaciones: goldens (`tests/test_n1_golden.py`) y humo (`scripts/n1_smoke.py`) idénticos bit a bit.
- Ningún parámetro se ajusta a un caso. Cada valor por defecto cita su fuente (`archivo:línea` de FMS/CROCO o LY04).
- Cada ecuación cita la referencia (`archivo:línea` o paper) en el docstring, con unidades. Índices (k, j, i) con k = 0 en el fondo; la superficie es k = N−1.
- Flujos positivos hacia el océano. `fw` = P − E [m s⁻¹]. `heat` = {"Q_ns", "Q_sw"} [W m⁻²]. τ [N m⁻²] en caras u (ny, nx+1) y v (ny+1, nx).
- Commits con rutas explícitas: nunca `git add validation/`, `-A` ni `.`, ni `logs/`, `pyOCEAN-literature-review-guide.docx`, npz grandes o salidas. Las tablas de referencia pequeñas (< 1 MB) van a `tests/data/`. Trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```

## Review Focus

1. **Viento casi nulo con estratificación muy estable o muy inestable:** los dos algoritmos dan flujos finitos, sin división por cero ni iteración divergente (piso de viento, ráfagas). Tests en Tasks 1 y 2.
2. **Columna más somera que `h_sfc`, o capa superior de 1 mm en la híbrida:** el promedio de superficie es el de toda la columna o el de las capas que caben en 1 m, finito; una columna seca no recibe nada. Test en Task 4.
3. **Tiempo del modelo fuera de la cobertura de la atmósfera:** al inicio lo rechaza el pre-flight; a mitad de corrida, error claro en ese paso (sin extrapolar). Tests en Tasks 3 y 4.
4. **ERA5 con NaN o valores de tierra junto a la costa:** se rellena la tierra antes de interpolar (`fill_land`); las celdas de océano del modelo quedan finitas. Test en Task 3.
5. **Evaporación mayor que el agua de una capa superior delgada:** pasa por el déficit de evaporación de N2a; volumen y sal se conservan. Test en Task 4.

---

### Task 1: NCAR en la forma de FMS, contra FMS compilado (criterio 1, parte NCAR)

**Files:**
- Create: `validation/bulk_points/fms_ncar_driver.F90`, `validation/bulk_points/build_reference.sh`, `validation/bulk_points/points.py`, `validation/bulk_points/README.md`, `tests/data/bulk_fms_ncar_reference.npz`
- Modify: `forcing/bulk.py` (nueva `bulk_ncar_fms`; `_SCHEMES`), `docs/design/n2b_refs/mom6_fms_bulk.md` (quitar los [verificar] resueltos)
- Test: `tests/test_bulk_fms.py`

**Interfaces:**
- Produces:
  - `bulk_ncar_fms(u10, v10, t_air, q_air, sst, p, *, u_ocean=0.0, v_ocean=0.0, z_u=10.0, z_t=10.0, z_q=10.0) -> TurbulentFluxes` (dataclass existente de `forcing/bulk.py`). t_air y sst en K, como FMS; el resto en SI.
  - `points.py::POINTS`: un `dict[str, np.ndarray]` con ≥ 200 puntos que cubren estable, inestable, viento 0.1–30 m s⁻¹, SST −1.8–32 °C y con y sin corriente.
  - `_SCHEMES["ncar_fms"] = bulk_ncar_fms`; `"ncar"` sigue siendo `bulk_ncar` (LY04/MITgcm) en `compute_surface_fluxes`, para no cambiar el comportamiento existente.

- [ ] **Step 1: Traer FMScoupler.** `git clone https://github.com/NOAA-GFDL/FMScoupler ~/pyOCEAN_refs/FMScoupler` y checkout de la versión compatible con FMS 2025.02.01 (anotar el commit). Leer `surface_flux.F90` (`ncar_ocean_fluxes` y su llamador), `ocean_albedo` y `ocean_rough` hasta resolver los [verificar] de `mom6_fms_bulk.md` §2: iteraciones, piso de viento, ráfagas, ψ, recorte de ζ, densidad del aire, q_sat. Actualizar el documento con `archivo:línea`.
- [ ] **Step 2: Driver de referencia.** `fms_ncar_driver.F90` llama a `ncar_ocean_fluxes` con los arrays de `POINTS`, leídos de un archivo binario o de texto que escribe `points.py`, y escribe τ, sensible, latente, evaporación, C_d, C_h, C_e y u\*. `build_reference.sh` compila contra los módulos de FMS que ya construye `validation/mom6_column/build_reference.sh` y genera `tests/data/bulk_fms_ncar_reference.npz`.
- [ ] **Step 3: Write the failing tests** en `tests/test_bulk_fms.py`:
  - `test_matches_fms_reference`: `bulk_ncar_fms` sobre `POINTS` iguala la tabla a 1e-10 relativo (o la tolerancia que el README justifique) en τ_x, τ_y, q_sens, q_lat y evap.
  - `test_calm_and_extreme_stability_finite` (Review Focus 1): viento 0, ΔT ±15 K → todo finito.
  - `test_ly04_differences_documented`: `bulk_ncar` contra `bulk_ncar_fms` en `POINTS`; la diferencia máxima relativa por campo queda por debajo de `points.LY04_VS_FMS_MAX_REL` (dict con la cifra medida y un comentario que nombra cada diferencia de algoritmo; la misma tabla va al README).
- [ ] **Step 4:** Run `pytest tests/test_bulk_fms.py -q`. Expected: FAIL (`bulk_ncar_fms` no existe).
- [ ] **Step 5:** Implementar `bulk_ncar_fms` línea por línea según el Fortran leído, con citas `surface_flux.F90:línea`.
- [ ] **Step 6:** Run `pytest tests/test_bulk_fms.py tests/test_bulk.py tests/test_surface_forcing.py -q`. Expected: PASS.
- [ ] **Step 7: MITgcm `pkg/exf` (criterio 1, punto 3).** Driver mínimo de `exf_bulkformulae` (o la rutina que use la versión local) sobre `POINTS`, comparado con `bulk_ncar`. Se reporta la diferencia en el README como (c) si no es exacta; sin test obligatorio si la rutina depende de bloques comunes que lo impidan (decirlo).
- [ ] **Step 8: Commit** (`forcing/bulk.py`, `tests/test_bulk_fms.py`, `tests/data/bulk_fms_ncar_reference.npz`, `validation/bulk_points/{fms_ncar_driver.F90,build_reference.sh,points.py,README.md}`, `docs/design/n2b_refs/mom6_fms_bulk.md`).

### Task 2: COARE portado de CROCO, contra CROCO compilado (criterio 1, parte COARE)

**Files:**
- Create: `forcing/bulk_coare.py`, `validation/bulk_points/croco_bulk_driver.F90`, `tests/data/bulk_croco_coare_reference.npz`
- Modify: `validation/bulk_points/build_reference.sh` (segundo driver), `validation/bulk_points/README.md`, `forcing/bulk.py` (`_SCHEMES["coare"]`)
- Test: `tests/test_bulk_coare.py`

**Interfaces:**
- Consumes: `POINTS` (Task 1).
- Produces:
  - `@dataclass CoareFluxes`: `tau_x, tau_y, q_sens, q_lat, q_lw, evap, cd, rho_air` (arrays en puntos T; q_lw neta hacia el océano, evap [m s⁻¹] positiva cuando sale agua).
  - `bulk_coare_croco(u10, v10, t_air_c, q_air, sst_c, lw_down, *, p_sfc=1.0e5, z_u=10.0, z_t=2.0, gustiness=True) -> CoareFluxes`. Viento absoluto, como CROCO; `p_sfc` constante por defecto, como CROCO (`croco_bulk.md` §1).
  - `_SCHEMES["coare"]`, un adaptador a `TurbulentFluxes` para `compute_surface_fluxes`.

- [ ] **Step 1: Driver de referencia.** Extraer de `~/pyOCEAN_refs/croco/OCEAN/bulk_flux.F` el bloque escalar (`:199-659`) y las funciones (`:758-972`) a una subrutina `bulk_point` (con `awk`, como `hybgen_column_mom6` en `validation/mom6_column/build_reference.sh`), con las opciones por defecto (`BULK_GUSTINESS`, `BULK_LW`), compilado en doble precisión (`-fdefault-real-8`, como `jobcomp`). Escribe las mismas salidas que `CoareFluxes` y genera el npz.
- [ ] **Step 2: Write the failing tests** en `tests/test_bulk_coare.py`:
  - `test_matches_croco_reference`: a 1e-10 relativo (o la tolerancia justificada en el README) en todos los campos de `CoareFluxes`.
  - `test_croco_quirks`: Charnock fijado con el viento inicial; ψ_t(0) = −0.0045; q_sat con 1e5 Pa (valores puntuales del documento, con `archivo:línea`).
  - `test_calm_and_extreme_stability_finite` (Review Focus 1).
- [ ] **Step 3:** Run `pytest tests/test_bulk_coare.py -q`. Expected: FAIL (módulo no existe).
- [ ] **Step 4:** Implementar el port línea por línea, con el pseudocódigo de `croco_bulk.md` §5 y citas `bulk_flux.F:línea`.
- [ ] **Step 5:** Run `pytest tests/test_bulk_coare.py tests/test_bulk.py -q`. Expected: PASS.
- [ ] **Step 6: Commit** (`forcing/bulk_coare.py`, `forcing/bulk.py`, `tests/test_bulk_coare.py`, `tests/data/bulk_croco_coare_reference.npz`, `validation/bulk_points/{croco_bulk_driver.F90,build_reference.sh,README.md}`).

### Task 3: proveedor de atmósfera y ERA5 real (criterio 2)

**Files:**
- Create: `forcing/atmosphere.py`, `validation/bulk_points/fetch_era5.py`
- Modify: `forcing/sources/era5.py` solo si una conversión resulta mal (con test que lo demuestre)
- Test: `tests/test_atmosphere.py`

**Interfaces:**
- Consumes: `ERA5Source`, `WRFSource`, `to_model_grid`, `fill_land`, `interp_time` (existentes).
- Produces:
  - `class AtmosphereProvider`:
    - constructor `(source_ds: xr.Dataset, grid: xr.Dataset, origin: np.datetime64)`, con `source_ds` canónico (lazy) en su malla original;
    - `__call__(t: float) -> dict[str, np.ndarray]`: los campos canónicos (`u10, v10, t2m, q2m, msl, sw_down, lw_down, precip`) en (ny, nx) en el tiempo `origin + t` [s], interpolados linealmente entre dos registros;
    - cada registro se lleva a la malla del modelo una sola vez (`fill_land` y luego `to_model_grid`), con caché de los dos registros vigentes;
    - fuera de la cobertura, `ValueError` con las fechas;
    - `coverage() -> tuple[float, float]` [s desde `origin`];
    - es función pura del tiempo: dos instancias iguales dan arrays idénticos para el mismo t, en cualquier orden de llamadas.
  - `era5_atmosphere(paths, grid, origin, t0, t1, bbox) -> AtmosphereProvider` y `wrf_atmosphere(paths, grid, origin, t0, t1) -> AtmosphereProvider`.
  - `fetch_era5.py`: descarga horaria con `cdsapi` (u10, v10, t2m, d2m, msl, ssrd, strd, tp y los flujos medios `msdwswrf`, `msdwlwrf`) para una caja y un periodo dados, a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/forcing/era5/`.

- [ ] **Step 1:** Descargar ERA5 de julio y agosto de 2017 sobre la plataforma noreste de EE. UU. (caja 38–43° N, 75–68° W; la del criterio 5). Si la API falla por credenciales, parar y reportar (no inventar datos).
- [ ] **Step 2: Write the failing tests** en `tests/test_atmosphere.py`:
  - `test_linear_in_time_exact`: con un dataset sintético de 3 registros, `provider(t)` iguala a `interp_time` llevado a la malla, a 1e-14, en t = registro, mitad y cuarto.
  - `test_pure_function_of_time`: llamadas en orden distinto (t2, t1, t2) dan arrays idénticos (`np.array_equal`).
  - `test_out_of_coverage_raises` (Review Focus 3).
  - `test_land_filled_before_interp` (Review Focus 4): un campo con NaN en tierra junto a la costa → celdas de océano finitas.
  - `test_era5_conversions_by_hand` (se salta si el archivo no existe, con `pytest.importorskip`/`skipif` del path): en un punto y una hora del archivo real, ssrd/3600 = sw_down, strd/3600 = lw_down, tp/3600 = precip [m s⁻¹] y q2m de d2m y msl (`specific_humidity_from_dewpoint`), contra valores calculados a mano en el test.
  - `test_era5_monthly_radiation_matches_mean_fluxes`: la media de julio de `sw_down` y `lw_down` iguala la de `msdwswrf`/`msdwlwrf` a 1 % (la diferencia es el centrado de los acumulados; documentar la cifra).
- [ ] **Step 3:** Run `pytest tests/test_atmosphere.py -q`. Expected: FAIL.
- [ ] **Step 4:** Implementar.
- [ ] **Step 5:** Run `pytest tests/test_atmosphere.py tests/test_era5.py tests/test_wrf.py tests/test_forcing_interp.py -q`. Expected: PASS.
- [ ] **Step 6: Commit** (`forcing/atmosphere.py`, `tests/test_atmosphere.py`, `validation/bulk_points/fetch_era5.py`, y `forcing/sources/era5.py` si cambió).

### Task 4: los flujos bulk en el modelo (criterios 3, 4 y 6)

**Files:**
- Create: `core/components/surface_fluxes.py`
- Modify: `core/model/driver.py` (`Driver.step`), `core/dynamics/sigma_baroclinic.py` (parámetro, docstring, `step` sin forzamiento), `core/model/preflight.py`, `core/model/registry.py`, `core/restart.py` (comprobación de configuración)
- Test: `tests/test_surface_fluxes_model.py`

**Interfaces:**
- Consumes: `AtmosphereProvider` (Task 3); `bulk_ncar_fms`, `bulk_ncar`, `bulk_kara` (Task 1, existentes); `bulk_coare_croco` (Task 2); `radiative_fluxes` (existente).
- Produces:
  - `@dataclass BulkFluxes` (configuración):
    - `atmosphere: AtmosphereProvider` y `scheme: str = "ncar"` ∈ {"ncar" (= `bulk_ncar_fms`), "ncar_ly04" (= `bulk_ncar`, MITgcm), "coare" (= `bulk_coare_croco`), "kara"};
    - `h_sfc: float = 1.0` [m] (MOM6 `HMIX_SFC_PROP`, `MOM.F90:2695`); `h_uv_sfc: float | None = None` (None = `h_sfc`; 0 = la capa superior, como MOM6 `HMIX_UV_SFC_PROP`);
    - `relative_wind: bool = True` en todos los esquemas; para imitar a CROCO (viento absoluto) se pasa `False` explícitamente;
    - `albedo: float = 0.066`, `emissivity: float = 0.97` (LY04, decisión 8 del spec); `"coare"` usa su propia onda larga (0.985 σT⁴, `croco_bulk.md` §4);
    - `z_u: float = 10.0`, `z_t: float = 2.0`, `z_q: float = 2.0` (ERA5).
  - `surface_average(h: np.ndarray, q: np.ndarray, h_sfc: float) -> np.ndarray`: (N, ny, nx) → (ny, nx), promedio ponderado por espesor sobre los `h_sfc` m superiores (k = N−1 hacia abajo); si la columna es más somera, sobre toda; `h_sfc == 0` es la capa superior; columnas secas → NaN (luego enmascaradas).
  - `class SurfaceFluxes` (componente): `forcing(st, t: float) -> tuple[tau_x, tau_y, fw, heat]`, más `last: dict[str, np.ndarray]` con `q_sens, q_lat, q_lw, q_sw, evap, precip, tau_x, tau_y, sst_bulk` en (ny, nx) del último paso.
    - SST, SSS, u y v de superficie del estado n; atmósfera en t = `st.time + dt/2` (decisión 2).
    - τ a las caras: media de los dos centros vecinos mojados; si un vecino es tierra o seco, el valor del mojado; envoltura periódica (decisión 7).
    - `heat = {"Q_ns": q_sens + q_lat + q_lw, "Q_sw": q_sw}`; `fw = precip − evap`. Celdas secas y tierra → 0.
  - `SigmaBaroclinic(..., surface_fluxes: BulkFluxes | None = None)`. Con él, `step(st)` no acepta `tau_x/tau_y/fw/heat` (error).
  - `Driver.step`: si `m.surface_fluxes` existe, sustituye los cuatro argumentos con `m._surface_fluxes.forcing(st, st.time + m.dt / 2)` **antes** de construir `StepContext`; si no, ninguna línea nueva se ejecuta.
  - Pre-flight, todo en un reporte (criterio 6): esquema desconocido; bulk sin `salinity` (E − P y la SSS de `"kara"` la necesitan); cobertura de la atmósfera que no incluye `[0, dt/2]` del primer paso; `h_sfc < 0`; albedo fuera de [0, 1]; emisividad fuera de (0, 1].
  - Registro: `q_sens`, `q_lat`, `q_lw`, `q_sw`, `evap`, `precip`, `tau_x_bulk`, `tau_y_bulk`, `sst_bulk` (y, x), diagnósticos.
  - Restart: comprueba `scheme`, `h_sfc`, `h_uv_sfc`, `relative_wind` y el `origin` de la atmósfera; no guarda estado (decisión 6).

- [ ] **Step 1: Write the failing tests** en `tests/test_surface_fluxes_model.py`, todos con un proveedor sintético de pocos registros (sin archivos):
  - `test_off_is_bit_for_bit`: σ, z\* e híbrido, 10 pasos con `surface_fluxes=None` contra el mismo sin el parámetro → `np.array_equal`.
  - `test_equals_prescribed`: con bulk y con los mismos `tau/heat/fw` que el componente devolvió, pasados a mano → estados idénticos bit a bit.
  - `test_heat_conservation`: Δ(Σ ρ₀ c_p T V) = Σ (Q_ns + Q_sw) A Δt a ≤ 1e-13 relativo del cambio, en σ, z\* e híbrido, con una columna de tierra y una seca.
  - `test_freshwater_and_salt_conservation`: Σ ΔV = Σ (P − E) A Δt al nivel de los tests de N2a, y la sal total constante.
  - `test_evaporation_exceeds_thin_top_layer` (Review Focus 5): evaporación fuerte sobre una capa superior de 1 mm en la híbrida → `n_limited_fw` > 0, volumen y sal conservados.
  - `test_sst_independent_of_coordinate` (criterio 4): columna uniforme en reposo, z\* de capas fijas contra σ → SST, flujos y T(z) a ≤ 1e-9 tras 20 pasos; la híbrida da los mismos flujos a la tolerancia que el test justifique en un comentario.
  - `test_surface_average_shallow_and_thin` (Review Focus 2): columna de 0.4 m → promedio de toda la columna; capa superior de 1 mm sobre una de 5 m → promedio de 1 m correcto; `h_sfc=0` → capa superior.
  - `test_dry_and_land_get_nothing`: celdas secas o de tierra quedan quietas bit a bit.
  - `test_coverage_ends_mid_run` (Review Focus 3): la atmósfera termina antes del paso 5 → `ValueError` en el paso 5 con la fecha.
  - `test_preflight_refuses_all_at_once`: esquema desconocido, sin salinidad, cobertura insuficiente y albedo 2 en un solo reporte; `step(st, tau_x=...)` con bulk activo levanta error.
  - `test_restart_bit_for_bit_with_bulk`: σ e híbrido, 6 pasos = 3 + restart + 3.
  - `test_threads_bit_for_bit`: 1 contra 5 hilos.
- [ ] **Step 2:** Run `pytest tests/test_surface_fluxes_model.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `pytest tests/test_surface_fluxes_model.py tests/test_surface_heat.py tests/test_n2a_freshwater.py tests/test_n1_golden.py tests/test_restart.py tests/test_preflight.py -q`. Expected: PASS, goldens sin cambios.
- [ ] **Step 5:** Run la suite completa `pytest tests/ -q`. Expected: 1886 + los nuevos passed, 2 xfailed.
- [ ] **Step 6: Commit.**

### Task 5: TEF en el restart (criterio 7)

**Files:**
- Modify: `diagnostics/tef.py` (`state()`/`load()` en `TEFTransect` y `TEFRegion`), `core/restart.py`
- Test: `tests/test_n2a_tef.py`

**Interfaces:**
- Produces: `TEFTransect.state() -> dict[str, np.ndarray]` y `.load(d)`, ídem `TEFRegion`; variables de restart con prefijo `tef_` y `tef_region_`, escritas solo si el modelo tiene `tef_face` o `tef_region`. Un restart sin ellas en un modelo que las usa: error claro.

- [ ] **Step 1: Write the failing test** `test_tef_restart_bit_for_bit`: el caso corto de Lorenz que ya usan los tests de TEF, N pasos continuos contra N/2 + restart + N/2 → `vol`, `salt`, `salt2`, `time` y todos los acumuladores de `TEFRegion` con `np.array_equal`.
- [ ] **Step 2:** Run `pytest tests/test_n2a_tef.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `pytest tests/test_n2a_tef.py tests/test_restart.py tests/test_n1_restart.py -q`. Expected: PASS.
- [ ] **Step 5: Commit.**

### Task 6: humo y costo (criterios 6 y 8)

**Files:**
- Create: `scripts/bulk_cost.py`
- Test: ninguno nuevo; el script verifica y reporta

**Interfaces:**
- Consumes: `BulkFluxes`, `AtmosphereProvider` (Tasks 3 y 4).

- [ ] **Step 1:** `scripts/n1_smoke.py` contra `main` → 9/9 hashes idénticos (bulk apagado). Si alguno cambia, parar: es un error de la Task 4.
- [ ] **Step 2:** `bulk_cost.py`, con el protocolo de `scripts/bbl_cost.py` (mediana de 50 pasos tras 10, 5 hilos, un proceso por modo, alternado, 3 rondas). Caso BayIW cut z\* con `salinity=True`:
  - con bulk `"ncar"` y un proveedor sintético con registros horarios en la malla completa;
  - contra el mismo caso con los `tau/heat/fw` constantes prescritos.
  - Reporta la razón, el tiempo del componente y la parte de lectura e interpolación.
- [ ] **Step 3:** Run el script (solo; nada más corriendo). Expected: razón ≤ 1.05. Si no se cumple, perfilar como en la corrección de costo de BBL antes de proponer cambios, y reportar sin bajar el criterio.
- [ ] **Step 4:** Commit del script; el registro va a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_logs/`.

### Task 7: columna ERA5 contra MITgcm o MOM6 (NCAR) y CROCO (COARE) (criterio 5)

**Files:**
- Create: `validation/bulk_column/` (`cases.py`, `run_pyocean.py`, `compare.py`, `README.md`, y los insumos de MITgcm/MOM6 y CROCO: `mitgcm/` o `mom6/`, `croco/`)
- Test: `tests/test_bulk_column.py` (lectura de resultados y definición del caso, como `tests/test_turbulence_columns.py`)

**Interfaces:**
- Consumes: el ERA5 de la Task 3; los modelos compilados (`~/pyOCEAN_refs/croco`, `~/pyOCEAN_refs/MITgcm` o MOM6).

- [ ] **Step 1: Caso.** Columna 5×5 periódica en un punto de la plataforma noreste de EE. UU. (el constructor elige el punto y lo justifica; por ejemplo cerca de la boya NDBC 44025, unos 36 m). Del 1 de julio al 15 de agosto de 2017 (≥ 30 días), KPP en los tres modelos, salinidad activa. El perfil inicial viene de una climatología (WOA) en el punto, documentada.
- [ ] **Step 2: Referencia NCAR.** MOM6 si su columna admite el bulk de FMS sin el modelo de hielo; si no, MITgcm con `pkg/exf` y KPP, y pyOCEAN con `"ncar_ly04"`. Justificar la elección en el README.
- [ ] **Step 3: Referencia COARE.** CROCO `SINGLE_COLUMN` con `BULK_FLUX` (opciones por defecto) y el ERA5 del punto en su formato `croco_blk.nc`; pyOCEAN con `"coare"`.
- [ ] **Step 4: Separar el acoplamiento de CROCO.** Una corrida extra de pyOCEAN con `"coare"`, `relative_wind=False` y `h_sfc=0` (capa superior), para separar esas elecciones de CROCO del resto.
- [ ] **Step 5: Write the tests** en `tests/test_bulk_column.py`: definición del caso (valores de `cases.py` iguales a los de los insumos de las referencias) y lectura de las series si existen (skip si no).
- [ ] **Step 6:** Correr las referencias y pyOCEAN, una a la vez, con salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column/`.
- [ ] **Step 7:** `compare.py`: SST, profundidad de la capa de mezcla y flujos medios por modelo. Criterio: flujos medios dentro del 2 % y SST dentro de 0.2 °C al final, o la diferencia explicada y clasificada (a)–(d).
- [ ] **Step 8: Commit** (código y README; no las salidas).

### Task 8: documentación y cierre (criterio 9; decisiones 4 y 11)

**Files:**
- Modify: `docs/manual/theory/` (nueva sección o archivo de flujos de superficie), `docs/manual/user-guide/configuration-reference.md`, `CLAUDE.md` (fila nueva de N2b y corrección de la fila N2a: TEOS-10 todavía no está en el 3D), `docs/design/vision_roadmap.md` (§7 marcador, §8 N2b y el orden de los cuatro proyectos), `docs/literature/reading_list.md` (LY04, Fairall et al. 2003, Edson et al. 2013 si se cita, López et al. 2020, Seijo-Ellis et al. 2024, Drenkard et al. 2025, Sasaki et al. 2026, Chang et al. 2026)
- Test: ninguno nuevo

- [ ] **Step 1:** Manual y guía de configuración: `BulkFluxes`, esquemas, convenciones, SST de 1 m, diferencias con MOM6 y CROCO clasificadas.
- [ ] **Step 2: Decisión 4 del spec.** Cerrar la viscosidad de respaldo para AB3 + WENO5 citando la corrida ya medida (Lorenz D σ + GLS a T_M2/498 con Smagorinsky 0.15: completa 150 ciclos, igual que el control; `CLAUDE.md`, fila Smagorinsky). Sin código.
- [ ] **Step 3:** `CLAUDE.md` (decisión 11 y fila N2b con las cifras de las Tasks 1–7), roadmap, `reading_list.md`.
- [ ] **Step 4:** Run la suite completa `pytest tests/ -q`. Expected: todo pasa, 2 xfailed.
- [ ] **Step 5: Commit.**
