# Mezcla por cizalla de Jackson et al. (2008) — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** agregar a pyOCEAN la mezcla vertical por cizalla de Jackson, Hallberg & Legg (2008), port de MOM6 `kappa_shear`, sumada a un cierre base. Validarla contra MOM6 compilado en columna y contra la verdad no hidrostática de Legg et al. (2006) en DOME.

**Architecture:** un kernel de columna (`physics/mixing/kappa_shear.py`: oráculo NumPy y Numba por columnas con fusión de capas delgadas) y una envoltura de cierre aditiva (`ShearAddedClosure` en `core/components/closure.py`). La envoltura suma κ_shear y ν_shear = Pr·κ_shear al cierre base (constante, 'none' o KPP) en la fase de trazadores, como KPP. Las métricas de Legg se agregan a `validation/dome/metrics.py` con sus valores no hidrostáticos digitalizados.

**Tech Stack:** Python, NumPy 1.26.4, Numba; Fortran de MOM6 compilado solo para la prueba de referencia (`validation/mom6_column/`).

**Spec:** `docs/superpowers/specs/2026-10-30-jackson-shear-mixing-design.md` (aprobado por Esteban el 2026-10-30). Lecturas de referencia: `docs/design/shear_refs/`.

## Global Constraints

- Solo Python (NumPy, Numba); sin dependencias nuevas; nunca `pip install`. numpy está fijado en 1.26.4.
- Toda corrida exporta `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, usa ≤ 5 hilos en los núcleos P, una corrida a la vez, y escribe en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`. Si una corrida pasa de ~24 h, se consulta a Esteban.
- Con `shear_mixing=None` (valor por defecto) el camino actual no cambia ni en el orden de las operaciones: 33 goldens (`tests/test_n1_golden.py`) y corridas de humo (`scripts/n1_smoke.py`) idénticos bit a bit.
- Cada ecuación cita Jackson et al. (2008) y `MOM_kappa_shear.F90:línea` en el docstring, con nombres de variables del artículo y unidades.
- Parámetros = valores por defecto de MOM6, conjunto 1 de JHL (`MOM_kappa_shear.F90:2106-2112`); nada se ajusta a un caso. Pr = 0 es sensibilidad.
- Rama corregida de Newton (`KAPPA_SHEAR_ITER_BUG` = False: dκ/dQ con `dz_Int`, `:1860-1866`).
- Índices (k, j, i) con k = 0 en el fondo; interfaces 0 = fondo, N = superficie. MOM6 va de arriba abajo: se invierte como en `kpp_column`.
- Commits con rutas explícitas; nunca `git add validation/`, `-A` ni `.`; nunca `logs/` ni `pyOCEAN-literature-review-guide.docx`, ni salidas de corridas. Trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```
- Anti-metas: GLS + cizalla se rechaza; κ0 solo suaviza la entrada y no se devuelve en κ; ningún benchmark cambia con el esquema apagado.

## Review Focus

1. **Columna con N² < 0** (inestable, p. ej. agua densa sobre ligera en el borde de la pluma): N² se recorta a 0, κ finito y acotado, sin NaN. Test en Task 1.
2. **Sin cizalla y estratificado:** κ_shear = ν_shear = 0, y el modelo con `shear_mixing='jackson'` da lo mismo que el cierre base solo, bit a bit. Test en Task 4.
3. **Columnas casi secas o con una sola capa con masa** (σ con `d_min`, híbrida con capas vacías): κ = ν = 0 y sin NaN; la columna seca queda quieta bit a bit. Tests en Tasks 3 y 4.
4. **Δt grande** (3600 s, como el `DT_THERM` de MOM6): la iteración termina dentro de sus topes (13 subpasos, 50 iteraciones) con κ finito. Test en Task 1.
5. **Fronteras periódicas y abiertas:** ν_shear llega a las caras con `centre_to_faces` respetando `per`, y las caras de borde abierto no reciben valores de fuera del dominio. Test en Task 4.

---

### Task 1: oráculo de columna en NumPy

**Files:**
- Create: `physics/mixing/kappa_shear.py`
- Test: `tests/test_kappa_shear.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class JacksonParams` con, en este orden y con el nombre de MOM6 en el comentario: `ri_crit=0.25` (RINO_CRIT), `shearmix_rate=0.089` (SHEARMIX_RATE), `fri_curvature=-0.97` (FRI_CURVATURE), `kappa_buoy_scale=0.82` (KAPPA_BUOY_SCALE_COEF), `tke_n_decay=0.24` (TKE_N_DECAY_CONST), `tke_shear_decay=0.14` (TKE_SHEAR_DECAY_CONST), `kd_seed=1.0` (KD_SEED_KAPPA_SHEAR), `kd_trunc_frac=0.01` (KD_TRUNC_KAPPA_SHEAR / KD_KAPPA_SHEAR_0), `tol_err=0.1` (KAPPA_SHEAR_TOL_ERR), `max_rino_it=50` (MAX_RINO_IT), `max_ks_it=13` (MAX_KAPPA_SHEAR_IT), `max_kap_src_chg=10.0` (KAPPA_SHEAR_MAX_KAP_SRC_CHG), `tke_background=0.0` (TKE_BACKGROUND), `lz_rescale=1.0` (LZ_RESCALE).
  - `kappa_shear_column(u, v, b, dz, f2: float, dt: float, kappa0: float, p: JacksonParams) -> tuple[np.ndarray, np.ndarray]`: u, v [m s⁻¹], b = −g(ρ−ρ₀)/ρ₀ [m s⁻²] y dz [m] de forma (N,), k = 0 en el fondo, sin capas delgadas (la fusión es de Task 3). Devuelve (κ promediado en Δt, Q promediado en Δt), ambos (N+1,) en interfaces [m² s⁻¹, m² s⁻²], con κ = 0 en el fondo y la superficie.

Port de `kappa_shear_column` (`MOM_kappa_shear.F90:866-1374`) y `find_kappa_tke` (`:1701-2041`): suavizado previo con κ0, celdas de interfaz h_Int, subpasos con búsqueda del paso y predictor–corrector, Picard → Newton, truncamiento en `kd_trunc_frac·kappa0`. **Decisión de Claude** (se documenta en el docstring): la flotabilidad entra como un solo trazador b, en lugar de T y S con ∂b/∂T y ∂b/∂S por interfaz como MOM6. Es exacto para la EOS lineal en T (DOME, Ilicak, la prueba de Task 2); con TEOS-10 es la linealización local.

- [ ] **Step 1: Write the failing tests** in `tests/test_kappa_shear.py`:
  - `test_supercritical_column_gives_zero`: N² ≥ Ri_c S² en todas las interfaces → `np.all(kappa == 0)`.
  - `test_shear_layer_mixes_inside_only`: capa de cizalla tanh con Ri_min = 0.1 en el centro → κ > 0 en el centro, κ = 0 donde Ri > 0.25 lejos de la capa, `kappa[0] == kappa[-1] == 0`.
  - `test_unstable_column_is_finite` (Review Focus 1): b decrece hacia arriba en dos interfaces → `np.isfinite(kappa).all()` y `kappa.max() < 10.0`.
  - `test_large_dt_terminates` (Review Focus 4): la capa de cizalla con dt = 3600 s → termina, κ finito, y el κ promediado no es mayor que el de dt = 600 s por más de 2× (efecto Hallberg 2000: la mezcla erosiona la cizalla dentro del paso).
  - `test_defaults_are_jhl_set_1`: `JacksonParams()` igual al conjunto 1 de JHL citado arriba.
- [ ] **Step 2:** Run `pytest tests/test_kappa_shear.py -q`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implementar `JacksonParams` y `kappa_shear_column` en `physics/mixing/kappa_shear.py`.
- [ ] **Step 4:** Run `pytest tests/test_kappa_shear.py -q`. Expected: 5 passed.
- [ ] **Step 5: Commit** `physics/mixing/kappa_shear.py tests/test_kappa_shear.py`.

### Task 2: columna contra MOM6 compilado (criterio 1)

**Files:**
- Create: `validation/mom6_column/kappa_shear_driver.F90`
- Modify: `validation/mom6_column/build_reference.sh` (extraer `kappa_shear_column` y lo que necesita, privados en `MOM_kappa_shear.F90`, a un módulo, como se hace con hybgen; compilar el driver), `validation/mom6_column/columns.py` (`kappa_shear_cases() -> list[dict]`, `write_kappa_shear_cases(path, cases)`, `read_kappa_shear_results(path) -> list[dict]`), `validation/mom6_column/README.md`
- Test: `tests/test_vertical_vs_mom6.py` (`test_kappa_shear_matches_mom6`, con `@needs("kappa_shear_driver")`)

**Interfaces:**
- Consumes: `kappa_shear_column`, `JacksonParams` (Task 1).

Casos (todos sin capas delgadas, EOS lineal en T):
- capa de cizalla tanh (Ri_min 0.1);
- chorro de fondo tipo overflow (u máximo a 100 m del fondo, interfaz densa encima);
- columna subcrítica en todo (κ ≡ 0);
- la capa de cizalla con dt = 3600 s.

Tolerancia: la fija el implementador a partir de `KAPPA_SHEAR_TOL_ERR` y de la diferencia observada, y la escribe en el test y en el README de `mom6_column` con la razón.

- [ ] **Step 1:** Escribir los casos y el test (con skip si el binario no existe). Run `pytest tests/test_vertical_vs_mom6.py -q -k kappa_shear`. Expected: SKIPPED.
- [ ] **Step 2:** Extender `build_reference.sh` y escribir el driver; correr `bash validation/mom6_column/build_reference.sh`. Expected: `kappa_shear_driver` en la lista "built:".
- [ ] **Step 3:** Run `pytest tests/test_vertical_vs_mom6.py -q`. Expected: todos pasan (remap, regrid y kappa_shear). Si kappa_shear no coincide, la causa se busca en el oráculo de Task 1, que se corrige ahí. Cada diferencia que quede se clasifica en el README.
- [ ] **Step 4: Commit** las rutas de "Files" (y `physics/mixing/kappa_shear.py` si se corrigió).

### Task 3: kernel Numba por columnas con fusión de capas delgadas

**Files:**
- Modify: `physics/mixing/kappa_shear.py`
- Test: `tests/test_kappa_shear.py`

**Interfaces:**
- Consumes: `kappa_shear_column` (oráculo), `thin_layer_threshold`, `active_layers` (`core/components/closure.py:52-118`; patrón de fusión de `KPPClosure._merged`, `:755-…`).
- Produces:
  - `kappa_shear_columns(u, v, b, dz, f2, dt: float, kappa0: float, h_thin, p: JacksonParams) -> tuple[np.ndarray, np.ndarray]`: u, v, b, dz de forma (N, M); f2 y h_thin de forma (M,). Devuelve (κ, Q) de forma (N+1, M). Numba `prange` sobre columnas. Las columnas secas (Σdz ≤ `H_MASSLESS`) dan 0.
  - `kappa_shear_columns_numpy(...)`: misma firma; recorre las columnas con fusión + `kappa_shear_column`. Es el oráculo.

Fusión como MOM6 (`MOM_kappa_shear.F90:254-303`): las capas delgadas se unen a la de arriba (o a la primera con masa); se suman h, u·h, v·h, b·h; se vuelve a las interfaces por interpolación lineal en la fracción de grosor. El umbral es el relativo de pyOCEAN, `thin_layer_threshold` (decisión del spec).

- [ ] **Step 1: Write the failing tests:**
  - `test_numba_matches_numpy_oracle`: 40 columnas variadas (incluye capas de 1 mm–2 cm y columnas secas) → `np.array_equal` entre Numba y NumPy.
  - `test_threads_bit_for_bit`: 1 y 5 hilos (`numba.set_num_threads`) → `np.array_equal`.
  - `test_thin_layers_do_not_change_kappa` (criterio 2): columna con y sin capas de 1 mm–2 cm intercaladas → κ en las interfaces comunes iguales a `rtol=1e-6`.
  - `test_dry_and_single_layer_columns` (Review Focus 3): columna seca y columna con una sola capa con masa → κ = Q = 0, finitos.
- [ ] **Step 2:** Run `pytest tests/test_kappa_shear.py -q`. Expected: los 4 nuevos FAIL.
- [ ] **Step 3:** Implementar `kappa_shear_columns_numpy` y `kappa_shear_columns`.
- [ ] **Step 4:** Run `pytest tests/test_kappa_shear.py tests/test_vertical_vs_mom6.py -q`. Expected: todo pasa.
- [ ] **Step 5: Commit.**

### Task 4: el cierre aditivo en el modelo

**Files:**
- Modify:
  - `core/components/closure.py` (`ShearAddedClosure`, `make_closure`);
  - `core/dynamics/sigma_baroclinic.py` (parámetros, `_PARAMS`, docstring);
  - `core/model/preflight.py`;
  - `core/restart.py`;
  - `core/model/registry.py`;
  - `core/model/driver.py`, solo si la suma lo exige.
- Test: `tests/test_shear_model.py`

**Interfaces:**
- Consumes: `kappa_shear_columns` (Task 3); `Closure`, `centre_to_faces`, `make_closure` (existentes).
- Produces:
  - `SigmaBaroclinic(..., shear_mixing: str | None = None, shear_prandtl: float = 1.0, jackson_params: JacksonParams | None = None)`. Acepta `shear_mixing` ∈ {None, 'jackson'} en σ y ALE.
  - `class ShearAddedClosure(Closure)`, que envuelve al cierre base.
    - `update_phase = "tracers"`, `stateful = True`.
    - `update` llama al del base y calcula κ_shear (N+1, M) y ν_shear = Pr·κ_shear.
    - `tracer_diffusivity()` = κ_base + κ_shear, donde κ_base es el del base, o el `kv` constante si el base devuelve None.
    - `viscosity(st)` = ν_base + `centre_to_faces(ν_shear)`, donde ν_base es el del base, o `nu_const` constante en las caras si el base devuelve None ('none': 0).
    - `state()`/`load()` agregan la clave `'shear_kv'` al dict de `SigmaState.closure`.
    - `state_fields()`/`load_state()` agregan `shear_kv` al restart, conservando los del base.
    - `remap` delega al base; ν_shear se conserva por índice de interfaz.
  - Registro: `kd_shear` y `kv_shear` (interfaces, m² s⁻¹), con unidades y descripción.
- Valores fijados:
  - κ0 = max(kv, 1e-7) m² s⁻¹;
  - f² en los centros = media de los 4 f² de las esquinas (`MOM_kappa_shear.F90:317-318`);
  - b = −g(ρ − ρ₀)/ρ₀ de la densidad de capa que ya recibe `update`;
  - el Δt de Jackson es el del modelo.
- Pre-flight, con todos los errores en un solo reporte:
  - `shear_mixing` con `closure='gls'` o `'parabolic'` → error;
  - valor desconocido → error;
  - `shear_prandtl` < 0 o no finito → error;
  - `jackson_params` con Ri_c ≤ 0 → error.

- [ ] **Step 1: Write the failing tests** in `tests/test_shear_model.py`:
  - `test_off_is_bit_for_bit`: modelo σ y ALE híbrido de 10 pasos con `shear_mixing=None` contra el mismo sin el parámetro → `np.array_equal` en η, u, v, T, h.
  - `test_zero_shear_equals_base` (Review Focus 2): columna estratificada en reposo con `'jackson'` contra el base → `np.array_equal`.
  - `test_shear_mixes_overflow_interface`: corriente de fondo estratificada con Ri < 0.25 en la interfaz, 20 pasos → κ_shear > 0 en la interfaz y la varianza de T baja más que con el base solo.
  - `test_prandtl_zero_gives_no_viscosity`: Pr = 0 → `viscosity` igual a la del base, bit a bit.
  - `test_kpp_plus_shear_adds`: base KPP → κ = κ_KPP + κ_shear a 1e-15.
  - `test_preflight_refuses_gls_and_bad_values`: los cuatro errores en un solo reporte.
  - `test_restart_bit_for_bit_with_shear`: 6 pasos = 3 + restart + 3, en ALE híbrido.
  - `test_dry_column_stays_still` (Review Focus 3): columna seca bit a bit quieta, como `tests/test_dry_column_closure.py`.
  - `test_periodic_and_open_faces` (Review Focus 5): canal periódico en x con frontera abierta en y → ν en las caras de envoltura iguales bit a bit, y sin valores de fuera del dominio en las caras abiertas.
- [ ] **Step 2:** Run `pytest tests/test_shear_model.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Run `pytest tests/test_shear_model.py tests/test_n1_golden.py tests/test_restart.py tests/test_preflight.py tests/test_closure_component.py -q`. Expected: todo pasa; goldens 33/33.
- [ ] **Step 5:** Run la suite completa `pytest tests/ -q`. Expected: 1797 + los nuevos passed, 2 xfailed.
- [ ] **Step 6: Commit.**

### Task 5: humo y costo

**Files:**
- Create: `scripts/shear_cost.py` (mediana de s/paso de 50 pasos tras 10 de calentamiento, 5 hilos, con y sin `'jackson'`, en BayIW cut z\* y DOME 10 km híbrido `kv_low` + Smagorinsky 0.15). El script **verifica** que el término esté activo (κ_shear máximo > 0) e imprime cuánto del paso se va en `update` del cierre.
- Modify: `validation/dome/README.md` (sección de costo).

- [ ] **Step 1:** `python scripts/n1_smoke.py` contra `main` en un worktree temporal del scratchpad (como en la tarea 3 de Smagorinsky). Expected: 9/9 hashes idénticos.
- [ ] **Step 2:** `python -I scripts/shear_cost.py`. Expected: imprime las razones. Si alguna pasa de 1.30, se perfila y se documenta qué parte cuesta (spec, criterio 7).
- [ ] **Step 3: Commit** `scripts/shear_cost.py validation/dome/README.md`.

### Task 6: métricas de Legg y opciones de los runners

**Files:**
- Create: `validation/dome/legg2006_nonhydro.json`
- Modify:
  - `validation/dome/metrics.py`;
  - `validation/dome/compare.py`;
  - `validation/dome/figures.py`;
  - `validation/dome/run_pyocean.py` (`--shear jackson`, `--shear-pr P`);
  - `validation/ilicak_benchmark/run_pyocean.py` (`--shear jackson`);
  - `validation/ilicak_benchmark/compare.py`.
- Test: `tests/test_dome_benchmark.py`, `tests/test_ilicak_benchmark.py`

**Interfaces:**
- Produces, en `metrics.py`, sobre las series diarias existentes (h, T, s, u, v 3-D):
  - `corner_entrainment(s, t0, t1, thr=0.01) -> float`: Legg ec. 14 en −50 < x′ < 50 km, −50 < y′ < 0 km;
  - `tracer_by_class(s, t0, t1, n_classes) -> (x, b_edges, S)`: Legg Fig. 4, Σ s por clase de flotabilidad contra x′;
  - α_E(x) y transporte ya existen (`entrainment`, `plume_transport`).
- Etiquetas de corrida:
  - DOME: `--shear jackson` agrega `_jackson` a la etiqueta, y `_jacksonpr0` con `--shear-pr 0`. Por ejemplo `kv_low_smag0.15_jackson`.
  - Ilicak: agrega `jackson`.
- `legg2006_nonhydro.json`: cada valor lleva `source` (figura y página del PDF), `method` ('text' o 'digitized') y, si está digitalizado, la incertidumbre de lectura. Contenido:
  - E de la esquina 8.6e-4 (texto, l. 727 de la copia `.md`);
  - E de 10 km z e isopicnal (Fig. 14);
  - transporte T(x) y α_E(x) (Fig. 3a, 3b);
  - trazador por clase (Fig. 4);
  - b_f (Fig. 15).

  Las figuras se leen renderizando las páginas del PDF de la biblioteca local con `pdftoppm` (fuera del repo).

- [ ] **Step 1: Write the failing tests:**
  - `test_corner_entrainment_synthetic`: pluma sintética con transporte que crece linealmente en la esquina → E igual al valor analítico a 1e-6.
  - `test_tracer_by_class_conserves`: Σ de `tracer_by_class` = Σ s·V en la ventana.
  - `test_jackson_tag`: `--shear jackson --shear-pr 0` → etiqueta `..._jacksonpr0`.
  - `test_legg_json_has_sources`: cada valor del JSON tiene `source` y `method`.
- [ ] **Step 2:** Run `pytest tests/test_dome_benchmark.py tests/test_ilicak_benchmark.py -q`. Expected: los nuevos FAIL.
- [ ] **Step 3:** Implementar y digitalizar. Agregar a `compare.py` las filas de E de la esquina y la tabla contra la verdad.
- [ ] **Step 4:** Run `python validation/dome/compare.py 10 --only hybrid` sobre las corridas existentes (`mitgcm_z_legg`, `pyocean_hybrid`, `pyocean_hybrid_gls`, `pyocean_hybrid_kv_low_smag0.15`). Expected: imprime E de la esquina para cada una sin error. MITgcm legg da un E del orden de la referencia de 10 km z de Legg.
- [ ] **Step 5:** Run los dos archivos de tests. Expected: todo pasa.
- [ ] **Step 6: Commit** las rutas de "Files".

### Task 7: corridas (una a la vez; el controlador las lanza en cadena)

Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/{dome/10km,ilicak_benchmark}/`; bitácoras en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/shear_logs/`. Antes de lanzar, verificar que no haya otra corrida en curso.

| # | Corrida | Comando |
|---|---|---|
| 1 | DOME híbrido `kv_low` + Smag 0.15 + Jackson Pr 1 (principal) | `python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson` |
| 2 | la misma con Pr 0 (sensibilidad) | `... --shear jackson --shear-pr 0` |
| 3 | DOME híbrido ν_v 0.5 + Jackson Pr 1 (secundaria, línea base `pyocean_hybrid`) | `python validation/dome/run_pyocean.py 10 hybrid --shear jackson` |
| 4 | DOME z\* `kv_low` + Smag 0.15 (línea base de 5) | `python validation/dome/run_pyocean.py 10 zstar --kv-low --smag 0.15` |
| 5 | DOME z\* `kv_low` + Smag 0.15 + Jackson Pr 1 | `... 10 zstar --kv-low --smag 0.15 --shear jackson` |
| 6 | DOME σ `kv_low` + Smag 0.15 (línea base de 7) | `python validation/dome/run_pyocean.py 10 sigma --kv-low --smag 0.15` |
| 7 | DOME σ `kv_low` + Smag 0.15 + Jackson Pr 1 | `... 10 sigma --kv-low --smag 0.15 --shear jackson` |
| 8 | Ilicak overflow z\* ν_h 100 + Jackson (sensibilidad) | `python validation/ilicak_benchmark/run_pyocean.py overflow zstar 100 --shear jackson` |

**Decisión de Claude:** las líneas base 4 y 6 no existen y hacen falta para clasificar el efecto de Jackson en z\* y σ. Si `--kv-low` del runner de DOME solo admite la híbrida, Task 6 lo extiende.

- [ ] **Step 1:** El controlador escribe la cadena en `shear_logs/chain.sh` y la lanza en segundo plano.
- [ ] **Step 2:** Revisar rc y bitácoras; si una corrida se detiene, guardar el post-mortem. No se re-ajusta nada.

### Task 8: análisis y documentación

**Files:**
- Modify:
  - `validation/dome/README.md` (sección Jackson: criterios 4 y 5 con números, cada diferencia de (a) a (d), "qué debe mejorar pyOCEAN");
  - `validation/ilicak_benchmark/README.md` (criterio 6);
  - `validation/mom6_column/README.md` (criterio 1, si falta);
  - `docs/manual/theory/turbulence-friction.md` (sección "Jackson shear mixing": ecuaciones, iteración, Δt, capas delgadas, Pr, suma al cierre base);
  - `docs/manual/user-guide/configuration-reference.md` (`shear_mixing`, `shear_prandtl`, `jackson_params`);
  - `CLAUDE.md` (fila nueva y conteo de tests);
  - `docs/design/vision_roadmap.md` §7 y §8;
  - `docs/literature/reading_list.md`: los papers del spec, con DOI, nombre de archivo y ★/⬜. Los 8 descargados son ✅; Hallberg 2000 va con [verificar].

- [ ] **Step 1:** Escribir. Cada número sale de una corrida de Task 7, de un test o de `legg2006_nonhydro.json`. La verdad digitalizada se marca como tal.
- [ ] **Step 2:** Run `pytest tests/ -q`. Expected: todo pasa; el conteo de CLAUDE.md coincide.
- [ ] **Step 3: Commit** las rutas de "Files".
