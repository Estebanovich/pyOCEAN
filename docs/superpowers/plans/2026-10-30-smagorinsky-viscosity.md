# Viscosidad horizontal de Smagorinsky (laplaciana, forma de tensor): plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** dar a pyOCEAN una viscosidad horizontal laplaciana de Smagorinsky en forma de tensor de esfuerzos pesada por grosor (MOM6), en σ y ALE, apagada por defecto, y medir su efecto en DOME, Lorenz e Ilicak.

**Architecture:**
- **Módulo nuevo `core/dynamics/hor_visc.py`.** Tiene la geometría (Δ², espaciados en h y q), un oráculo NumPy y un kernel Numba del operador completo: tasas de deformación, ν en h y q con cota, esfuerzos y divergencia. Los dos son bit a bit iguales.
- **`Momentum.terms`** agrega el término `"hviscosity"` con este operador cuando `smagorinsky > 0`, en lugar del ∇²u constante. Si `smagorinsky == 0`, el camino de `ah` no cambia en nada.
- **`SigmaBaroclinic`** recibe `smagorinsky`. `SigmaState` tiene el contador `n_visc_limited`. Restart, registro y pre-flight lo conocen.
- **Los runners de benchmark** reciben `--smag C`.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-smagorinsky-viscosity-design.md` (aprobado por Esteban, 2026-10-30). Las definiciones, la cota reescalada para AB3 y los casos de benchmark del spec son vinculantes.

**Fuentes** (`archivo:línea` dentro):
- `docs/design/smag_refs/mom6_hor_visc.md` (fórmulas, `hq`, cotas, fronteras);
- `docs/design/smag_refs/croco_mitgcm_visc.md`;
- `docs/design/smag_refs/pyocean_current.md` (dónde está el `ah` actual, máscaras, periódicas, hilos);
- biblioteca local: Ilicak 2012 (eq. 11–15, Fig. 19), Adcroft 2019 (OM4 §2.2.5).

## Global Constraints

- **`smagorinsky=0` (por defecto) = bit a bit.** La matriz dorada de N1 (`tests/test_n1_golden.py`, `scripts/n1_make_golden.py --check` 31/31) y `scripts/n1_smoke.py` contra `main` quedan idénticas. El costo por paso de los casos del smoke sube ≤ 1 %.
- **Fórmulas exactas** (spec, "Definiciones"):
  - D_T = ∂u/∂x − ∂v/∂y en h; D_S = ∂v/∂x + ∂u/∂y en q;
  - |S|_h = √(D_T² + ¼ Σ₄ D_S²); |S|_q = √(D_S² + ¼ Σ₄ D_T²); la divergencia no entra;
  - Δ² = 2 dx² dy² / (dx² + dy²);
  - ν = max(ah, C Δ² |S|);
  - cota ν ≤ 0.8 · (6/11) / (4 Δt (1/dx² + 1/dy²)) · hrat_min;
  - σ_xx = ν h D_T y σ_xy = ν h_q D_S, con h_q la media de tipo armónico de MOM6 (0 si una cara se anula);
  - tendencia = divergencia del esfuerzo / (h_cara + h_neglect).
- **Fronteras:**
  - tierra y caras secas: D_S = 0 en sus esquinas (deslizamiento libre);
  - borde abierto: D_S = 0 en sus esquinas y tendencia 0 en sus caras;
  - periódicas: pad-crop con halo 2 (`core/model/periodic.py`).
- **Sin ajuste:** C = 0.15 en los benchmarks. La sensibilidad con C = 0.5 solo en DOME; C = 0.5 también en Ilicak Fig. 19(b), porque es el valor del artículo.
- **Regla de cierre del criterio 5 (Esteban):** si DOME híbrido + GLS sigue truncando con C = 0.15, se documenta la causa y queda abierto. No se cambia C ni se agrega biharmónico.
- **Estilo:** solo Python, type hints, docstrings con la física y su cita (Smagorinsky 1963; Griffies & Hallberg 2000; MOM6 `MOM_hor_visc.F90:línea`). Numba `parallel=True, cache=True` con `prange` por filas (k, j), resultado bit a bit para cualquier número de hilos, con oráculo NumPy en los tests.
- **Corridas:**
  - ≤ 5 hilos, solo P-cores, una a la vez;
  - exportar `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`;
  - salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`;
  - consultar a Esteban si una corrida pasa de ~24 h.
- **Commits:** con rutas explícitas (nunca `git add validation/`, `-A` ni `.`; nunca `logs/` ni `pyOCEAN-literature-review-guide.docx`). Terminan con:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```

## Review Focus

1. **Capas evanescentes en ALE** (h = 0 en una o más caras alrededor de una esquina, columnas híbridas con capas sin masa): sin NaN ni Inf, tendencia finita, h_q = 0 y la cota con hrat_min → 0 da ν = 0 ahí. Test en Task 1.
2. **Mojado y secado durante la corrida** (una cara pasa a seca entre pasos en σ con `d_min`): la cara seca no recibe tendencia y sus esquinas tienen D_S = 0. Test en Task 2.
3. **Malla estirada no uniforme** (dx variable como BayIW): Δ² y la cota se calculan por celda, y ν coincide con la fórmula de referencia del test. Test en Task 1.
4. **`ah > 0` con `smagorinsky > 0` donde el flujo está quieto:** ν = ah exacto (el máximo), y el operador de tensor con ν constante reproduce el decaimiento analítico. Test en Task 1.
5. **Número de hilos:** el resultado con 1 y con 5 hilos es bit a bit igual. Test en Task 1.

---

### Task 1: operador de Smagorinsky (geometría, oráculo NumPy, kernel Numba)

**Files:**
- Create: `core/dynamics/hor_visc.py`
- Test: `tests/test_hor_visc.py`

**Interfaces:**
- Produces:
  - `smag_geometry(hg: xr.Dataset) -> dict[str, np.ndarray]`: claves `dx_t, dy_t, dx_u, dy_u, dx_v, dy_v, area_t, area_u, area_v, dx_q, dy_q, area_q, delta2_t, delta2_q, inv_lap_t, inv_lap_q`. `inv_lap_* = 1/(1/dx² + 1/dy²)`. En q, dx_q es la media de los dx_v vecinos en i y dy_q la de los dy_u vecinos en j; en los bordes se usa el único vecino.
  - `smagorinsky_numpy(u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, C, ah, dt, h_neglect) -> tuple[fu, fv, nu_t, nu_q, n_limited]`.
  - `smagorinsky(...)`: misma firma y salida, kernel Numba.
  - Formas:
    - u (N, ny, nx+1), v (N, ny+1, nx);
    - h_t (N, ny, nx), h_u como u, h_v como v;
    - `mask_q` (ny+1, nx+1) bool, True donde la esquina tiene sus 4 caras mojadas;
    - `open_u` (ny, nx+1) y `open_v` (ny+1, nx) bool, caras de borde abierto;
    - nu_t (N, ny, nx), nu_q (N, ny+1, nx+1);
    - `n_limited` int (puntos h y q donde actuó la cota).

- [ ] **Step 1: escribir los tests que fallan** en `tests/test_hor_visc.py`:
  - `test_nu_matches_mom6_formula_on_stretched_grid`: malla 7×6 estirada (dx creciente 1.0–1.6 km, dy 0.8–1.2 km), u y v con un campo de corte aleatorio con semilla fija, h uniforme, ah = 0, Δt pequeño (cota inactiva). En el test se calculan D_T, D_S, |S| y Δ² con NumPy explícito según "Global Constraints", y se pide `nu_t`, `nu_q` = C Δ² |S| con `rtol=1e-13`.
  - `test_constant_nu_shear_wave_decays_analytically`: C = 0, ah = 100, caja 40 × 16 con paredes de deslizamiento libre, h uniforme, u = U cos(π y/L_y) en las caras u y v = 0. Este campo cumple D_S = 0 en las paredes en y. En las caras u que están a ≥ 3 celdas de las paredes en x se pide fu = −ah (π/L_y)² u, con error relativo ≤ 2 (π Δy/L_y)², es decir O(Δy²).
  - `test_energy_dissipation_nonnegative`: 20 campos aleatorios con semilla, h aleatorio en [0, 50] m con 30 % de ceros, C = 0.15, ah = 1. Se pide Σ (h_u u fu area_u) + Σ (h_v v fv area_v) ≤ 1e-12 · escala.
  - `test_vanished_layers_finite_and_zero_nu`: Review Focus 1. Una esquina con una cara h = 0: `nu_q` = 0 en esa esquina, todo finito.
  - `test_quiet_flow_gives_background_ah`: Review Focus 4. u = v = 0 con ah = 7: nu_t == 7 y nu_q == 7 exactos, fu = fv = 0.
  - `test_bound_rescaled_for_ab3_and_counted`: C = 1e6 y Δt = 600 s. Se pide nu_t ≤ 0.8·(6/11)/(4·dt)·inv_lap_t·hrat_min (+ 1e-12 relativo), y `n_limited` igual al número de puntos activos donde C Δ²|S| > cota.
  - `test_open_edge_faces_get_no_tendency`: `open_u[:, 0]` = True. Se pide fu[:, :, 0] == 0 y D_S = 0 en sus esquinas (equivalente: el resultado no cambia si se altera v en la fila de esquinas de ese borde).
  - `test_numba_equals_numpy_any_threads`: Review Focus 5. Kernel == oráculo con `np.array_equal` en las cinco salidas, con 1 y 5 hilos (`numba.set_num_threads`).
- [ ] **Step 2: correr los tests y verlos fallar.** Run: `pytest tests/test_hor_visc.py -q`. Expected: ImportError / FAIL.
- [ ] **Step 3: implementar `core/dynamics/hor_visc.py`.**
  - Seguir `MOM_hor_visc.F90` por las líneas de `smag_refs/mom6_hor_visc.md` §1, §3, §4 y §5: h_q armónico, hrat_min, divisor h_cara + h_neglect.
  - `h_neglect` = el `H_NEGLECT` de `core/vertical/remap.py`.
  - Docstring con las ecuaciones en LaTeX y las citas.
  - El kernel Numba fusiona lo que pueda; hacen falta al menos dos pasadas, porque ν_q necesita D_T de los vecinos.
- [ ] **Step 4: correr los tests.** Run: `pytest tests/test_hor_visc.py -q`. Expected: todos pasan.
- [ ] **Step 5: commit.** `git add core/dynamics/hor_visc.py tests/test_hor_visc.py`, mensaje "Smagorinsky operator: MOM6 stress-tensor form, AB3 bound, NumPy oracle + Numba kernel".

### Task 2: integración en el modelo (σ y ALE), estado, restart, pre-flight

**Files:**
- Modify:
  - `core/components/momentum.py` (`terms`, término `"hviscosity"`, ~:204-215);
  - `core/dynamics/sigma_baroclinic.py` (parámetro, `_PARAMS` ~:1853, `SigmaState` ~:141);
  - `core/restart.py` (:76 config, :185 y :268 contador);
  - `core/model/registry.py` (:46);
  - `core/model/preflight.py`.
- Test: `tests/test_smagorinsky_model.py`

**Interfaces:**
- Consumes: `smag_geometry`, `smagorinsky` (Task 1).
- Produces:
  - `SigmaBaroclinic(..., smagorinsky: float = 0.0)`, también asignable después de construido, como `ah`;
  - `SigmaState.n_visc_limited: int = 0`, que acumula `n_limited` de cada paso;
  - variable de registro `nu_smag`: diagnóstico (N, ny, nx) en h [m² s⁻¹], solo si se pide en la salida;
  - `Momentum.last_nu_t` con el ν_t del último paso, para los diagnósticos de Task 4.
- Grosores que se pasan al operador:
  - **σ:** h_t = D_t · ds[k], h_u = Du · ds[k], h_v = Dv · ds[k];
  - **ALE:** h_t = st.h; h_u y h_v de `mean_face_thickness(st.h)` (la media de las dos celdas recortada, la misma de la viscosidad vertical).
- Máscaras:
  - `wet_u` y `wet_v` del paso (mojado y secado);
  - `mask_q` = esquina con sus 4 caras mojadas, recalculada en cada paso;
  - `open_u` y `open_v` desde `self.open`.
- Periódicas: halo 2 con `per.apply` sobre (u, v, h_t, h_u, h_v, máscaras). La geometría se rellena una vez.

- [ ] **Step 1: escribir los tests que fallan** en `tests/test_smagorinsky_model.py`:
  - `test_zero_smagorinsky_is_bitwise`: con `smagorinsky=0` el hash de 10 pasos de `zstar_vi_arakawa` y de `sigma_gls` de `tests/n1_cases.py` es igual al dorado.
  - `test_doubly_periodic_conserves_momentum`: ALE z\*, periódico en x e y, u y v aleatorios, h uniforme, C = 0.15, 50 pasos sin Coriolis ni presión (ρ uniforme). Se pide que Σ h u area cambie ≤ 1e-12 relativo y que la energía cinética no crezca.
  - `test_sigma_and_ale_fixed_layers_agree`: columna de capas fijas, ALE z\* con η = 0 contra σ con el mismo campo; tendencias iguales ≤ 1e-12.
  - `test_drying_face_gets_no_viscous_tendency`: Review Focus 2. Caso de mojado y secado en σ con una cara que se seca en el paso n; su tendencia viscosa es 0 y no hay NaN.
  - `test_restart_bit_for_bit_with_smagorinsky`: 4 pasos continuos contra 2 pasos, restart y 2 pasos, en ALE híbrido con C = 0.15; igualdad exacta, incluido `n_visc_limited`.
  - `test_preflight_rejects_negative_smagorinsky`: `smagorinsky=-0.1` da un error de pre-flight que nombra el argumento.
  - `test_open_box_runs_and_edge_faces_zero`: caja ALE abierta al oeste (helper `_ale_open_box` de `tests/test_sigma_ale.py`); 20 pasos con C = 0.15, finito, y el término viscoso es 0 en las caras de borde (`keep_terms`).
- [ ] **Step 2: correr y ver fallar.** Run: `pytest tests/test_smagorinsky_model.py -q`. Expected: TypeError (argumento desconocido) o FAIL.
- [ ] **Step 3: implementar.**
  - Con `smagorinsky > 0`, `terms` agrega `("hviscosity", fu, fv, "+")` en la misma posición del orden.
  - Con `smagorinsky == 0`, el bloque `if self.ah > 0` queda tal cual.
  - El restart escribe y compara `smagorinsky` en la config y guarda `n_visc_limited` como atributo, igual que `n_limited`.
- [ ] **Step 4: correr los tests nuevos, la matriz dorada y la suite completa.** Run: `pytest tests/ -q -p no:cacheprovider`. Expected: todo pasa, y la matriz dorada sin cambios.
- [ ] **Step 5: commit** con las rutas de "Files". Mensaje: "Smagorinsky viscosity in σ and ALE (off by default): term, state counter, restart, pre-flight".

### Task 3: costo y smoke contra `main`

**Files:**
- Create: `scripts/smag_cost.py`, que mide s/paso con y sin Smagorinsky.
- Modify: `validation/bayiw_cut/README.md` (sección corta de costo).

- [ ] **Step 1: smoke.** Crear un worktree temporal de `main` y correr `scripts/n1_smoke.py` en `main` y en la rama, y luego `--compare`. Expected: 9/9 hashes idénticos y costo ≤ 1.01× (repetir una vez si un caso sale > 1.01×; si se repite, reportarlo como en N3).
- [ ] **Step 2: costo con el término activo.** `scripts/smag_cost.py` mide BayIW cut z\* y DOME 10 km híbrido + GLS, 50 pasos después de 10 de calentamiento, con 5 hilos, C = 0 y C = 0.15. Expected: se reporta la razón. El objetivo del spec es ≤ 1.10×; si se pasa, se perfila el kernel una vez y se documenta.
- [ ] **Step 3: commit** `scripts/smag_cost.py validation/bayiw_cut/README.md`.

### Task 4: opciones de benchmark (`--smag`) en DOME, Lorenz e Ilicak

**Files:**
- Modify:
  - `validation/dome/run_pyocean.py`, `validation/dome/cases.py` (caso `kv_low` para la híbrida);
  - `validation/lorenz_estuary/run_pyocean.py`, `validation/lorenz_estuary/case.py`;
  - `validation/ilicak_benchmark/run_pyocean.py` (caso `overflow_smag`: ν_h = 100, C = 0.5, z\*).
- Test: `tests/test_dome_benchmark.py`, `tests/test_lorenz_estuary.py`, `tests/test_ilicak_benchmark.py` (un test cada uno).

**Interfaces:**
- Produces:
  - `run_pyocean.py --smag C`: float, por defecto 0. Llega a `SigmaBaroclinic(smagorinsky=C)`.
  - La etiqueta de la corrida lleva `_smag{C}`, por ejemplo `pyocean_hybrid_gls_smag0.15`.
  - `series.npz` y `summary.json` guardan `n_visc_limited` y max ν.

- [ ] **Step 1: tests que fallan.**
  - `test_dome_smag_option_reaches_model`: construir el modelo DOME con `--smag 0.15` (función de construcción del runner, sin correr); `m.smagorinsky == 0.15`.
  - Lo mismo para Lorenz (`test_lorenz_smag_option_reaches_model`).
  - Para Ilicak, `test_ilicak_overflow_smag_case`: el caso existe con ν_h = 100, C = 0.5 y z\*.
- [ ] **Step 2: verlos fallar.** Run: `pytest tests/test_dome_benchmark.py tests/test_lorenz_estuary.py tests/test_ilicak_benchmark.py -q -k smag`. Expected: FAIL.
- [ ] **Step 3: implementar las opciones**, sin cambiar las corridas existentes cuando `--smag` no se da.
- [ ] **Step 4: correr los tres archivos de test.** Expected: pasan.
- [ ] **Step 5: commit** con las rutas de "Files".

### Task 5: corridas de benchmark (una a la vez)

Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/{dome,lorenz_estuary,ilicak}/smag/`. Antes de cada corrida, verificar que no haya otra corriendo.

- [ ] **Step 1: DOME 10 km híbrido + GLS, C = 0.15**, 50 días (~20 min). Se reporta: n_trunc (criterio 5), n_visc_limited, max|u|, b_f, trayectoria Y(x), transporte y centroide, contra `pyocean_hybrid_gls` y MITgcm z (Legg). Cada diferencia se clasifica (a)–(d).
- [ ] **Step 2: DOME, sensibilidad C = 0.5**, misma configuración. Se reporta como sensibilidad.
- [ ] **Step 3: DOME `kv_low` híbrido**, ν_v = 1e-4, `ah` = 50, C = 0.15, 50 días (criterio 6). Se reporta si completa, y si no, el día, el lugar y la causa (post-mortem).
- [ ] **Step 4: Lorenz D σ + GLS, Δt = T_M2/498, C = 0.15**, 150 ciclos (criterio 7), contra la corrida de T_M2/746 en los mismos ciclos: S_max, |u|max, Me y la relación exacta (≤ 5e-10).
- [ ] **Step 5: Ilicak overflow Fig. 19(b)**, z\*, ν_h = 100 + C = 0.5, contra la corrida ν_h = 100 existente. Se reportan RPE, mezcla espuria y extremos nuevos; se espera menos mezcla (criterio 8).
- [ ] **Step 6: actualizar `compare.py` y las figuras** de DOME e Ilicak para que lean las corridas `_smag`. Commit solo de código y README, nunca de salidas.

### Task 6: documentación y literatura

**Files:**
- Modify:
  - `validation/dome/README.md`, `validation/lorenz_estuary/README.md`, `validation/ilicak_benchmark/README.md` (resultados de Task 5 con clases y "qué debe mejorar pyOCEAN");
  - `docs/manual/theory/momentum.md` (sección "Smagorinsky horizontal viscosity" con las ecuaciones, la cota AB3 y las fronteras);
  - `docs/manual/user-guide/configuration-reference.md` (`smagorinsky`);
  - `CLAUDE.md` (fila nueva y conteo de tests);
  - `docs/design/vision_roadmap.md` §7 y §8;
  - `docs/literature/reading_list.md`: Smagorinsky 1963 y Griffies & Hallberg 2000, con DOI marcado **[verificar]** si viene de memoria, ★ porque se usan en el código, ⬜.

- [ ] **Step 1:** escribir los cambios. Cada número viene de una corrida de Task 5 o de un test.
- [ ] **Step 2:** correr la suite completa. Expected: todo pasa; el conteo de CLAUDE.md coincide.
- [ ] **Step 3:** commit con las rutas de "Files".
