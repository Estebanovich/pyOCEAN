# N1: contrato de componentes y driver con orden fijo — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** sacar de `SigmaBaroclinic` un contrato de componentes y un driver de 10 fases con orden fijo, sin cambiar ni un bit de la física.

**Architecture:** el plan tiene tres capas.
- **Componentes:** especificación (dataclass) → `check()` → `build()`, en `core/components/`.
- **Núcleos:** funciones puras, que en su mayoría ya existen.
- **Driver:** `core/model/driver.py` llama a los componentes en el orden actual del paso. `SigmaBaroclinic` queda como fachada con la misma firma y conserva todos los nombres privados como alias.

La red de seguridad es una matriz dorada de hashes del estado, generada desde `main` antes de mover código.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, SciPy (SuperLU, CG), netCDF4, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-n1-component-core-design.md`. El mapa del código actual, con `file:line`, está en `docs/design/n1_refs/pyocean.md`; abreviaturas: SB = `core/dynamics/sigma_baroclinic.py`, SWD = `core/dynamics/sigma_wd.py`, WD = `core/dynamics/wetting_drying.py`.

## Global Constraints

- **Bits.** Ningún cambio de física ni de tolerancias. Cada tarea de extracción termina con `pytest tests/test_n1_golden.py` en verde: hash SHA-256 igual en cada campo, en cada paso registrado.
- **Expresiones movidas tal cual.** Las expresiones aritméticas se mueven carácter por carácter:
  - el orden de suma de `_tendencies` (`SB:1579-1614`);
  - `np.tensordot` en `_baroclinic_part` y `(ds*u).sum(0)` en los transportes (no se unifican);
  - la expresión θ/θ_bc (`SB:2281`);
  - los núcleos `_rk_stage`/`_ab3_kernel`.
- **Numba.** Las condiciones del despacho NumPy/Numba no cambian:
  - `_diffuse_h` usa Numba solo en ALE;
  - `ReferenceProfile` usa Numba a partir de 1000 puntos;
  - `_rhs` usa Numba solo con WENO5.

  Tampoco cambian las banderas `fastmath`/`parallel`. Todo módulo nuevo usa la lista `_THREADS` de SB (`from core.dynamics import sigma_baroclinic as sb; sb._THREADS`), no una copia.
- **Referencias.** Nada copia `h` ni `T` entre fases: las cachés `_tt_cache`/`_zc_cache` dependen de la identidad del arreglo. Nada escribe `h` in situ.
- **Compatibilidad.** Todos los nombres privados y de módulo listados en `pyocean.md` §4 siguen existiendo en `sigma_baroclinic.py`, como métodos delegados o reexportaciones. Los tests existentes no cambian, salvo el monkeypatch de `sb.open_face_transport` (`tests/test_sigma_ale.py:720`), que pasa a apuntar al módulo nuevo (Task 6).
- **Hilos.** Como mucho 5 hilos, solo P-cores; las corridas de humo, una a la vez.
- **Estilo.** PEP 8, type hints y docstrings con la física y su cita (CLAUDE.md). Índices (k, j, i), con k = 0 en el fondo.
- **Commits.** Rutas explícitas en cada commit; nunca `git add validation/` ni `git add -A`. Todos terminan con:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton
  ```
- **Rama.** `n1-component-core`; la especificación ya está en `27dd228`.

## Refinamiento de la especificación §3 (decidido al escribir el plan)

La especificación pide que todo lo que cruza pasos viva en `SigmaState`. En N1 esto se aplica así:

| Dato | Dónde | Por qué |
|---|---|---|
| GLS (`tke`, `eps`, `akv`) | `SigmaState.closure` (dict de arreglos) | cambia cada paso; el componente lo sincroniza por referencia con `core.gls` antes y después de `core.step` |
| `Qx`, `Qy`, `Fz`, `omega`, `g_hist` | `SigmaState` (ya están) | — |
| geometría `fs.last_*` | registro `BarotropicFluxes` que devuelve la fase 4 | se escribe y se lee dentro del mismo paso |
| `_ext`, `_T_in`, `_tr_in` | argumentos entre fases | se escriben y se leen dentro del mismo paso |
| LU desfasado | se queda en `WetDryFreeSurface` y se declara en `state_fields()` para el restart | moverlo obligaría a refactorizar en cada paso (costo y bits); se mueve en N2, al partir `SigmaWetDry.step` |
| `T_ext`, `h_ext`, `tr_ext`, `rho_ref` | configuración del componente, fijada en `initial_state` | no cambian durante la corrida; los tests los asignan después de `initial_state` (`test_sigma_ale.py:127-141, 731-733, 865`) |

## Review Focus

1. **Restart a mitad de la historia de AB3.** Escribir en el paso 2 (AB2) y leer debe seguir con AB3 en el paso 3, igual que una corrida continua. Lo cubre Task 13, `test_restart_mid_ab_startup`.
2. **Modelo armado y reusado para dos corridas.** Un `initial_state` nuevo sobre el mismo modelo debe dar los mismos bits que un modelo recién creado; el LU desfasado no debe filtrarse de una corrida a otra salvo como hoy. Lo cubre Task 11, `test_driver_reuse_model_two_runs`.
3. **Fase 4 sin Flather** (bordes cerrados): `BarotropicFluxes.open_*` debe ser `None` y las fases 5 y 6 no deben tocarlo. Lo cubre Task 7, `test_free_surface_closed_has_no_open_fluxes`.
4. **σ con `open_boundary_state` y sin él.** El σ evalúa el exterior solo con proveedor; ALE lo evalúa siempre. Lo cubre la matriz dorada, con los casos `sigma_obc_state` y `sigma_open_noprovider`.
5. **Argumentos ignorados que hoy pasan en silencio.** Deben emitir `PyOceanConfigWarning` sin cambiar los bits. Lo cubre Task 12, `test_ignored_args_warn_and_keep_bits`.

---

### Task 0: matriz dorada y restart de referencia desde `main`

**Files:**
- Create: `tests/n1_cases.py` (constructores de casos pequeños)
- Create: `scripts/n1_make_golden.py` (genera hashes y estados)
- Create: `tests/data/n1_golden/hashes.json`, `tests/data/n1_golden/<case>.npz` (estado en el paso 10)
- Create: `tests/data/n1_restart_main.nc`
- Create: `tests/test_n1_golden.py`

**Interfaces:**
- Produces:
  - `tests.n1_cases.CASES: dict[str, Callable[[], tuple[SigmaBaroclinic, SigmaState, Callable[[int], tuple[ndarray | None, ndarray | None]]]]]`. Cada entrada devuelve el modelo, el estado inicial y el viento por paso.
  - `tests.n1_cases.state_hashes(st: SigmaState) -> dict[str, str]`: SHA-256 de `np.ascontiguousarray(a).tobytes()` para cada arreglo de `SigmaState`. Incluye `g_hist` aplanado como `g_hist.{i}.{u|v}`, `tr`, `closure.*` cuando exista, y los escalares `time`, `n_trunc`, `n_limited` con `repr`.
  - `tests.n1_cases.run_case(name: str, steps=(1, 2, 3, 10)) -> dict[int, dict[str, str]]`.
  - `tests.n1_cases.CASES` también cubre el GLS: en `main` su estado vive en `m.core.gls`, así que `state_hashes` acepta `model=` y hashea `model.core.gls.{tke,eps,akv}` cuando `st.closure` no existe. Así el mismo JSON sirve antes y después de Task 2.

- [ ] **Step 1: escribir `tests/n1_cases.py` con unos 30 casos**

  Mallas de como mucho 12 × 8 × 6 celdas, Δt estable y `set_threads(1)` dentro de `run_case`. Reutiliza el estilo de `_ale_box` (`tests/test_sigma_ale.py:22`) y `_box` (`tests/test_sigma_baroclinic.py:21`). Batimetría con pendiente (para que el PGF y la híbrida trabajen), T estratificada, f₀ = 1e-4 y viento no nulo en la mitad de los casos.

  Casos, con nombre = clave:
  - **σ:** `sigma_jacobian`, `sigma_twoterm`, `sigma_fv`, `sigma_gls`, `sigma_parabolic`, `sigma_loglaw`, `sigma_pc`, `sigma_upwind3_adv`, `sigma_centred_adv`, `sigma_open_noprovider` (Flather + Orlanski + esponja, sin proveedor), `sigma_obc_state` (radiating con proveedor), `sigma_kh_kv`, `sigma_rho_ref_auto`.
  - **z\*:** `zstar_vi_arakawa`, `zstar_vi_godunov`, `zstar_flux`, `zstar_upwind_thick`, `zstar_pc`, `zstar_quadratic_drag`, `zstar_sponge_interior` (con `sponge_tr`), `zstar_tracers2`, `zstar_mixdiag`, `zstar_obc_radiating`, `zstar_obc_clamped_transport` (`OBData.transport`).
  - **híbrida:** `hybrid_default`, `hybrid_cutoff` (`thin_layer_cutoff>0`), `hybrid_coupling_none`, `hybrid_hsrc_init` (`initial_state(h_src=...)`), `hybrid_open_radiating`.
  - **ALE σ:** `alesigma_fv`.
  - **Reutilización:** `zstar_reuse`. Mismo modelo, dos `initial_state` seguidos; se registran los hashes de la segunda corrida (para `test_driver_reuse_model_two_runs`, Task 11).

- [ ] **Step 2: escribir `scripts/n1_make_golden.py`**

  Recorre `CASES`. Para cada caso guarda los hashes de los pasos 1, 2, 3 y 10 en `hashes.json` (con el commit de origen en `"_source": "<sha>"`) y el estado del paso 10 en `<case>.npz` (solo para diagnosticar). Además escribe `tests/data/n1_restart_main.nc` con `write_restart` del caso `zstar_vi_arakawa` en el paso 2, y en `hashes.json["_restart_continue"]` los hashes del paso 3 continuado desde ese restart.

- [ ] **Step 3: generar desde `main` y confirmar el determinismo**

  Run:
  ```bash
  git stash -u   # solo si hay cambios sin commit
  python scripts/n1_make_golden.py && python scripts/n1_make_golden.py --check
  ```
  Expected: `--check` recalcula y dice `30/30 identical`. Si un caso no es determinista entre corridas, se quita de la matriz y se anota en el JSON (`"_excluded"`).

- [ ] **Step 4: escribir `tests/test_n1_golden.py`**

  ```python
  @pytest.mark.parametrize("name", sorted(CASES))
  def test_golden(name):
      got = run_case(name)
      ref = GOLDEN[name]
      for step, hashes in ref.items():
          assert got[int(step)] == hashes, f"{name} step {step}: {diff_fields(got[int(step)], hashes)}"

  def test_restart_from_main_file_continues_bit_for_bit(): ...  # lee n1_restart_main.nc, un paso, compara con _restart_continue
  ```

- [ ] **Step 5: comprobar el tamaño y el tiempo**

  Run: `pytest tests/test_n1_golden.py -q && du -sh tests/data/n1_golden tests/data/n1_restart_main.nc`
  Expected: todo pasa en menos de 60 s; los datos ocupan menos de 3 MB.

- [ ] **Step 6: commit**

  ```bash
  git add tests/n1_cases.py tests/test_n1_golden.py scripts/n1_make_golden.py tests/data/n1_golden tests/data/n1_restart_main.nc
  git commit -m "N1 task 0: golden state-hash matrix and reference restart from main"
  ```

---

### Task 1: contrato y registro de variables

**Files:**
- Create: `core/model/__init__.py`, `core/model/contract.py`, `core/model/registry.py`
- Create: `core/components/__init__.py`
- Test: `tests/test_model_contract.py`

**Interfaces:**
- Produces, en `core/model/contract.py`:
  - `class Role(str, Enum)`, con los miembros `EOS, VERTICAL_COORDINATE, PGF, MOMENTUM, TIME_STEPPER, FREE_SURFACE, CLOSURE, BOTTOM_DRAG, LAYER_TRANSPORT, TRACER_ADVECTION, TRACER_DIFFUSION, OPEN_BOUNDARY, INTERIOR_SPONGE, SURFACE_FLUXES, DIAGNOSTICS`.
  - `@dataclass(frozen=True) class Issue: level: Literal["error", "warning"]; component: str; message: str`.
  - `def param(default, *, units: str, doc: str, cite: str = "", benchmark: str = "") -> Any`: devuelve `dataclasses.field(default=..., metadata={...})`.
  - `class Component(Protocol)`, con `name: str`, `role: Role`, `reads: tuple[str, ...]`, `writes: tuple[str, ...]`, `def check(self, ctx: "BuildContext") -> list[Issue]` y `def state_fields(self) -> dict[str, np.ndarray]`.
  - `@dataclass class BuildContext`, con los campos `vertical: Literal["sigma_eulerian", "ale"]`, `regrid: str | None`, `open_edges: tuple[str, ...]`, `has_state_provider: bool`, `dt: float`, `n_tracers: int` y `given: frozenset[str]` (los argumentos que el usuario pasó explícitamente).
  - `class PyOceanConfigWarning(UserWarning)`.
  - `class ConfigurationError(ValueError)`: su mensaje lista todos los errores.
  - `def raise_issues(issues: list[Issue]) -> None`: emite los avisos con `warnings.warn(..., PyOceanConfigWarning)` y luego lanza una sola `ConfigurationError` si hay algún error.
- Produces, en `core/model/registry.py`:
  - `@dataclass(frozen=True) class VarSpec: name; dims: tuple[str, ...]; location: Literal["t", "u", "v", "w", "q", "scalar"]; units: str; standard_name: str | None; kind: Literal["prognostic", "history", "component_state", "diagnostic"]`.
  - `REGISTRY: dict[str, VarSpec]`, con una entrada por cada campo de `SigmaState` (más `closure.tke`, `closure.eps`, `closure.akv`). Las unidades y los nombres CF salen de `diagnostics/output.py`, donde existen.

- [ ] **Step 1: escribir los tests**

  Cubren:
  - `raise_issues` con dos errores y un aviso: lanza una sola excepción cuyo mensaje contiene los dos errores, y el aviso se emite como `PyOceanConfigWarning`.
  - `param` guarda `units`, `doc` y `cite` en los metadatos.
  - `REGISTRY` cubre todos los `dataclasses.fields(SigmaState)`, salvo `time`, y cada `VarSpec.kind` es válido.
- [ ] **Step 2:** `pytest tests/test_model_contract.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: implementar `contract.py` y `registry.py`** con las firmas de arriba.
- [ ] **Step 4:** `pytest tests/test_model_contract.py tests/test_n1_golden.py -q` → PASS.
- [ ] **Step 5: commit**

  ```bash
  git add core/model core/components tests/test_model_contract.py
  git commit -m "N1 task 1: component contract (Role, Issue, param, Component) and variable registry"
  ```

---

### Task 2: el estado de GLS pasa a `SigmaState.closure`

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py:106-124` (campo nuevo), `:2266-2314` (sincronización)
- Modify: `core/restart.py:111-116` (lee y escribe desde `st.closure`; mismos nombres de variable en el archivo)
- Test: `tests/test_n1_golden.py` (sin cambios), `tests/test_restart.py`

**Interfaces:**
- Produces:
  - `SigmaState.closure: dict[str, np.ndarray] | None = None`, con las claves `"tke"`, `"eps"`, `"akv"` cuando `closure == "gls"`.
  - `SigmaBaroclinic._sync_closure_in(st)` y `_sync_closure_out() -> dict | None`. Asignan y leen por referencia los arreglos de `self.core.gls`, sin copias.
  - `state_hashes` (Task 0) ya hashea `st.closure` cuando existe, y produce las mismas claves que hasheaba de `model.core.gls`.

- [ ] **Step 1: implementar.** `initial_state` llena `closure` desde `core.gls`. `step` llama a `_sync_closure_in(st)` antes de `core.step` y pone `closure=_sync_closure_out()` en el estado nuevo. `write_restart` y `read_restart` leen y escriben `st.closure` con los nombres de variable de hoy.
- [ ] **Step 2: comprobar.** `pytest tests/test_n1_golden.py tests/test_restart.py tests/test_gls.py -q` → PASS (`sigma_gls` idéntico).
- [ ] **Step 3: commit**

  ```bash
  git add core/dynamics/sigma_baroclinic.py core/restart.py tests/n1_cases.py
  git commit -m "N1 task 2: GLS state carried in SigmaState.closure"
  ```

---

### Task 3: componentes EOS y PGF

**Files:**
- Create: `core/components/eos.py`, `core/components/pgf.py`
- Modify: SB: `density` (`:1351-1353`), `_baroclinic_pgf` (`:1449-1521`) y `_fv_pgf` (`:1523-1533`) delegan; `ReferenceProfile`/`reference_profile` se reexportan
- Test: `tests/test_components_pgf.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class LinearEOS: rho0: float; t_coef: float; t_ref: float`, con los métodos `density(T) -> ndarray` (la misma expresión que `SB:1352`) y `check(ctx) -> list[Issue]`.
  - `class PressureGradient` (`role=Role.PGF`), con `scheme: Literal["jacobian", "twoterm", "fv"]`, `rho_ref: ReferenceProfile | None` (mutable: se fija en `initial_state`) y `def __call__(self, geom: "LayerGeometry", eta, T, h) -> tuple[ndarray, ndarray]`. Devuelve exactamente lo que hoy devuelve `_baroclinic_pgf`/`_fv_pgf`.
  - `LayerGeometry` es un `NamedTuple` con lo que esos métodos leen de `self`: `grid`, `dsigma`, `bathy` y los callables de `layer_heights` (los que tienen caché), `face_depths`, `mean_face_thickness`. Se construye en SB y se pasa; las cachés siguen siendo del modelo.
  - `SigmaBaroclinic.rho_ref` pasa a ser una propiedad con setter que lee y escribe `self._pgf.rho_ref`. Los tests asignan `m.rho_ref`, y eso sigue funcionando.

- [ ] **Step 1: escribir el test** `test_pgf_component_matches_model_method`. Con el modelo del seamount (`tests/test_sigma_baroclinic.py:165`) en los tres esquemas, compara `np.array_equal(PressureGradient(...)(geom, eta, T, h), m._baroclinic_pgf(...))`. Hacia el final de esta tarea, ambos caminos pasan por el componente, así que el test protege contra que la delegación pierda argumentos.
- [ ] **Step 2: mover el código**, tal cual, a los módulos nuevos; los métodos de SB delegan.
- [ ] **Step 3:** `pytest tests/test_components_pgf.py tests/test_n1_golden.py tests/test_sigma_baroclinic.py tests/test_vertical_pgf.py -q` → PASS.
- [ ] **Step 4: commit** (con rutas explícitas).

---

### Task 4: componente Momentum (términos con nombre)

**Files:**
- Create: `core/components/momentum.py`
- Modify: SB: `_tendencies` (`:1579-1614`), `_coriolis` (`:1535-1549`) y `_vector_invariant` (`:1551-1569`) delegan
- Test: `tests/test_components_momentum.py`

**Interfaces:**
- Consumes: `PressureGradient.__call__` (Task 3).
- Produces:
  - `class FluxFormMomentum` con `adv: Literal["flux", "upwind3", "centred"]` y `ah: float`.
  - `class VectorInvariantMomentum` con `ke_scheme: Literal["arakawa", "godunov"]` y `ah: float`.
  - En las dos clases: `def terms(self, st: SigmaState, geom, pgf_xy) -> list[tuple[str, ndarray, ndarray, Literal["+", "-"]]]`. Devuelve los términos **en el orden de suma de hoy**: `("pgf", …, "+")`, `("coriolis" | "vector_invariant", …, "+")`, `("advection", …, "+")`, `("hviscosity", …, "+")`, `("vadvection", …, "-")`. Un término que hoy no se calcula no aparece en la lista.
  - `def sum_terms(terms, wet_u, wet_v) -> tuple[ndarray, ndarray]`. Reproduce la acumulación in situ de `SB:1579-1614`: el primer término se copia y los demás se suman o restan in situ en el orden dado, y al final se enmascara.
  - `ke_scheme` acepta `"godunov"`. `"gudonov"` se normaliza a `"godunov"` dentro de SB (el aviso llega en Task 12).

- [ ] **Step 1: test** `test_sum_terms_equals_tendencies`. Para `zstar_vi_arakawa`, `zstar_flux`, `sigma_upwind3_adv` y `sigma_centred_adv` de `n1_cases`, compara `sum_terms(comp.terms(...)) == m._tendencies(st)` con `np.array_equal`.
- [ ] **Step 2: mover el código.** `_tendencies` queda como `return sum_terms(self._momentum.terms(st, self._geom(), self._pgf(...)), *wet)`.
- [ ] **Step 3:** `pytest tests/test_components_momentum.py tests/test_n1_golden.py tests/test_vector_invariant.py tests/test_sigma_ale.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 5: componente TimeStepper (AB3)

**Files:**
- Create: `core/components/time_stepper.py` (`ab3_combine` y `AB3` se mueven aquí; SB los reexporta)
- Modify: SB `:2253-2261`
- Test: `tests/test_components_time_stepper.py`

**Interfaces:**
- Produces: `class AB3Momentum` con `def combine(self, g_hist: list, gu, gv) -> tuple[ndarray, ndarray, list]`. Devuelve `gu_ab`, `gv_ab` y la historia nueva, con AB1, AB2 y AB3 al arrancar, exactamente como en `SB:2254-2261`. El docstring declara que los trazadores tienen su propio esquema (`tracer_time`); ver la especificación §8.6.

- [ ] **Step 1: test.** Tres llamadas seguidas reproducen `ab3_combine` a mano con `np.array_equal` y la historia queda con 1, 2 y 3 elementos.
- [ ] **Step 2: implementar y delegar.**
- [ ] **Step 3:** `pytest tests/test_components_time_stepper.py tests/test_n1_golden.py tests/test_timestepping.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 6: componente OpenBoundary

**Files:**
- Create: `core/components/open_boundary.py`
- Modify: SB: los métodos `_evaluate_exterior` (`:1787-1850`), `_exterior_T`, `_open_velocity` (`:1764-1785`), `_radiate` (`:1715-1753`), `_baroclinic_part` (`:1705-1712`), `_ext_stack` (`:1951-1963`), `_open_face_transports` (`:1905-1922`) y `_flather_args` delegan. `_bc_mask`, `_ext_cols`, `_bc_u`, `_bc_v`, `T_ext`, `h_ext` y `tr_ext` pasan a ser propiedades con setter hacia el componente (`test_open_boundary_state.py:196-203` las asigna).
- Modify: `tests/test_sigma_ale.py:720`. El monkeypatch pasa de `sb.open_face_transport` a `core.components.open_boundary.open_face_transport`. Es el único cambio permitido en los tests existentes.
- Test: `tests/test_components_open_boundary.py`

**Interfaces:**
- Produces:
  - `@dataclass class Exterior`: los mismos campos que hoy guarda `self._ext`, más `T_in: ndarray | None` y `tr_in: ndarray | None`.
  - `class OpenBoundary`, con estos métodos:
    - `evaluate(t, hu=None, hv=None, wu=None, wv=None) -> Exterior | None`;
    - `flather_args(ext) -> dict`;
    - `relax_velocity(u_new, v_new, ext, fluxes: "BarotropicFluxes", wu=None, wv=None) -> None` (in situ, declarado `writes_inplace=("u", "v")`);
    - `inflow_stack(ext, stack_shape) -> ndarray`;
    - `open_face_transports(Qx, Qy, ext, fluxes, h) -> None` (in situ);
    - `tracer_sponge(Tstack, h, ext) -> ndarray`.
  - Este módulo hace `from core.vertical.ale import open_face_transport` y lo llama por su nombre de módulo, para que el monkeypatch actualizado funcione.
  - `self._ext`, `self._T_in` y `self._tr_in` desaparecen como estado. SB conserva propiedades de solo lectura con el último `Exterior`, para el test `m._T_in` (`test_sigma_baroclinic.py:572`).

- [ ] **Step 1: test** `test_open_boundary_component_has_no_hidden_step_state`. Después de `step()`, el componente no tiene atributos que cambien entre pasos, salvo la configuración: se comparan los `vars(comp)` antes y después de dos pasos de `zstar_obc_radiating`.
- [ ] **Step 2: mover el código.** Las fases que hoy leen `self._ext` y `self._T_in` reciben `Exterior` como argumento.
- [ ] **Step 3:** `pytest tests/test_components_open_boundary.py tests/test_n1_golden.py tests/test_open_boundary_state.py tests/test_sigma_ale.py tests/test_sigma_baroclinic.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 7: FreeSurface (envoltura), BottomDrag y espesor viscoso

**Files:**
- Create: `core/components/free_surface.py`, `core/components/bottom_drag.py`
- Modify: SB `:2140-2154` (ALE) y `:2272-2275` (σ); `_bed_drag_rates` (`:1852-1876`) delega
- Test: `tests/test_components_free_surface.py`

**Interfaces:**
- Consumes: `Exterior` y `OpenBoundary.flather_args` (Task 6).
- Produces:
  - `@dataclass(frozen=True) class BarotropicFluxes`, con `wet_u`, `wet_v`, `Du`, `Dv`, `open_Qu`, `open_Qv`, `open_Du`, `open_Dv` (cada uno `ndarray | None`) y `clip_volume: float`. Son las copias de referencia de `fs.last_*` (`WD:309-342`) tomadas justo después de `core.step`.
  - `class FreeSurfaceVerticalViscosity` (`role=Role.FREE_SURFACE`), que envuelve `SigmaWetDry`. Su método es `step(eta, u, v, gu_ab, gv_ab, tu, tv, ext, visc: dict, closure_state) -> tuple[eta, u, v, BarotropicFluxes, closure_state]`, y su `state_fields()` devuelve el LU desfasado (`fs.preconditioner_state`).
  - `class QuadraticBedDrag` con `rates(u, v, hu, hv, ...) -> dict`, y `class LogLawDrag`. Este último es solo declarativo en N1: sigue dentro de `SWD.column_step`.
  - Todos los lectores de `fs.last_*` en SB (`:1750`, `:1912-1922`, `:2167-2168`, `:2277-2292`) leen ahora `BarotropicFluxes`.

- [ ] **Step 1: test** `test_free_surface_closed_has_no_open_fluxes`. Con bordes cerrados, los campos `open_*` son `None` y `step` no falla.
- [ ] **Step 2: implementar la envoltura y cambiar los lectores.**
- [ ] **Step 3:** `pytest tests/test_components_free_surface.py tests/test_n1_golden.py tests/test_sigma_wd.py tests/test_wetting_drying.py tests/test_sigma_ale.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 8: componente LayerTransport

**Files:**
- Create: `core/components/layer_transport.py`
- Modify: SB: el bloque σ (`:2276-2297`) y el ALE (`:2166-2185`) pasan a llamar al componente
- Test: `tests/test_components_layer_transport.py`

**Interfaces:**
- Consumes: `BarotropicFluxes` (Task 7) y `OpenBoundary.open_face_transports` (Task 6).
- Produces:
  - `@dataclass class LayerFluxes`, con `Qx`, `Qy`, `Fz`, `V0`, `V1`, `h_dagger: ndarray | None`, `n_limited: int` y `omega: ndarray | None`.
  - `class SigmaLayerTransport` con `theta` y `theta_bc`, y método `__call__(u_old, u_new, v_old, v_new, eta_old, eta_new, fluxes, ext, ob) -> LayerFluxes`. La expresión de `SB:2281` se mueve tal cual.
  - `class ALELayerTransport` con `thickness_flux: Literal["ppm", "upwind"]` y el mismo `__call__`, más `h`, `hu` y `hv`.

- [ ] **Step 1: test.** Para `sigma_kh_kv` y `zstar_upwind_thick`, `LayerFluxes.Fz` cierra la continuidad: `V1 − V0 = −Δt ∇·Q` en cada columna con `np.allclose(atol=1e-12)`. Es un chequeo de la física, no de bits.
- [ ] **Step 2: mover el código.**
- [ ] **Step 3:** `pytest tests/test_components_layer_transport.py tests/test_n1_golden.py tests/test_vertical_ale.py tests/test_sigma_ale.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 9: TracerAdvection y TracerDiffusion

**Files:**
- Create: `core/components/tracers.py`
- Modify: SB: los drivers `_advect` (`:2094-2118`), `_advect_fct` (`:1993-2026` y su driver), `_advect_numpy` (`:2079-2084`), `_rhs`, `_faces_h`, `_upwind_faces_h`, `_open_faces`, `_diffuse_h`, `_diffuse_h_numpy` (`:1645-1703`) y los pasos kv (`:2201-2205`, `:2304-2308`) delegan. Los núcleos de módulo (`_weno5_faces`, `_rk_stage`, `_fct_*`, `fct_coefficients_numpy`) se quedan en SB.
- Test: `tests/test_components_tracers.py`

**Interfaces:**
- Consumes: `LayerFluxes` (Task 8) y `Exterior.T_in`/`tr_in` (Task 6).
- Produces:
  - `class TracerAdvection` con `scheme`, `time: Literal["ssprk3", "pc"]` y `limiter: Literal["fct", "none"]`. Su método es `advect(stack, fluxes: LayerFluxes, ok, inflow: ndarray | None, h=None) -> ndarray`: el valor de entrada llega **como argumento**, no por `self`.
  - `class TracerDiffusion` con `kh` y `kv`. Su método es `apply(T_stack, h_or_D, ...) -> ndarray`, y conserva el despacho de `_diffuse_h` (Numba solo si se pasa `h`).
  - `def mix_num_advection(...)`: mueve `SB:2195`/`:2301` tal cual.

- [ ] **Step 1: test** `test_tracer_advection_inflow_is_explicit`. Se llama a `advect` dos veces con `inflow` distintos sobre `zstar_obc_radiating`: el resultado cambia solo en las celdas del borde y no depende del estado del modelo.
- [ ] **Step 2: mover el código.**
- [ ] **Step 3:** `pytest tests/test_components_tracers.py tests/test_n1_golden.py tests/test_sigma_baroclinic.py tests/test_sigma_ale.py tests/test_advection.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 10: VerticalCoordinate e InteriorSponge

**Files:**
- Create: `core/components/vertical_coordinate.py`, `core/components/sponge.py`
- Modify: SB: `_setup_ale` (`:1164-1228`), la geometría de capas (`:1333-1360`), `_regrid_remap` (`:1395-1410`), `_transport_thickness` (`:1419-1427`, **con la caché por identidad**), `_mean_face_thickness`, `_edge_face_depths`, el remapeo de velocidad (`:1435-1446`), `truncate_cfl` y su error (`:2224-2240`), y `_interior_sponge` (`:1878-1903`). Todos delegan.
- Test: `tests/test_components_vertical.py`

**Interfaces:**
- Produces:
  - `class SigmaEulerian` y `class ALECoordinate` (con `rule`, `params`, `remap_scheme`, `h_thin_bed`, `init_iterations` y `max_truncations`). Métodos comunes:
    - `layer_heights(h) -> ndarray`, con caché `_zc_cache`;
    - `transport_thickness(h) -> tuple`, con caché `_tt_cache`;
    - `mean_face_thickness(h)`;
    - `edge_face_depths()`;
    - `regrid_remap(h_dagger, stack, eta) -> tuple[h_new, stack_new, mix_remap]`. ALE solamente; llama a `rg.regrid_columns` **por atributo del módulo** (`from core.vertical import regrid as rg`), para que el monkeypatch de `test_sigma_ale.py:391` siga funcionando;
    - `remap_velocity(u, v, h_dagger, h_new) -> tuple`;
    - `truncate(u, v, h, dt, n_trunc) -> tuple[u, v, n_trunc]`.
  - El caso σ implementa lo mismo con D·Δσ y los métodos solo ALE lanzan `NotImplementedError`. El driver no los llama.
  - `class InteriorSponge` con `rate`, `T`, `zw` y `tr`, y método `apply(T_stack, h) -> ndarray`.

- [ ] **Step 1: test** `test_transport_thickness_cache_identity_preserved`. Es el equivalente, sobre el componente, de `tests/test_sigma_ale.py:832-836`: dos llamadas con el mismo `h` devuelven el mismo objeto.
- [ ] **Step 2: mover el código.**
- [ ] **Step 3:** `pytest tests/test_components_vertical.py tests/test_n1_golden.py tests/test_sigma_ale.py tests/test_vertical_regrid.py tests/test_vertical_remap.py -q` → PASS.
- [ ] **Step 4: commit.**

---

### Task 11: el driver de 10 fases y la fachada

**Files:**
- Create: `core/model/driver.py`
- Modify: SB: `step` (`:2246-2314`) y `_step_ale` (`:2120-2243`) quedan como `return self._driver.step(st, tau_x, tau_y)`. El `__init__` arma los componentes y el driver.
- Test: `tests/test_model_driver.py`

**Interfaces:**
- Consumes: todos los componentes de las tareas 3 a 10.
- Produces:
  - `PHASES: tuple[str, ...] = ("prepare", "explicit_momentum", "time_combine", "implicit_and_free_surface", "boundary_velocity", "layer_transport", "tracers", "vertical_coordinate", "interior_sponge", "finalize")`.
  - `class Driver`, con este `__init__`: `(components: dict[Role, Any], *, dt, rho0, vertical, dt_dyn=None, dt_tracer=None, dt_therm=None, debug=False, keep_terms=False)`. Sus métodos y atributos:
    - `step(st, tau_x=None, tau_y=None) -> SigmaState`;
    - `phase_hashes: list[dict[str, str]]`, que se llena solo con `debug=True` (un dict por paso, con hashes `state_hashes` de los arreglos vivos al final de cada fase);
    - `last_terms: dict[str, tuple[ndarray, ndarray]] | None`, que se llena solo con `keep_terms=True` y guarda referencias, no copias.
  - La tabla de §4 de la especificación es el contrato: cada fase llama a los métodos de los componentes en el orden de `pyocean.md` §3. Las fases 8 y 9 no hacen nada en σ.
  - `SigmaBaroclinic(..., debug=False)` es el único argumento nuevo de la fachada, y es opcional.

- [ ] **Step 1: tests**
  - `test_driver_phase_order`: con `debug=True`, en un caso σ y uno ALE, `list(driver.phase_hashes[-1])` es igual a `PHASES` (las fases que no hacen nada en σ también aparecen).
  - `test_keep_terms_does_not_change_bits`: `zstar_vi_arakawa`, 3 pasos con y sin `keep_terms`, hashes iguales.
  - `test_driver_reuse_model_two_runs`: dos `initial_state` + 5 pasos sobre el mismo modelo dan los mismos hashes que un modelo nuevo para la segunda corrida en σ con solver 'direct'. En ALE con CG + LU desfasado, los hashes deben ser iguales a los de `main` haciendo lo mismo; Task 0 agrega el caso `zstar_reuse` para tenerlos.
- [ ] **Step 2: implementar el driver.** Mover la coreografía de `step`/`_step_ale` al driver, fase por fase.
- [ ] **Step 3: correr toda la suite.** `pytest tests/ -q -x` → 1388 + los tests nuevos en PASS, con los mismos xfail.
- [ ] **Step 4: commit.**

---

### Task 12: pre-flight `check()`

**Files:**
- Create: `core/model/preflight.py`
- Modify: SB `__init__`. Los chequeos de `:1038-1220` se convierten en `Issue` de cada componente; `__init__` llama a `raise_issues(preflight(components, ctx))`. Las mismas condiciones dan el mismo tipo de excepción (`ValueError`; `ConfigurationError` es subclase), así que los `match=` de `tests/test_sigma_ale.py:44` siguen funcionando: cada mensaje conserva el texto de hoy al inicio.
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `Issue`, `raise_issues` y `BuildContext` (Task 1).
- Produces:
  - `def preflight(components: dict[Role, Any], ctx: BuildContext) -> list[Issue]`.
  - Tabla de avisos de §5 de la especificación (`PyOceanConfigWarning`), cuando el argumento está en `ctx.given`:
    - en σ: `regrid`, `regrid_params`, `remap_scheme`, `h_thin_bed`, `init_iterations`, `max_truncations`, `thin_layer_cutoff`, `mix_diagnostic`, `thickness_flux` y `thin_layer_coupling="mom6"`;
    - `ke_scheme` con `momentum="flux"`;
    - `sponge_T` o `sponge_zw` sin `sponge_rate`;
    - `ke_scheme="gudonov"`, con el mensaje `"ke_scheme='gudonov' is a misspelling kept for compatibility; use 'godunov'"`.
  - Error nuevo: `adv` fuera de `{"flux", "upwind3", "centred"}`.
  - `ctx.given` se calcula en SB comparando los argumentos recibidos con los valores por defecto de la firma (`inspect.signature`).
  - Error si `dt_dyn`, `dt_tracer` y `dt_therm` no son iguales: "N1 supports a single rate".

- [ ] **Step 1: tests**
  - `test_preflight_collects_all_errors`: dos errores a la vez (`pgf="bad"` y `adv="bad"`) aparecen en una sola excepción.
  - `test_ignored_args_warn_and_keep_bits`: el caso `sigma_jacobian` con `regrid="zstar"` emite `PyOceanConfigWarning` y da los mismos hashes de la matriz dorada.
  - `test_gudonov_alias_warns`.
  - `test_error_message_names_fix`: el mensaje de `adv="bad"` contiene `"valid: 'flux', 'upwind3', 'centred'"`.
- [ ] **Step 2: implementar.**
- [ ] **Step 3:** `pytest tests/test_preflight.py tests/test_n1_golden.py tests/test_sigma_ale.py tests/test_sigma_baroclinic.py -q -W error::DeprecationWarning` → PASS.
- [ ] **Step 4: comprobar que la suite no emite avisos nuevos.** `pytest tests/ -q 2>&1 | grep -c PyOceanConfigWarning` → 0, salvo los tests que los provocan a propósito. Si un test existente provoca uno, se le agrega `pytest.mark.filterwarnings` y se anota en el commit.
- [ ] **Step 5: commit.**

---

### Task 13: restart por secciones y accesores públicos

**Files:**
- Modify: `core/restart.py`:
  - `write_restart` agrega a cada variable el atributo `component="<role>"`, con los mismos nombres de variable de hoy;
  - `_config` arma la huella con los parámetros de cada componente, sin leer `_sp_T`/`_sp_zw` (estos pasan por `InteriorSponge`);
  - `read_restart` acepta archivos con o sin `component`.
- Modify: SB. Accesores públicos `face_thickness(h)`, `edge_face_depths()` y `pressure_gradient(eta, T, h=None)`, documentados. Los nombres privados delegan en ellos.
- Modify: `validation/bayiw_cut/run_pyocean.py:33-34`, `validation/bayiw_2d/run_pyocean.py:36`, `validation/internal_wave/run_pyocean.py:28`, `validation/seamount/run_seamount.py:71` y `docs/manual/figures/make_pgf.py:127-129`, para usar los accesores públicos.
- Test: `tests/test_restart.py` (casos nuevos)

**Interfaces:**
- Produces: `SigmaBaroclinic.face_thickness(h) -> tuple`, `edge_face_depths() -> tuple` y `pressure_gradient(eta, T, h=None) -> tuple[ndarray, ndarray]`.

- [ ] **Step 1: tests**
  - `test_restart_has_component_sections`: cada variable del archivo nuevo tiene el atributo `component`.
  - `test_restart_mid_ab_startup`: con `zstar_vi_arakawa`, escribir en el paso 2, leer y dar un paso tiene que ser bit a bit igual que la corrida continua en el paso 3.
  - `test_restart_from_main_file_continues_bit_for_bit` (Task 0) sigue en verde.
- [ ] **Step 2: implementar y actualizar los runners.**
- [ ] **Step 3:** `pytest tests/test_restart.py tests/test_n1_golden.py -q` → PASS. Para cada runner modificado, `python -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" <file>` y `grep -n "\._" <file>` sin usos privados del modelo.
- [ ] **Step 4: commit.** Con rutas explícitas a cada runner, nunca `validation/`.

---

### Task 14: corridas de humo, costo y documentación

**Files:**
- Create: `scripts/n1_smoke.py`. Corre, uno a la vez, en `main` (worktree temporal en `../pyOCEAN_main_n1`) y en la rama, y compara hashes del estado final:
  - corte BayIW z\* e híbrida, 1 h;
  - lock exchange de Ilicak z\*/σ/híbrida, 30 min;
  - DOME a 10 km, 1 día;
  - onda interna, 1 h;
  - seamount σ, 1 día.

  Reutiliza los runners de `validation/*` importando sus funciones de construcción del caso; si un runner no expone una, se agrega una función `build_case(...)` en ese runner. También mide s/paso (mediana de 20 pasos después de 5 de calentamiento, 5 hilos) en el corte BayIW z\* e híbrida y en el overflow de Ilicak.
- Modify: `CLAUDE.md` (fila nueva en la tabla de la Fase 2 y nota en la etapa N), `docs/design/vision_roadmap.md` §8 (N1 ✅) y §9 (la arquitectura real), `docs/manual/architecture/code-map.md` y `docs/manual/architecture/time-step.md` (las 10 fases).
- Modify: el docstring de cada módulo de `core/components/`. Cada uno declara su contrato (rol, lecturas y escrituras con su nivel de tiempo, parámetros con unidades y cita) y cita las referencias de `docs/design/n1_refs/` con `file:line`.

- [ ] **Step 1: crear el worktree de `main`.** `git worktree add ../pyOCEAN_main_n1 main`
- [ ] **Step 2: correr el humo.** `python scripts/n1_smoke.py --ref ../pyOCEAN_main_n1 --out /private/tmp/claude-501/-Users-estebancruz-pyOCEAN/f74d1368-f589-4534-beba-fb43855817b8/scratchpad/n1_smoke`
  Expected: `8/8 identical` y costo de la rama ≤ 1.02 × `main` en los tres casos de costo. Si falla el costo, se perfila la fase con `debug=False` antes de seguir; no se cierra N1 por encima del 2 %.
- [ ] **Step 3: correr la suite completa.** `pytest tests/ -q` → PASS (1388 + nuevos), con los mismos xfail.
- [ ] **Step 4: escribir la documentación** de la lista de arriba.
- [ ] **Step 5: quitar el worktree.** `git worktree remove ../pyOCEAN_main_n1`
- [ ] **Step 6: commit.**

  ```bash
  git add scripts/n1_smoke.py CLAUDE.md docs/design/vision_roadmap.md docs/manual/architecture/code-map.md docs/manual/architecture/time-step.md core/components
  ```
