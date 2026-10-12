# ALE Plan B — Integración en el modelo: h pronóstico, paso lagrangiano, gradiente de volumen finito y escalón

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que `SigmaBaroclinic` corra con coordenada vertical ALE (σ, z\* o híbrida). Esto incluye:
- grosor de capa pronóstico;
- paso lagrangiano, regeneración de malla y remapeo cada paso;
- gradiente de presión de volumen finito.

Se valida en el escalón 600 → 164 m de la boca de BayIW y en el monte submarino.

**Architecture:**
- **El kit de columna del Plan A** (`core/vertical/remap.py`, `regrid.py`) se completa con:
  - la transición somera por ventana y los objetivos ligeros arriba (decisiones del 2026-10-11);
  - dos módulos nuevos: `ale.py` (geometría de capas en las caras C) y `pgf.py` (gradiente de volumen finito de Adcroft et al. 2008, como MOM6).
- **`SigmaWetDry`** acepta la geometría de capas en las caras.
- **`SigmaBaroclinic`** gana el modo `vertical_coordinate="ale"` con su propio `_step_ale`. El modo σ actual (`"sigma_eulerian"`) no cambia ni un bit.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md`. Cubre:
- §2: estado, paso y código;
- §3: ventana y objetivos ligeros;
- §5: gradiente de volumen finito y capas delgadas;
- §6.2: lo que aplica sin fronteras abiertas;
- §7: tests unitarios y de integración 1, 2 y 4 (parcial);
- §9: fases 3 y 4, más la parte de la fase 7 que no necesita fronteras.

**Planes siguientes:**
- **C:** fronteras abiertas bajo ALE (§6.1).
- **D:** validación con modelos completos, BayIW con z\* e híbrida, PQM, restart ALE, salida CF con h y verificador de dominio.

## Global Constraints

- **Convención:** k = 0 en el fondo. Las interfaces son alturas z_w (N+1 valores), con z_w[0] = −H, z_w[N] = η y z = 0 en el reposo. En las caras, u es (N, ny, nx+1) y v es (N, ny+1, nx).
- **El modo σ no cambia:** con `vertical_coordinate="sigma_eulerian"` (por defecto), los 980 tests actuales pasan sin tocarlos y el camino de código σ queda idéntico. Las funciones nuevas reciben `h=None` y en ese caso usan exactamente las fórmulas actuales.
- **Estado ALE:** `SigmaState.h` es (N, ny, nx) en m, con Σ_k h = H + η en las celdas mojadas, y h = 0 en tierra y en celdas secas.
- **Geometría en las caras:** h_cara,k = min(h_k,L, h_k,R), como `hFacW` de MITgcm (`model/src/ini_masks_etc.F:250-252`, `useMin4hFacEdges`). Las fracciones son w_k = h_cara,k / Σ h_cara.
  - **Media de profundidad:** ū = Σ w_k u_k.
  - **Transporte por capa:** Q_k = L·D_f·w_k·(θ u_k^{n+1} + (1 − θ) u_k^n), con D_f la profundidad total aguas arriba de `WetDryFreeSurface`.
  - Así, Σ Q_k es exactamente el flujo de la superficie libre.
  - En σ, w_k = Δσ_k exactamente.
- **Alcance de ALE en este plan:**
  - sin fronteras abiertas ni esponja (Plan C);
  - `z0=None` y `closure` en {"constant", "none"} (el arrastre de pared sobre capas que se desvanecen queda para después);
  - las celdas con H ≤ 0 solo con la regla σ o con la híbrida con zona σ somera (`d_hybrid > 0`).
  - Fuera de este alcance, el constructor da `ValueError` con una explicación.
- **Grosores:**
  - mínimo de regeneración `H_MIN = 1e-3` m (efectivo min(H_MIN, D/N));
  - capa de fondo "delgada" para la velocidad: `H_THIN_BED = 0.01` m (HYCOM `hybgen.F90:1757-1764`);
  - piso de dz para el solver de Thomas: `DZ_FLOOR = 1e-6` m;
  - `H_NEGLECT = 1e-30` m en denominadores.
- **Hilos:** como mucho 5 (núcleos P). `@njit(cache=True)` en kernels sin llamadas entre módulos. Un kernel que llama a otro módulo (p. ej. `remap._plm_slope_wa`) va sin `cache=True`: la caché de Numba solo revisa el archivo propio (lección del Plan A). `prange` solo en drivers.
- **Entorno:** numpy debe quedar en 1.26.4. No se instala nada.
- **Commits** solo cuando Esteban lo autorice. Los pasos "Commit" se ejecutan solo si autorizó commits por tarea al elegir la ejecución. Los mensajes terminan con las dos líneas de atribución de la sesión.
- **Documentación:** cada ecuación lleva su referencia (artículo y/o `archivo:línea` del modelo de referencia). Docstrings con la física.

## Review Focus

1. **Columnas que se secan o casi se secan** con ALE σ (playa con H ≤ 0). Se espera: sin NaN, h ≥ 0, Σh = max(H + η, 0) y un trazador uniforme que sigue uniforme. → Test en Task 6.
2. **Capas desvanecidas junto a capas gruesas en un escalón con z\*.** Se espera:
   - h† nunca negativo;
   - sin NaN en el gradiente de volumen finito;
   - velocidad cero en las capas de fondo delgadas de cada cara, sin cambiar la media Σ w u.
   - → Tests en Task 2 y Task 8.
3. **Perfil de densidad no monótono durante la corrida** con la regla híbrida (agua cálida bajo agua fría). Se espera: la regla monotoniza, no falla y conserva el calor. → Test en Task 7.
4. **`rho_ref` dado como función cualquiera**, no como `ReferenceProfile`, con `pgf="fv"`. Se espera: se tabula y da el mismo resultado que el perfil equivalente. → Test en Task 5.
5. **Restart de un modelo ALE.** Se espera un error claro, no un archivo sin h que al leerse da un estado incoherente. → Test en Task 5.

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `core/vertical/regrid.py` (modificar) | `transition_weight`, `window_shallow_sigma` (sustituye a `blend_shallow_sigma`), `target_densities(..., n_light)`, validación de H ≤ 0 por regla |
| `core/vertical/ale.py` (nuevo) | Geometría de capas en las caras C: `face_thickness`, `face_fractions`, `zero_thin_bed_velocity`, `lagrangian_thickness` |
| `core/vertical/pgf.py` (nuevo) | Gradiente de presión de volumen finito: `pressure_edges_plm`, kernel por caras, `fv_pgf` |
| `core/dynamics/sigma_wd.py` (modificar) | `column_step(..., dz)` y `step(..., wu, wv, hu, hv)` |
| `core/dynamics/sigma_baroclinic.py` (modificar) | `SigmaState.h`; configuración ALE; geometría desde h; `pgf="fv"`; `_step_ale` |
| `core/restart.py` (modificar) | Error claro para modelos ALE |
| `tests/test_vertical_regrid.py` (modificar) | Ventana, objetivos ligeros, validación por regla |
| `tests/test_vertical_ale.py` (nuevo) | Geometría de caras |
| `tests/test_vertical_pgf.py` (nuevo) | Gradiente de volumen finito |
| `tests/test_sigma_wd.py` (modificar) | Geometría de capas en `SigmaWetDry` |
| `tests/test_sigma_ale.py` (nuevo) | Modelo ALE: configuración, conservación, regresión ALE-σ, híbrida, escalón |
| `tests/test_restart.py` (modificar) | Error de restart ALE |
| `validation/mom6_column/compare_hybgen.py` (modificar) | Usa la ventana |
| `validation/seamount/run_seamount.py`, `README.md` (modificar) | Opciones `--pgf fv` y `--coord zstar`; resultados |
| `CLAUDE.md`, `docs/design/vision_roadmap.md` §7 (modificar) | Filas de estado y marcador |

---

### Task 1: Transición somera por ventana, objetivos ligeros y validación por regla

**Files:**
- Modify: `core/vertical/regrid.py` (sustituir `blend_shallow_sigma`, rama híbrida de `_regrid_columns`, `target_densities`, validación de `regrid_columns`, docstring del módulo)
- Modify: `tests/test_vertical_regrid.py` (sustituir `test_blend_shallow_sigma_endpoints_and_middle`)
- Modify: `validation/mom6_column/compare_hybgen.py` (función `hycom1`)

**Interfaces:**
- Consumes: `zstar_column`, `sigma_column`, `hycom1_column`, `enforce_min_thickness`, `regrid_columns` (Plan A).
- Produces:
  - `transition_weight(D, d_sigma, d_hybrid) -> float`;
  - `window_shallow_sigma(zw_hyb, zw_sig, zw_zs, H, D, d_sigma, d_hybrid, h_min, zw) -> None`;
  - `target_densities(rho_top, n_layers, n_light=0, d_rho_light=0.01) -> ndarray (n_layers+1,)`;
  - `regrid_columns` acepta H ≤ 0 con la regla σ y con la híbrida en su zona σ.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_vertical_regrid.py`, sustituir `test_blend_shallow_sigma_endpoints_and_middle` por:

```python
def _window_targets(H, n=10):
    """Isopycnals crowded towards the surface (depths H s², s = 0 … 1), so they differ from uniform σ."""
    return 1024.0 + 0.02 * H * np.linspace(0.0, 1.0, n + 1) ** 2


def _window_case(H, d_sigma=50.0, d_hybrid=100.0, n=10):
    """Hybrid, σ and z* interfaces of a linearly stratified column of depth H (η = 0)."""
    zw_t = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    h, rho = np.diff(zw_t), 1024.0 - 0.02 * zc
    dz = np.full(n, 2.0)                                                  # z* floor: top 20 m
    targets = _window_targets(H, n)
    hyb, sig, zs = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
    rg.hycom1_column(H, 0.0, h, rho, targets, dz, np.full(n, np.inf), rg.H_MIN, hyb)
    rg.sigma_column(H, 0.0, np.full(n, 1.0 / n), rg.H_MIN, sig)
    rg.zstar_column(H, 0.0, dz, rg.H_MIN, zs)
    out = np.empty(n + 1)
    rg.window_shallow_sigma(hyb, sig, zs, H, H, d_sigma, d_hybrid, rg.H_MIN, out)
    return hyb, sig, zs, out


def test_window_is_sigma_in_shallow_water_and_hybrid_in_deep_water():
    hyb, sig, _, out = _window_case(40.0)                               # D ≤ d_sigma
    np.testing.assert_array_equal(out, sig)
    hyb, sig, _, out = _window_case(120.0)                              # D ≥ d_hybrid
    np.testing.assert_array_equal(out, hyb)


def test_window_keeps_the_isopycnals_that_fit_in_the_window():
    H = 75.0                                                            # q = 0.5
    hyb, sig, zs, out = _window_case(H)
    q = 0.5
    lo, hi = q * (-H) + (1 - q) * sig, q * zs + (1 - q) * sig
    assert ((out >= lo - 1e-12) & (out <= hi + 1e-12)).all()
    inside = (hyb > lo) & (hyb < hi)
    inside[[0, -1]] = False
    # depths: hybrid 2, 4, 6.75, 12, … (z* floor, then H s²), window tops 4.75, 9.5, …: 3 deep
    # interfaces stay on their isopycnal, 6 shallow ones are clipped (a position blend would move all)
    assert inside.sum() == 3 and (~inside[1:-1]).sum() == 6
    np.testing.assert_allclose(out[inside], hyb[inside], atol=1e-12)    # those stay on their isopycnal
    clipped = ~inside
    clipped[[0, -1]] = False
    np.testing.assert_allclose(out[clipped], np.clip(hyb, lo, hi)[clipped], atol=1e-12)
    assert (np.diff(out) > 0).all()
    np.testing.assert_allclose(out[-1] - out[0], H, rtol=1e-14)


@pytest.mark.parametrize("H, other", [(50.0 + 1e-6, "sig"), (100.0 - 1e-6, "hyb")])
def test_window_is_continuous_at_both_ends_of_the_transition(H, other):
    # no jump in the grid when the total depth crosses d_sigma or d_hybrid (tide, wetting–drying)
    hyb, sig, _, out = _window_case(H)
    ref = sig if other == "sig" else hyb
    assert np.abs(out - ref).max() < 1e-4


def test_regrid_columns_hybrid_with_transition_matches_the_window():
    hyb, sig, zs, out = _window_case(75.0)
    n = 10
    zw_t = np.linspace(-75.0, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    h3 = np.diff(zw_t)[:, None, None].repeat(2, 1)
    r3 = (1024.0 - 0.02 * zc)[:, None, None].repeat(2, 1)
    params = dict(dz_top=np.full(n, 2.0), rho_target=_window_targets(75.0, n),
                  dsigma=np.full(n, 1.0 / n), d_sigma=50.0, d_hybrid=100.0)
    zw3 = rg.regrid_columns("hybrid", np.full((2, 1), 75.0), np.zeros((2, 1)), h3, r3, params, np.ones((2, 1), bool))
    np.testing.assert_allclose(zw3[:, 0, 0], out, atol=1e-12)


def test_light_top_targets_keep_the_mixed_layer_in_z():
    # 40 m mixed layer over a linear pycnocline: with targets that start at the surface density the
    # first interface sinks to the base of the mixed layer (HYCOM1 lumps it); targets lighter than
    # any water (OM4 style) keep the top interfaces on their z* floor
    H, n = 201.0, 10
    zw_t = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    rho = np.where(zc > -40.0, 1024.0, 1024.0 + 0.02 * (-40.0 - zc))
    h = np.diff(zw_t)
    t0 = rg.target_densities(rho, n)
    t3 = rg.target_densities(rho, n, n_light=3)
    assert t3.shape == (n + 1,) and (np.diff(t3) > 0).all() and (t3[:3] < rho.min()).all()
    np.testing.assert_allclose(t3[3:], np.linspace(rho.min(), rho.max(), n - 2))
    np.testing.assert_array_equal(t0, rg.target_densities(rho, n, n_light=0))
    z0 = _hybrid(H, 0.0, h, rho, t0)[::-1]                              # surface first
    z3 = _hybrid(H, 0.0, h, rho, t3)[::-1]
    assert z0[1] <= -35.0                                               # lumped mixed layer
    np.testing.assert_allclose(z3[1:4], -np.cumsum(DELR)[:3], atol=1e-12)   # on the z* floor


def test_target_densities_rejects_bad_n_light():
    with pytest.raises(ValueError):
        rg.target_densities(np.array([1024.0, 1025.0]), 4, n_light=5)


def test_sigma_rule_accepts_floodable_columns_and_hybrid_only_in_its_sigma_zone():
    H = np.array([[-0.5, 10.0]])
    eta = np.array([[0.7, 0.0]])                                        # flooded land: D = 0.2 m
    h3 = np.ones((10, 1, 2))
    wet = np.ones((1, 2), bool)
    zw = rg.regrid_columns("sigma", H, eta, h3, None, dict(dsigma=np.full(10, 0.1)), wet)
    np.testing.assert_allclose(zw[-1] - zw[0], (H + eta)[0], rtol=1e-12)
    r3 = np.full((10, 1, 2), 1025.0)
    base = dict(dz_top=np.ones(10), rho_target=np.linspace(1024, 1026, 11), dsigma=np.full(10, 0.1))
    with pytest.raises(ValueError):
        rg.regrid_columns("hybrid", H, eta, h3, r3, base, wet)                       # no σ zone
    zw = rg.regrid_columns("hybrid", H, eta, h3, r3, dict(base, d_sigma=5.0, d_hybrid=20.0), wet)
    np.testing.assert_allclose(np.diff(zw[:, 0, 0]), 0.02, rtol=1e-12)               # pure σ there
    with pytest.raises(ValueError):
        rg.regrid_columns("zstar", H, eta, h3, None, base, wet)
```

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_vertical_regrid.py -q 2>&1 | tail -15`
Expected: FAIL. `window_shallow_sigma` no existe (AttributeError), `target_densities()` no acepta `n_light` (TypeError) y la regla σ rechaza H ≤ 0 (ValueError).

- [ ] **Step 3: Implementar en `core/vertical/regrid.py`**

1. Sustituir `blend_shallow_sigma` por:

```python
@njit(cache=True)
def transition_weight(D, d_sigma, d_hybrid):  # pragma: no cover
    """q = clip((D − d_σ)/(d_hyb − d_σ), 0, 1): 0 = pure σ (shallow), 1 = pure hybrid (deep)."""
    if d_hybrid <= d_sigma:
        return 1.0 if D > d_sigma else 0.0
    return min(max((D - d_sigma) / (d_hybrid - d_sigma), 0.0), 1.0)


@njit(cache=True)
def window_shallow_sigma(zw_hyb, zw_sig, zw_zs, H, D, d_sigma, d_hybrid, h_min, zw):  # pragma: no cover
    """
    Shallow-water σ transition by a window (spec §3, decision of 2026-10-11).  With
    q = transition_weight(D, d_σ, d_hyb), interface j keeps its hybrid position inside

        L_j = q (−H) + (1 − q) σ_j  ≤  z_j  ≤  U_j = q z*_j + (1 − q) σ_j

    and is clipped to the nearer bound otherwise; then the minimum thickness.  q = 0: pure σ
    (wetting–drying); q = 1: the hybrid grid unchanged; in between, the isopycnals that fit in
    the window are kept.  Continuous in D; interfaces never cross (monotone bounds and input);
    Σh = D.  HYCOM (MOM_hybgen_regrid.F90:590-620) instead blends the minimum thicknesses
    dp0k/ds0k with a weight that makes them add up to D, so between dsns and dpns its column
    is pure σ and follows no isopycnal.
    """
    n = zw.size - 1
    q = transition_weight(D, d_sigma, d_hybrid)
    if q >= 1.0:
        for j in range(n + 1):
            zw[j] = zw_hyb[j]
        return
    for j in range(n + 1):
        lo = q * (-H) + (1.0 - q) * zw_sig[j]
        hi = q * zw_zs[j] + (1.0 - q) * zw_sig[j]
        zw[j] = min(max(zw_hyb[j], lo), hi)
    zw[0] = -H
    zw[n] = zw_sig[n]
    enforce_min_thickness(zw, h_min)
```

2. En `_regrid_columns`, sustituir la rama `else:` (híbrida) por:

```python
        else:
            q = transition_weight(H[j, i] + eta[j, i], d_sigma, d_hybrid) if d_hybrid > 0.0 else 1.0
            if q <= 0.0:                                     # σ zone: no hybrid grid needed (H may be ≤ 0)
                sigma_column(H[j, i], eta[j, i], dsigma, h_min, col)
            else:
                hycom1_column(H[j, i], eta[j, i], np.ascontiguousarray(h[:, j, i]),
                              np.ascontiguousarray(rho[:, j, i]), rho_target, dz_top, max_h_top, h_min, col)
                if q < 1.0:
                    sig = np.empty(n + 1)
                    zs = np.empty(n + 1)
                    sigma_column(H[j, i], eta[j, i], dsigma, h_min, sig)
                    zstar_column(H[j, i], eta[j, i], dz_top, h_min, zs)
                    hyb = col.copy()
                    window_shallow_sigma(hyb, sig, zs, H[j, i], H[j, i] + eta[j, i], d_sigma, d_hybrid,
                                         h_min, col)
```

3. Sustituir `target_densities` por:

```python
def target_densities(rho_top: np.ndarray, n_layers: int, n_light: int = 0, d_rho_light: float = 0.01) -> np.ndarray:
    """
    Interface target densities (surface first) [kg m⁻³]: the first ``n_light`` lighter than any
    water, ρ_min − d_rho_light·(n_light … 1), so that the hybrid rule keeps those interfaces on
    their z* floor and the mixed layer stays resolved in z (OM4 style, Adcroft et al. 2019; see
    validation/mom6_column/README.md); the rest evenly spaced over the profile range.
    """
    if not 0 <= n_light < n_layers:
        raise ValueError(f"n_light must be in [0, {n_layers - 1}], got {n_light}")
    rho_top = np.asarray(rho_top, float)
    lo, hi = np.nanmin(rho_top), np.nanmax(rho_top)
    light = lo - d_rho_light * np.arange(n_light, 0, -1)
    return np.r_[light, np.linspace(lo, hi, n_layers + 1 - n_light)]
```

4. En `regrid_columns`, sustituir la comprobación `Hw, Dw = ...` / `if not (...): raise ValueError(...)` por:

```python
    Hw, Dw = H[wet], (H + eta)[wet]
    if not (np.isfinite(Hw).all() and np.isfinite(Dw).all() and (Dw > 0).all()):
        raise ValueError("wet columns need finite H and D = H + eta > 0")
    if rule == "zstar" and not (Hw > 0).all():
        raise ValueError("z* needs H > 0 in every wet column (cells that can dry: sigma, or hybrid with d_hybrid > 0)")
    if rule == "hybrid":
        d_s, d_h = float(params.get("d_sigma", 0.0)), float(params.get("d_hybrid", 0.0))
        if ((Hw <= 0) & ~((d_h > 0) & (Dw <= d_s))).any():
            raise ValueError("hybrid needs H > 0 outside its shallow σ zone (d_hybrid > 0 and D ≤ d_sigma)")
```

y actualizar su docstring: "Raises ValueError for a missing required input, a wet column with non-finite H or η, D ≤ 0, H ≤ 0 with z\* (or with the hybrid rule outside its σ zone), or non-finite ρ (hybrid)."

5. En el docstring del módulo, cambiar "optional shallow-water σ transition (HYCOM hybgen.F90:392-428, applied to the interfaces)" por "optional shallow-water σ transition by a window (window_shallow_sigma; spec §3)".

- [ ] **Step 4: Actualizar `validation/mom6_column/compare_hybgen.py`**

En `hycom1()`, sustituir el bloque `if blend:` por:

```python
    if blend:
        sig, zs, out = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        rg.sigma_column(c["H"], 0.0, np.full(n, 1.0 / n), H_MIN, sig)
        rg.zstar_column(c["H"], 0.0, dz, H_MIN, zs)
        rg.window_shallow_sigma(zw, sig, zs, c["H"], c["H"], 50.0, 100.0, H_MIN, out)
        zw = out
```

(La columna "somera 30 m" tiene q = 0, así que su resultado es σ, igual que antes.)

- [ ] **Step 5: Correr los tests y ver que pasan**

Run: `pytest tests/test_vertical_regrid.py tests/test_vertical_vs_mom6.py -q 2>&1 | tail -5`
Expected: PASS. 29 − 1 + 8 = 36 tests de regrid (el parametrizado cuenta 2) y 2 contra MOM6.

Si el volumen `/Volumes/ESTEBAN_MIT` está montado y existe el ejecutable de hybgen:
- Run: `python validation/mom6_column/compare_hybgen.py > /dev/null && git diff --stat validation/mom6_column/hybgen_comparison.txt`
- Expected: sin cambios en `hybgen_comparison.txt`.

- [ ] **Step 6: Suite completa y commit**

Run: `pytest tests/ -q -x 2>&1 | tail -3`
Expected: todo pasa (987 + 1 xfail).

```bash
git add core/vertical/regrid.py tests/test_vertical_regrid.py validation/mom6_column/compare_hybgen.py
git commit -m "ALE regrid: shallow sigma window transition, light top targets, per-rule H checks"
```

---

### Task 2: Geometría de capas en las caras C (`core/vertical/ale.py`)

**Files:**
- Create: `core/vertical/ale.py`
- Create: `tests/test_vertical_ale.py`

**Interfaces:**
- Consumes: nada del modelo.
- Produces (todas con k = 0 en el fondo):
  - `H_THIN_BED = 0.01`, `DZ_FLOOR = 1e-6`;
  - `face_thickness(h, active) -> (hu (N,ny,nx+1), hv (N,ny+1,nx))`;
  - `face_fractions(hf) -> w` (misma forma, Σ_k w = 1);
  - `zero_thin_bed_velocity(u, hf, w, h_thin=H_THIN_BED) -> u'`;
  - `lagrangian_thickness(h, divq, dt, area, tol=1e-9) -> h†`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_vertical_ale.py`:

```python
"""
Tests for core/vertical/ale.py — layer geometry of the ALE coordinate on the C grid:
face thickness (thinner of the two cells, MITgcm hFacW), face fractions, zero velocity in
thin bed layers (HYCOM hybgen) and the Lagrangian continuity guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import ale


def test_face_thickness_is_the_thinner_cell_and_fractions_are_dsigma_for_sigma_layers():
    ds = np.array([0.1, 0.2, 0.3, 0.4])
    D = np.array([[10.0, 30.0, 20.0]])
    h = D[None] * ds[:, None, None]
    hu, hv = ale.face_thickness(h, np.ones((1, 3), bool))
    np.testing.assert_allclose(hu[:, 0, 1], 10.0 * ds)                  # min of the two columns
    np.testing.assert_allclose(hu[:, 0, 0], h[:, 0, 0])                 # domain-edge face: inner cell
    np.testing.assert_allclose(hu[:, 0, 3], h[:, 0, 2])
    np.testing.assert_allclose(ale.face_fractions(hu), np.broadcast_to(ds[:, None, None], hu.shape), rtol=1e-14)
    assert hv.shape == (4, 2, 3)


def test_face_next_to_land_has_no_thickness():
    h = np.full((3, 1, 3), 5.0)
    active = np.array([[True, False, True]])
    hu, _ = ale.face_thickness(h, active)
    assert (hu[:, 0, 1] == 0).all() and (hu[:, 0, 2] == 0).all()
    np.testing.assert_allclose(ale.face_fractions(hu)[:, 0, 1], 1.0 / 3)  # defined, no NaN


def test_zstar_step_face_has_vanished_bed_layers():
    # z* across a 600 → 164 m step: layers below the shallow bed are 1 mm on one side, so the face
    # carries nothing there (MITgcm partial cells)
    deep = np.full(6, 100.0)                                             # 600 m, bed first
    shallow = np.array([1e-3, 1e-3, 1e-3, 63.997, 50.0, 50.0])           # 164 m, three collapsed layers
    h = np.stack([deep, shallow], axis=1)[:, None, :]
    hu, _ = ale.face_thickness(h, np.ones((1, 2), bool))
    np.testing.assert_allclose(hu[:3, 0, 1], 1e-3)
    assert ale.face_fractions(hu)[:3, 0, 1].max() < 1e-4


def test_zero_thin_bed_velocity_keeps_the_depth_mean():
    hf = np.array([1e-3, 5e-3, 2.0, 3.0, 5.0])[:, None, None]
    w = ale.face_fractions(hf)
    u = np.array([0.3, -0.2, 0.1, 0.05, 0.02])[:, None, None]
    out = ale.zero_thin_bed_velocity(u, hf, w)
    assert out[0] == 0 and out[1] == 0                                   # thin run from the bed
    np.testing.assert_allclose((w * out).sum(0), (w * u).sum(0), rtol=1e-14)
    np.testing.assert_allclose(np.diff(out[2:, 0, 0]), np.diff(u[2:, 0, 0]))   # shear above unchanged


def test_zero_thin_bed_velocity_ignores_thin_layers_above_a_thick_one_and_all_thin_columns():
    hf = np.array([2.0, 1e-3, 3.0])[:, None, None]
    u = np.array([0.1, 0.2, 0.3])[:, None, None]
    np.testing.assert_array_equal(ale.zero_thin_bed_velocity(u, hf, ale.face_fractions(hf)), u)
    hf = np.full((3, 1, 1), 1e-3)                                        # very shallow: all thin
    np.testing.assert_array_equal(ale.zero_thin_bed_velocity(u, hf, ale.face_fractions(hf)), u)


def test_lagrangian_thickness_updates_and_rejects_an_emptied_layer():
    h = np.full((2, 1, 1), 1.0)
    area = np.full((1, 1), 100.0)
    out = ale.lagrangian_thickness(h, np.array([10.0, -10.0])[:, None, None], 2.0, area)
    np.testing.assert_allclose(out[:, 0, 0], [0.8, 1.2])
    with pytest.raises(RuntimeError, match="layer 0"):
        ale.lagrangian_thickness(h, np.array([100.0, 0.0])[:, None, None], 2.0, area)
    tiny = ale.lagrangian_thickness(h, np.array([50.0 + 1e-11, 0.0])[:, None, None], 2.0, area)
    assert tiny[0, 0, 0] == 0.0                                          # round-off below zero → 0
```

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_vertical_ale.py -q 2>&1 | tail -5`
Expected: FAIL con `ImportError: cannot import name 'ale'`.

- [ ] **Step 3: Implementar `core/vertical/ale.py`**

```python
"""
Layer geometry of the ALE vertical coordinate on the Arakawa C grid (spec
docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §2, §5).

Convention: layer thickness h (N, ny, nx) [m], k = 0 at the bed; u-faces (N, ny, nx+1),
v-faces (N, ny+1, nx).

Face thickness: the thinner of the two neighbouring cells,

    h_k,f = min(h_k,L, h_k,R),

as the partial cells of MITgcm (hFacW = min of the two hFacC, model/src/ini_masks_etc.F:250-252,
useMin4hFacEdges): a layer that has vanished on one side of a step carries no transport.  Face
fractions w_k = h_k,f / Σ_k h_k,f weight the depth mean ū = Σ w_k u_k of the free surface, so the
layer transports Q_k = L D_f w_k u_k add up to the free-surface flux L D_f ū exactly.  For
σ-layers w_k = Δσ_k.

References: Adcroft, A., Hill, C. & Marshall, J. (1997). Mon. Wea. Rev. 125, 2293–2315 (shaved
cells / partial steps) [verificar].  HYCOM hybgen.F90:1757-1764 (thin near-bottom layers).
"""

from __future__ import annotations

import numpy as np

H_THIN_BED = 0.01      # [m] bed layers thinner than this carry no velocity (HYCOM hybgen.F90:1757-1764)
DZ_FLOOR = 1e-6        # [m] smallest thickness passed to the implicit vertical solvers


def face_thickness(h: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer thickness at u- and v-faces [m]: min(h_L, h_R) between two active cells, the inner
    cell on the domain edge, 0 next to land.  h (N, ny, nx), active (ny, nx) bool.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    n, ny, nx = h.shape
    hu = np.zeros((n, ny, nx + 1))
    hv = np.zeros((n, ny + 1, nx))
    hu[:, :, 1:-1] = np.where((active[:, :-1] & active[:, 1:])[None], np.minimum(h[:, :, :-1], h[:, :, 1:]), 0.0)
    hu[:, :, 0], hu[:, :, -1] = h[:, :, 0], h[:, :, -1]
    hv[:, 1:-1, :] = np.where((active[:-1, :] & active[1:, :])[None], np.minimum(h[:, :-1, :], h[:, 1:, :]), 0.0)
    hv[:, 0, :], hv[:, -1, :] = h[:, 0, :], h[:, -1, :]
    return hu, hv


def face_fractions(hf: np.ndarray) -> np.ndarray:
    """w_k = h_k,f / Σ_k h_k,f (uniform 1/N on faces with no water, so ū stays defined)."""
    tot = hf.sum(axis=0)
    wet = tot > 0
    return np.where(wet[None], hf / np.where(wet, tot, 1.0)[None], 1.0 / hf.shape[0])


def zero_thin_bed_velocity(u: np.ndarray, hf: np.ndarray, w: np.ndarray, h_thin: float = H_THIN_BED) -> np.ndarray:
    """
    Zero velocity in the run of face layers thinner than ``h_thin`` that starts at the bed
    (HYCOM hybgen.F90:1757-1764: "thin near-bottom layer, zero total current").  The removed
    part Σ_thin w u is spread over the other layers in proportion to w, so the depth mean
    Σ w u — and with it the free-surface flux — is unchanged.  Faces whose layers are all thin
    are left alone.
    """
    thin = np.cumprod(hf < h_thin, axis=0).astype(bool)
    keep = ~thin
    w_keep = np.where(keep, w, 0.0).sum(axis=0)
    ok = w_keep > 0
    removed = np.where(thin, w * u, 0.0).sum(axis=0)
    shift = np.where(ok, removed / np.where(ok, w_keep, 1.0), 0.0)
    return np.where(thin & ok[None], 0.0, u + np.where(keep, shift[None], 0.0))


def lagrangian_thickness(h: np.ndarray, divq: np.ndarray, dt: float, area: np.ndarray,
                         tol: float = 1e-9) -> np.ndarray:
    """
    Layer continuity without flux through the interfaces (ω ≡ 0, spec §2 step 1):

        h_k† = h_k − Δt (∇·Q_k) / A

    divq (N, ny, nx) [m³ s⁻¹], area (ny, nx) [m²].  Values in [−tol, 0) are round-off and set
    to 0; a layer emptied beyond that means Δt is too long for the outflow (CFL) and raises.
    """
    out = h - dt * divq / np.where(area > 0, area, 1.0)[None]
    bad = out < -tol
    if bad.any():
        k, j, i = (int(a[0]) for a in np.nonzero(bad))
        raise RuntimeError(f"layer {k} of cell (j={j}, i={i}) emptied in the Lagrangian step: "
                           f"h† = {out[k, j, i]:.3e} m (reduce dt)")
    return np.maximum(out, 0.0)
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `pytest tests/test_vertical_ale.py -q 2>&1 | tail -3`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add core/vertical/ale.py tests/test_vertical_ale.py
git commit -m "ALE: C-grid face geometry (min thickness, fractions, thin bed layers, Lagrangian continuity)"
```

---

### Task 3: `SigmaWetDry` con geometría de capas en las caras

**Files:**
- Modify: `core/dynamics/sigma_wd.py` (`column_step` y `step`)
- Modify: `tests/test_sigma_wd.py` (añadir 2 tests al final)

**Interfaces:**
- Consumes: nada nuevo.
- Produces:
  - `SigmaWetDry.column_step(u, G_k, depth, nu=None, tau=None, dz=None)`;
  - `SigmaWetDry.step(eta, u, v, G_u=None, G_v=None, flather_eta=None, flather_ubar=None, flather_vbar=None, tau_u=None, tau_v=None, wu=None, wv=None, hu=None, hv=None)`.
  - Por defecto (`None`) el resultado es idéntico bit a bit al actual.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir al final de `tests/test_sigma_wd.py`:

```python
def _wd_basin():
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=12, ny=10, dx=1000.0, dy=1000.0, f0=1e-4)
    m = SigmaWetDry(hg, np.full((10, 12), 20.0), dt=30.0, n_layers=4, z0=None, closure="constant", nu_const=1e-3)
    x = (np.arange(12) + 0.5) * 1000.0
    y = (np.arange(10) + 0.5) * 1000.0
    eta = 0.3 * np.exp(-((x[None] - 6e3) ** 2 + (y[:, None] - 5e3) ** 2) / 3e3**2)
    rng = np.random.default_rng(1)
    u = 0.05 * rng.standard_normal((4, 10, 13))
    v = 0.05 * rng.standard_normal((4, 11, 12))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1]] = 0.0
    return m, eta, u, v


def test_explicit_sigma_geometry_reproduces_the_default_step_bit_for_bit():
    m, eta, u, v = _wd_basin()
    e0, u0, v0 = m.step(eta, u, v)
    Du, Dv = m.face_depths(eta)
    ds = m.dsigma[:, None, None]
    flo = lambda D: np.maximum(np.nan_to_num(D, nan=m.d_min), m.d_min)                 # noqa: E731
    e1, u1, v1 = m.step(eta, u, v, wu=np.broadcast_to(ds, u.shape), wv=np.broadcast_to(ds, v.shape),
                        hu=flo(Du)[None] * ds, hv=flo(Dv)[None] * ds)
    assert np.array_equal(e0, e1) and np.array_equal(u0, u1) and np.array_equal(v0, v1)


def test_layer_transports_with_face_fractions_add_up_to_the_free_surface_flux():
    m, eta, u, v = _wd_basin()
    w = np.array([0.05, 0.15, 0.3, 0.5])[:, None, None]                 # not the σ fractions
    wu, wv = np.broadcast_to(w, u.shape), np.broadcast_to(w, v.shape)
    Du, Dv = m.face_depths(eta)
    e1, u1, v1 = m.step(eta, u, v, wu=wu, wv=wv, hu=np.nan_to_num(Du)[None] * w, hv=np.nan_to_num(Dv)[None] * w)
    th, fs = m.fs.theta, m.fs
    un, vn = np.where(fs.last_wet_u, u, 0.0), np.where(fs.last_wet_v, v, 0.0)
    Qx = (fs.Lu * fs.last_Du)[None] * wu * (th * u1 + (1 - th) * un)
    Qy = (fs.Lv * fs.last_Dv)[None] * wv * (th * v1 + (1 - th) * vn)
    div = (Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:] - Qy[:, :-1]).sum(0)
    np.testing.assert_allclose(fs.area * (e1 - eta), -m.dt * div, atol=1e-9 * fs.area.max() * 0.3)
```

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_sigma_wd.py -q 2>&1 | tail -5`
Expected: FAIL con `TypeError: step() got an unexpected keyword argument 'wu'`.

- [ ] **Step 3: Implementar**

1. En `column_step`, añadir `dz: np.ndarray | None = None` como último parámetro. Documentar en el docstring: "dz : (N, ...) layer thickness at the faces [m] (default D Δσ_k)". Sustituir la línea `dz = np.broadcast_to(D * self.dsigma[:, None], uu.shape)` por:

```python
        if dz is None:
            dz = np.broadcast_to(D * self.dsigma[:, None], uu.shape)
        else:
            dz = np.asarray(dz, float).reshape(n, -1)
```

2. En `step`, añadir los parámetros `wu`, `wv`, `hu`, `hv` (todos `np.ndarray | None = None`). Documentarlos en el docstring:
   - "wu, wv : layer fractions at the faces for the depth mean ū = Σ w_k u_k (default Δσ_k; ALE: core.vertical.ale.face_fractions)";
   - "hu, hv : layer thickness at the faces for the implicit vertical viscosity [m] (default D Δσ_k)".

   Sustituir las líneas desde `u_hat = ...` hasta `ubar_hat, vbar_hat = ...` por:

```python
        wu_ = ds if wu is None else wu
        wv_ = ds if wv is None else wv
        u_hat = self.column_step(u, G_u, Du, nu_u, tau=tau_u, dz=hu)
        v_hat = self.column_step(v, G_v, Dv, nu_v, tau=tau_v, dz=hv)
        ubar_n, vbar_n = (wu_ * u).sum(0), (wv_ * v).sum(0)
        ubar_hat, vbar_hat = (wu_ * u_hat).sum(0), (wv_ * v_hat).sum(0)
```

   La corrección barotrópica uniforme `u_hat + (ubar_new − ubar_hat)` sigue siendo correcta porque Σ_k w_k = 1.

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `pytest tests/test_sigma_wd.py tests/test_sigma_baroclinic.py -q 2>&1 | tail -3`
Expected: todo pasa (11 + 2 en `test_sigma_wd.py`; 37 en `test_sigma_baroclinic.py` sin cambios).

- [ ] **Step 5: Commit**

```bash
git add core/dynamics/sigma_wd.py tests/test_sigma_wd.py
git commit -m "SigmaWetDry: optional face-layer fractions and thicknesses (ALE geometry)"
```

---

### Task 4: Gradiente de presión de volumen finito (`core/vertical/pgf.py`)

**Files:**
- Create: `core/vertical/pgf.py`
- Create: `tests/test_vertical_pgf.py`

**Interfaces:**
- Consumes: `remap._plm_slope_wa`, `remap._plm_monotonized_slope`, `remap.H_NEGLECT` (Plan A).
- Produces:
  - `pressure_edges_plm(h, T, T_bot, T_top) -> None`, por columna, k = 0 en el fondo;
  - `fv_pgf(h, H, eta, T, du, dv, rho0, alpha, t0, ref_z, ref_rho, g, mass_weight=True) -> (gx (N,ny,nx+1), gy (N,ny+1,nx))` en m s⁻². Es la parte baroclínica: la parte ρ₀gη la resuelve la superficie libre.

**Algoritmo** (MOM6 commit 0e9a561; rutas relativas a `src/`). Dentro del kernel se trabaja con la superficie primero, como MOM6:
- e[0] = η y e[K+1] = e[K] − h[K], con e[N] = −H.
- **Bordes PLM de T:**
  - `ALE/MOM_ALE.F90:1512-1596`, `pressure_reconstruction_plm`;
  - pendiente `PLM_slope_wa` y `PLM_monotonized_slope` (`ALE/PLM_functions.F90:22-65, 124-159`);
  - celdas extremas por extrapolación h2: el borde interior es la interpolación lineal entre centros (`PLM_functions.F90:164-183`). `BOUNDARY_EXTRAPOLATION_PRESSURE = True` es el valor por defecto (`core/MOM_PressureForce_FV.F90:2186`).
- **Anomalía de densidad** ρ′(T, z) = ρ₀ − α(T − T₀) − ρ_ref(z):
  - con ρ_ref = ρ₀ constante es la de MOM6 (`MOM_EOS_linear.F90:76-86`);
  - con un perfil ρ_ref(z) se resta la presión de una columna de referencia que depende solo de z: su integral de contorno ∮P_ref(z) dz = 0, así que la resta es exacta.
- **Integrales por columna**, con 5 puntos de Boole de arriba abajo (`core/MOM_density_integrals.F90:579-623`):
  - z_n = e_t − n·dz/4 y T_n = ((4 − n)T_t + n·T_b)/4;
  - B = (7(r₀ + r₄) + 32(r₁ + r₃) + 12 r₂)/90;
  - dpa = g·dz·B;
  - intz = ½·g·dz²·(B − (16(r₃ − r₁) + 7(r₄ − r₀))/90).
- **Integral horizontal** a lo largo de la capa (`:642-751`):
  - sub-columnas m = 1..3 con w_L = (4 − m)/4;
  - dz_x = w_L dz_L + w_R dz_R; tope e_x interpolado; T de las esquinas interpolado;
  - I_m = g·dz_x·B;
  - intx_dpa = (7(dpa_L + dpa_R) + 32(I₁ + I₃) + 12 I₂)/90.
- **Ponderación por masa en el fondo** (`:652-685`; `MASS_WEIGHT_IN_PRESSURE_GRADIENT`, activada aquí por defecto como pide el §5 de la especificación; en MOM6 está apagada por defecto, `MOM_PressureForce_FV.F90:2124-2149`):
  - hW = max(0, z_fondo,L − e_R[k], z_fondo,R − e_L[k]), es decir, la capa de un lado queda entera bajo el fondo del otro;
  - si hW > 0: hW ·= ((h_L − h_R)/(h_L + h_R))²;
  - las esquinas se mezclan hacia la celda gruesa: T_tl = (hW h_R T_tR + (hW h_L + h_L h_R) T_tL)/(hW(h_L + h_R) + h_L h_R), y lo mismo para T_tr, T_bl y T_br;
  - las columnas extremas (dpa_L, dpa_R) no se ponderan.
- **Presión en las interfaces y fuerza** (`MOM_PressureForce_FV.F90:1337-1343, 1532-1558, 1793-1802`):
  - pa_top = g·η·(ρ_ref(η/2) − ρ₀): la anomalía en la superficie, sin la parte ρ₀gη;
  - pa(K+1) = pa(K) + dpa;
  - intx_pa(0) = ½(pa_L + pa_R)(0), intx_pa(K+1) = intx_pa(K) + intx_dpa;
  - la fuerza es

    PFu_k = [(pa_L h_L + intz_L) − (pa_R h_R + intz_R) + (h_R − h_L)·intx_pa(K) − (e_R[K+1] − e_L[K+1])·intx_dpa] · 2/(ρ₀ Δx (h_L + h_R + h_neglect)).
- **Exactitud:** con EOS lineal y T lineal en la capa, los integrandos son a lo sumo cuadráticos y Boole es exacto. El esquema es exacto para ρ lineal en z con cualquier geometría de capas.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_vertical_pgf.py`:

```python
"""
Tests for core/vertical/pgf.py — finite-volume pressure-gradient force (Adcroft, Hallberg &
Harrison 2008; MOM6 MOM_PressureForce_FV / MOM_density_integrals) with a linear EOS.
"""

from __future__ import annotations

import numpy as np

from core.vertical import pgf

RHO0, ALPHA, T0, G = 1000.0, 0.2, 0.0, 9.81
CONST_REF = (np.array([-1e4, 1e4]), np.array([RHO0, RHO0]))


def _layers(H, eta, n, rng=None):
    """Interfaces bed first: uniform, or random monotone (tilted, some 1 mm layers) with rng."""
    ny, nx = H.shape
    if rng is None:
        frac = np.broadcast_to(np.linspace(0, 1, n + 1)[:, None, None], (n + 1, ny, nx))
    else:
        f = rng.uniform(0.0, 1.0, (n, ny, nx))
        f[rng.uniform(size=f.shape) < 0.2] = 1e-5                       # some vanished layers
        frac = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(f, 0) / f.sum(0)], 0)
    D = H + eta
    zw = -H[None] + D[None] * frac
    return np.diff(zw, axis=0), 0.5 * (zw[1:] + zw[:-1])


def _call(h, H, eta, T, dx=1000.0, ref=CONST_REF, mass_weight=True):
    ny, nx = H.shape
    du, dv = np.full((ny, nx + 1), dx), np.full((ny + 1, nx), dx)
    return pgf.fv_pgf(h, H, eta, T, du, dv, RHO0, ALPHA, T0, ref[0], ref[1], G, mass_weight=mass_weight)


def test_plm_edges_are_exact_for_a_linear_profile_on_an_irregular_column():
    h = np.array([3.0, 1.0, 1e-3, 7.0, 2.0])
    zc = np.cumsum(h) - 0.5 * h
    T = 4.0 + 0.3 * zc
    tb, tt = np.empty(5), np.empty(5)
    pgf.pressure_edges_plm(h, T, tb, tt)
    np.testing.assert_allclose(tb, 4.0 + 0.3 * (zc - 0.5 * h), atol=1e-12)
    np.testing.assert_allclose(tt, 4.0 + 0.3 * (zc + 0.5 * h), atol=1e-12)


def test_no_force_for_horizontally_uniform_linear_stratification_on_tilted_layers():
    rng = np.random.default_rng(3)
    ny, nx, n = 4, 7, 12
    H = rng.uniform(50.0, 400.0, (ny, nx))
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n, rng)
    T = 20.0 + 0.02 * zc                                                 # ρ linear in z
    gx, gy = _call(h, H, eta, T)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


def test_exact_for_density_linear_in_x_and_z_over_a_flat_bed():
    # ρ′ = a x + b z: PGF_x = −(g/ρ₀) ∂/∂x ∫_z^0 ρ′ dz′ = (g a/ρ₀) z, averaged over each layer
    ny, nx, n = 3, 8, 6
    H = np.full((ny, nx), 30.0)
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n)
    xc = (np.arange(nx) + 0.5) * 1000.0
    a, b = 1e-4, -0.02
    rho = RHO0 + a * xc[None, None, :] + b * zc
    gx, gy = _call(h, H, eta, (RHO0 - rho) / ALPHA)
    np.testing.assert_allclose(gx[:, :, 1:-1], np.broadcast_to(G * a / RHO0 * zc[:, :, :1], gx[:, :, 1:-1].shape),
                               rtol=1e-10, atol=1e-16)
    assert np.abs(gy).max() < 1e-15


def test_reference_profile_equal_to_the_water_gives_no_force():
    ny, nx, n = 2, 5, 8
    H = np.array([[100.0, 160.0, 220.0, 300.0, 380.0]] * ny)
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n)
    zz = np.linspace(-400.0, 0.0, 801)
    rho_of = lambda z: RHO0 + 3.0 * (1 - np.exp(z / 100.0))           # noqa: E731
    T = (RHO0 - rho_of(zc)) / ALPHA
    gx0, _ = _call(h, H, eta, T)                                         # without the profile: σ-type error
    gx1, _ = _call(h, H, eta, T, ref=(zz, rho_of(zz)))
    assert np.abs(gx1).max() < 1e-3 * np.abs(gx0).max()


def test_mass_weighting_hides_the_value_of_a_layer_vanished_below_the_neighbour_bed():
    # z*-like step: left column 100 m with its two deepest layers vanished at the bed; right
    # column 300 m.  Right layer 1 lies wholly below the left bed, so the force on it must not
    # depend on the (meaningless) T of the vanished left layer
    n = 4
    H = np.array([[100.0, 300.0]])
    eta = np.zeros((1, 2))
    h = np.array([[1e-3, 1e-3, 50.0 - 2e-3, 50.0], [100.0, 100.0, 50.0, 50.0]]).T[:, None, :]
    T = np.full((n, 1, 2), 10.0)

    def force_change(mw):
        T2 = T.copy()
        T2[1, 0, 0] = 25.0                                               # garbage in the vanished layer
        return abs(_call(h, H, eta, T2, mass_weight=mw)[0][1, 0, 1] - _call(h, H, eta, T, mass_weight=mw)[0][1, 0, 1])

    assert force_change(False) > 0
    assert force_change(True) < 1e-3 * force_change(False)
```

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_vertical_pgf.py -q 2>&1 | tail -5`
Expected: FAIL con `ImportError: cannot import name 'pgf'`.

- [ ] **Step 3: Implementar `core/vertical/pgf.py`**

```python
"""
Finite-volume pressure-gradient force (Adcroft, Hallberg & Harrison 2008), Boussinesq, linear
equation of state ρ = ρ₀ − α (T − T₀), on any layer geometry (σ, z*, hybrid).

The force on layer k at a u-face is the contour integral of the pressure around the layer's
vertical section between the two cell centres, divided by its area:

    PFu_k = −(1/(ρ₀ h̄ Δx)) ∮ p dz,   h̄ = (h_L + h_R)/2

evaluated with T piecewise linear (PLM) in each layer and Boole's 5-point rule in the vertical and
along the layer (exact for a linear EOS).  Only the baroclinic part is returned: the ρ₀ g η part
is the free-surface gradient of the barotropic solver (the force is linear in p and ∮(η − z) dz
gives exactly −g δ_x η).  A reference profile ρ_ref(z) is subtracted (pressure anomaly pa); its
pressure depends on z only, so ∮ P_ref dz = 0 and the subtraction is exact — it only removes
the large cancelling terms.  Bottom mass weighting (MOM6 MASS_WEIGHT_IN_PRESSURE_GRADIENT)
replaces the corner values of a layer that lies wholly below the neighbour's bed by values
weighted towards the thicker cell.

MOM6 source (commit 0e9a561, src/): core/MOM_PressureForce_FV.F90:1146-1198 (interfaces),
:1250-1260 (surface anomaly), :1337-1343 (pa), :1532-1558 (intx_pa), :1793-1802 (force);
core/MOM_density_integrals.F90:579-623 (column integrals), :642-751 (along the layer),
:652-685 (mass weighting); ALE/MOM_ALE.F90:1512-1596 and ALE/PLM_functions.F90:22-183 (PLM).

References: Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the
pressure gradient force using analytic integration. Ocean Modell. 22, 106–113.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.vertical.remap import H_NEGLECT, _plm_monotonized_slope, _plm_slope_wa

N_REF = 8193          # points of the uniform ρ_ref(z) table


# Not disk-cached: calls remap._plm_slope_wa / _plm_monotonized_slope (Numba's cache only checks this file).
@njit
def pressure_edges_plm(h, T, T_bot, T_top):  # pragma: no cover
    """
    Bottom and top edge values of T in each layer (bed first) for the pressure integrals: limited
    PLM in the interior (White & Adcroft 2008; MOM6 pressure_reconstruction_plm, MOM_ALE.F90:1512-1596)
    and unlimited h2 extrapolation in the bed and top layers (PLM_functions.F90:164-183).
    """
    n = h.size
    if n == 1:
        T_bot[0] = T[0]
        T_top[0] = T[0]
        return
    slp = np.zeros(n)
    for k in range(1, n - 1):
        slp[k] = _plm_slope_wa(h[k - 1], h[k], h[k + 1], H_NEGLECT, T[k - 1], T[k], T[k + 1])
    for k in range(1, n - 1):
        m = _plm_monotonized_slope(T[k - 1], T[k], T[k + 1], slp[k - 1], slp[k], slp[k + 1])
        T_bot[k] = T[k] - 0.5 * m
        T_top[k] = T[k] + 0.5 * m
    e = (T[1] * h[0] + T[0] * h[1]) / (h[0] + h[1] + H_NEGLECT)
    T_top[0] = e
    T_bot[0] = 2.0 * T[0] - e
    e = (T[n - 1] * h[n - 2] + T[n - 2] * h[n - 1]) / (h[n - 2] + h[n - 1] + H_NEGLECT)
    T_bot[n - 1] = e
    T_top[n - 1] = 2.0 * T[n - 1] - e


@njit(parallel=True)
def _edges_columns(h, T, T_bot, T_top):  # pragma: no cover
    n, ny, nx = h.shape
    for c in prange(ny * nx):
        j, i = c // nx, c % nx
        tb = np.empty(n)
        tt = np.empty(n)
        pressure_edges_plm(np.ascontiguousarray(h[:, j, i]), np.ascontiguousarray(T[:, j, i]), tb, tt)
        T_bot[:, j, i] = tb
        T_top[:, j, i] = tt


@njit(inline="always")
def _ref(z, z0, dz, rr):  # pragma: no cover
    """ρ_ref(z) from a uniform table (constant beyond its ends)."""
    x = (z - z0) / dz
    m = rr.size
    if x <= 0.0:
        return rr[0]
    if x >= m - 1:
        return rr[m - 1]
    i = int(x)
    f = x - i
    return rr[i] + f * (rr[i + 1] - rr[i])


@njit(inline="always")
def _boole_mean(e_top, dz, t_top, t_bot, rho0, alpha, t0, z0, dzr, rr):  # pragma: no cover
    """(B, r₀, …, r₄): Boole mean of ρ′ over a layer and its 5 point values, top to bottom (no allocation)."""
    r0 = rho0 - alpha * (t_top - t0) - _ref(e_top, z0, dzr, rr)
    r1 = rho0 - alpha * ((3.0 * t_top + t_bot) / 4.0 - t0) - _ref(e_top - 0.25 * dz, z0, dzr, rr)
    r2 = rho0 - alpha * ((t_top + t_bot) / 2.0 - t0) - _ref(e_top - 0.5 * dz, z0, dzr, rr)
    r3 = rho0 - alpha * ((t_top + 3.0 * t_bot) / 4.0 - t0) - _ref(e_top - 0.75 * dz, z0, dzr, rr)
    r4 = rho0 - alpha * (t_bot - t0) - _ref(e_top - dz, z0, dzr, rr)
    b = (7.0 * (r0 + r4) + 32.0 * (r1 + r3) + 12.0 * r2) / 90.0
    return b, r0, r1, r2, r3, r4


@njit(parallel=True)
def _fv_pgf_x(e, h, Tt, Tb, dx, rho0, alpha, t0, z0, dzr, rr, g, mass_weight, out):  # pragma: no cover
    """
    Force on the interior u-faces (columns i−1, i).  e (N+1, ny, nx), h, Tt, Tb (N, ny, nx):
    SURFACE FIRST (MOM6 order); dx (ny, nx+1) [m]; out (N, ny, nx+1) [m s⁻²].
    """
    n, ny, nx = h.shape
    for j in prange(ny):
        for i in range(1, nx):
            L = i - 1
            R = i
            paL = g * e[0, j, L] * (_ref(0.5 * e[0, j, L], z0, dzr, rr) - rho0)
            paR = g * e[0, j, R] * (_ref(0.5 * e[0, j, R], z0, dzr, rr) - rho0)
            ixpa = 0.5 * (paL + paR)
            for k in range(n):
                hL = h[k, j, L]
                hR = h[k, j, R]
                bL, a0, a1, _, a3, a4 = _boole_mean(e[k, j, L], hL, Tt[k, j, L], Tb[k, j, L], rho0, alpha, t0,
                                                    z0, dzr, rr)
                bR, c0, c1, _, c3, c4 = _boole_mean(e[k, j, R], hR, Tt[k, j, R], Tb[k, j, R], rho0, alpha, t0,
                                                    z0, dzr, rr)
                dpaL = g * hL * bL
                dpaR = g * hR * bR
                izL = 0.5 * g * hL * hL * (bL - (16.0 * (a3 - a1) + 7.0 * (a4 - a0)) / 90.0)
                izR = 0.5 * g * hR * hR * (bR - (16.0 * (c3 - c1) + 7.0 * (c4 - c0)) / 90.0)
                ttl, tbl, ttr, tbr = Tt[k, j, L], Tb[k, j, L], Tt[k, j, R], Tb[k, j, R]
                if mass_weight:
                    hw = max(0.0, e[n, j, L] - e[k, j, R], e[n, j, R] - e[k, j, L])
                    if hw > 0.0:
                        a = hL + H_NEGLECT
                        c = hR + H_NEGLECT
                        hw *= ((a - c) / (a + c)) ** 2
                        iD = 1.0 / (hw * (a + c) + a * c)
                        ttl, ttr = (hw * c * Tt[k, j, R] + (hw * a + a * c) * Tt[k, j, L]) * iD, \
                            (hw * a * Tt[k, j, L] + (hw * c + a * c) * Tt[k, j, R]) * iD
                        tbl, tbr = (hw * c * Tb[k, j, R] + (hw * a + a * c) * Tb[k, j, L]) * iD, \
                            (hw * a * Tb[k, j, L] + (hw * c + a * c) * Tb[k, j, R]) * iD
                I1 = 0.0
                I2 = 0.0
                I3 = 0.0
                for m in range(1, 4):
                    wl = (4 - m) / 4.0
                    wr = 1.0 - wl
                    dzx = wl * hL + wr * hR
                    bx = _boole_mean(wl * e[k, j, L] + wr * e[k, j, R], dzx, wl * ttl + wr * ttr,
                                     wl * tbl + wr * tbr, rho0, alpha, t0, z0, dzr, rr)[0]
                    if m == 1:
                        I1 = g * dzx * bx
                    elif m == 2:
                        I2 = g * dzx * bx
                    else:
                        I3 = g * dzx * bx
                ixdpa = (7.0 * (dpaL + dpaR) + 32.0 * (I1 + I3) + 12.0 * I2) / 90.0
                out[k, j, i] = ((paL * hL + izL) - (paR * hR + izR) + (hR - hL) * ixpa
                                - (e[k + 1, j, R] - e[k + 1, j, L]) * ixdpa) \
                    * 2.0 / (rho0 * dx[j, i] * (hL + hR + H_NEGLECT))
                paL += dpaL
                paR += dpaR
                ixpa += ixdpa


def fv_pgf(h: np.ndarray, H: np.ndarray, eta: np.ndarray, T: np.ndarray, du: np.ndarray, dv: np.ndarray,
           rho0: float, alpha: float, t0: float, ref_z: np.ndarray, ref_rho: np.ndarray, g: float,
           mass_weight: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Baroclinic finite-volume pressure-gradient force at u- and v-faces [m s⁻²] (module docstring).

    h, T : (N, ny, nx) layer thickness [m] and temperature [°C], k = 0 at the bed
    H, eta : (ny, nx) bed depth and surface height [m] (land: any finite value, masked by the caller)
    du (ny, nx+1), dv (ny+1, nx) : centre-to-centre distances across the faces [m]
    ref_z, ref_rho : reference profile ρ_ref(z) (increasing z) [m], [kg m⁻³]; constant ρ₀ = no profile
    Returns gx (N, ny, nx+1), gy (N, ny+1, nx) with zeros on the domain-edge faces.
    """
    h = np.ascontiguousarray(h, float)
    T = np.ascontiguousarray(T, float)
    n, ny, nx = h.shape
    tb, tt = np.empty_like(h), np.empty_like(h)
    _edges_columns(h, T, tb, tt)
    hs, Tt, Tb = h[::-1].copy(), tt[::-1].copy(), tb[::-1].copy()          # surface first
    e = np.empty((n + 1, ny, nx))
    e[n] = -np.asarray(H, float)
    for k in range(n - 1, -1, -1):                                         # MOM_PressureForce_FV.F90:1195-1198
        e[k] = e[k + 1] + hs[k]
    lo = min(float(np.min(ref_z)), float(e[n].min())) - 1.0
    hi = max(float(np.max(ref_z)), float(e[0].max())) + 1.0
    zz = np.linspace(lo, hi, N_REF)
    rr = np.interp(zz, np.asarray(ref_z, float), np.asarray(ref_rho, float))
    dzr = zz[1] - zz[0]
    gx = np.zeros((n, ny, nx + 1))
    _fv_pgf_x(e, hs, Tt, Tb, np.ascontiguousarray(du, float), rho0, alpha, t0, lo, dzr, rr, g, mass_weight, gx)
    sw = lambda a: np.ascontiguousarray(np.swapaxes(a, -1, -2))                     # noqa: E731
    gyT = np.zeros((n, nx, ny + 1))
    _fv_pgf_x(sw(e), sw(hs), sw(Tt), sw(Tb), sw(dv), rho0, alpha, t0, lo, dzr, rr, g, mass_weight, gyT)
    return gx[::-1].copy(), np.swapaxes(gyT, -1, -2)[::-1].copy()
```

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `pytest tests/test_vertical_pgf.py -q 2>&1 | tail -3`
Expected: 5 passed. Si `test_no_force_for_horizontally_uniform_linear_stratification_on_tilted_layers` no llega a 1e-13, hay que imprimir el máximo:
- un valor del orden de 1e-14 a 1e-12 es redondeo: anotar un `Ruling:` con el valor medido y subir la tolerancia a 10× ese valor;
- un valor mayor es un error del port: usar superpowers:systematic-debugging.

- [ ] **Step 5: Commit**

```bash
git add core/vertical/pgf.py tests/test_vertical_pgf.py
git commit -m "Finite-volume pressure-gradient force (Adcroft et al. 2008, as MOM6) with bottom mass weighting"
```

---

### Task 5: `SigmaBaroclinic` en modo ALE: configuración, estado h, geometría y `pgf="fv"`

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`
- Modify: `core/restart.py` (`write_restart`, `read_restart`)
- Create: `tests/test_sigma_ale.py`
- Modify: `tests/test_restart.py` (1 test)

**Interfaces:**
- Consumes: Tasks 1–4. `rg.regrid_columns`, `rm.remap_columns`, `rm.PCM/PLM/PPM_H4`, `ale.*`, `fv_pgf`.
- Produces:
  - `SigmaState.h` (N, ny, nx) o `None`;
  - parámetros nuevos de `SigmaBaroclinic(...)`: `vertical_coordinate="sigma_eulerian"`, `regrid="sigma"`, `regrid_params=None`, `remap_scheme="ppm_h4"`, `h_thin_bed=0.01`, `init_iterations=10`; además, `pgf` acepta `"fv"`;
  - atributos `self.ale`, `self.regrid_rule`, `self.regrid_params`, `self.remap_code`;
  - métodos:
    - `_initial_h(eta) -> h`;
    - `_regrid_remap(eta, h_src, fields, T) -> (h_new, list)`;
    - `_remap_faces(vel, hf_src, hf_tgt) -> vel`;
    - `layer_heights(eta, h=None)`, `layer_volumes(eta, h=None)`;
    - `_baroclinic_pgf(eta, T, h=None)`, `_diffuse_h(T, V, eta, h=None)`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_sigma_ale.py`:

```python
"""
Tests for the ALE mode of core/dynamics/sigma_baroclinic.py (spec
docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §2, §5, §7): configuration,
prognostic layer thickness, Lagrangian step + regrid + remap, finite-volume pressure gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, reference_profile
from core.grid.horizontal import create_cartesian_grid
from core.vertical import regrid as rg

RHO0, TCOEF = 1000.0, 0.28


def _ale_box(rule="zstar", nx=32, nz=10, depth=20.0, dx=2000.0, dt=30.0, bathy=None, params=None, **kw):
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.full((1, nx), depth) if bathy is None else bathy
    if params is None:
        params = dict(dz_top=np.full(nz, np.nanmax(H) / nz))
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                           nu_const=1e-6, theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid=rule,
                           regrid_params=params, **kw)


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("kw, match", [
    (dict(vertical_coordinate="zz"), "vertical_coordinate"),
    (dict(regrid="isopycnal"), "regrid"),
    (dict(remap_scheme="pqm"), "remap_scheme"),
    (dict(open_boundaries=("west",)), "open boundaries"),
    (dict(z0=1e-3), "z0"),
    (dict(closure="gls"), "closure"),
    (dict(params={}), "dz_top"),
])
def test_ale_configuration_errors(kw, match):
    base = dict(rule="zstar")
    if "regrid" in kw:
        base["rule"] = kw.pop("regrid")
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    args = dict(dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale", regrid=base["rule"],
                regrid_params=dict(dz_top=np.full(4, 5.0)))
    if "params" in kw:
        args["regrid_params"] = kw.pop("params")
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), **args)


def test_zstar_rejects_cells_that_can_dry():
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.linspace(-1.0, 10.0, 8)[None]
    with pytest.raises(ValueError, match="dry"):
        SigmaBaroclinic(hg, H, dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale",
                        regrid="zstar", regrid_params=dict(dz_top=np.full(4, 2.5)))


def test_ale_initial_state_has_the_rule_grid_and_sigma_mode_has_no_h():
    m = _ale_box(bathy=np.array([[20.0, 13.0, 7.0, 20.0]]), nx=4, nz=4, params=dict(dz_top=np.full(4, 5.0)))
    st = m.initial_state(np.zeros((1, 4)), np.full((4, 1, 4), 10.0))
    np.testing.assert_allclose(st.h.sum(0), [[20.0, 13.0, 7.0, 20.0]], rtol=1e-14)
    np.testing.assert_allclose(st.h[:, 0, 0], 5.0, rtol=1e-14)
    # 13 m column on 5 m levels (surface first 5, 5, partial 3); the deepest level collapses at the bed
    np.testing.assert_allclose(st.h[::-1, 0, 1][:3], [5.0, 5.0, 3.0 - 1e-3], atol=1e-12)
    assert abs(st.h[0, 0, 1] - 1e-3) < 1e-12
    z = m.layer_heights(np.zeros((1, 4)), st.h)
    np.testing.assert_allclose(z[-1, 0, 0], -2.5)
    sig = SigmaBaroclinic(create_cartesian_grid(nx=4, ny=1, dx=1e3, dy=1e3, f0=0.0), np.full((1, 4), 20.0),
                          dt=10.0, n_layers=4)
    assert sig.initial_state(np.zeros((1, 4)), np.full((4, 1, 4), 10.0)).h is None and not sig.ale


def test_fv_pgf_is_exact_for_linear_stratification_over_a_seamount():
    from tests.test_sigma_baroclinic import _seamount_model
    m, z = _seamount_model("fv")
    T = 20.0 + 0.005 * z
    gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


def test_fv_pgf_error_over_the_seamount_is_not_larger_than_the_jacobian():
    # spec §7: "error del seamount ≤ al actual" — both with the reference profile of the state
    from tests.test_sigma_baroclinic import _seamount_model
    err = {}
    for scheme in ("jacobian", "fv"):
        m, z = _seamount_model(scheme)
        rho = RHO0 + 3.0 * (1 - np.exp(z / 1000.0))
        T = (RHO0 - rho) / 0.2
        m.rho_ref = reference_profile(z, rho)
        gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
        err[scheme] = max(np.abs(gx).max(), np.abs(gy).max())
    assert err["fv"] <= err["jacobian"]


def test_fv_pgf_tabulates_any_callable_reference_profile():
    from tests.test_sigma_baroclinic import _seamount_model
    m, z = _seamount_model("fv")
    rho = RHO0 + 3.0 * (1 - np.exp(z / 1000.0))
    T = (RHO0 - rho) / 0.2
    prof = reference_profile(z, rho)
    m.rho_ref = prof
    g1 = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)[0]
    m.rho_ref = lambda zz: prof(zz)
    g2 = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)[0]
    np.testing.assert_allclose(g2, g1, atol=1e-10)
```

En `tests/test_restart.py`, añadir:

```python
def test_restart_refuses_an_ale_model(tmp_path):
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=6, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, 6), 20.0), dt=10.0, n_layers=4, z0=None, closure="constant",
                        vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(4, 5.0)))
    st = m.initial_state(np.zeros((1, 6)), np.full((4, 1, 6), 10.0))
    with pytest.raises(NotImplementedError, match="ALE"):
        write_restart(tmp_path / "r.nc", m, st)
    with pytest.raises(NotImplementedError, match="ALE"):
        read_restart(tmp_path / "r.nc", m)
```

(Comprobar los imports de `test_restart.py`: `write_restart`, `read_restart`, `SigmaBaroclinic`, `pytest` y `np`. Añadir los que falten.)

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_sigma_ale.py tests/test_restart.py -q 2>&1 | tail -8`
Expected: FAIL. `vertical_coordinate` no es un parámetro (TypeError) y `pgf="fv"` da ValueError.

- [ ] **Step 3: Implementar en `core/dynamics/sigma_baroclinic.py`**

1. **Imports**, debajo de los actuales:

```python
from core.vertical import regrid as rg
from core.vertical import remap as rm
from core.vertical.ale import DZ_FLOOR, face_fractions, face_thickness, lagrangian_thickness, zero_thin_bed_velocity
from core.vertical.pgf import fv_pgf
```

   y la constante `REMAP_SCHEMES = {"pcm": rm.PCM, "plm": rm.PLM, "ppm_h4": rm.PPM_H4}` junto a `AB3`.

2. **`SigmaState`:** añadir, después de `T`:

```python
    h: np.ndarray | None = None                              # ALE layer thickness (N,ny,nx) [m], k = 0 at the bed
```

   Los campos siguientes ya tienen valor por defecto, así que el orden del dataclass es válido. `read_restart` construye `SigmaState` por nombre.

3. **Constructor:** añadir los parámetros `vertical_coordinate: str = "sigma_eulerian", regrid: str = "sigma", regrid_params: dict | None = None, remap_scheme: str = "ppm_h4", h_thin_bed: float = 0.01, init_iterations: int = 10`, y documentarlos en el docstring de la clase:

```
    vertical_coordinate : 'sigma_eulerian' (default; σ-layers with vertical advection) or 'ale'
          (Lagrangian step + regrid + remap each step; spec docs/superpowers/specs/2026-10-11-ale-...)
    regrid : ALE grid rule 'sigma' | 'zstar' | 'hybrid' (core.vertical.regrid)
    regrid_params : rule parameters (dz_top, rho_target, max_h_top, d_sigma, d_hybrid, h_min; dsigma
          defaults to the model's)
    remap_scheme : 'pcm' | 'plm' | 'ppm_h4' (default)
    h_thin_bed : bed face layers thinner than this [m] carry no velocity (HYCOM)
    init_iterations : regrid–remap passes of the initial state with the hybrid rule
    pgf : also 'fv', finite volume (Adcroft et al. 2008; core.vertical.pgf)
```

   Cambiar la validación de `pgf` a `if pgf not in ("twoterm", "jacobian", "fv"):`. Al final de `__init__`, después de `self.area = self.fs.area`, añadir:

```python
        self._setup_ale(vertical_coordinate, regrid, regrid_params, remap_scheme, z0, closure, h_thin_bed,
                        init_iterations, act)
```

   y el método:

```python
    def _setup_ale(self, vertical_coordinate, regrid, regrid_params, remap_scheme, z0, closure, h_thin_bed,
                   init_iterations, act) -> None:
        """Check and store the ALE configuration (spec §2, §6.2); scope limits of ALE Plan B raise."""
        if vertical_coordinate not in ("sigma_eulerian", "ale"):
            raise ValueError(f"unknown vertical_coordinate {vertical_coordinate!r}; use 'sigma_eulerian' or 'ale'")
        self.ale = vertical_coordinate == "ale"
        if not self.ale:
            return
        if regrid not in ("sigma", "zstar", "hybrid"):
            raise ValueError(f"unknown regrid rule {regrid!r}; use 'sigma', 'zstar' or 'hybrid'")
        if remap_scheme not in REMAP_SCHEMES:
            raise ValueError(f"unknown remap_scheme {remap_scheme!r}; use one of {sorted(REMAP_SCHEMES)}")
        if self.open:
            raise ValueError("open boundaries with vertical_coordinate='ale' are not implemented yet (ALE Plan C)")
        if z0 is not None:
            raise ValueError("vertical_coordinate='ale' needs z0=None for now: the law-of-the-wall drag uses "
                             "the bed layer, which can vanish under z*")
        if closure not in ("constant", "none"):
            raise ValueError("vertical_coordinate='ale' needs closure 'constant' or 'none' for now "
                             "(GLS and the parabolic ν assume σ-layers)")
        params = dict(regrid_params or {})
        params.setdefault("dsigma", self.ds)
        for key in {"sigma": (), "zstar": ("dz_top",), "hybrid": ("dz_top", "rho_target")}[regrid]:
            if key not in params:
                raise ValueError(f"regrid={regrid!r} needs regrid_params[{key!r}]")
        floodable = act & (np.nan_to_num(np.asarray(self.H, float), nan=1.0) <= 0)
        if floodable.any() and (regrid == "zstar" or (regrid == "hybrid" and float(params.get("d_hybrid", 0.0)) <= 0)):
            raise ValueError("cells with H ≤ 0 can dry: use regrid='sigma', or 'hybrid' with a shallow σ zone "
                             "(regrid_params d_sigma, d_hybrid > 0) covering them")
        self.regrid_rule, self.regrid_params = regrid, params
        self.remap_code = REMAP_SCHEMES[remap_scheme]
        self.h_thin_bed, self.init_iterations = float(h_thin_bed), int(init_iterations)
```

4. **Geometría.** Sustituir `initial_state`, `layer_heights`, `layer_volumes`, `tracer_content` y `rpe` por:

```python
    def initial_state(self, eta: np.ndarray, T: np.ndarray) -> SigmaState:
        """
        Resting state; T on the layers of ``layer_heights(eta)``.  With ``rho_ref="auto"`` the reference
        profile is taken from this T.  ALE: h is the rule's grid (z* for 'hybrid', followed by
        ``init_iterations`` regrid–remap passes towards the isopycnal grid).
        """
        ny, nx = self.H.shape
        if isinstance(self.rho_ref, str) and self.rho_ref == "auto":
            z = self.layer_heights(eta)
            act = np.broadcast_to(self.fs.active[None], z.shape)
            self.rho_ref = reference_profile(z[act], self.density(np.asarray(T, float))[act])
        self.T_ext = np.array(T, float)                                # inflow / sponge target at open boundaries
        st = SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                        v=np.zeros((self.n, ny + 1, nx)), T=np.array(T, float),
                        h=self._initial_h(np.asarray(eta, float)) if self.ale else None)
        if self.ale and self.regrid_rule == "hybrid":
            for _ in range(self.init_iterations):
                st.h, (st.T,) = self._regrid_remap(st.eta, st.h, [st.T], st.T)
        return st

    def layer_heights(self, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """
        Height of the layer centres [m] (z = 0 at the rest surface), (N, ny, nx): σ-layers
        z_k = −H + D σ_c,k, or the centres of thicknesses h (ALE; default the initial grid).
        """
        if h is None and self.ale:
            h = self._initial_h(np.asarray(eta, float))
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            return -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]
        zw = -np.nan_to_num(self.H)[None] + np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, axis=0)])
        return 0.5 * (zw[1:] + zw[:-1])

    def layer_volumes(self, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Layer volumes [m³]: D Δσ_k A, or h_k A (ALE)."""
        if h is not None:
            return h * self.area[None]
        D = np.maximum(np.nan_to_num(self.H + eta, nan=0.0), 0.0)
        return D[None] * self.ds[:, None, None] * self.area[None]

    def tracer_content(self, st: SigmaState, power: int = 1) -> float:
        V = self.layer_volumes(st.eta, st.h)
        return float(np.sum(V * st.T**power))

    def rpe(self, st: SigmaState) -> float:
        """Reference potential energy [J] (flat-bottom basin: area = wet area)."""
        V = self.layer_volumes(st.eta, st.h)
        act = np.broadcast_to(self.fs.active[None], V.shape)
        return reference_potential_energy(self.density(st.T)[act], V[act], area=float(self.area[self.fs.active].sum()))

    # ------------------------------------------------------------------ ALE helpers
    def _initial_h(self, eta: np.ndarray) -> np.ndarray:
        """ALE layer thicknesses of the rule's grid at η [m] (z* for 'hybrid', σ where H ≤ 0); 0 on dry/land."""
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.fs.active & (H + eta > 0)
        h0 = np.zeros((self.n,) + H.shape)
        rule = "zstar" if self.regrid_rule == "hybrid" else self.regrid_rule
        sig = wet & (H <= 0) if rule == "zstar" else np.zeros_like(wet)
        for r, mask in ((rule, wet & ~sig), ("sigma", sig)):
            if mask.any():
                zw = rg.regrid_columns(r, H, eta, h0, None, self.regrid_params, mask)
                h0 = np.where(mask[None], np.diff(zw, axis=0), h0)
        return h0

    def _regrid_remap(self, eta: np.ndarray, h_src: np.ndarray, fields: list, T: np.ndarray):
        """Regrid with the rule (spec §3) at surface η and remap ``fields`` (spec §4): (h_new, remapped)."""
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.fs.active & (H + eta > 0) & (h_src.sum(axis=0) > 0)
        rho = self.density(T) if self.regrid_rule == "hybrid" else None
        zw = rg.regrid_columns(self.regrid_rule, H, eta, h_src, rho, self.regrid_params, wet)
        h_new = np.where(wet[None], np.diff(zw, axis=0), h_src)
        numba.set_num_threads(_THREADS[0])
        return h_new, [rm.remap_columns(h_src, f, h_new, self.remap_code, wet) for f in fields]

    def _remap_faces(self, vel: np.ndarray, hf_src: np.ndarray, hf_tgt: np.ndarray) -> np.ndarray:
        """
        Remap face velocities between two face-layer geometries in fraction space: thicknesses
        w_src S → w_tgt S with the same column total S, so Σ w u (the depth mean of the free
        surface) is conserved exactly (spec §2 step 4).
        """
        tot = hf_src.sum(axis=0)
        wet = tot > 0
        S = np.where(wet, tot, 1.0)[None]
        return rm.remap_columns(face_fractions(hf_src) * S, vel, face_fractions(hf_tgt) * S, self.remap_code, wet)
```

5. **Gradiente de presión.** Al principio de `_baroclinic_pgf`, cambiar la firma a `_baroclinic_pgf(self, eta, T, h=None)` y sustituir las líneas `n = self.n` … `z = -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]` por:

```python
        if self.pgf == "fv":
            return self._fv_pgf(eta, T, h)
        n = self.n
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            h = D[None] * self.ds[:, None, None]
            z = -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]   # cell-centre height (z = 0 at rest)
        else:
            z = self.layer_heights(eta, h)
```

   Añadir el método:

```python
    def _fv_pgf(self, eta: np.ndarray, T: np.ndarray, h: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Finite-volume PGF (core.vertical.pgf) on σ-layers (h=None) or ALE layers; ρ_ref tabulated."""
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            h = D[None] * self.ds[:, None, None]
        ref = self.rho_ref
        if ref is None or isinstance(ref, str):
            zr, rr = np.array([-1.0, 1.0]), np.full(2, self.rho0)
        elif isinstance(ref, ReferenceProfile):
            zr, rr = ref.z, ref.rho
        else:
            zr = np.linspace(-float(np.nanmax(self.H)) - 10.0, 10.0, 4001)
            rr = np.asarray(ref(zr), float)
        numba.set_num_threads(_THREADS[0])
        gx, gy = fv_pgf(h, np.nan_to_num(self.H), np.nan_to_num(eta), np.nan_to_num(T), self.fs.du, self.fs.dv,
                        self.rho0, self.t_coef, self.t_ref, zr, rr, self.g)
        return (np.where(np.isfinite(self.fs.H_u)[None], gx, 0.0),
                np.where(np.isfinite(self.fs.H_v)[None], gy, 0.0))
```

   (`ReferenceProfile` se define en este mismo módulo antes de la clase. `rho_ref == "auto"` antes de `initial_state` no tiene perfil todavía, así que se usa ρ₀.)

6. **`_tendencies`:** cambiar `gu, gv = self._baroclinic_pgf(st.eta, st.T)` por `gu, gv = self._baroclinic_pgf(st.eta, st.T, st.h)`, y `V = self.layer_volumes(st.eta)` por `V = self.layer_volumes(st.eta, st.h)`.

7. **`_diffuse_h`:** cambiar la firma a `_diffuse_h(self, T, V, eta, h=None)`:
   - en la línea de `Tw`, `self.layer_heights(eta)` pasa a `self.layer_heights(eta, h)`;
   - sustituir las dos asignaciones `Fx[..., 1:-1] = ...` y `Fy[:, 1:-1, :] = ...` por:

```python
        if h is None:
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu * np.nan_to_num(Du) / self.fs.du)[None, :, 1:-1] * ds \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv * np.nan_to_num(Dv) / self.fs.dv)[None, 1:-1, :] * ds \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
        else:                                                        # ALE: face layer thickness (min of the cells)
            hu, hv = face_thickness(h, self.fs.active)
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu / self.fs.du)[None, :, 1:-1] * hu[..., 1:-1] \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv / self.fs.dv)[None, 1:-1, :] * hv[:, 1:-1, :] \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
```

   Actualizar el docstring: "with h_{u,k} = D_u Δσ_k (σ) or the face layer thickness min(h_L, h_R) (ALE)".

- [ ] **Step 4: Implementar el error de restart en `core/restart.py`**

Al principio de `write_restart` y de `read_restart`:

```python
    if getattr(model, "ale", False):
        raise NotImplementedError("restart of an ALE model (vertical_coordinate='ale') needs h and is not "
                                  "implemented yet (ALE Plan D)")
```

- [ ] **Step 5: Correr los tests y ver que pasan**

Run: `pytest tests/test_sigma_ale.py tests/test_restart.py tests/test_sigma_baroclinic.py -q 2>&1 | tail -5`
Expected: todo pasa. Son 13 tests nuevos en `test_sigma_ale.py` (7 de configuración + 6), 7 + 1 en restart y 37 en σ sin cambios.

Si `test_fv_pgf_error_over_the_seamount_is_not_larger_than_the_jacobian` falla:
1. imprimir los dos errores;
2. comprobar el port contra el §"Algoritmo" de la Task 4 con superpowers:systematic-debugging;
3. si el port es correcto, el resultado es un hallazgo, no un fallo. Anotar `Ruling:` con los dos valores y cambiar la aserción a `err["fv"] <= 1.5 * err["jacobian"]` solo si Esteban lo aprueba. Es una de las paradas "plan roto" de la ejecución.

- [ ] **Step 6: Suite completa y commit**

Run: `pytest tests/ -q -x 2>&1 | tail -3`
Expected: todo pasa.

```bash
git add core/dynamics/sigma_baroclinic.py core/restart.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "SigmaBaroclinic: ALE configuration, prognostic h, geometry from h, finite-volume PGF option"
```

---

### Task 6: Paso ALE (lagrangiano + regeneración + remapeo)

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (`step` y nuevo `_step_ale`)
- Modify: `tests/test_sigma_ale.py` (añadir tests)

**Interfaces:**
- Consumes: Task 5 (`_regrid_remap`, `_remap_faces`, `layer_volumes(eta, h)`) y Tasks 2–3.
- Produces: `SigmaBaroclinic.step(st)` en modo ALE devuelve un `SigmaState` con:
  - `h` (Σh = H + η^{n+1});
  - `Fz` = 0;
  - `omega` = None;
  - `mix_num` = mezcla numérica de la advección más la del remapeo [T² m³ s⁻¹].

**El paso** (spec §2):
1. Tendencias y AB3 como en σ, con la geometría de h.
2. `core.step` con w y h en las caras de hⁿ; después, velocidad cero en las capas de fondo delgadas (sin cambiar Σ w u).
3. Q_k = L·D_f·w_k·(θu^{n+1} + (1 − θ)uⁿ); h† = hⁿ − Δt ∇·Q_k / A; advección SSP-RK3 de [T, T²] con Fz = 0 y V de A·hⁿ a A·h†.
4. Difusión horizontal y vertical sobre h†.
5. Regeneración de la malla y remapeo de [T, T²] (T² para el diagnóstico de mezcla del remapeo).
6. Remapeo de u y v en espacio de fracciones, de las caras de h† a las de h^{n+1}; velocidad cero en capas de fondo delgadas.

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_sigma_ale.py`:

```python
# ---------------------------------------------------------------- the ALE step
@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_uniform_tracer_stays_uniform_and_is_conserved_while_the_surface_sloshes(rule):
    m = _ale_box(rule)
    x = (np.arange(32) + 0.5) * 2000.0
    eta = 0.5 * np.cos(np.pi * x / 64e3)[None, :]
    st = m.initial_state(eta, np.full((10, 1, 32), 7.0))
    c0 = m.tracer_content(st)
    for _ in range(200):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-13
    assert np.abs(st.eta).max() > 0.1


@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_resting_stratification_over_a_flat_bottom_stays_at_rest(rule):
    m = _ale_box(rule)
    z = m.layer_heights(np.zeros((1, 32)))
    st = m.initial_state(np.zeros((1, 32)), 5.0 + 0.5 * (z + 20.0))
    for _ in range(100):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-12 and np.abs(st.eta).max() < 1e-12


def test_online_mixing_includes_the_remap_and_matches_the_global_variance_loss():
    nx, nz = 40, 10
    m = _ale_box("zstar", nx=nx, nz=nz, dx=1000.0, dt=20.0, ah=1.0)
    T = np.where(np.arange(nx) < nx // 2, 10.0, 0.0)[None, None, :].repeat(nz, 0)
    st = m.initial_state(np.zeros((1, nx)), T)
    for _ in range(50):
        var0 = m.tracer_content(st, power=2)
        st = m.step(st)
        lost = (var0 - m.tracer_content(st, power=2)) / m.dt
        assert np.isclose(st.mix_num.sum(), lost, rtol=1e-9, atol=1e-12 * abs(var0))
    assert st.mix_num.sum() > 0


def test_sigma_ale_on_a_drying_beach_stays_positive_and_uniform():
    # review focus 1: cells that dry and flood under ALE σ
    nx = 30
    H = np.linspace(4.0, -0.5, nx)[None, :]
    m = _ale_box("sigma", nx=nx, nz=4, dx=200.0, dt=5.0, bathy=H, params={})
    x = np.arange(nx)
    eta = np.maximum(0.4 * np.cos(np.pi * x / nx)[None, :], -H + 0.0)
    st = m.initial_state(eta, np.full((4, 1, nx), 3.0))
    c0 = m.tracer_content(st)
    for _ in range(300):
        st = m.step(st)
        assert np.isfinite(st.h).all() and (st.h >= 0).all()
        np.testing.assert_allclose(st.h.sum(0), np.maximum(H + st.eta, 0.0), atol=1e-9)
    wet = (H + st.eta)[0] > 0.05
    np.testing.assert_allclose(st.T[:, 0, wet], 3.0, atol=1e-10)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-10      # drying cells: h† round-off clipped at 0
```

- [ ] **Step 2: Correr los tests y ver que fallan**

Run: `pytest tests/test_sigma_ale.py -q 2>&1 | tail -8`
Expected: FAIL en los 6 tests nuevos. `step` sigue el camino σ y no actualiza `h` (`st.h` queda fijo, Σh ≠ H + η).

- [ ] **Step 3: Implementar**

En `step`, justo después de las líneas `tu = ...` / `tv = ...`, añadir:

```python
        if self.ale:
            return self._step_ale(st, gu_ab, gv_ab, tu, tv, hist)
```

y el método:

```python
    def _step_ale(self, st: SigmaState, gu_ab: np.ndarray, gv_ab: np.ndarray, tu, tv, hist: list) -> SigmaState:
        """
        ALE step (spec §2): (1) Lagrangian step with no flux through the layer interfaces —
        free surface with the depth mean ū = Σ w_k u_k on the face fractions of hⁿ, layer transports
        Q_k = L D_f w_k (θ u_k^{n+1} + (1−θ) u_k^n) that add up to the free-surface flux, layer
        continuity h† = hⁿ − Δt ∇·Q_k / A, flux-form tracer advection with Fz = 0; (2) vertical
        physics on h†; (3) regrid with the rule; (4) conservative remap of T (and T² for the
        mixing diagnostic) and of u, v in face-fraction space.  Velocity is zero in thin bed
        layers of each face (HYCOM hybgen.F90:1757-1764).
        """
        dt, th, n = self.dt, self.fs.theta, self.n
        act = self.fs.active
        hu, hv = face_thickness(st.h, act)
        wu, wv = face_fractions(hu), face_fractions(hv)
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                               wu=wu, wv=wv, hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR))
        u_new = zero_thin_bed_velocity(u_new, hu, wu, self.h_thin_bed)
        v_new = zero_thin_bed_velocity(v_new, hv, wv, self.h_thin_bed)
        un = np.where(self.fs.last_wet_u[None], st.u, 0.0)
        vn = np.where(self.fs.last_wet_v[None], st.v, 0.0)
        Qx = (self.fs.Lu * self.fs.last_Du)[None] * wu * (th * u_new + (1 - th) * un)
        Qy = (self.fs.Lv * self.fs.last_Dv)[None] * wv * (th * v_new + (1 - th) * vn)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        h_lag = lagrangian_thickness(st.h, divh, dt, self.area)
        V0, V1 = self.layer_volumes(st.eta, st.h), self.layer_volumes(eta_new, h_lag)
        Fz = np.zeros((n + 1,) + self.H.shape)                              # Lagrangian: no flux through interfaces
        ok = np.broadcast_to(act[None], V1.shape) & (V1 > 0)
        C = self._advect(np.stack([st.T, st.T**2]), V0, V1, Qx, Qy, Fz, ok)
        mix = V1 * (C[1] - C[0] ** 2) / dt                                  # advection part
        T_lag = C[0]
        if self.kh > 0:
            T_lag = self._diffuse_h(T_lag, V1, eta_new, h_lag)
        if self.kv > 0:
            dz = np.maximum(h_lag, DZ_FLOOR).reshape(n, -1)
            T_lag = vertical_implicit(T_lag.reshape(n, -1), dz, np.full((n + 1, dz.shape[1]), self.kv),
                                      np.zeros(dz.shape[1]), dt).reshape(T_lag.shape)
        h_new, (T_new, T2_new) = self._regrid_remap(eta_new, h_lag, [T_lag, T_lag**2], T_lag)
        mix = mix + self.layer_volumes(eta_new, h_new) * (T2_new - T_new**2) / dt   # remap part
        hu_lag, hv_lag = face_thickness(h_lag, act)
        hu1, hv1 = face_thickness(h_new, act)
        u_new = zero_thin_bed_velocity(self._remap_faces(u_new, hu_lag, hu1), hu1, face_fractions(hu1), self.h_thin_bed)
        v_new = zero_thin_bed_velocity(self._remap_faces(v_new, hv_lag, hv1), hv1, face_fractions(hv1), self.h_thin_bed)
        return SigmaState(eta=eta_new, u=u_new, v=v_new, T=T_new, h=h_new, g_hist=hist, omega=None, mix_num=mix,
                          Qx=Qx, Qy=Qy, Fz=Fz, time=st.time + dt)
```

Notas para quien implemente:
- **`mix` y la difusión:** el término de advección se toma antes de la difusión explícita, como en el modo σ (allí `mix` se calcula antes de `_diffuse_h`).
- **El test de mezcla usa `ah` y no `kh`:** la varianza perdida coincide con advección + remapeo.
- **Si se usa `kh` o `kv`,** la difusión física también reduce la varianza y no entra en `mix_num`, igual que en el modo σ.

- [ ] **Step 4: Correr los tests y ver que pasan**

Run: `pytest tests/test_sigma_ale.py -q 2>&1 | tail -5`
Expected: todo pasa.

Si la conservación del contenido queda entre 1e-13 y 1e-12:
1. imprimir el error relativo;
2. revisar si `lagrangian_thickness` recortó algún valor (añadir temporalmente un print del mínimo de `h - dt*divq/A`);
3. si es redondeo de la suma, anotar `Ruling:` con el valor y usar 1e-12, la tolerancia del test σ equivalente de lock exchange.

- [ ] **Step 5: Suite completa y commit**

Run: `pytest tests/ -q -x 2>&1 | tail -3`
Expected: todo pasa.

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "SigmaBaroclinic ALE step: Lagrangian step, regrid, conservative remap of tracers and velocities"
```

---

### Task 7: Regresión ALE-σ contra el modelo σ y prueba de la regla híbrida en el modelo

**Files:**
- Modify: `tests/test_sigma_ale.py` (añadir tests)
- Modify: `core/dynamics/sigma_baroclinic.py` solo si los tests destapan un error (con TDD)

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces: tests de regresión (spec §7, integración 1).

- [ ] **Step 1: Escribir los tests**

```python
# ---------------------------------------------------------------- regression ALE-σ vs σ (spec §7, integration 1)
def test_ale_sigma_lock_exchange_matches_benjamin_and_conserves_heat():
    nx, nz, depth, dt = 128, 20, 20.0, 20.0
    m = _ale_box("sigma", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={})
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF
    st = m.initial_state(np.zeros((1, nx)), T)
    c0, r0 = m.tracer_content(st), m.rpe(st)
    x = (np.arange(nx) + 0.5) * 500.0
    half = 7.5 / TCOEF
    fronts = {}
    for n in range(int(6 * 3600 / dt)):
        st = m.step(st)
        t = (n + 1) * dt
        if t in (2 * 3600.0, 6 * 3600.0):
            fronts[t] = (x[st.T[0, 0] < half].max() - 32e3, 32e3 - x[st.T[-1, 0] > half].min())
    U = 0.5 * np.sqrt(9.81 * 5.0 / RHO0 * depth)
    for k in range(2):
        speed = (fronts[6 * 3600.0][k] - fronts[2 * 3600.0][k]) / (4 * 3600)
        assert abs(speed / U - 1) < 0.10
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-12
    assert m.rpe(st) >= r0


def test_ale_sigma_internal_wave_matches_the_eulerian_sigma_model():
    # mode-1 internal wave in a closed 60 km channel (N = 0.02 s⁻¹, 200 m): after 2 h the ALE-σ
    # temperature differs from the Eulerian σ model by much less than the wave signal
    from tests.test_sigma_baroclinic import _channel
    m_s, st_s = _channel(60, 30e3)
    m_a, st_a = _channel(60, 30e3, vertical_coordinate="ale", regrid="sigma")
    T0 = st_s.T.copy()
    for _ in range(60):
        st_s = m_s.step(st_s)
        st_a = m_a.step(st_a)
    signal = np.abs(st_s.T - T0).max()
    assert signal > 0.05
    assert np.abs(st_a.T - st_s.T).max() < 0.1 * signal


# ---------------------------------------------------------------- hybrid rule in the model
def test_hybrid_ale_survives_an_unstable_patch_and_conserves_heat():
    # review focus 3: warm water under cold water in some columns; the regrid monotonizes
    nx, nz, depth = 40, 10, 20.0
    T0 = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF
    T0[:2, 0, 5:9] = 12.0 / TCOEF                                       # warm (light) at the bed
    rho = RHO0 - TCOEF * T0
    params = dict(dz_top=np.full(nz, 2.0), rho_target=rg.target_densities(rho[:, 0, :].ravel(), nz, n_light=2))
    m = _ale_box("hybrid", nx=nx, nz=nz, depth=depth, dx=500.0, dt=20.0, params=params)
    st = m.initial_state(np.zeros((1, nx)), T0)
    c0 = m.tracer_content(st)
    for _ in range(300):
        st = m.step(st)
        assert np.isfinite(st.T).all() and (st.h >= 0).all()
        np.testing.assert_allclose(st.h.sum(0), depth + st.eta, rtol=1e-12)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-12
```

`_channel` (en `tests/test_sigma_baroclinic.py`) pasa `**kw` a `SigmaBaroclinic` y fija `z0=None` y `closure="constant"`, así que acepta el modo ALE sin cambios.

- [ ] **Step 2: Correr los tests**

Run: `pytest tests/test_sigma_ale.py -q 2>&1 | tail -8`
Expected: PASS. Estos tests comprueban comportamiento ya implementado en la Task 6. Para verificar que discriminan, aplicar una mutación temporal en `_step_ale`: saltar el remapeo del trazador (`T_new = T_lag` después de `_regrid_remap`). Con ella deben fallar los tests de calor de lock exchange y de la híbrida. Anotar el resultado en el ledger y revertir la mutación.

Si un test falla sin mutación, usar superpowers:systematic-debugging, corregir con un test RED → GREEN y anotar la corrección.

- [ ] **Step 3: Registrar la comparación ALE-σ / σ**

Run:

```bash
python - <<'EOF'
import numpy as np, sys
sys.path.insert(0, ".")
from tests.test_sigma_baroclinic import _box
from tests.test_sigma_ale import _ale_box, TCOEF
for name, m in (("sigma", _box(nx=128, nz=20, depth=20.0, dx=500.0, dt=20.0)),
                ("ale-sigma", _ale_box("sigma", nx=128, nz=20, depth=20.0, dx=500.0, dt=20.0, params={}))):
    T = np.where(np.arange(128) < 64, 5.0, 10.0)[None, None, :].repeat(20, 0) / TCOEF
    st = m.initial_state(np.zeros((1, 128)), T); r0 = m.rpe(st)
    for _ in range(1080): st = m.step(st)
    print(name, "RPE gain [J]", m.rpe(st) - r0)
EOF
```

Expected: dos números positivos. Anotarlos en el ledger como "lock exchange 6 h: ganancia de RPE σ = …, ALE-σ = …". Van a `CLAUDE.md` en la Task 8. No es un criterio de aprobación, sino información para el Plan D (mezcla espuria del remapeo frente al WENO vertical, spec §6.2).

- [ ] **Step 4: Commit**

```bash
git add tests/test_sigma_ale.py
git commit -m "ALE regression: sigma-ALE lock exchange and internal wave vs Eulerian sigma; hybrid smoke test"
```

---

### Task 8: Escalón 600 → 164 m, monte submarino y cierre del Plan B

**Files:**
- Modify: `tests/test_sigma_ale.py` (test del escalón)
- Modify: `validation/seamount/run_seamount.py`, `validation/seamount/README.md`
- Modify: `CLAUDE.md` (filas de estado y total de tests), `docs/design/vision_roadmap.md` §7
- Modify: la memoria `project_status.md` (fuera del repositorio)

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: el test de integración 2 del §7 de la especificación, la validación del monte submarino con ALE y la documentación.

- [ ] **Step 1: Escribir el test del escalón**

Añadir a `tests/test_sigma_ale.py`:

```python
# ---------------------------------------------------------------- bay-mouth step (spec §7, integration 2)
def _step_section(coord, nz=30):
    """2-D section across a 600 → 164 m step (BayIW bay mouth), at rest with a tanh pycnocline at 80 m."""
    nx, dx = 40, 1000.0
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.where(np.arange(nx) < nx // 2, 600.0, 164.0)[None, :]
    kw = dict(pgf="jacobian")
    if coord == "zstar":
        kw = dict(pgf="fv", vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, 600.0 / nz)))
    m = SigmaBaroclinic(hg, H, dt=60.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, rho_ref="auto", **kw)
    z = m.layer_heights(np.zeros((1, nx)))
    return m, m.initial_state(np.zeros((1, nx)), 15.0 + 5.0 * np.tanh((z + 80.0) / 30.0))


def test_bay_mouth_step_spurious_flow_sigma_vs_zstar():
    # σ-layers over rx0 = 0.57 drive a spurious current (the BayIW problem); z* with the
    # finite-volume PGF keeps the resting state
    umax = {}
    for coord in ("sigma", "zstar"):
        m, st = _step_section(coord)
        for _ in range(720):                                            # 12 h
            st = m.step(st)
        assert np.isfinite(st.u).all()
        if coord == "zstar":
            assert (st.h >= 0).all()
        umax[coord] = np.abs(st.u).max()
    assert umax["zstar"] < 0.05 * umax["sigma"]
    assert umax["zstar"] < 1e-3
```

- [ ] **Step 2: Correr el test y registrar los valores**

Run: `pytest tests/test_sigma_ale.py::test_bay_mouth_step_spurious_flow_sigma_vs_zstar -q -s 2>&1 | tail -5`

(Añadir temporalmente `print(umax)` antes de las aserciones para registrar los valores, y quitarlo después.)

Expected: PASS. Anotar en el ledger los dos `umax`.

Si z\* no queda por debajo del 5 % de σ:
1. usar superpowers:systematic-debugging;
2. localizar la capa y la cara del máximo (celda parcial, capa desvanecida, ponderación por masa);
3. clasificar el resultado según `CLAUDE.md` (a)–(d) y corregir solo si es (a) o (b), con un test RED → GREEN.

La especificación pide "nivel de redondeo" para z\*. Si queda un error pequeño y explicado por la celda parcial, es un hallazgo para Esteban (parada "plan roto"), no una tolerancia que se ajusta en silencio.

- [ ] **Step 3: Monte submarino con ALE**

En `validation/seamount/run_seamount.py`:
- añadir `ap.add_argument("--coord", default="sigma", choices=["sigma", "zstar"])`;
- pasar `coord` a `setup(dx, n_layers, dt, pgf, coord)`;
- dentro de `setup`, con `coord == "zstar"`, construir con `vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(n_layers, H0 / n_layers))`;
- calcular las alturas con `z = m.layer_heights(np.zeros((ny, nx)))` en lugar de la fórmula σ;
- en `main`, con `--ref`, usar `z = m.layer_heights(st.eta, st.h)` en lugar de la fórmula σ;
- poner el sufijo `tag = args.pgf + ("_ref" if args.ref else "") + ("_zstar" if args.coord == "zstar" else "")`;
- añadir `--pgf fv --coord zstar` a la sección Usage del docstring.

Run (una corrida a la vez, cada una con los 5 núcleos P):

```bash
cd /Users/estebancruz/pyOCEAN
python -c "import numba; print(numba.config.NUMBA_NUM_THREADS)"
for args in "--pgf fv --ref" "--pgf fv --ref --coord zstar" "--pgf fv --coord zstar"; do
  NUMBA_NUM_THREADS=5 python validation/seamount/run_seamount.py $args | tail -3
done
```

Expected: tres resúmenes en `validation/seamount/out/summary_*.txt`. Añadirlos a la tabla de `validation/seamount/README.md`, con sus filas y una nota breve:
- FV + perfil en σ frente al jacobiano + perfil (0.17 / 0.21 / 0.41 cm/s);
- z\* con y sin perfil;
- clasificación (a)–(d) de cualquier diferencia;
- lista "qué debe mejorar pyOCEAN".

Si el kernel de `run_seamount.py` usa `set_threads`, poner `set_threads(5)`. Si no, `NUMBA_NUM_THREADS=5` basta para los kernels con `prange`.

- [ ] **Step 4: Documentación**

1. En `CLAUDE.md`, tabla de la fase 2, añadir las filas:

```
| ALE face geometry (min face thickness = MITgcm hFacW, fractions, thin bed layers = HYCOM, Lagrangian continuity) | `core/vertical/ale.py` | `tests/test_vertical_ale.py` | ✅ 6/6 |
| Finite-volume PGF (Adcroft et al. 2008, port of MOM6 MOM_PressureForce_FV + density integrals, PLM T, Boole, bottom mass weighting; exact for linear ρ on any layers) | `core/vertical/pgf.py` | `tests/test_vertical_pgf.py` | ✅ 5/5 |
| ALE mode of the stratified model (prognostic h, Lagrangian step + regrid + remap each step, σ/z*/hybrid, FV PGF; closed domains, z0=None, constant ν; ALE-σ lock exchange = Benjamin, RPE gain σ … / ALE-σ …; bay-mouth step 600→164 m: max|u| σ … vs z* … ) | `core/dynamics/sigma_baroclinic.py` | `tests/test_sigma_ale.py` | ✅ N/N |
```

   Rellenar los "…" y N con los valores de las Tasks 7 y 8. Actualizar también:
   - la fila de regrid ("ALE grid rules …") para mencionar la ventana y los objetivos ligeros, con su nuevo conteo;
   - la fila de `SigmaWetDry` (+2);
   - la de restart (+1);
   - el total "**Phase 2 in progress — N total tests passing (+1 strict xfail).**" con el número que dé `pytest tests/ -q | tail -1`.

2. En `docs/design/vision_roadmap.md` §7, añadir o actualizar las filas del marcador:
   - "Gradiente de presión FV (seamount)" con los valores del Step 3;
   - "Escalón de la boca de BayIW (σ vs z\*)" con los del Step 2.

3. En la memoria (`/Users/estebancruz/.claude/projects/-Users-estebancruz-pyOCEAN/memory/project_status.md`), sección "Current (2026-10-11)": Plan B terminado, con los números clave. Lo siguiente es el Plan C (fronteras abiertas bajo ALE).

- [ ] **Step 5: Suite completa y commit**

Run: `pytest tests/ -q 2>&1 | tail -3`
Expected: todo pasa (+1 xfail).

```bash
git add tests/test_sigma_ale.py validation/seamount/run_seamount.py validation/seamount/README.md CLAUDE.md docs/design/vision_roadmap.md
git commit -m "ALE Plan B: bay-mouth step test, seamount with FV PGF and z*, docs"
```
