# ALE Plan A — Kit de columna (remapeo y reglas de malla) validado contra MOM6

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Funciones por columna, compiladas con Numba, para:
- remapear conservativamente (PCM, PLM, PPM_H4);
- generar mallas z\*, σ e híbrida z\*–isopicna (HYCOM1).

Cada función se valida contra las rutinas Fortran originales de MOM6.

**Architecture:**
- Paquete nuevo `core/vertical/` con dos módulos puros:
  - `remap.py`: reconstrucción y remapeo de una columna, más un driver 3-D con prange;
  - `regrid.py`: reglas de malla por columna, más un driver 3-D.
- No tocan el modelo: la integración en `SigmaBaroclinic` es el Plan B.
- La validación de nivel 1 compila las rutinas de MOM6 en pequeños programas Fortran (`validation/mom6_column/`). Los tests de pytest comparan pyOCEAN contra su salida y se saltan si la compilación de referencia no existe.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba 0.59.1, pytest; gfortran 15 + MPI + netCDF-Fortran (Homebrew) para el arnés; MOM6 (commit 0e9a561) y FMS 2025.02.01.

**Spec:** `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` (§3, §4, §7, §8 nivel 1, §9 fases 1–2).

**Planes siguientes** (se escriben al terminar cada uno):
- B: integración en el modelo (h pronóstico, paso lagrangiano, gradiente de volumen finito, escalón);
- C: fronteras abiertas;
- D: validación con modelos completos, BayIW y cierre. PQM entra en el Plan D, junto al test de mezcla espuria, según el §4 de la especificación.

## Global Constraints

- **Columnas** ordenadas con k = 0 en el fondo (convención del modelo). Las interfaces se dan como alturas z_w (N+1 valores, del fondo z_w[0] = −H a la superficie z_w[N] = η, con z = 0 en el reposo). MOM6 usa la superficie primero: los drivers invierten el orden.
- **Σh** del destino = Σh del origen (las reglas no cambian el grosor total). El remapeo conserva Σ h·u al redondeo cuando las sumas coinciden.
- **Grosor mínimo** por defecto `H_MIN = 1e-3` m, efectivo min(H_MIN, D/N).
- **Grosor despreciable** para denominadores: `H_NEGLECT = 1e-30` m. En reconstrucciones de borde: `H_NEGLECT_EDGE = 1e-10` m (los valores de MOM6 con answer_date ≥ 2019).
- **Fórmulas de MOM6** en su variante `answer_date ≥ 20190101` (expresiones "menos sensibles al redondeo").
- **Hilos:** como mucho 5 (núcleos P). Numba con `@njit(cache=True)` y `prange` solo en los drivers 3-D. Nada de E-cores ni prioridad reducida.
- **Entorno:** numpy debe quedar en 1.26.4. No se instala nada con pip. Las herramientas del sistema (Homebrew) solo con aprobación explícita de Esteban (Task 5).
- **Commits** solo cuando Esteban lo autorice. Los pasos "Commit" de cada tarea se ejecutan solo si autorizó commits por tarea al elegir la ejecución. Los mensajes terminan con las dos líneas de atribución de la sesión.
- **Documentación:** cada ecuación lleva su referencia (artículo y/o `archivo:línea` de MOM6). Docstrings con la física.

## Review Focus

1. **Columnas con capas de grosor cero** en medio o en los extremos (z\* en aguas someras). Se espera: sin NaN ni división por cero, conservación exacta, y el valor de una capa vacía de destino igual al valor de la reconstrucción en esa posición. → Tests en Task 1 y Task 4.
2. **Columnas de 1, 2 o 3 capas activas** (bahía somera, plataforma). Se espera: el esquema baja de orden automáticamente (PCM con 1, PLM con ≤ 3) sin error. → Test en Task 4.
3. **Perfil de densidad no monótono o completamente mezclado** en la regla híbrida. Se espera: se monotoniza, no falla la búsqueda, y las interfaces caen en el piso z\*. → Test en Task 7.
4. **Densidades objetivo fuera del rango de la columna.** Se espera: las interfaces van al fondo o a la superficie y el grosor mínimo las separa sin cruzarse. → Test en Task 7.
5. **Σh del destino distinto del origen por redondeo** (~1e-13 m). Se espera: no se pierde ni se crea contenido más allá de ese redondeo y no hay valores fuera de rango. → Test en Task 1.

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `core/vertical/__init__.py` (nuevo) | Exporta la API pública del paquete |
| `core/vertical/remap.py` (nuevo) | Reconstrucciones PCM/PLM/PPM_H4 y remapeo conservativo por columna; driver 3-D `remap_columns` |
| `core/vertical/regrid.py` (nuevo) | Reglas `zstar_column`, `sigma_column`, `hycom1_column`, mezcla σ somera, densidades objetivo; driver 3-D `regrid_columns` |
| `tests/test_vertical_remap.py` (nuevo) | Tests unitarios del remapeo |
| `tests/test_vertical_regrid.py` (nuevo) | Tests unitarios de las reglas |
| `validation/mom6_column/build_reference.sh` (nuevo) | Compila FMS + MOM6 (objetos) y los drivers Fortran |
| `validation/mom6_column/remap_driver.F90` (nuevo) | Llama a `remapping_core_h` de MOM6 sobre columnas leídas de un archivo |
| `validation/mom6_column/regrid_driver.F90` (nuevo) | Llama a `build_zstar_column`, `build_sigma_column` y `build_and_interpolate_grid` de MOM6 |
| `validation/mom6_column/hybgen_driver.F90` (nuevo) | Llama al puerto de `hybgen` de MOM6 (`MOM_hybgen_regrid`) en columnas típicas |
| `validation/mom6_column/columns.py` (nuevo) | Columnas de prueba compartidas (formato de texto) y lectura de resultados |
| `validation/mom6_column/README.md` (nuevo) | Cómo compilar, qué se compara, resultados |
| `tests/test_vertical_vs_mom6.py` (nuevo) | Comparación pyOCEAN vs MOM6 (se salta sin la compilación) |
| `CLAUDE.md`, `docs/design/vision_roadmap.md` (modificar) | Filas de estado y tabla de §7 |

---

### Task 1: PCM y remapeo conservativo de una columna

**Files:**
- Create: `core/vertical/__init__.py`, `core/vertical/remap.py`
- Test: `tests/test_vertical_remap.py`

**Interfaces:**
- Produces:
  - `remap.PCM = 0`, `remap.PLM = 1`, `remap.PPM_H4 = 2`
  - `remap.SCHEMES: dict[str, int]` = `{"pcm": 0, "plm": 1, "ppm_h4": 2}`
  - `remap.H_NEGLECT = 1e-30`, `remap.H_NEGLECT_EDGE = 1e-10`
  - `remap.reconstruct(h: f8[:], u: f8[:], scheme: int, h_neglect: float, h_neglect_edge: float, coef: f8[:, 3]) -> None`. Coeficientes del polinomio en ξ ∈ [0, 1] de cada celda: p(ξ) = c0 + c1 ξ + c2 ξ².
  - `remap.remap_column(h_src: f8[:], u_src: f8[:], h_tgt: f8[:], scheme: int, u_tgt: f8[:], h_neglect: float = 1e-30, h_neglect_edge: float = 1e-10) -> None`

- [ ] **Step 1: Write the failing tests**

```python
"""
Tests for core/vertical/remap.py — conservative 1-D vertical remapping
(White & Adcroft 2008; MOM6 MOM_remapping.F90 remapping_core_h).
Columns are ordered with k = 0 at the bed; any consistent order works.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import remap as rm


def _remap(h0, u0, h1, scheme):
    u1 = np.empty(len(h1))
    rm.remap_column(np.asarray(h0, float), np.asarray(u0, float), np.asarray(h1, float), scheme, u1)
    return u1


@pytest.mark.parametrize("scheme", [rm.PCM])
def test_identity_when_grid_unchanged(scheme):
    rng = np.random.default_rng(0)
    h = rng.uniform(0.5, 3.0, 12)
    u = rng.normal(10, 2, 12)
    np.testing.assert_allclose(_remap(h, u, h, scheme), u, rtol=1e-14)


@pytest.mark.parametrize("scheme", [rm.PCM])
def test_conservation_to_roundoff(scheme):
    rng = np.random.default_rng(1)
    h0 = rng.uniform(0.0, 4.0, 20)
    h0[[3, 4, 11]] = 0.0                                   # vanished source layers
    u0 = rng.normal(15, 3, 20)
    h1 = rng.uniform(0.1, 2.0, 15)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, scheme)
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * abs((h0 * u0).sum())


def test_pcm_piecewise_constant_values():
    # source: two cells (1 m of 1.0, 1 m of 3.0); target: one 2 m cell and the reverse split
    np.testing.assert_allclose(_remap([1, 1], [1.0, 3.0], [2.0], rm.PCM), [2.0])
    np.testing.assert_allclose(_remap([1, 1], [1.0, 3.0], [0.5, 1.5], rm.PCM), [1.0, (0.5 + 3.0) / 1.5])


def test_vanished_target_takes_local_value():
    u1 = _remap([1.0, 1.0], [1.0, 3.0], [1.0, 0.0, 1.0], rm.PCM)
    assert np.isfinite(u1).all()
    np.testing.assert_allclose(u1[[0, 2]], [1.0, 3.0])
    assert 1.0 <= u1[1] <= 3.0


def test_roundoff_mismatch_in_total_thickness():
    h0 = np.array([1.0, 2.0, 3.0])
    u0 = np.array([4.0, 5.0, 6.0])
    h1 = np.array([3.0, 3.0 + 1e-13])                       # target 1e-13 m thicker
    u1 = _remap(h0, u0, h1, rm.PCM)
    assert np.all((u1 >= 4.0 - 1e-12) & (u1 <= 6.0 + 1e-12))
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) < 1e-11
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_remap.py -q`
Expected: ERROR, `ModuleNotFoundError: No module named 'core.vertical'`.

- [ ] **Step 3: Write the implementation**

`core/vertical/__init__.py`:

```python
"""
General vertical coordinate (ALE) column toolkit: conservative remapping and
grid-generation rules (docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md).
"""
```

`core/vertical/remap.py`:

```python
"""
Conservative vertical remapping of one column between two grids with the same
total thickness (vertical Lagrangian-remap method; Griffies et al. 2020).

Each source cell gets a polynomial reconstruction p(ξ), ξ ∈ [0, 1] across the
cell, with the cell mean as its average:

    PCM     p = ū                                              (piecewise constant)
    PLM     p = u_L + (u_R − u_L) ξ        White & Adcroft (2008); MOM6 PLM_functions.F90
    PPM_H4  p = u_L + [4(ū−u_L) + 2(ū−u_R)] ξ + 3[(u_R−ū) + (u_L−ū)] ξ²
            with explicit 4th-order edge values; Colella & Woodward (1984) limiter;
            MOM6 PPM_functions.F90:23-120, regrid_edge_values.F90:215-350

The target means come from integrating p over the union of both grids.  In
every source cell, the thickest sub-cell takes the residual u0·h0 − Σ(other
sub-cells) (the "Hallberg trick", MOM6 MOM_remapping.F90:942-961), so the
column content Σ h u is conserved to round-off.  Top and bottom cells are PCM
(no extrapolation; MOM6 default REMAP_BOUNDARY_EXTRAP = False).  Short columns
reduce the order: 1 active cell → PCM, ≤ 3 → PLM (MOM6 MOM_remapping.F90:453-460).

Columns can be in any consistent order (the model uses k = 0 at the bed).

References
----------
White, L. & Adcroft, A. (2008). J. Comput. Phys. 227, 7394–7422. doi:10.1016/j.jcp.2008.04.026
White, L., Adcroft, A. & Hallberg, R. (2009). J. Comput. Phys. 228, 8665–8692.
Colella, P. & Woodward, P. (1984). J. Comput. Phys. 54, 174–201.
Griffies, S.M. et al. (2020). JAMES 12, e2019MS001954.
MOM6 source: ~/pyOCEAN_refs/MOM6/src/ALE (commit 0e9a561).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

PCM, PLM, PPM_H4 = 0, 1, 2
SCHEMES = {"pcm": PCM, "plm": PLM, "ppm_h4": PPM_H4}
H_NEGLECT = 1e-30          # [m] negligible thickness for cell reconstructions
H_NEGLECT_EDGE = 1e-10     # [m] negligible thickness for edge values


@njit(cache=True)
def _effective_scheme(n_active: int, scheme: int) -> int:
    """Lower the order on short columns (MOM6 MOM_remapping.F90:453-460)."""
    if n_active <= 1:
        return PCM
    if n_active <= 3 and scheme > PLM:
        return PLM
    return scheme


@njit(cache=True)
def reconstruct(h, u, scheme, h_neglect, h_neglect_edge, coef):  # pragma: no cover
    """
    Polynomial coefficients coef[k] = (c0, c1, c2) of p(ξ) = c0 + c1 ξ + c2 ξ² in cell k [units of u].
    h : cell thicknesses [m];  u : cell means.
    """
    n = h.size
    for k in range(n):
        coef[k, 0] = u[k]
        coef[k, 1] = 0.0
        coef[k, 2] = 0.0


@njit(cache=True)
def _average(c0, c1, c2, xa, xb):
    """Mean of p(ξ) over [xa, xb] ⊂ [0, 1]; the point value when xb ≈ xa."""
    if xb - xa <= 1e-14:
        x = 0.5 * (xa + xb)
        return c0 + x * (c1 + x * c2)
    return c0 + 0.5 * c1 * (xa + xb) + c2 * (xa * xa + xa * xb + xb * xb) / 3.0


@njit(cache=True)
def remap_column(h_src, u_src, h_tgt, scheme, u_tgt, h_neglect=H_NEGLECT,
                 h_neglect_edge=H_NEGLECT_EDGE):  # pragma: no cover
    """
    Conservative remap of cell means u_src on thicknesses h_src [m] to thicknesses
    h_tgt [m] (same order; Σh_tgt = Σh_src up to round-off); result in u_tgt.
    """
    n0 = h_src.size
    n1 = h_tgt.size
    n_active = 0
    for k in range(n0):
        if h_src[k] > 0.0:
            n_active += 1
    coef = np.empty((n0, 3))
    reconstruct(h_src, u_src, _effective_scheme(n_active, scheme), h_neglect, h_neglect_edge, coef)

    # target interfaces (cumulative, starting at 0)
    t_edge = np.empty(n1 + 1)
    t_edge[0] = 0.0
    for j in range(n1):
        t_edge[j + 1] = t_edge[j] + h_tgt[j]
    content = np.zeros(n1)
    piece = np.empty(n1 + 1)
    piece_j = np.empty(n1 + 1, np.int64)

    a = 0.0                      # bottom of the current source cell
    j = 0
    last = -1                    # last source cell with h > 0
    for i in range(n0):
        hi = h_src[i]
        if hi <= 0.0:
            continue
        last = i
        b = a + hi
        npc = 0
        thick = -1.0
        ithick = 0
        while j < n1:
            lo = max(a, t_edge[j])
            hi_t = t_edge[j + 1] if j < n1 - 1 else max(t_edge[n1], b)   # last target absorbs a longer source
            up = min(b, hi_t)
            L = up - lo
            if L > 0.0:
                piece[npc] = L * _average(coef[i, 0], coef[i, 1], coef[i, 2], (lo - a) / hi, (up - a) / hi)
                piece_j[npc] = j
                if L > thick:
                    thick = L
                    ithick = npc
                npc += 1
            if hi_t <= b and j < n1 - 1:
                j += 1
            else:
                break
        if npc > 0:
            s = 0.0
            for p in range(npc):
                if p != ithick:
                    s += piece[p]
            piece[ithick] = u_src[i] * hi - s                      # Hallberg trick: exact cell content
            for p in range(npc):
                content[piece_j[p]] += piece[p]
        a = b

    # target thicker than source (round-off): extend with the top value of the last source cell
    if last >= 0 and t_edge[n1] > a:
        top = coef[last, 0] + coef[last, 1] + coef[last, 2]
        content[n1 - 1] += (t_edge[n1] - a) * top

    for jj in range(n1):
        if h_tgt[jj] > 0.0:
            u_tgt[jj] = content[jj] / h_tgt[jj]
        else:                                                     # vanished target: local value
            x = t_edge[jj]
            aa = 0.0
            val = u_src[last] if last >= 0 else 0.0
            for i in range(n0):
                if h_src[i] <= 0.0:
                    continue
                if x <= aa + h_src[i]:
                    xi = (x - aa) / h_src[i]
                    val = coef[i, 0] + xi * (coef[i, 1] + xi * coef[i, 2])
                    break
                aa += h_src[i]
            u_tgt[jj] = val
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_vertical_remap.py -q`
Expected: 6 passed.

- [ ] **Step 5: Mutation check**
  - Comment out the line `piece[ithick] = u_src[i] * hi - s` and run `test_conservation_to_roundoff`. It must fail. Restore the line.
  - Change `max(t_edge[n1], b)` to `t_edge[n1 + 0]` and run `test_roundoff_mismatch_in_total_thickness`. It must fail. Restore.

- [ ] **Step 6: Commit** (if authorised)

```bash
git add core/vertical/__init__.py core/vertical/remap.py tests/test_vertical_remap.py
git commit -m "ALE column toolkit: PCM reconstruction and conservative column remap"
```

### Task 2: PLM (White & Adcroft 2008, as in MOM6)

**Files:**
- Modify: `core/vertical/remap.py` (add `_plm_slope_wa`, `_plm_monotonized_slope`, PLM branch of `reconstruct`)
- Test: `tests/test_vertical_remap.py`

**Interfaces:**
- Consumes: `reconstruct`, `remap_column` (Task 1)
- Produces: `reconstruct(..., scheme=PLM, ...)` with MOM6 `PLM_reconstruction` coefficients (`PLM_functions.F90:186-262`)

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_plm_exact_for_linear_profile_interior():
    # u = 2 + 3 z sampled as cell means on a nonuniform grid: interior PLM reproduces the line
    h = np.array([1.0, 0.5, 2.0, 1.0, 1.5])
    z = np.concatenate([[0.0], np.cumsum(h)])
    u = 2 + 3 * 0.5 * (z[1:] + z[:-1])
    coef = np.empty((5, 3))
    rm.reconstruct(h, u, rm.PLM, rm.H_NEGLECT, rm.H_NEGLECT_EDGE, coef)
    for k in (1, 2, 3):
        np.testing.assert_allclose(coef[k, 0], 2 + 3 * z[k], rtol=1e-12)           # left edge
        np.testing.assert_allclose(coef[k, 0] + coef[k, 1], 2 + 3 * z[k + 1], rtol=1e-12)
    np.testing.assert_allclose(coef[[0, 4], 1], 0.0)                               # PCM at the ends


@pytest.mark.parametrize("scheme", [rm.PCM, rm.PLM])
def test_no_new_extrema(scheme):
    rng = np.random.default_rng(3)
    h0 = rng.uniform(0.2, 2.0, 30)
    u0 = np.where(np.arange(30) < 15, 10.0, 20.0) + rng.normal(0, 0.5, 30)
    h1 = rng.uniform(0.1, 1.0, 50)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, scheme)
    assert u1.min() >= u0.min() - 1e-12 and u1.max() <= u0.max() + 1e-12


def test_plm_less_diffusive_than_pcm_on_smooth_profile():
    h0 = np.full(40, 1.0)
    z = np.arange(40) + 0.5
    u0 = np.sin(2 * np.pi * z / 40)
    h1 = np.full(40, 1.0)
    shift = 0.3                                   # remap to a grid shifted by 0.3 cells, then back
    hs = np.concatenate([[shift], np.full(39, 1.0), [1.0 - shift]])
    err = {}
    for scheme in (rm.PCM, rm.PLM):
        u = u0.copy()
        for _ in range(10):
            us = _remap(h0, u, hs, scheme)
            u = _remap(hs, us, h1, scheme)
        err[scheme] = np.abs(u - u0).max()
    assert err[rm.PLM] < 0.5 * err[rm.PCM]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_remap.py -q -k "plm or extrema"`
Expected: FAIL. `test_plm_exact_for_linear_profile_interior` fails, because the coefficients are PCM.

- [ ] **Step 3: Implement** (add to `remap.py` before `reconstruct`, and replace its body)

```python
_EPS = np.finfo(np.float64).eps


@njit(cache=True)
def _plm_slope_wa(h_l, h_c, h_r, h_neglect, u_l, u_c, u_r):
    """Limited PLM slope (difference across the cell), MOM6 PLM_functions.F90:22-68."""
    sigma_r = u_r - u_c
    sigma_l = u_c - u_l
    sigma_c = 2.0 * (u_r - u_l) * (h_c / (h_l + 2.0 * h_c + h_r + h_neglect))
    u_min = min(u_l, u_c, u_r)
    u_max = max(u_l, u_c, u_r)
    if sigma_l * sigma_r > 0.0:
        s = min(abs(sigma_c), 2.0 * min(u_c - u_min, u_max - u_c))
        slope = s if sigma_c >= 0.0 else -s
    else:
        slope = 0.0
    if u_c - 0.5 * abs(slope) < u_min or u_c + 0.5 * abs(slope) > u_max:
        slope = slope * (1.0 - _EPS)
    if abs(slope) < 1e-140:
        slope = 0.0
    return slope


@njit(cache=True)
def _plm_monotonized_slope(u_l, u_c, u_r, s_l, s_c, s_r):
    """Edge-monotonic PLM slope, MOM6 PLM_functions.F90:122-158."""
    almost_two = 2.0 * (1.0 - _EPS)
    e_r = u_l + 0.5 * s_l
    e_l = u_r - 0.5 * s_r
    slp = abs(s_c)
    edge = u_c - 0.5 * s_c
    if (edge - e_r) * (u_c - edge) < 0.0:
        edge = 0.5 * (edge + e_r)
        slp = min(slp, abs(edge - u_c) * almost_two)
    edge = u_c + 0.5 * s_c
    if (edge - u_c) * (e_l - edge) < 0.0:
        edge = 0.5 * (edge + e_l)
        slp = min(slp, abs(edge - u_c) * almost_two)
    return slp if s_c >= 0.0 else -slp


@njit(cache=True)
def reconstruct(h, u, scheme, h_neglect, h_neglect_edge, coef):  # pragma: no cover
    """
    Polynomial coefficients coef[k] = (c0, c1, c2) of p(ξ) = c0 + c1 ξ + c2 ξ² in cell k [units of u].
    h : cell thicknesses [m];  u : cell means.  PLM: MOM6 PLM_reconstruction (PLM_functions.F90:186-262).
    """
    n = h.size
    for k in range(n):
        coef[k, 0] = u[k]
        coef[k, 1] = 0.0
        coef[k, 2] = 0.0
    if scheme == PCM or n < 3:
        return
    if scheme == PLM:
        slp = np.zeros(n)
        mslp = np.zeros(n)
        for k in range(1, n - 1):
            slp[k] = _plm_slope_wa(h[k - 1], h[k], h[k + 1], h_neglect, u[k - 1], u[k], u[k + 1])
        for k in range(1, n - 1):
            mslp[k] = _plm_monotonized_slope(u[k - 1], u[k], u[k + 1], slp[k - 1], slp[k], slp[k + 1])
        almost_one = 1.0 - _EPS
        for k in range(1, n - 1):
            u_l = u[k] - 0.5 * mslp[k]
            u_r = u[k] + 0.5 * mslp[k]
            coef[k, 0] = u_l
            coef[k, 1] = u_r - u_l
            edge = coef[k, 1] + coef[k, 0]
            # Fortran u(k+1) - 0.5*sign(mslp(k+1), slp(k+1)): |mslp| with the sign of slp
            e_r = u[k + 1] - 0.5 * (abs(mslp[k + 1]) if slp[k + 1] >= 0.0 else -abs(mslp[k + 1]))
            if (edge - u[k]) * (e_r - edge) < 0.0:
                coef[k, 1] = coef[k, 1] * almost_one
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_vertical_remap.py -q`
Expected: all pass (9).

- [ ] **Step 5: Mutation check.** Replace `2.0 * min(u_c - u_min, u_max - u_c)` with `10.0 * min(...)` and run `test_no_new_extrema`. It must fail. Restore.

- [ ] **Step 6: Commit** (if authorised): `git commit -m "ALE remap: PLM reconstruction (White & Adcroft 2008, MOM6 PLM_reconstruction)"`

### Task 3: PPM_H4

**Files:**
- Modify: `core/vertical/remap.py` (add `_solve4`, `_end_values_h4`, `_edges_h4`, `_bound_edges`, `_check_discontinuous`, PPM branch)
- Test: `tests/test_vertical_remap.py`

**Interfaces:**
- Consumes: `reconstruct`, `_plm_*` (Task 2)
- Produces: `reconstruct(..., scheme=PPM_H4, ...)`. Coefficients as in MOM6 `PPM_reconstruction` (`PPM_functions.F90:23-48`) after `edge_values_explicit_h4` (`regrid_edge_values.F90:215-350`, answer_date ≥ 2019) and `PPM_limiter_standard` (`PPM_functions.F90:52-120`).

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_ppm_exact_for_quadratic_interior():
    # cell means of u = 1 + z + z²: the 4th-order edges are exact, and so are interior polynomials
    # where the profile is monotonic (no limiter action)
    h = np.array([1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2])
    z = np.concatenate([[0.0], np.cumsum(h)]) + 1.0          # z > 0: monotonic
    F = lambda x: x + x**2 / 2 + x**3 / 3                     # antiderivative of 1 + z + z²
    u = (F(z[1:]) - F(z[:-1])) / h
    coef = np.empty((7, 3))
    rm.reconstruct(h, u, rm.PPM_H4, rm.H_NEGLECT, rm.H_NEGLECT_EDGE, coef)
    for k in range(2, 5):
        np.testing.assert_allclose(coef[k, 0], 1 + z[k] + z[k] ** 2, rtol=1e-10)
        np.testing.assert_allclose(coef[k].sum(), 1 + z[k + 1] + z[k + 1] ** 2, rtol=1e-10)


def test_ppm_no_new_extrema_and_conservation():
    rng = np.random.default_rng(4)
    h0 = rng.uniform(0.2, 2.0, 40)
    u0 = np.where(np.arange(40) < 20, 10.0, 20.0) + rng.normal(0, 0.3, 40)
    h1 = rng.uniform(0.1, 1.0, 60)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, rm.PPM_H4)
    assert u1.min() >= u0.min() - 1e-12 and u1.max() <= u0.max() + 1e-12
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * abs((h0 * u0).sum())


def test_ppm_less_diffusive_than_plm():
    h0 = np.full(40, 1.0)
    u0 = np.sin(2 * np.pi * (np.arange(40) + 0.5) / 40)
    hs = np.concatenate([[0.3], np.full(39, 1.0), [0.7]])
    err = {}
    for scheme in (rm.PLM, rm.PPM_H4):
        u = u0.copy()
        for _ in range(10):
            u = _remap(hs, _remap(h0, u, hs, scheme), h0, scheme)
        err[scheme] = np.abs(u - u0).max()
    assert err[rm.PPM_H4] < err[rm.PLM]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_remap.py -q -k ppm`
Expected: FAIL (PPM_H4 currently falls back to PCM).

- [ ] **Step 3: Implement** (add these functions, then the PPM branch at the end of `reconstruct`)

```python
H_MIN_FRAC = 1e-5          # MOM6 regrid_edge_values.F90:27


@njit(cache=True)
def _solve4(A, b):
    """Gaussian elimination with partial pivoting for a 4×4 system (in place copies)."""
    M = A.copy()
    x = b.copy()
    for c in range(4):
        p = c
        for r in range(c + 1, 4):
            if abs(M[r, c]) > abs(M[p, c]):
                p = r
        if p != c:
            for q in range(4):
                M[c, q], M[p, q] = M[p, q], M[c, q]
            x[c], x[p] = x[p], x[c]
        for r in range(c + 1, 4):
            f = M[r, c] / M[c, c]
            for q in range(c, 4):
                M[r, q] -= f * M[c, q]
            x[r] -= f * x[c]
    for c in range(3, -1, -1):
        s = x[c]
        for q in range(c + 1, 4):
            s -= M[c, q] * x[q]
        x[c] = s / M[c, c]
    return x


@njit(cache=True)
def _end_values_h4(dz, u):
    """
    Cubic C(x) = C0 + C1 x + C2 x² + C3 x³, x from the column end, whose means over the
    first 4 cells equal u (MOM6 end_value_h4, regrid_edge_values.F90:636-749, solved here
    directly: ∫ C dx over each cell = u_i dz_i).  Returns (C(0), C(dz[0])).
    """
    A = np.empty((4, 4))
    b = np.empty(4)
    x0 = 0.0
    for i in range(4):
        x1 = x0 + dz[i]
        for j in range(4):
            A[i, j] = (x1 ** (j + 1) - x0 ** (j + 1)) / (j + 1)
        b[i] = u[i] * dz[i]
        x0 = x1
    C = _solve4(A, b)
    d = dz[0]
    return C[0], C[0] + d * (C[1] + d * (C[2] + d * C[3]))


@njit(cache=True)
def _edges_h4(h, u, h_neglect, e):
    """Explicit 4th-order edge values e[k] = (left, right), MOM6 regrid_edge_values.F90:215-350."""
    n = h.size
    for i in range(2, n - 1):
        h0 = h[i - 2]
        h1 = h[i - 1]
        h2 = h[i]
        h3 = h[i + 1]
        if h0 + h1 == 0.0 or h1 + h2 == 0.0 or h2 + h3 == 0.0:
            h_min = H_MIN_FRAC * max(h_neglect, (h0 + h1) + (h2 + h3))
            h0 = max(h_min, h0)
            h1 = max(h_min, h1)
            h2 = max(h_min, h2)
            h3 = max(h_min, h3)
        I_h12 = 1.0 / (h1 + h2)
        I_den_et2 = 1.0 / (((h0 + h1) + h2) * (h0 + h1))
        I_h012 = (h0 + h1) * I_den_et2
        I_den_et3 = 1.0 / ((h1 + (h2 + h3)) * (h2 + h3))
        I_h123 = (h2 + h3) * I_den_et3
        et1 = ((1.0 + (h1 * I_h012 + (h0 + h1) * I_h123)) * I_h12 * (h2 * (h2 + h3)) * u[i - 1]
               + (1.0 + (h2 * I_h123 + (h2 + h3) * I_h012)) * I_h12 * (h1 * (h0 + h1)) * u[i])
        et2 = (h1 * (h2 * (h2 + h3)) * I_den_et2) * (u[i - 1] - u[i - 2])
        et3 = (h2 * (h1 * (h0 + h1)) * I_den_et3) * (u[i] - u[i + 1])
        e[i, 0] = (et1 + (et2 + et3)) / ((h0 + h1) + (h2 + h3))
        e[i - 1, 1] = e[i, 0]
    dz = np.empty(4)
    ut = np.empty(4)
    for i in range(4):
        dz[i] = max(h_neglect, h[i])
        ut[i] = u[i]
    e[0, 0], e[0, 1] = _end_values_h4(dz, ut)
    e[1, 0] = e[0, 1]
    for i in range(4):
        dz[i] = max(h_neglect, h[n - 1 - i])
        ut[i] = u[n - 1 - i]
    e[n - 1, 1], e[n - 1, 0] = _end_values_h4(dz, ut)
    e[n - 2, 1] = e[n - 1, 0]


@njit(cache=True)
def _bound_edges(h, u, e):
    """MOM6 bound_edge_values, answer_date ≥ 2019 (regrid_edge_values.F90:40-103)."""
    n = h.size
    for k in range(n):
        km1 = max(0, k - 1)
        kp1 = min(k + 1, n - 1)
        slope_x_h = 0.0
        if (h[km1] + h[kp1]) + 2.0 * h[k] > 0.0:
            sl = u[k] - u[km1]
            sc = (u[kp1] - u[km1]) * (h[k] / ((h[km1] + h[kp1]) + 2.0 * h[k]))
            sr = u[kp1] - u[k]
            if sl * sr > 0.0:
                m = min(abs(sl), abs(sc), abs(sr))
                slope_x_h = m if sc >= 0.0 else -m
        sgn = 1.0 if slope_x_h >= 0.0 else -1.0
        if (u[km1] - e[k, 0]) * (e[k, 0] - u[k]) < 0.0:
            e[k, 0] = u[k] - sgn * min(abs(slope_x_h), abs(e[k, 0] - u[k]))
        if (u[kp1] - e[k, 1]) * (e[k, 1] - u[k]) < 0.0:
            e[k, 1] = u[k] + sgn * min(abs(slope_x_h), abs(e[k, 1] - u[k]))
        e[k, 0] = max(min(e[k, 0], max(u[km1], u[k])), min(u[km1], u[k]))
        e[k, 1] = max(min(e[k, 1], max(u[kp1], u[k])), min(u[kp1], u[k]))


@njit(cache=True)
def _check_discontinuous(u, e):
    """MOM6 check_discontinuous_edge_values (regrid_edge_values.F90:134-152)."""
    for k in range(u.size - 1):
        if (e[k + 1, 0] - e[k, 1]) * (u[k + 1] - u[k]) < 0.0:
            a = 0.5 * (e[k, 1] + e[k + 1, 0])
            a = max(min(a, max(u[k], u[k + 1])), min(u[k], u[k + 1]))
            e[k, 1] = a
            e[k + 1, 0] = a
```

PPM branch, appended at the end of `reconstruct` (replacing nothing else):

```python
    if scheme == PPM_H4 and n >= 4:
        e = np.empty((n, 2))
        _edges_h4(h, u, h_neglect_edge, e)
        _bound_edges(h, u, e)
        _check_discontinuous(u, e)
        for k in range(1, n - 1):                    # Colella & Woodward (1984) limiter
            u_l = u[k - 1]
            u_c = u[k]
            u_r = u[k + 1]
            el = e[k, 0]
            er = e[k, 1]
            if (u_r - u_c) * (u_c - u_l) <= 0.0:
                el = u_c
                er = u_c
            else:
                expr1 = 3.0 * (er - el) * ((u_c - el) + (u_c - er))
                expr2 = (er - el) * (er - el)
                if expr1 > expr2:
                    el = u_c + 2.0 * (u_c - er)
                    el = max(min(el, max(u_l, u_c)), min(u_l, u_c))
                elif expr1 < -expr2:
                    er = u_c + 2.0 * (u_c - el)
                    er = max(min(er, max(u_r, u_c)), min(u_r, u_c))
            if abs(er - el) < max(1e-60, _EPS * abs(u_c)):
                el = u_c
                er = u_c
            coef[k, 0] = el
            coef[k, 1] = 4.0 * (u_c - el) + 2.0 * (u_c - er)
            coef[k, 2] = 3.0 * ((er - u_c) + (el - u_c))
```

Also change the early return at the start of `reconstruct` to `if scheme == PCM or n < 3: return`, and run the PLM branch only `if scheme == PLM` (already so). The PPM branch needs n ≥ 4; `_effective_scheme` guarantees that for remap calls.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_vertical_remap.py -q`
Expected: all pass (12).

- [ ] **Step 5: Mutation check.** Delete the `elif expr1 < -expr2:` branch and run `test_ppm_no_new_extrema_and_conservation`. It must fail. If it passes, increase the jump in the test profile until it fails, then restore.

- [ ] **Step 6: Commit** (if authorised): `git commit -m "ALE remap: PPM_H4 reconstruction (explicit H4 edges + Colella-Woodward limiter, as MOM6)"`

### Task 4: 3-D driver, short columns and vanished layers

**Files:**
- Modify: `core/vertical/remap.py` (add `remap_columns`)
- Test: `tests/test_vertical_remap.py`

**Interfaces:**
- Consumes: `remap_column`
- Produces: `remap_columns(h_src: f8[N0, ny, nx], f_src: f8[N0, ny, nx], h_tgt: f8[N1, ny, nx], scheme: int, wet: bool[ny, nx]) -> f8[N1, ny, nx]`. Dry or masked columns return `f_src` when N0 == N1, otherwise NaN. Uses `numba.prange` over columns; the thread count is set by the caller with `numba.set_num_threads`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_short_columns_reduce_order():
    for n in (1, 2, 3):
        h0 = np.ones(n)
        u0 = np.arange(n, dtype=float)
        u1 = _remap(h0, u0, h0, rm.PPM_H4)
        np.testing.assert_allclose(u1, u0, rtol=1e-14)


def test_zstar_like_column_with_many_vanished_bottom_layers():
    # 50 layers, only the top 6 thick (shallow column in z*): no NaN, conservation
    h0 = np.r_[np.full(44, 0.0), np.linspace(5, 20, 6)]
    u0 = np.r_[np.full(44, 12.0), np.linspace(12, 18, 6)]
    h1 = np.r_[np.full(44, 1e-3), np.linspace(5, 20, 6)]
    h1[-1] -= 44e-3
    for scheme in (rm.PCM, rm.PLM, rm.PPM_H4):
        u1 = _remap(h0, u0, h1, scheme)
        assert np.isfinite(u1).all()
        assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * (h0 * u0).sum()


def test_remap_columns_matches_single_column():
    rng = np.random.default_rng(5)
    N, ny, nx = 8, 3, 4
    h0 = rng.uniform(0.5, 2.0, (N, ny, nx))
    f = rng.normal(10, 1, (N, ny, nx))
    h1 = rng.uniform(0.5, 2.0, (N, ny, nx))
    h1 *= h0.sum(0) / h1.sum(0)
    wet = np.ones((ny, nx), bool)
    wet[0, 0] = False
    out = rm.remap_columns(h0, f, h1, rm.PPM_H4, wet)
    ref = np.empty(N)
    rm.remap_column(h0[:, 1, 2], f[:, 1, 2], h1[:, 1, 2], rm.PPM_H4, ref)
    np.testing.assert_allclose(out[:, 1, 2], ref, rtol=0, atol=0)
    np.testing.assert_array_equal(out[:, 0, 0], f[:, 0, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_remap.py -q -k "short or vanished or columns"`
Expected: `test_remap_columns_matches_single_column` fails (`AttributeError: remap_columns`). Fix any other failure in `remap_column` before continuing.

- [ ] **Step 3: Implement**

```python
@njit(cache=True, parallel=True)
def _remap_columns(h_src, f_src, h_tgt, scheme, wet, out):  # pragma: no cover
    n0, ny, nx = h_src.shape
    n1 = h_tgt.shape[0]
    for c in prange(ny * nx):
        j = c // nx
        i = c % nx
        if not wet[j, i]:
            for k in range(n1):
                out[k, j, i] = f_src[k, j, i] if n0 == n1 else np.nan
            continue
        col = np.empty(n1)
        remap_column(np.ascontiguousarray(h_src[:, j, i]), np.ascontiguousarray(f_src[:, j, i]),
                     np.ascontiguousarray(h_tgt[:, j, i]), scheme, col)
        for k in range(n1):
            out[k, j, i] = col[k]


def remap_columns(h_src: np.ndarray, f_src: np.ndarray, h_tgt: np.ndarray, scheme: int,
                  wet: np.ndarray) -> np.ndarray:
    """
    Remap every wet column of f_src (N0, ny, nx) from thicknesses h_src to h_tgt (N1, ny, nx) [m].
    Dry columns: f_src copied when N0 == N1, else NaN.  Threads: numba.set_num_threads (≤ 5).
    """
    out = np.empty(h_tgt.shape)
    _remap_columns(np.ascontiguousarray(h_src, float), np.ascontiguousarray(f_src, float),
                   np.ascontiguousarray(h_tgt, float), int(scheme), np.asarray(wet, np.bool_), out)
    return out
```

- [ ] **Step 4: Run all remap tests**

Run: `python -m pytest tests/test_vertical_remap.py -q`
Expected: all pass (15).

- [ ] **Step 5: Commit** (if authorised): `git commit -m "ALE remap: threaded 3-D driver; short and vanished columns"`

### Task 5: Reference build (FMS + MOM6) and remap comparison against MOM6

**Prerequisite (ask Esteban first):** installing `autoconf automake libtool` with Homebrew (`brew install autoconf automake libtool`). FMS and MOM6 use autotools, and they are not installed. If Esteban declines, mark Tasks 5, 8 and 9 as blocked in the README and continue with Tasks 6–7. Do not attempt a hand-written build of FMS.

**Files:**
- Create: `validation/mom6_column/build_reference.sh`, `validation/mom6_column/remap_driver.F90`, `validation/mom6_column/columns.py`, `validation/mom6_column/README.md`
- Test: `tests/test_vertical_vs_mom6.py`

**Interfaces:**
- Consumes: `remap.remap_column`, `remap.SCHEMES`
- Produces:
  - `validation/mom6_column/columns.py`:
    - `write_remap_cases(path, cases)`
    - `read_remap_results(path) -> list[np.ndarray]`
    - `remap_cases() -> list[dict]` (keys `h0, u0, h1, scheme`)
  - The executables `~/pyOCEAN_refs/build/column/remap_driver` and (Task 8) `regrid_driver`.

- [ ] **Step 1: Build script** (`validation/mom6_column/build_reference.sh`)

```bash
#!/bin/bash
# Build FMS + MOM6 (objects and modules) with the MOM6 .testing machinery, then the column drivers.
# Output: ~/pyOCEAN_refs/build/{.testing build tree, column/}.  Uses ≤ 5 make jobs (P-cores).
set -euo pipefail
REF=$HOME/pyOCEAN_refs
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$REF/MOM6/.testing"
make -j5 FMS_URL="file://$REF/FMS" FMS_COMMIT=2025.02.01 build/symmetric/MOM6
B=$REF/MOM6/.testing/build/symmetric
OBJS=$(ls "$B"/*.o | grep -v -E "/(MOM_driver|ocean_solo|MOM_sum_driver)\.o$")
FMSLIB=$(ls "$REF"/MOM6/.testing/deps/lib/libFMS.a)
mkdir -p "$REF/build/column"
for d in remap_driver regrid_driver hybgen_driver; do
  [ -f "$HERE/$d.F90" ] || continue
  mpif90 -O2 -fdefault-real-8 -fdefault-double-8 -I"$B" -I"$REF/MOM6/.testing/deps/include" \
         "$HERE/$d.F90" $OBJS "$FMSLIB" $(nf-config --flibs) -o "$REF/build/column/$d"
done
echo "built: $(ls "$REF/build/column")"
```

The `.testing` Makefile clones FMS from `FMS_URL` and builds it with autotools into `deps/`. Check the real variable names in `~/pyOCEAN_refs/MOM6/.testing/Makefile` (lines 80–200). Adjust `B`, the excluded driver objects and the library path to what that Makefile produces. These are the only lines that may need adapting, and the README must document the adaptation.

- [ ] **Step 2: Run the build**

Run: `bash validation/mom6_column/build_reference.sh 2>&1 | tail -20`
Expected: `build/symmetric/MOM6` is built, followed by `built:`. The driver list is empty at this point because the driver source does not exist yet. If FMS or MOM6 fail to compile, stop and report the error to Esteban; do not patch MOM6 or FMS.

- [ ] **Step 3: Shared test columns** (`validation/mom6_column/columns.py`)

```python
"""
Test columns for the pyOCEAN ↔ MOM6 column comparisons (remapping and regridding),
written as plain text for the Fortran drivers.  Order: MOM6 convention (k = 1 at
the surface), so the drivers need no reordering; pyOCEAN calls receive the same
arrays (remapping is order-agnostic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCHEME_NAMES = {0: "PCM", 1: "PLM", 2: "PPM_H4"}


def remap_cases() -> list[dict]:
    rng = np.random.default_rng(2026)
    cases = []
    for scheme in (0, 1, 2):
        for n0, n1 in ((10, 10), (25, 40), (50, 50), (4, 7)):
            h0 = rng.uniform(0.0, 5.0, n0)
            h0[rng.random(n0) < 0.15] = 0.0                       # vanished layers
            h0[0] = max(h0[0], 0.5)
            u0 = np.cumsum(rng.normal(0.2, 0.5, n0)) + 10.0
            h1 = rng.uniform(0.0, 5.0, n1)
            h1[0] = max(h1[0], 0.5)
            h1 *= h0.sum() / h1.sum()
            cases.append(dict(h0=h0, u0=u0, h1=h1, scheme=scheme))
    return cases


def write_remap_cases(path: Path, cases: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{SCHEME_NAMES[c['scheme']]}\n{len(c['h0'])} {len(c['h1'])}\n")
            for a in (c["h0"], c["u0"], c["h1"]):
                f.write(" ".join(f"{v:.17e}" for v in a) + "\n")


def read_remap_results(path: Path) -> list[np.ndarray]:
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    return [np.array(lines[1 + i].split(), float) for i in range(n)]
```

- [ ] **Step 4: Remap driver** (`validation/mom6_column/remap_driver.F90`)

```fortran
!> Reads columns written by columns.py and remaps them with MOM6 remapping_core_h.
!! Usage: remap_driver in.txt out.txt
program remap_driver
  use MOM_remapping, only : remapping_CS, initialize_remapping, remapping_core_h
  implicit none
  type(remapping_CS) :: CS
  integer :: ncase, c, n0, n1
  character(len=16) :: scheme
  character(len=512) :: fin, fout
  real, allocatable :: h0(:), u0(:), h1(:), u1(:)
  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  do c = 1, ncase
    read(10,*) scheme ; read(10,*) n0, n1
    allocate(h0(n0), u0(n0), h1(n1), u1(n1))
    read(10,*) h0 ; read(10,*) u0 ; read(10,*) h1
    call initialize_remapping(CS, trim(scheme), boundary_extrapolation=.false., &
         om4_remap_via_sub_cells=.false., answer_date=99991231, nk=max(n0,n1), &
         h_neglect=1.0e-30, h_neglect_edge=1.0e-10)
    call remapping_core_h(CS, n0, h0, u0, n1, h1, u1)
    write(11,'(*(es25.17))') u1
    deallocate(h0, u0, h1, u1)
  enddo
  close(10) ; close(11)
end program remap_driver
```

Check the `initialize_remapping` keywords against `MOM_remapping.F90:1658-1699` before compiling. If MOM6 requires an `end_remapping` or a `CS` re-initialisation per scheme, follow its unit test `config_src/drivers/unit_tests/test_MOM_remapping.F90`.

- [ ] **Step 5: Rebuild with the driver**

Run: `bash validation/mom6_column/build_reference.sh 2>&1 | tail -3`
Expected: `built: remap_driver`.

- [ ] **Step 6: Write the comparison test** (`tests/test_vertical_vs_mom6.py`)

```python
"""
Level-1 validation (spec §8): pyOCEAN column routines against the original MOM6
Fortran (compiled by validation/mom6_column/build_reference.sh).  Skipped when the
reference build is absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "mom6_column"))
import columns  # noqa: E402

from core.vertical import remap as rm  # noqa: E402

BUILD = Path.home() / "pyOCEAN_refs" / "build" / "column"
needs = lambda exe: pytest.mark.skipif(not (BUILD / exe).exists(), reason=f"MOM6 reference {exe} not built")


@needs("remap_driver")
def test_remap_matches_mom6(tmp_path):
    cases = columns.remap_cases()
    columns.write_remap_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "remap_driver"), str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_remap_results(tmp_path / "out.txt")
    for c, r in zip(cases, ref):
        u1 = np.empty(len(c["h1"]))
        rm.remap_column(c["h0"], c["u0"], c["h1"], c["scheme"], u1)
        np.testing.assert_allclose(u1, r, rtol=1e-11, atol=1e-11,
                                   err_msg=f"{columns.SCHEME_NAMES[c['scheme']]} n0={len(c['h0'])}")
```

- [ ] **Step 7: Run it**

Run: `python -m pytest tests/test_vertical_vs_mom6.py -q`
Expected: 1 passed. If a case differs:
- Print the case and both results.
- The likely sources are, in this order:
  1. the end values `_end_values_h4` (closed form in MOM6 versus our 4×4 solve; the difference should be ≤ 1e-12);
  2. the treatment of vanished target cells;
  3. how a source longer than the target is absorbed.
- Fix pyOCEAN to match MOM6 unless MOM6 is clearly wrong. In that case, document the difference in the README and loosen only that case, with a comment.

- [ ] **Step 8: README** (`validation/mom6_column/README.md`). Spanish, with:
  - purpose;
  - how to build (prerequisites, `build_reference.sh`, the adaptations made);
  - what is compared;
  - a result table: case, scheme, max |Δ|.

- [ ] **Step 9: Commit** (if authorised): `git add validation/mom6_column tests/test_vertical_vs_mom6.py && git commit -m "ALE remap validated against MOM6 remapping_core_h (level 1)"`

### Task 6: z\* and σ rules

**Files:**
- Create: `core/vertical/regrid.py`
- Test: `tests/test_vertical_regrid.py`

**Interfaces:**
- Produces:
  - `regrid.H_MIN = 1e-3`
  - `regrid.zstar_column(H: float, eta: float, dz_top: f8[N], h_min: float, zw: f8[N+1]) -> None`
    - `dz_top`: nominal thicknesses, surface first (e.g. MITgcm `delR`);
    - `zw`: output interface heights, bed first.
  - `regrid.sigma_column(H: float, eta: float, dsigma: f8[N], h_min: float, zw: f8[N+1]) -> None`
    - `dsigma`: fractions bed first, Σ = 1.
  - `regrid.thickness(zw) -> h` (= `np.diff(zw)`)

- [ ] **Step 1: Write the failing tests**

```python
"""
Tests for core/vertical/regrid.py — grid-generation rules per column (spec §3):
z* (Adcroft & Campin 2004; MOM6 coord_zlike.F90), σ (coord_sigma.F90) and the
hybrid z*–isopycnal HYCOM1 rule (coord_hycom.F90).  Interfaces zw are heights,
bed first (zw[0] = −H, zw[N] = η).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import regrid as rg

DELR = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 70.0])   # surface first, Σ = 201


def _zstar(H, eta, dz=DELR, h_min=rg.H_MIN):
    zw = np.empty(dz.size + 1)
    rg.zstar_column(H, eta, dz, h_min, zw)
    return zw


def test_zstar_full_depth_equals_scaled_levels():
    H, eta = 201.0, 0.5
    zw = _zstar(H, eta)
    s = (H + eta) / H
    expected = eta - s * np.r_[0.0, np.cumsum(DELR)]           # surface first
    np.testing.assert_allclose(zw, expected[::-1], rtol=1e-14, atol=1e-12)


def test_zstar_shallow_column_collapses_bottom_layers():
    H, eta = 40.0, 0.0
    zw = _zstar(H, eta)
    h = np.diff(zw)
    assert zw[0] == -H and zw[-1] == eta
    np.testing.assert_allclose(h.sum(), H + eta, rtol=1e-14)
    assert (h >= rg.H_MIN * (1 - 1e-12)).all()
    np.testing.assert_allclose(h[-4:][::-1], DELR[:4], rtol=1e-12)      # top 4 levels unchanged (Σ = 11 m)
    assert np.count_nonzero(np.isclose(h, rg.H_MIN)) >= 4              # collapsed at the bed


def test_zstar_min_thickness_capped_by_depth():
    zw = _zstar(0.005, 0.0)                                             # 5 mm of water, 10 layers
    h = np.diff(zw)
    np.testing.assert_allclose(h.sum(), 0.005, rtol=1e-12)
    assert (h > 0).all()


def test_sigma_reproduces_fractions():
    ds = np.array([0.1, 0.2, 0.3, 0.4])                                 # bed first
    zw = np.empty(5)
    rg.sigma_column(100.0, 2.0, ds, rg.H_MIN, zw)
    np.testing.assert_allclose(np.diff(zw), 102.0 * ds, rtol=1e-14)
    assert zw[0] == -100.0 and zw[-1] == 2.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_regrid.py -q`
Expected: ERROR (`ModuleNotFoundError` / `ImportError` for `regrid`).

- [ ] **Step 3: Implement** (`core/vertical/regrid.py`, first part)

```python
"""
Grid-generation (regridding) rules of the ALE vertical coordinate, per column
(spec docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §3).

Convention: interface heights zw [m], bed first: zw[0] = −H, zw[N] = η (z = 0 at
rest).  No rule changes the column thickness D = H + η.  Minimum thickness
h_min,eff = min(h_min, D/N), inflated upward from the bed (MOM6 coord_zlike.F90:137-143,
coord_sigma.F90:76-81).

z*   Adcroft & Campin (2004): z = η + s z*, s = D/H, nominal levels Δz_k (surface first,
     e.g. MITgcm delR); levels below the bed collapse to h_min at the bed, the deepest
     active level is truncated — the partial cell of MITgcm.  MOM6 build_zstar_column.
σ    Fixed fractions Δσ_k (bed first): h_k = D Δσ_k.  MOM6 build_sigma_column.

References: Adcroft, A. & Campin, J.-M. (2004). Ocean Modell. 7, 269–284.
MOM6 source ~/pyOCEAN_refs/MOM6/src/ALE (commit 0e9a561).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

H_MIN = 1e-3     # [m] minimum layer thickness after regridding (MOM6 MIN_THICKNESS default)


@njit(cache=True)
def zstar_column(H, eta, dz_top, h_min, zw):  # pragma: no cover
    """z* interfaces (bed first) for depth H [m], surface η [m], nominal Δz surface first [m]."""
    n = dz_top.size
    D = H + eta
    hm = min(h_min, D / n)
    s = D / H
    e = np.empty(n + 1)                     # surface first, heights
    e[0] = eta
    for k in range(n):
        e[k + 1] = e[k] - s * dz_top[k]
    e[n] = -H
    for k in range(n - 1, -1, -1):
        if e[k] < e[k + 1] + hm:
            e[k] = e[k + 1] + hm
    for j in range(n + 1):
        zw[j] = e[n - j]


@njit(cache=True)
def sigma_column(H, eta, dsigma, h_min, zw):  # pragma: no cover
    """σ interfaces (bed first): fractions dsigma bed first, Σ = 1."""
    n = dsigma.size
    D = H + eta
    hm = min(h_min, D / n)
    zw[0] = -H
    for j in range(n):
        zw[j + 1] = zw[j] + D * dsigma[j]
        if zw[j + 1] < zw[j] + hm:
            zw[j + 1] = zw[j] + hm
    zw[n] = eta


def thickness(zw: np.ndarray) -> np.ndarray:
    """Layer thicknesses [m] from interface heights (first axis, bed first)."""
    return np.diff(zw, axis=0)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_vertical_regrid.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit** (if authorised): `git commit -m "ALE regrid: z* and sigma column rules (MOM6 coord_zlike / coord_sigma)"`

### Task 7: Hybrid rule (HYCOM1 + z\* floor + maximum thickness + shallow σ) and 3-D driver

**Files:**
- Modify: `core/vertical/regrid.py`
- Test: `tests/test_vertical_regrid.py`

**Interfaces:**
- Consumes: `zstar_column`, `sigma_column`, `remap._bound_edges` (P1M uses `bound_edge_values`)
- Produces:
  - `regrid.hycom1_column(H, eta, h_bed: f8[N], rho_bed: f8[N], rho_target: f8[N+1], dz_top: f8[N], max_h_top: f8[N], h_min, zw: f8[N+1]) -> None`
    - `rho_bed`: layer densities [kg m⁻³] at the reference pressure (computed by the caller with the compressibility trick, Plan B);
    - `rho_target`: interface target densities, surface first, increasing downward; the first and last are unused;
    - `max_h_top`: maximum thickness per layer, surface first; `np.inf` disables it.
  - `regrid.blend_shallow_sigma(zw_hyb: f8[N+1], zw_sig: f8[N+1], D: float, d_sigma: float, d_hybrid: float, zw: f8[N+1]) -> None`
    - q = clip((D − d_σ)/(d_hyb − d_σ), 0, 1); zw = q·zw_hyb + (1 − q)·zw_sig. This is the HYCOM shallow-water σ transition (`hybgen.F90:392-428`) applied to interfaces, which is how it maps onto the HYCOM1 rule. At the end-points it gives pure σ (D ≤ d_σ) and the pure hybrid (D ≥ d_hyb), as in HYCOM.
  - `regrid.enforce_min_thickness(zw, h_min) -> None`: top-down clamp, then bottom-up inflation, with h_min,eff = min(h_min, D/N).
  - `regrid.target_densities(rho_top: f8[:], n_layers: int) -> f8[n_layers+1]`: evenly spaced from min to max of the profile.
  - `regrid.regrid_columns(rule: str, H: f8[ny,nx], eta: f8[ny,nx], h: f8[N,ny,nx], rho: f8[N,ny,nx] | None, params: dict, wet: bool[ny,nx]) -> zw: f8[N+1,ny,nx]` with `rule` in {"zstar", "sigma", "hybrid"}. `params` holds `dz_top`, `dsigma`, `rho_target`, `max_h_top`, `d_sigma`, `d_hybrid` and `h_min` as needed.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _hybrid(H, eta, h, rho, targets, dz=DELR, max_h=None, h_min=rg.H_MIN):
    n = h.size
    zw = np.empty(n + 1)
    mh = np.full(n, np.inf) if max_h is None else max_h
    rg.hycom1_column(H, eta, h, rho, targets, dz, mh, h_min, zw)
    return zw


def _linear_column(H=201.0, n=10, rho_top=1024.0, drho_dz=0.02):
    zw = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    return np.diff(zw), rho_top - drho_dz * zc                          # bed first, ρ grows downward


def test_hybrid_interfaces_on_target_densities():
    h, rho = _linear_column()
    # targets (surface first) whose z positions lie below the z* floor
    z_t = -np.array([0.0, 30, 60, 80, 100, 120, 140, 160, 170, 178, 201.0])   # interior cells only
    targets = 1024.0 - 0.02 * z_t
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    np.testing.assert_allclose(zw[::-1][1:-1], z_t[1:-1], atol=1e-9)    # interior interfaces


def test_hybrid_respects_zstar_floor():
    h, rho = _linear_column()
    targets = np.full(11, 1024.0)                                       # all targets lighter than the column
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    zstar = np.empty(11)
    rg.zstar_column(201.0, 0.0, DELR, rg.H_MIN, zstar)
    np.testing.assert_allclose(zw, zstar, atol=1e-9)                    # pushed down to the z* floor


def test_hybrid_unstratified_column_does_not_fail():
    h = np.full(10, 20.1)
    rho = np.full(10, 1025.0)                                           # well mixed
    targets = np.linspace(1024.0, 1027.0, 11)
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    assert np.isfinite(zw).all() and (np.diff(zw) > 0).all()


def test_hybrid_targets_out_of_range_keep_layers_separate():
    h, rho = _linear_column()
    targets = np.linspace(1000.0, 1100.0, 11)                           # far outside the column range
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    assert (np.diff(zw) >= rg.H_MIN * (1 - 1e-12)).all()
    np.testing.assert_allclose(np.diff(zw).sum(), 201.0, rtol=1e-14)


def test_hybrid_nonmonotonic_profile_is_monotonized():
    h, rho = _linear_column()
    rho[3] += 0.5                                                       # a dense blob above lighter water
    zw = _hybrid(201.0, 0.0, h, rho, np.linspace(1024.0, 1028.0, 11))
    assert (np.diff(zw) > 0).all()


def test_hybrid_maximum_layer_thickness():
    h, rho = _linear_column()
    targets = np.r_[1024.0, np.full(9, 1028.0), 1028.1]                 # would put everything at the bed
    zw = _hybrid(201.0, 0.0, h, rho, targets, max_h=np.full(10, 25.0))
    assert (np.diff(zw) <= 25.0 + 1e-9).all()
    np.testing.assert_allclose(np.diff(zw).sum(), 201.0, rtol=1e-14)


def test_blend_shallow_sigma_endpoints_and_middle():
    a = np.array([-10.0, -6.0, 0.0])
    b = np.array([-10.0, -4.0, 0.0])
    zw = np.empty(3)
    rg.blend_shallow_sigma(a, b, 5.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, b)                                   # D ≤ d_sigma: pure σ
    rg.blend_shallow_sigma(a, b, 40.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, a)                                   # D ≥ d_hybrid: pure hybrid
    rg.blend_shallow_sigma(a, b, 20.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, 0.5 * (a + b))


def test_regrid_columns_hybrid_matches_single_column():
    h, rho = _linear_column()
    ny, nx = 2, 3
    H = np.full((ny, nx), 201.0)
    eta = np.zeros((ny, nx))
    h3 = np.repeat(h[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    r3 = np.repeat(rho[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    targets = np.linspace(1024.0, 1028.0, 11)
    params = dict(dz_top=DELR, rho_target=targets, max_h_top=np.full(10, np.inf),
                  dsigma=np.full(10, 0.1), d_sigma=0.0, d_hybrid=0.0, h_min=rg.H_MIN)
    zw3 = rg.regrid_columns("hybrid", H, eta, h3, r3, params, np.ones((ny, nx), bool))
    np.testing.assert_allclose(zw3[:, 1, 2], _hybrid(201.0, 0.0, h, rho, targets), rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_vertical_regrid.py -q`
Expected: the new tests fail (`AttributeError: hycom1_column`).

- [ ] **Step 3: Implement** (append to `regrid.py`; add the hybrid description to the module docstring)

Docstring addition:

```text
Hybrid (HYCOM1 of MOM6, OM4; Adcroft et al. 2019): interface target densities
(surface first, increasing downward).  The layer densities are monotonized
(ρ_k = min(ρ_k, ρ_k+1) upward from the bed, coord_hycom.F90:218), reconstructed
P1M with H2 edges (edges bounded and averaged, P1M_functions.F90:29-61), and
each target is located by inverting the linear cell polynomial
(regrid_interp.F90:378-511).  Then: z* floor (an interface is never above its
z* position, coord_hycom.F90:255-267), maximum layer thickness (:270-279),
optional shallow-water σ transition (HYCOM hybgen.F90:392-428, applied to the
interfaces) and the minimum thickness.
```

Code:

```python
from core.vertical.remap import _bound_edges


@njit(cache=True)
def enforce_min_thickness(zw, h_min):  # pragma: no cover
    """Keep layers ≥ min(h_min, D/N): clamp from the top, then inflate from the bed."""
    n = zw.size - 1
    hm = min(h_min, (zw[n] - zw[0]) / n)
    for j in range(n - 1, 0, -1):
        if zw[j] > zw[j + 1] - hm:
            zw[j] = zw[j + 1] - hm
    for j in range(1, n):
        if zw[j] < zw[j - 1] + hm:
            zw[j] = zw[j - 1] + hm


@njit(cache=True)
def _p1m_h2_positions(h_top, rho_top, targets, x_new):  # pragma: no cover
    """
    Depths (positive down from the surface, 0 … D) of target interface densities, MOM6
    build_and_interpolate_grid with INTERPOLATION_SCHEME = P1M_H2 and no boundary
    extrapolation.  h_top, rho_top surface first; targets surface first (n+1).
    """
    n = h_top.size
    e = np.empty((n, 2))
    e[0, 0] = rho_top[0]
    e[n - 1, 1] = rho_top[n - 1]
    for k in range(1, n):                                    # H2 edges (regrid_edge_values.F90:168-194)
        if h_top[k - 1] + h_top[k] == 0.0:
            e[k, 0] = 0.5 * (rho_top[k - 1] + rho_top[k])
        else:
            e[k, 0] = (rho_top[k - 1] * h_top[k] + rho_top[k] * h_top[k - 1]) / (h_top[k - 1] + h_top[k])
        e[k - 1, 1] = e[k, 0]
    _bound_edges(h_top, rho_top, e)
    for k in range(n - 1):                                   # average_discontinuous_edge_values
        if e[k, 1] != e[k + 1, 0]:
            a = 0.5 * (e[k, 1] + e[k + 1, 0])
            e[k, 1] = a
            e[k + 1, 0] = a
    xg = np.empty(n + 1)
    xg[0] = 0.0
    for k in range(n):
        xg[k + 1] = xg[k] + h_top[k]
    x_new[0] = 0.0
    x_new[n] = xg[n]
    for kk in range(1, n):
        t = targets[kk]
        if t <= e[0, 0]:
            x_new[kk] = xg[0]
            continue
        found = False
        for k in range(1, n):
            if t >= e[k - 1, 1] and t <= e[k, 0]:
                x_new[kk] = xg[k]
                found = True
                break
        if found:
            continue
        if t >= e[n - 1, 1]:
            x_new[kk] = xg[n]
            continue
        for k in range(n):
            if e[k, 0] < t < e[k, 1]:
                xi = (t - e[k, 0]) / (e[k, 1] - e[k, 0])
                x_new[kk] = xg[k] + min(max(xi, 0.0), 1.0) * h_top[k]
                break


@njit(cache=True)
def hycom1_column(H, eta, h_bed, rho_bed, rho_target, dz_top, max_h_top, h_min, zw):  # pragma: no cover
    """Hybrid z*–isopycnal interfaces (bed first) — see the module docstring."""
    n = h_bed.size
    D = H + eta
    h_top = h_bed[::-1].copy()
    rho = rho_bed[::-1].copy()
    for k in range(n - 2, -1, -1):                           # monotonic (coord_hycom.F90:218)
        rho[k] = min(rho[k], rho[k + 1])
    x = np.empty(n + 1)
    _p1m_h2_positions(h_top, rho, rho_target, x)
    s = D / H                                                # z* floor (coord_hycom.F90:255-267)
    nominal = 0.0
    for k in range(1, n + 1):
        nominal += dz_top[k - 1] * s
        x[k] = min(max(x[k], nominal), D)
    for k in range(1, n):                                    # maximum layer thickness (:270-279)
        x[k] = min(x[k], x[k - 1] + max_h_top[k - 1])
    x[n] = D
    for j in range(n + 1):
        zw[j] = eta - x[n - j]
    zw[0] = -H
    zw[n] = eta
    enforce_min_thickness(zw, h_min)


@njit(cache=True)
def blend_shallow_sigma(zw_hyb, zw_sig, D, d_sigma, d_hybrid, zw):  # pragma: no cover
    """HYCOM shallow-water σ transition applied to interfaces (hybgen.F90:392-428)."""
    if d_hybrid <= d_sigma:
        q = 1.0 if D > d_sigma else 0.0
    else:
        q = min(max((D - d_sigma) / (d_hybrid - d_sigma), 0.0), 1.0)
    for j in range(zw.size):
        zw[j] = q * zw_hyb[j] + (1.0 - q) * zw_sig[j]


def target_densities(rho_top: np.ndarray, n_layers: int) -> np.ndarray:
    """Interface target densities (surface first) evenly spaced over the profile range [kg m⁻³]."""
    rho_top = np.asarray(rho_top, float)
    return np.linspace(np.nanmin(rho_top), np.nanmax(rho_top), n_layers + 1)


@njit(cache=True, parallel=True)
def _regrid_columns(code, H, eta, h, rho, dz_top, dsigma, rho_target, max_h_top,
                    d_sigma, d_hybrid, h_min, wet, zw):  # pragma: no cover
    n, ny, nx = h.shape
    for c in prange(ny * nx):
        j = c // nx
        i = c % nx
        if not wet[j, i]:
            continue
        col = np.empty(n + 1)
        if code == 0:
            zstar_column(H[j, i], eta[j, i], dz_top, h_min, col)
        elif code == 1:
            sigma_column(H[j, i], eta[j, i], dsigma, h_min, col)
        else:
            hycom1_column(H[j, i], eta[j, i], np.ascontiguousarray(h[:, j, i]),
                          np.ascontiguousarray(rho[:, j, i]), rho_target, dz_top, max_h_top, h_min, col)
            if d_hybrid > 0.0:
                sig = np.empty(n + 1)
                sigma_column(H[j, i], eta[j, i], dsigma, h_min, sig)
                hyb = col.copy()
                blend_shallow_sigma(hyb, sig, H[j, i] + eta[j, i], d_sigma, d_hybrid, col)
        for k in range(n + 1):
            zw[k, j, i] = col[k]


def regrid_columns(rule: str, H: np.ndarray, eta: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
                   params: dict, wet: np.ndarray) -> np.ndarray:
    """
    Target interface heights zw (N+1, ny, nx) [m], bed first, for rule "zstar", "sigma" or "hybrid".
    params: dz_top (z* and hybrid floor), dsigma (σ and the shallow transition), rho_target,
    max_h_top, d_sigma, d_hybrid [m] (0 = no transition), h_min [m].  Dry columns: NaN.
    """
    code = {"zstar": 0, "sigma": 1, "hybrid": 2}[rule]
    n = h.shape[0]
    zw = np.full((n + 1,) + h.shape[1:], np.nan)
    f = lambda key, default: np.ascontiguousarray(params.get(key, default), float)
    _regrid_columns(code, np.nan_to_num(np.asarray(H, float)), np.asarray(eta, float), np.ascontiguousarray(h, float),
                    np.ascontiguousarray(h if rho is None else rho, float), f("dz_top", np.ones(n)),
                    f("dsigma", np.full(n, 1.0 / n)), f("rho_target", np.zeros(n + 1)),
                    f("max_h_top", np.full(n, np.inf)), float(params.get("d_sigma", 0.0)),
                    float(params.get("d_hybrid", 0.0)), float(params.get("h_min", H_MIN)),
                    np.asarray(wet, np.bool_), zw)
    return zw
```

- [ ] **Step 4: Run all regrid tests**

Run: `python -m pytest tests/test_vertical_regrid.py -q`
Expected: all pass (12). If `test_hybrid_interfaces_on_target_densities` is off by a fraction of a layer, check the H2 edges and the cell search. On a linear profile, P1M is exact.

- [ ] **Step 5: Mutation checks**
  - Remove the z\* floor loop and run `test_hybrid_respects_zstar_floor`. It must fail.
  - Remove the monotonization loop and run `test_hybrid_nonmonotonic_profile_is_monotonized`. It must fail; if it does not, strengthen the blob until it does.
  - Restore both.

- [ ] **Step 6: Commit** (if authorised): `git commit -m "ALE regrid: hybrid HYCOM1 rule with z* floor, max thickness, shallow sigma; 3-D driver"`

### Task 8: Regrid comparison against MOM6 (z\*, σ, P1M interpolation)

**Files:**
- Create: `validation/mom6_column/regrid_driver.F90`
- Modify: `validation/mom6_column/columns.py` (add `regrid_cases`, `write_regrid_cases`, `read_regrid_results`), `tests/test_vertical_vs_mom6.py`, `validation/mom6_column/README.md`

**Interfaces:**
- Consumes: `regrid.zstar_column`, `regrid.sigma_column`, `regrid._p1m_h2_positions` (Task 7); the build from Task 5
- Produces:
  - `columns.regrid_cases() -> list[dict]` (keys `kind` ∈ {"zstar", "sigma", "interp"}, `H`, `eta`, `dz`, `h`, `rho`, `targets`)
  - the executable `regrid_driver`

- [ ] **Step 1: Cases and I/O** (append to `columns.py`)

```python
def regrid_cases() -> list[dict]:
    rng = np.random.default_rng(7)
    dz = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 70.0])
    cases = []
    for H, eta in ((201.0, 0.3), (164.0, -0.2), (40.0, 0.0), (11.0, 0.01)):
        cases.append(dict(kind="zstar", H=H, eta=eta, dz=dz))
        cases.append(dict(kind="sigma", H=H, eta=eta, dz=np.full(10, 0.1)))
    for _ in range(6):
        h = rng.uniform(0.5, 30.0, 12)
        rho = np.sort(1024.0 + rng.uniform(0.0, 4.0, 12))           # surface first, increasing
        rho[rng.integers(0, 12)] -= 0.3                              # a non-monotonic glitch is fine for interp
        targets = np.linspace(1023.5, 1028.5, 13)
        cases.append(dict(kind="interp", h=h, rho=np.sort(rho), targets=targets))
    return cases


def write_regrid_cases(path: Path, cases: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(c["kind"] + "\n")
            if c["kind"] in ("zstar", "sigma"):
                f.write(f"{len(c['dz'])} {c['H']:.17e} {c['eta']:.17e}\n")
                f.write(" ".join(f"{v:.17e}" for v in c["dz"]) + "\n")
            else:
                f.write(f"{len(c['h'])}\n")
                for a in (c["h"], c["rho"], c["targets"]):
                    f.write(" ".join(f"{v:.17e}" for v in a) + "\n")


def read_regrid_results(path: Path) -> list[np.ndarray]:
    return read_remap_results(path)
```

- [ ] **Step 2: Driver** (`validation/mom6_column/regrid_driver.F90`)

```fortran
!> MOM6 column builders on cases written by columns.py (surface first, depths positive down
!! for build_and_interpolate_grid; interface heights for z* / sigma as MOM6 returns them).
program regrid_driver
  use coord_zlike,   only : zlike_CS, init_coord_zlike, set_zlike_params, build_zstar_column
  use coord_sigma,   only : sigma_CS, init_coord_sigma, set_sigma_params, build_sigma_column
  use regrid_interp, only : interp_CS_type, set_interp_scheme, set_interp_extrap, &
                            set_interp_answer_date, build_and_interpolate_grid
  implicit none
  type(zlike_CS), pointer :: zCS => null()
  type(sigma_CS), pointer :: sCS => null()
  type(interp_CS_type) :: iCS
  integer :: ncase, c, n
  character(len=16) :: kind
  character(len=512) :: fin, fout
  real :: H, eta
  real, allocatable :: dz(:), zi(:), h(:), rho(:), tgt(:), x0(:), h1(:), x1(:)
  call get_command_argument(1, fin) ; call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old') ; open(11, file=trim(fout), status='replace')
  read(10,*) ncase ; write(11,'(i6)') ncase
  call set_interp_scheme(iCS, "P1M_H2") ; call set_interp_extrap(iCS, .false.)
  call set_interp_answer_date(iCS, 99991231)
  do c = 1, ncase
    read(10,*) kind
    if (trim(kind) == 'zstar' .or. trim(kind) == 'sigma') then
      read(10,*) n, H, eta ; allocate(dz(n), zi(n+1)) ; read(10,*) dz
      if (trim(kind) == 'zstar') then
        call init_coord_zlike(zCS, n, dz) ; call set_zlike_params(zCS, min_thickness=1.0e-3)
        call build_zstar_column(zCS, H, H+eta, zi)
        deallocate(zCS) ; nullify(zCS)
      else
        call init_coord_sigma(sCS, n, dz) ; call set_sigma_params(sCS, min_thickness=1.0e-3)
        call build_sigma_column(sCS, H, H+eta, zi)        ! heights, surface first (as z*)
        deallocate(sCS) ; nullify(sCS)
      endif
      write(11,'(*(es25.17))') zi
      deallocate(dz, zi)
    else
      read(10,*) n ; allocate(h(n), rho(n), tgt(n+1), x0(n+1), h1(n), x1(n+1))
      read(10,*) h ; read(10,*) rho ; read(10,*) tgt
      x0(1) = 0.0
      block ; integer :: k ; do k = 1, n ; x0(k+1) = x0(k) + h(k) ; enddo ; end block
      call build_and_interpolate_grid(iCS, rho, n, h, x0, tgt, n, h1, x1, 1.0e-30, 1.0e-10)
      write(11,'(*(es25.17))') x1
      deallocate(h, rho, tgt, x0, h1, x1)
    endif
  enddo
  close(10) ; close(11)
end program regrid_driver
```

Before compiling, check these against the cited sources and fix the driver to match:
- the argument lists of `init_coord_zlike` / `init_coord_sigma` and `set_*_params` (`coord_zlike.F90:30-60`, `coord_sigma.F90:25-60`);
- that `build_zstar_column` and `build_sigma_column` both return absolute interface heights, surface first (`coord_zlike.F90:120-143`, `coord_sigma.F90:71-82`);
- the `build_and_interpolate_grid` order of arguments (`regrid_interp.F90:333-360`).


- [ ] **Step 3: Comparison test** (append to `tests/test_vertical_vs_mom6.py`)

```python
from core.vertical import regrid as rg  # noqa: E402


@needs("regrid_driver")
def test_regrid_matches_mom6(tmp_path):
    cases = columns.regrid_cases()
    columns.write_regrid_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "regrid_driver"), str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_regrid_results(tmp_path / "out.txt")
    for c, r in zip(cases, ref):
        if c["kind"] == "zstar":
            zw = np.empty(len(c["dz"]) + 1)
            rg.zstar_column(c["H"], c["eta"], c["dz"], 1e-3, zw)
            np.testing.assert_allclose(zw[::-1], r, atol=1e-10, err_msg=f"zstar H={c['H']}")
        elif c["kind"] == "sigma":
            zw = np.empty(len(c["dz"]) + 1)
            rg.sigma_column(c["H"], c["eta"], c["dz"][::-1], 1e-3, zw)
            np.testing.assert_allclose(zw[::-1], r, atol=1e-10, err_msg=f"sigma H={c['H']}")
        else:
            x = np.empty(len(c["h"]) + 1)
            rg._p1m_h2_positions(c["h"], c["rho"], c["targets"], x)
            np.testing.assert_allclose(x, r, atol=1e-9, err_msg="P1M_H2 interpolation")
```

- [ ] **Step 4: Build and run**

Run: `bash validation/mom6_column/build_reference.sh | tail -1 && python -m pytest tests/test_vertical_vs_mom6.py -q`
Expected: 2 passed. If there are differences, apply the rule from Task 5 Step 7: match MOM6 unless MOM6 is clearly wrong, and record any accepted difference in the README.

- [ ] **Step 5: README results table** (add the regrid rows). **Commit** (if authorised): `git commit -m "ALE regrid validated against MOM6 coord_zlike/coord_sigma/regrid_interp (level 1)"`

### Task 9: Hybrid rule vs HYCOM `hybgen` (MOM6 port), quantitative comparison

This comparison is quantitative, not to round-off: HYCOM1 and hybgen are different algorithms (spec §8). Its result informs the decision whether to port hybgen later.

**Files:**
- Create: `validation/mom6_column/hybgen_driver.F90`, `validation/mom6_column/compare_hybgen.py`
- Modify: `validation/mom6_column/README.md`

**Interfaces:**
- Consumes: `regrid.hycom1_column`, `regrid.blend_shallow_sigma`, `regrid.sigma_column`; MOM6 `MOM_hybgen_regrid` (`hybgen_regrid`, column version)
- Produces: a table in the README (per column type: max and mean |Δz| of the interfaces, in m and as a fraction of the local layer thickness).

- [ ] **Step 1: Read the MOM6 hybgen port interface.** Read `~/pyOCEAN_refs/MOM6/src/ALE/MOM_hybgen_regrid.F90`. Identify:
  - the column routine (the "column_regrid" part named in the code-review notes) and its arguments: thicknesses, densities, targets, `dp00`/`ds00` parameters;
  - how its control structure is initialised without a parameter file. If it needs `MOM_file_parser`, construct the structure fields directly in the driver.

  Write the driver so that it calls that column routine on columns from `compare_hybgen.py`. The columns are:
  - stratified 600 m;
  - well-mixed 600 m;
  - shallow 30 m;
  - 164 m with a light bottom layer;
  - a BayIW boundary column with the MITgcm T profile.

  Use the same target densities and a z-grid equal to MITgcm `delR` for both rules.

- [ ] **Step 2: `compare_hybgen.py`.** It:
  - writes the columns;
  - runs `hybgen_driver`;
  - computes our `hycom1_column` (with `blend_shallow_sigma` for the shallow case, d_σ = 50 m, d_hyb = 100 m);
  - prints and writes `validation/mom6_column/hybgen_comparison.txt` with, per column, max |Δz|, mean |Δz| and the layer index of the maximum.

- [ ] **Step 3: Run it and document it.** Run `python validation/mom6_column/compare_hybgen.py`. Add the table and an interpretation paragraph to the README:
  - where the rules differ (bottom layer, transition zone, cushion);
  - whether the differences are larger than one layer thickness.
  
  A difference larger than a layer in the interior is the evidence that would justify porting hybgen (spec §3, out of scope).

- [ ] **Step 4: Commit** (if authorised): `git commit -m "ALE: hybrid rule compared with HYCOM hybgen (MOM6 port) on typical columns"`

### Task 10: Close Plan A

**Files:**
- Modify: `CLAUDE.md` (Build Status rows), `docs/design/vision_roadmap.md` (§7 scorecard rows)

- [ ] **Step 1: Full test suite**

Run: `NUMBA_NUM_THREADS=5 OMP_NUM_THREADS=1 python -m pytest tests/ -q -W ignore::RuntimeWarning -p no:cacheprovider 2>&1 | tail -3`
Expected: 926 + the new tests passed, 1 xfailed, and no failures. Report any failure by name.

- [ ] **Step 2: CLAUDE.md rows.** Add two rows to the Phase 2 table and update the total count:
  - "ALE column remapping PCM/PLM/PPM_H4 (= MOM6 remapping_core_h to 1e-11)" with `core/vertical/remap.py`;
  - "ALE grid rules z*, σ, hybrid HYCOM1 (= MOM6 coord_* to 1e-10)" with `core/vertical/regrid.py`.

- [ ] **Step 3: Roadmap scorecard (§7).** Add "ALE column toolkit validated vs MOM6 (level 1)" ✅ and "hybrid vs hybgen" with its quantitative result.

- [ ] **Step 4: Commit** (if authorised): `git commit -m "ALE Plan A complete: column toolkit validated against MOM6"`
