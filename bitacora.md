# Bitácora de desarrollo — pyOCEAN

Registro cronológico de decisiones de diseño, problemas encontrados y soluciones.
Cada entrada corresponde a una sesión de trabajo.

---

## Sesión 1 — 2026-03-23

**Módulos completados:** `core/grid/horizontal.py`, `core/grid/vertical.py`

### Horizontal grid

Implementado el grid Arakawa C con dos backends:
- `create_cartesian_grid` — dominio plano con f-plane/β-plane
- `create_spherical_grid` — coordenadas esféricas con `f = 2Ω sin φ`

Convención de staggering (igual que MITgcm/MOM6/ROMS):
```
q---v---q---v---q
|       |       |
u   T   u   T   u
|       |       |
q---v---q---v---q
```

Tamaños de dimensión para nx × ny celdas T:
- `i_t`: 0…nx-1, `j_t`: 0…ny-1
- `i_u`: 0…nx (nx+1 caras U — incluye pared este)
- `j_v`: 0…ny (ny+1 caras V — incluye pared norte)

Todo en `xr.Dataset` con atributos CF-compliant.

### Vertical grid

z-levels con z ≤ 0 (convención oceanográfica: superficie en z=0).
Tres constructores: `create_z_grid`, `uniform_z_grid`, `stretched_z_grid`.

`compute_hfac` implementa partial cells (MITgcm §2.10):
- `hfac_t(k,j,i)` — fracción húmeda de celda T
- `hfac_u`, `hfac_v` — mínimo de celdas adyacentes (no flujo a través de tierra)
- Celdas con `hfac < hfac_min=0.1` se colapsan a tierra

**28 + 37 = 65 tests, todos pasando.**

---

## Sesión 2 — 2026-03-24

**Módulos completados:** `core/state.py`, `physics/eos/`, `core/dynamics/pressure.py`

### ModelState (`core/state.py`)

Contenedor de variables prognosticas sobre el C-grid:

| Variable | Dims | Descripción |
|----------|------|-------------|
| `u` | `(k, j_t, i_u)` | Velocidad zonal [m/s] |
| `v` | `(k, j_v, i_t)` | Velocidad meridional [m/s] |
| `w` | `(k_f, j_t, i_t)` | Velocidad vertical [m/s] en interfaces |
| `T` | `(k, j_t, i_t)` | Temperatura potencial [°C] |
| `S` | `(k, j_t, i_t)` | Salinidad [PSU] |
| `eta` | `(j_t, i_t)` | Altura de la superficie libre [m] |

Constructor principal: `ModelState.from_grids(h_grid, v_grid)`.

Operaciones implementadas para el timestepper Adams–Bashforth 2:
- `copy()` — deep copy independiente para el estado en t-1
- `zeros_like(other)` — arrays de tendencia ∂φ/∂t
- `__add__`, `__mul__`, `__rmul__` — permite `state_new = state_old + dt * tendency`
- `to_dataset()` — exporta a `xr.Dataset` para NetCDF

**52 tests pasando.**

### Ecuación de estado (`physics/eos/`)

Dos opciones seleccionables con el argumento `eos`:

**TEOS-10** (`eos="teos10"`): EOS no-lineal completa vía `gsw.rho(SA, CT, p)`.
- Aproximación para dominio idealizado: SA ≈ SP (PSU), CT ≈ θ (error < 0.01°C)
- Adecuado para simulaciones realistas

**EOS Lineal** (`eos="linear"`): primera expansión de Taylor alrededor de (T₀, S₀, ρ₀):
```
ρ = ρ₀ [1 − α(T − T₀) + β(S − S₀)]     (Vallis 2006, eq. 1.156)
```
- Valores por defecto: ρ₀=1025, α=2×10⁻⁴ 1/°C, β=7.4×10⁻⁴ 1/PSU, T₀=10°C, S₀=35 PSU
- Completamente configurable por kwargs
- Preferida para Phase 1 por transparencia y velocidad

Función `compute_density(state, z_grid, eos, **kwargs)`:
- Presión aproximada: `p_dbar(k) = ρ₀ g |z_t[k]| / 1e4`
- Retorna `xr.DataArray ρ(k, j_t, i_t)` con coordenadas del estado

También acepta `eos=callable` para EOS personalizada.

**Decisión de diseño**: usar EOS lineal por defecto en Phase 1 (validación seiches).
La diferencia con TEOS-10 en superficie es ~1.8 kg/m³ con T=10°C, S=35 PSU — aceptable para dinámica idealizada. Se cambia a TEOS-10 en Phase 2.

**36 tests pasando.**

### Presión hidrostática (`core/dynamics/pressure.py`)

Integración downward de `∂p/∂z = −ρg` (Vallis 2006, eq. 2.75):

```
p_face[0]   = 0
p_face[k+1] = p_face[k] + ρ[k] g dz[k]         (cumsum)
p[k]        = ρ₀ g η + p_face[k] + ½ ρ[k] g dz[k]
```

Término barotropico `ρ₀ g η` incluido en la misma función (no separado), para que el gradiente de presión capture la pendiente de la superficie libre correctamente.

**Gradiente de presión en el C-grid:**
```
PGF_x[k, j, i+½] = −(1/ρ₀) (p[k,j,i+1] − p[k,j,i]) / dx_u[j, i+½]
PGF_y[k, j+½, i] = −(1/ρ₀) (p[k,j+1,i] − p[k,j,i]) / dy_v[j+½, i]
```

Usa `dx_u(j_t, i_u)` y `dy_v(j_v, i_t)` del grid horizontal — no espaciados escalares.
Esto hace el código compatible con grids esféricos desde el principio.

Caras de frontera (i_u=0, i_u=nx, j_v=0, j_v=ny) se dejan en cero (pared sólida).

**Solver CG para superficie libre implícita** (`surface_pressure_cg`):
Resuelve la ecuación de Helmholtz 2D (Dukowicz & Smith 1994):
```
∇·(H ∇η) − η/(g Δt²) = RHS
```
Ensamble de matriz sparse + `scipy.sparse.linalg.cg`. BC Dirichlet η=0 en fronteras.
No se usa en Phase 1 (superficie libre explícita); listo para Phase 2+.

**Problema encontrado**: `scipy.sparse.linalg.cg` ≥ 1.12 cambió `tol=` por `rtol=`.
Solución: usar `rtol=` directamente.

**Problema encontrado**: el scaffold genera `physics/eos/` como paquete, no como módulo.
Crear `physics/eos.py` genera conflicto silencioso (Python prefiere el paquete).
Solución: código en `physics/eos/__init__.py`.

**34 tests pasando.**

---

## Pendiente — Phase 1

| Módulo | Descripción |
|--------|-------------|
| `core/dynamics/momentum.py` | Ecuaciones de momento con Coriolis y PGF |
| `core/timestepping/ab2.py` | Adams–Bashforth 2 para u, v, T, S; Forward-Euler η |
| `numerics/advection/` | Advección upwind centrada (Phase 1) |
| `physics/mixing/kpp.py` | Mezcla vertical KPP (Large et al. 1994) |
| `boundaries/obcs.py` | Condiciones de frontera abiertas (Orlanski + sponge) |
| `forcing/surface.py` | Forzamiento superficial (bulk formulae) |
| `diagnostics/io.py` | Escritura NetCDF4 vía xarray |
| Notebook validación | Seiche rectangular vs MITgcm BayIW_Rectan |

---

## Notas de arquitectura

### Flujo de datos por timestep (diseño objetivo)

```
t=n → t=n+1:

1. compute_density(state, z_grid, eos)           → rho(k,j,i)
2. hydrostatic_pressure(rho, eta, z_grid)        → p(k,j,i)
3. pressure_gradient_x/y(p, h_grid)             → pgf_x, pgf_y
4. momentum_tendency(state, pgf_x, pgf_y, ...)  → du/dt, dv/dt
5. tracer_tendency(state, ...)                   → dT/dt, dS/dt
6. ab2_step(state, tendency, state_prev, dt)     → state_new
7. update_eta(state_new, h_grid, dt)             → eta_new
8. apply_bcs(state_new, h_grid)                  → enforce walls
```

### Convenciones numéricas

- Índice k=0: capa más superficial. k=nz-1: más profunda.
- z > 0 es el espacio por encima del océano; z_t[k] ≤ 0 siempre.
- Profundidad de batimetría positiva hacia abajo (depth > 0 = océano, depth = 0 = tierra).
- Presión en Pa internamente; se convierte a dbar sólo para llamadas a gsw.
