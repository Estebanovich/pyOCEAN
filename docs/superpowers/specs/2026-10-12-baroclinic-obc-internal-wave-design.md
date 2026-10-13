# Forzamiento baroclínico en fronteras abiertas y benchmark `internal_wave` de MITgcm

**Fecha:** 2026-10-12. **Estado:** diseño aprobado por Esteban en la conversación, sección por sección. Falta revisar el texto escrito.
**Rama:** `obc-internal-wave`, desde `main` (5196341).
**Contexto:** es el paso 2 del orden re-acordado (`2026-10-12-bayiw-2d-section-design.md` §1) y el alcance B del Plan C (`2026-10-12-ale-open-boundaries-design.md`). Hoy las fronteras abiertas de pyOCEAN suponen un exterior en reposo:
- Flather hacia η = 0, ū = 0;
- Orlanski y la esponja llevan u′ hacia 0;
- el perfil exterior de T queda fijo desde el estado inicial.

Para imponer una onda baroclínica (y, más adelante, anidar en HYCOM o forzar mareas internas en PAC_MEX), el exterior tiene que variar en el tiempo.

**Alcance elegido por Esteban:** general, enfoque 1 (un proveedor como función).

## 1. Estado exterior e interfaz

### 1.1 `OBData` ampliado (`boundaries/obcs/__init__.py`)

Por borde:
- **Campos que ya existen:** `side`, `T (nz, n)`, `S`, `u_normal (nz, n)` y `eta (n)`.
- **Campos nuevos, opcionales:**
  - `zw (nz+1, n)`: las interfaces en z de la malla de los datos [m], de la superficie al fondo;
  - `ubar (n)`: la velocidad normal media [m s⁻¹].
- **Significado:** `u_normal` es la velocidad normal *total*; el modelo separa ū y u′ con sus propios pesos de cara. Si falta `eta` o `ubar`, valen 0.
- El modelo z de la Fase 1, que usa `OBData` sin `zw`, no cambia.

### 1.2 Argumentos nuevos de `SigmaBaroclinic`

- **`open_boundary_state`:** una función `t [s] → {borde: OBData}`, o `None`.
  - Con `None` el comportamiento es el actual: exterior en reposo con el perfil inicial. La corrida σ queda idéntica bit a bit y la ALE no cambia.
- **`open_boundary_mode`:** `{borde: 'radiating' | 'clamped'}`; el valor por defecto es `'radiating'` en todos los bordes abiertos.
- **`obc_tau_in`, `obc_tau_out`:** las escalas de relajación de u′ hacia el exterior en el modo `'radiating'` [s].
  - Por defecto 1 día y 10 días, los valores típicos de CROCO (`OCEAN/read_inp.F:2576-2577`, en días).
  - Solo actúan con `open_boundary_state`.

### 1.3 `interpolated_boundary(times, states)` (`forcing/`)

- Devuelve una función que interpola linealmente en el tiempo entre listas de `OBData` con la misma forma, y se queda en el extremo fuera del rango.
- Es la base de los adaptadores HYCOM/ESPC. xarray queda solo en la entrada/salida (decisión 2026-10-11).

### 1.4 Uso dentro del paso

- El modelo evalúa el estado en tⁿ⁺¹ (y en el tiempo de cada etapa intermedia del esquema de los trazadores σ).
- T y `u_normal` se remapean de la malla `zw` a las capas actuales de la celda o cara del borde:
  - **ALE:** con `exterior_on_layers` (`core/vertical/ale.py`), ya probado;
  - **σ:** con la misma rutina sobre las capas σ.
- La partición ū/u′ de los datos exteriores usa los pesos de cara del modelo: Σ w u′ₑₓₜ = 0.

### 1.5 Validación al construir el modelo y en la primera evaluación

Se lanza un `ValueError` con un mensaje claro si:
- algún borde de `open_boundary_state` o de `open_boundary_mode` no está en `open_boundaries`;
- las formas son inconsistentes entre sí o con el borde;
- `zw` no es monótono o no cubre la profundidad del borde;
- hay un modo desconocido.

## 2. Qué hace cada modo en el borde

Notación: ⋅ₑₓₜ es el estado exterior en tⁿ⁺¹, remapeado a las capas del borde.

### 2.1 `'radiating'`

Generaliza el Plan C según CROCO `OCEAN/u3dbc.F:111-155` (Marchesiello, McWilliams & Shchepetkin 2001, Ocean Modell. 3, 1–20).

- **Barotrópico:** Flather hacia (ηₑₓₜ, ūₑₓₜ) en lugar de (0, 0).
- **u′:**
  - Orlanski como ahora (`_radiate`).
  - Después, u′_B ← (1 − α) u′_B + α u′ₑₓₜ, con α = Δt/τ_in cuando la onda entra (r = 0) y Δt/τ_out cuando sale.
  - Sin `open_boundary_state`, α = 0, así que no hay cambio.
- **Esponja:** relaja u′ hacia u′ₑₓₜ y, si `sponge_tracer`, T hacia Tₑₓₜ (hoy lo hace hacia 0 y el perfil inicial).
- **T entrante:** la advección usa Tₑₓₜ(tⁿ⁺¹) en las caras de entrada.

### 2.2 `'clamped'`

Dirichlet, como el `obcs_calc.F` de `internal_wave` en MITgcm y `M3_FRC_BRY` de CROCO (`u3dbc.F:96-108`).

- u′ del borde = u′ₑₓₜ y T entrante = Tₑₓₜ, sin Orlanski.
- **Barotrópico:** Flather hacia (ηₑₓₜ, ūₑₓₜ).
- MITgcm fija también el transporte total; con ū = 0 eso es transporte nulo. Se clasifica (c): el forzamiento de `internal_wave` tiene media vertical nula.

## 3. Benchmark `internal_wave`

### 3.1 Configuración

Idéntica a `~/pyOCEAN_refs/MITgcm/verification/internal_wave/input`:
- **Malla:** 60 columnas de dx = 218.03 m (`delXvar`, Lx = 13.08 km), una fila y 20 niveles de 10 m.
- **Fondo:** H = 200 m; pendiente 0.03 desde x = 2.5 km hasta 40 m de profundidad (`topog.slope`); la última columna es tierra.
- **Física:** N² = 1e-6, f = 0, EOS lineal (α = 2e-4, `Tref` del `data`), ν_h = κ_h = 1e-2, ν_z = κ_z = 1e-3, deslizamiento libre (`no_slip` = .FALSE.), Δt = 500 s.
- **Detalles de MITgcm:** `hFacMin` = 0.2 (celdas parciales) y `nonlinFreeSurf` = 3.
- **Oeste `'clamped'`:**
  - u′ y T de `code/obcs_calc.F` (modo 1, u₀ = 0.024 m/s, período 44 567 s, perfil vertical de media nula);
  - se reproducen como una función de Python línea por línea, con la fórmula de kx y el factor `cos(kx·1.5·Δx)`.
- **Este:** pared. La frontera de MITgcm cae sobre tierra.
- **Corridas:**
  - MITgcm sin cambios en el código, con su `code/` y su `input/`, compilado sin MPI con el optfile de Esteban en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/`.
  - **10 períodos (5.2 días)** en lugar de los 100 pasos de la verificación, con salida MNC cada 30 min o menos.
- **pyOCEAN:** z\* (20 capas), híbrida (20 capas, objetivos del perfil lineal) y σ como referencia, con Δt = 500 s (se reduce si es inestable y se reporta).

### 3.2 Prueba analítica (solo pyOCEAN)

- **Configuración:** un canal plano de 200 m con el oeste `'clamped'` y el este `'radiating'` con esponja.
- **Solución:** la onda progresiva u = u₀ cos(πz/H) sin(ωt − kx), con k = πω/(NH), una longitud de onda de ≈2.8 km.
- **Criterio:** amplitud ±5 % y fase ±10° en el interior, después de 3 períodos.
- Prueba a la vez el modo impuesto y la salida de la onda por la radiación.

### 3.3 Métricas contra MITgcm

Análisis armónico a ω sobre los últimos 4 períodos:
- amplitud y fase de u y de T′ en la sección x–z, y su error rms relativo;
- flujo de energía baroclínica ⟨p′u′⟩ integrado en la vertical en x = 2, 6 y 10 km (plano, pendiente y zona somera);
- 2.º armónico (2ω) en la zona somera (peralte no lineal);
- salud: truncamientos, max|Σh − H − η| y calor.

### 3.4 Criterios

- Parte plana: amplitud ±10 % y fase ±10°.
- Pendiente y zona somera: flujo de energía ±15 %.
- Toda diferencia se clasifica de (a) a (d); si es (a), me detengo y se lo reporto a Esteban.

## 4. Pruebas

- **`OBData`:** validación de `zw` y de las formas; `interpolated_boundary` (lineal entre tiempos y constante fuera del rango).
- **`'radiating'` sin estado:** idéntico al código actual (σ bit a bit; ALE con las huellas de `tests/data/ale_perf_reference.json`).
- **`'radiating'` con estado:**
  - un exterior constante con u′ ≠ 0 lleva el borde hacia u′ₑₓₜ con la escala τ_in;
  - Flather hacia ηₑₓₜ, ūₑₓₜ eleva η en un canal cerrado del otro lado.
- **`'clamped'`:** u′ del borde = u′ₑₓₜ y Σ w u′ = 0.
- **Onda progresiva analítica** (§3.2), en z\* y en la híbrida.
- **Errores** del §1.5.

## 5. Entregables

- **Código:** `boundaries/obcs` (`OBData`), `forcing/` (`interpolated_boundary`) y `core/dynamics/sigma_baroclinic.py` (los modos y el estado exterior).
- **`validation/internal_wave/`:** `make_case.py` (entradas y la función de borde), `run_mitgcm.sh`, `run_pyocean.py`, `compare.py` y `README.md` con las tablas, la clasificación y "qué debe mejorar pyOCEAN".
- **Documentación:** una fila en el marcador (`docs/design/vision_roadmap.md` §7), `CLAUDE.md` y la memoria.

## 6. Fuera de alcance

- Los adaptadores HYCOM/ESPC a `OBData` dependientes del tiempo (siguiente paso de anidamiento).
- La salinidad en el exterior; queda como trazador pasivo.
- Imponer el transporte total en el modo `'clamped'` (frontera de transporte para el solver barotrópico).
- La BayIW 3-D recortada (paso 3 del orden).
