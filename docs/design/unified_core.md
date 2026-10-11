# Núcleo 3-D unificado (plan aprobado el 2026-10-11)

**Orden acordado con Esteban:**
1. Unificar el núcleo 3-D.
2. Decidir el backend (JAX o Numba) con la medición del prototipo.
3. Configuración por archivo y documentación.
4. Etapa C (ALE híbrido).

Este documento detalla el paso 1.

## Situación

Hay tres líneas de modelo que no se comunican:

| Línea | Archivos | Tiene | Le falta |
|---|---|---|---|
| Modelo z (fase 1) | `core/state.py`, `core/dynamics/{pressure,momentum}.py`, `physics/mixing/kpp.py`, `forcing/surface.py`, `boundaries/obcs` | T/S, TEOS-10 o EOS lineal, KPP, forzamiento de superficie, OBC Orlanski | mojado-secado, superficie libre implícita moderna, validación regional |
| 2DH de marea | `core/dynamics/{free_surface,wetting_drying}.py`, `forcing/tides.py`, `validation/pacmex_tides` | marea TPXO + potencial + SAL, Flather, mojado-secado, deriva log-law, verificador de dominio | estratificación |
| σ estratificado | `core/dynamics/{sigma_wd,sigma_baroclinic}.py` | jacobiano + ρ_ref, WENO5, Coriolis, viento, OBC + esponja, σ estirada, difusión, restart, Numba | salinidad, EOS no lineal, flujos de calor y agua dulce, flotabilidad en GLS, KPP, marea, frontera desde un modelo padre |

## Decisión: el núcleo es el modelo σ

Es el que tiene la numérica más moderna y validada (CROCO-like). Se le agregan las piezas de las otras dos líneas, cada una con su test y su benchmark:

| # | Pieza | Origen | Validación |
|---|---|---|---|
| 1 | Pila genérica de trazadores (T, S y pasivos), varianza por trazador | nuevo | constancia y conservación por trazador; lock exchange con S |
| 2 | EOS lineal (T, S) y TEOS-10 con p ≈ −ρ₀ g z | `physics/eos` | gradiente de presión exacto en reposo con ρ_ref(z) de (T, S); seamount con TEOS-10 |
| 3 | Flotabilidad en GLS: producción B = −K_ρ N², difusividad de trazadores K_t con funciones de estabilidad de Canuto A | port de CROCO `gls_mixing.F` | Kato–Phillips (profundización de la capa de mezcla por viento sobre estratificación lineal), contra la solución empírica y GOTM/CROCO |
| 4 | Flujos de superficie: Q_net, onda corta penetrante (Paulson & Simpson), E−P sobre S, esfuerzo de `bulk_ncar`/`bulk_kara` | `forcing/bulk.py`, `forcing/surface.py` | balance de calor de columna exacto; convección por enfriamiento |
| 5 | KPP en columnas σ (alternativa a GLS) | `physics/mixing/kpp.py` | mismo Kato–Phillips; tesis: KPP + espectros de ondas internas |
| 6 | Marea 3-D: Flather con η/ū de TPXO, potencial + SAL como fuerza barotrópica | `forcing/tides.py` | PAC_MEX en 3-D homogéneo = 2DH (≤ redondeo de la deriva); después estratificado (etapa E) |
| 7 | Frontera desde un modelo padre: T/S/ū/u′ de HYCOM/ESPC dependientes del tiempo como `T_ext` y velocidad objetivo de la esponja; marea + baja frecuencia en Flather | `forcing/ocean_init.boundary_data` | nido del Golfo de California contra HYCOM-IOA |
| 8 | Medida de reflexión de ondas internas λ = F_r/F_in (Siyanbola et al. 2023) y variante "pretty good sponge" (Lavelle & Thacker 2008) | nuevo | canal modo 1 (ya existe) |

**Estado interno (aprobado el 2026-10-11):**
- Dentro del modelo, el estado son arreglos NumPy en `SigmaState`, compatibles con JAX y Numba. Es lo que hacen MITgcm, ROMS, MOM6, Veros, Oceananigans y FESOM2-JAX.
- xarray se usa solo en la entrada y salida: malla, forzamiento, `OutputWriter` y restart.
- La guía de CLAUDE.md ya está actualizada.

**Registro único de variables (paso 3):**
- **Qué es:** un módulo `core/variables.py` con una entrada por variable: nombre, dimensiones, ubicación en la malla C (t/u/v/w), unidades, nombre estándar CF y descripción. Es el equivalente de `register_diag_field` de MOM6 y del registro de variables de Veros.
- **Quién lo usa:** `diagnostics/output.py` (hoy con su tabla `FIELDS`), `core/restart.py` (hoy `_FIELDS`/`_OPTIONAL`) y `diagnostics/postmortem.py` dejan de tener listas propias y toman los metadatos de ahí.
- **Validación:** un test exige que toda variable de `SigmaState` esté registrada y que la salida y el restart sean CF-válidos.

El modelo z de la fase 1 queda como referencia histórica, con sus tests, y no se extiende.

## Orden dentro del paso 1

Primero lo que BayIW y PAC_MEX necesitan:
1. Trazadores genéricos y salinidad.
2. EOS.
3. Flotabilidad en GLS.
4. Flujos de superficie.
5. Marea 3-D.
6. Frontera desde un modelo padre.
7. KPP.
8. Reflexión.

Cada pieza se cierra con su benchmark y su fila en la tabla de §7 de `vision_roadmap.md`.
