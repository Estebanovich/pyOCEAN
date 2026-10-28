# Benchmark: estuario idealizado de Lorenz et al. (2021)

- **Spec:** `docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md` §6
- **Plan:** `docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md`, Task 8
- **Artículo:** Lorenz, Klingbeil & Burchard (2021), *Impact of evaporation and precipitation on estuarine mixing*, JPO 51, 1319–1333, doi:10.1175/jpo-d-20-0158.1. Copia local: `Lorenz_2021 - Impact of Evaporation and Precipitation on Estuarine Mixing.md` (§4b l. 838-857, Tabla 1 l. 881-906, pie de la Fig. 2 l. 827-837, Tabla 2 l. 923-1030); la posición de los transectos se leyó de la Fig. 2 del PDF (p. 1324).

## Qué es

Un canal 2-D con evaporación, precipitación, río y marea, donde la mezcla total del modelo dentro del estuario debe ser **exactamente** la de la relación (32) del artículo, evaluada con TEF en el transecto. GETM lo cumple "down to machine accuracy" (Tabla 2). En pyOCEAN prueba juntos la salinidad, el agua dulce real, el balance de varianza discreto (Task 6) y el TEF dentro del modelo (Task 7). Después se comparan las magnitudes con la Tabla 2 de GETM.

## Configuración

| | Valor | Fuente |
|---|---|---|
| Canal | 100 km, dx = dy = 200 m, 500 × 1 celdas | §4b |
| Fondo | lineal de 15 m (océano, oeste) a 5 m (río, este); H en los centros de celda | §4b |
| Vertical | 60 capas σ uniformes | §4b |
| Frontera oceánica | S = 30 g/kg; M2 de 0.5 m en B–D, η_ext = a sin(ωt), ū_ext = 0 | §4b; la forma de imponerla: **[no dado en el artículo]** |
| Agua dulce | río, E integrada y P integrada: 5.0 m³/s cada uno (Tabla 1) | §4b, Tabla 1 |
| Zonas | A: E en la mitad interior (x ≥ 50 km). B–D: E en el tercio medio (celdas 167–332), P en el tercio este (333–499) | pie de la Fig. 2 |
| Transectos | borde oeste de la zona de evaporación: cara 250 (x = 50 km) en A, cara 167 (x = 33.4 km) en B–D; la región es todo lo que está al este | líneas discontinuas de la Fig. 2 |
| T | constante (10 °C), α = 0 | §4b ("temperature is kept constant"); el valor: **[no dado en el artículo]** |
| Integración | 500 ciclos M2 hasta el estado cuasi periódico + promedio de 100 ciclos (600 en total); en A, sin marea, el ciclo M2 es solo la unidad de tiempo | §4b l. 851-857 |
| TEF | 250 bins de 0 a 40 g/kg, cada paso, con los flujos efectivos de la advección | §4b l. 846-849 |

| Tabla 1 | Q_r | Q_evap | Q_precip | M2 |
|---|---|---|---|---|
| A | 0 | −5 | 0 | 0 |
| B | 5 | −5 | 0 | 0.5 m |
| C | 0 | −5 | 5 | 0.5 m |
| D | 5 | −5 | 5 | 0.5 m |

**Parámetros que el artículo no da** (ninguno ajustado a la Tabla 2):

| Parámetro | Valor | Elección |
|---|---|---|
| ρ₀ | 1025 kg/m³ | **[no dado en el artículo]**; `rho_0` de GETM [verificar] |
| β (EOS lineal) | 0.78 kg m⁻³ (g/kg)⁻¹, S₀ = 30 | **[no dado en el artículo]**; `dsr0` de la EOS lineal de GETM [verificar]. GETM puede usar UNESCO por defecto (β ≈ 0.77 a S = 30, T = 10 °C) |
| κ_v de S (las dos configuraciones) y ν (ALE-σ) | 1e-4 m²/s constante | **[no dado en el artículo]**; GETM usa k-ε de GOTM para todo y fondo `avmback = avhback = 0` [verificar]; 1e-4 es el `nu_const` por defecto de pyOCEAN, la constante más simple |
| ν_h | 0 | **[no dado en el artículo]**; `Am = 0` de GETM [verificar] |
| κ_h | 0 | lo exige la identidad exacta (el flujo difusivo en el transecto no tiene volumen y TEF no lo puede clasificar) |
| Rugosidad del fondo | z0 = 1e-3 m (σ + GLS, ley log); C_d = [κ/(ln(D/z0) − 1)]² = 2.37e-3 con D = 10 m sobre la velocidad media de la columna (ALE-σ, `hbbl` = 20 m) | **[no dado en el artículo]**; el mismo z0 en las dos configuraciones (derivado, no ajustado) |
| Esponja oeste | 10 celdas (2 km), perfil MITgcm, τ = (1200, 60) s | **[no dado en el artículo]**; GETM impone S en la frontera sin esponja; es la frontera probada de pyOCEAN (Flather + Orlanski + esponja), a ≥ 31 km de los transectos |
| Estado inicial | reposo, η = 0, S = 30 en todo el canal | **[no dado en el artículo]** |
| f | 0 (corte 2-D de una fila) | **[no dado en el artículo]** |
| Δt | T_M2/373 ≈ 119.9 s en las dos configuraciones (número entero de pasos por ciclo) | búsqueda abajo |

**Configuraciones de pyOCEAN** (spec §6, Esteban):

| | `sigma_gls` | `ale_sigma` |
|---|---|---|
| Coordenada | σ euleriana | ALE con regla σ, PGF de volumen finito |
| Cierre | GLS k-ε (Canuto A) en el momento; S con κ_v constante | ν y κ constantes |
| Fondo | ley logarítmica, z0 = 1e-3 m | arrastre cuadrático, C_d = 2.37e-3 |
| Trazadores | WENO5, SSP-RK3 | WENO5 + FCT, SSP-RK3, remapeo PPM_H4 |

**Diferencias ya conocidas antes de correr:**
- **GLS no actúa sobre los trazadores (b).** En `sigma_gls`, la difusividad vertical de S es la constante `kv`, aunque GLS cambie ν (N2a Review Focus 5; `tests/test_lorenz_estuary.py::test_gls_config_uses_constant_tracer_kv`). Además nuestro GLS no tiene término de flotabilidad (`physics/mixing/gls.py:12`). En GETM el k-ε mezcla también la sal. Es física faltante conocida; no se "corrige" en N2a: la diferencia medida es el insumo de N3.
- **Advección (c).** GETM usa TVD Superbee (Pietrzak 1998); pyOCEAN, WENO5 (más FCT en ALE).
- **Marea en la frontera (c, por confirmar).** Flather hacia η_ext = 0.5 sin ωt con ū_ext = 0 deja una amplitud de **0.16 m en la celda oceánica y 0.50 m en la cabeza** (B, ALE-σ, cuarto ciclo). El canal mide ≈ 0.225 λ_M2, cerca del cuarto de onda, así que la respuesta de la boca es casi un nodo. Si GETM impone η = 0.5 m en la frontera (elevación prescrita), su marea en la cabeza sería mucho mayor. El artículo solo dice "an M2 tidal amplitude of 0.5 m".

**Supuestos de la identidad exacta** (`case.check_identity_assumptions`, se verifican al construir): κ_h = 0; ninguna celda de esponja en la región; toda cara abierta de la región es un río `'clamped'` de agua dulce (S = 0) que solo entra; salinidad, balance de varianza y TEF activos.

## Búsqueda de Δt (5 hilos, Δt = T_M2/N)

Estable = campos finitos, S en un rango físico y sin truncamientos CFL en ALE, durante 4–10 ciclos M2 desde el estado inicial.

| Config | Exp. | Δt [s] (N) | Resultado |
|---|---|---|---|
| `sigma_gls` | B | 596 (75) | inestable a las 25 h |
| `sigma_gls` | B, D | 452 (99), 361 (124) | estables 10 ciclos |
| `sigma_gls` | A | 452, 361, 240 (186) | inestable a las 22–25 h: la advección vertical explícita de S en la capa superior (≈ 0.13 m) bajo el chorro superficial de evaporación (u ≈ 0.09 m/s en x ≈ 60 km) |
| `sigma_gls` | A, C | 120 (373) | estables 10 ciclos |
| `ale_sigma` | B | 600 (75), 361 (124) | > 1000 truncamientos CFL |
| `ale_sigma` | B | 240 (186) | 631 truncamientos en 4 ciclos |
| `ale_sigma` | B | 200 (224) | 7 truncamientos en 10 ciclos |
| `ale_sigma` | A, B, C, D | 180 (248) | limpio, 10 ciclos |

Se usa **Δt = T_M2/373 ≈ 119.9 s** en las dos configuraciones: el mayor estable que se encontró para σ + GLS (lo limita A), con margen 1.5× en ALE-σ. El estado cuasi periódico (cientos de ciclos) puede ser más exigente que estos 10 ciclos; el runner revisa que los campos sean finitos cada ¼ de ciclo y guarda un checkpoint cada 25 ciclos.

## Cómo reproducirlo

```bash
python validation/lorenz_estuary/run_pyocean.py B sigma_gls --hours 2           # costo: 2 h simuladas, no escribe nada
python validation/lorenz_estuary/run_pyocean.py B sigma_gls                     # 600 ciclos, promedio de los últimos 100
python validation/lorenz_estuary/run_pyocean.py B sigma_gls --resume            # sigue desde el último checkpoint
python validation/lorenz_estuary/compare.py                                     # Tabla 2: GETM contra pyOCEAN
```

Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary/<exp>_<config>/`, nunca al repositorio: `summary.json`, `section.npz`, `series.npz` y el checkpoint (`restart.nc` + `restart_extra.pkl`, con los acumuladores de TEF que `core/restart.py` no guarda). Reanudar reproduce bit a bit la corrida sin interrumpir (verificado en B, las dos configuraciones, 0.5 ciclos).

## Resultados

Pendiente: las 8 corridas, la Tabla 2 de pyOCEAN contra GETM, la clasificación (a)–(d) de cada diferencia y la lista "Qué debe mejorar pyOCEAN" (Task 8, paso 4).
