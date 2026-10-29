# Columnas de turbulencia: GLS y KPP de pyOCEAN contra CROCO (`SINGLE_COLUMN`)

N3, tarea 8 (spec `docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md`, criterios 1 y 3, ítem 6.1).
Los siete casos de una columna de CROCO, con sus valores exactos (`docs/design/n2c_refs/croco_column_cases.md`,
`cases.py`). Nada se ajustó.

## Cómo correr

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
# CROCO (≈ 15 s de compilación y 1-20 s de corrida por caso)
validation/turbulence_columns/croco/build_croco.sh kato_phillips gls     # gls | kpp | kppsfc | kppraw
validation/turbulence_columns/croco/run_croco.sh   kato_phillips gls
# pyOCEAN (una corrida a la vez)
python -m validation.turbulence_columns.run_pyocean kato_phillips sigma gls [--production energy]
python -m validation.turbulence_columns.run_pyocean --matrix             # las 26 corridas, ≈ 50 min
python -m validation.turbulence_columns.compare                          # todas las tablas de abajo
```

Salidas: `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/{croco,pyocean}/` (nunca en el repo). Ejecutables de
CROCO: `~/pyOCEAN_refs/croco_runs/scm/<caso>_<cierre>/`.

## Configuración

- **CROCO** (`croco/build_croco.sh`): el `cppdefs.h` del código con `#undef REGIONAL`, `#define SINGLE_COLUMN` y una
  sola clave de caso; `param.h`, `ana_*.F`, `analytical.F` y `prsgrd.F` sin tocar; el `jobcomp` de `tidal_flat3d`
  (cambios para arm64 de `validation/thacker/README.md`). El `croco.in.<CASO>` de `TEST_CASES` se pasa en la línea de
  comandos (evita la errata `KATO_PHILLIPS` de `read_inp.F:127`); solo cambian las rutas de salida, `NINFO` y los
  campos del historial (Akv, Akt, HBL, esfuerzos, `TKE GLS Lscale`). `SINGLE_COLUMN` no define `AVERAGES` (solo el
  bloque REGIONAL, `cppdefs.h:310`), así que la referencia son las fotos horarias de `scm_his.nc`.
  - KPP: `kpp` = `#undef GLS_MIXING`, `LMD_MIXING` con el preset regional (`LMD_SKPP`, `LMD_BKPP`, `LMD_RIMIX`,
    `LMD_CONVEC`, `LMD_NONLOCAL`, `cppdefs.h:255-259`); `kppsfc` = solo `LMD_SKPP` + `LMD_NONLOCAL` (el análogo del
    `'kpp'` de pyOCEAN); en el ciclo diurno además `DIURNAL_SRFLUX`, que solo pone Ric = 0.15
    (`lmd_skpp2005.F:97-103`, McWilliams et al. 2009).
  - **Error de CROCO corregido localmente en `kpp`/`kppsfc` (clase d).** Con EOS lineal, `alfabeta.F:133` (y `:138`)
    devuelve α = |Tcoef| en kg m⁻³ K⁻¹; la rama no lineal divide entre ρ₀ (`alfabeta.F:116-118`) y
    `lmd_skpp2005.F:180-186` necesita α en K⁻¹ (B₀ = g α stflx en m² s⁻³). El flujo de flotabilidad de KPP sale
    ρ₀ = 1024 veces mayor: en Willis–Deardorff la convección mezcla los 50 m en 6 h (h_bl = 50 m, SST 19.37 °C, contra
    15.35 m y 20.77 °C corregido; `kppraw` guarda la versión sin corregir). Solo KPP llama `alfabeta_tile`
    (`lmd_skpp1994.F:177`, `lmd_skpp2005.F:167`); la versión 1994 tiene el mismo problema (probado). En Kato–Phillips
    no hay flujo de calor y no cambia nada. El directorio de corrida lleva una copia de `alfabeta.F` con α = |Tcoef|/ρ₀.
- **pyOCEAN** (`cases.py`): caja doblemente periódica 5 × 5 (Δx = 2 km), σ uniforme, Δt = 30 s, EOS lineal
  (ρ₀ = 1024, en EKBBL 1.22; Tcoef = 0.2, T₀ = 16), c_p = 3985, GLS k-ε Canuto A estratificada con los pisos de CROCO
  (ν ≥ 1e-5, κ_t ≥ 1e-6), KPP con los valores de CVMix (`KPPParams()`, sin fondo). `z0` = Zob (ALE: `hbbl` = ½ Δz, así
  la capa del fondo lleva sola el esfuerzo, como en σ). Coordenadas: `sigma`, `zstar` (ALE z*, capas fijas) e
  `hybrid` (ALE HYCOM1 con las isopicnas iniciales como blancos y la malla σ como piso z*: en la capa de mezcla las
  capas quedan en z*, en su base las interfaces cuya densidad se mezcló colapsan — malla que se mueve y capas casi
  vacías).
  - Dos forzamientos de CROCO no tienen argumento en `step()` y los añade el arnés sin tocar el modelo: la fuerza de
    cuerpo de los FORCED_* (`prsgrd.F:455-462`: ∂v/∂t += f U_G, −g∂ζ/∂y o −g∂ζ/∂y cos ωt) como término explícito con
    nombre `"body_force"` (último de `Momentum.terms`, pasa por AB3 como el gradiente de presión de CROCO), y el
    pequeño flujo de calor del fondo de los tres casos estratificados (`analytical.F:133-136`), que entra en la capa
    más baja con más de 1 mm (`H_MASSLESS`, el piso de capas delgadas de la GLS; en σ y z* la del fondo), para que una
    capa de fondo casi vacía del híbrido no reciba Δt·btflx/h₀ → ∞. `build` verifica que el gancho de la fuerza de
    cuerpo funcionó (el último término es `"body_force"` y cambia la tendencia de v en F) y si no, falla.
  - T inicial de los FORCED_* = 20 °C (rama por defecto, `ana_initial.F:1222`; `croco_column_cases.md` §4 decía T₀ = 16,
    sin efecto: T es uniforme).

**Métricas** (`compare.py`): D_m = profundidad de la primera interfaz, desde arriba, con k < 1e-5 m² s⁻²
(Warner et al. 2005), con log k interpolado; D_m(T) = interfaz con el mayor ∂T/∂z (para KPP, sin k), sin contar las
capas de menos de 1 mm ni el 10 % inferior de la columna (allí el flujo de calor del fondo de CROCO enfría la celda del
fondo y da un gradiente mayor que un salto de arrastre débil: sin esa exclusión CROCO-GLS daba 49.5 m en Kato y 149 m en
el ciclo diurno); ley de Price
D_m = 1.05 u* N₀^(−1/2) t^(1/2) (Warner et al. 2005, ec. 60; 1.05 = (2 R_v)^{1/4} con el número de Richardson global
R_v ≈ 0.6 de Price 1979, l. 30, 481 y 599 de la copia local); errores de perfil = RMS(pyOCEAN − CROCO)/rango de CROCO.

## Resultados

### 1. Kato–Phillips (criterio 1: D_m a ±5 % de CROCO y de Price)

| corrida | D_m 6 h | 12 h | 24 h | 30 h | vs Price (30 h) | vs CROCO (30 h) |
|---|---|---|---|---|---|---|
| Price 1979 | 15.43 | 21.82 | 30.86 | 34.51 | — | — |
| CROCO GLS | 15.88 | 22.61 | 32.11 | 36.04 | +4.4 % | — |
| pyOCEAN σ GLS | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN z* GLS | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN híbrido GLS | 15.91 | 22.70 | 32.16 | 36.07 | +4.5 % | +0.1 % |
| pyOCEAN σ / z* GLS, P = `energy` | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN híbrido GLS, P = `energy` | 15.91 | 22.71 | 32.26 | 36.07 | +4.5 % | +0.1 % |

**Pasa** en las tres coordenadas: a 0.1 % de CROCO y +4.4 % de Price (CROCO mismo da +4.4 %). Perfiles de T, u, k, ε,
ν, κ_t a 30 h: σ y z* = CROCO a < 1e-3 del rango; híbrido ≤ 0.8 %.

### 2. Los otros seis casos GLS contra CROCO (criterio 1: perfiles de T, u, k, ε)

RMS(pyOCEAN σ − CROCO)/rango(CROCO) a la mitad / al final de la corrida (z* da lo mismo, §3):

| caso | T | u | v | k | ε | ν | κ_t | diagnóstico (CROCO / pyOCEAN σ) |
|---|---|---|---|---|---|---|---|---|
| willis_deardorff | 0.000/0.000 | — | — | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | D_m(k) 72 h 12.46 / 12.46 m, SST 20.85 / 20.85 |
| diurnal_cycle | 0.000/0.000 | — | — | 0.002/0.002 | 0.001/0.001 | 0.004/0.004 | 0.003/0.003 | SST día 8: rango 0.340 / 0.339 K, media 21.947 / 21.947; D_m 06 h 26.68 / 26.67 m |
| forced_ekbbl | — | 0.007/0.006 | 0.001/0.002 | 0.005/0.003 | 0.009/0.010 | 0.000/0.000 | 0.039/0.040 | ū 16.66 / 16.68 m/s; ángulo en la capa del fondo **75.8° / 62.8°**; k máx 0.131 / 0.122 |
| forced_dbleek | — | 0.001/0.001 | 0.001/0.002 | 0.001/0.001 | 0.000/0.000 | 0.002/0.008 | 0.001/0.007 | ū 0.84 / 0.84 mm/s, v̄ 5.07 / 5.09 mm/s, ángulo 63.1° / 63.4° |
| forced_nonrotbbl | — | 0.000/0.000 | 0.009/0.009 | 0.000/0.000 | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | v̄ −0.1341 / −0.1331 m/s (ley log: −0.126); ν máx 4.13e-3 / 4.13e-3 |
| forced_oscnonrotbbl | — | 0.000/0.000 | 0.007/0.002 | 0.048/0.029 | 0.034/0.009 | 0.010/0.008 | 0.011/0.009 | v̄ último ciclo: amplitud 0.1113 / 0.1109 m/s, fase −125.5° / −125.5° |

Teoría de convección penetrante (Willis–Deardorff, modelo de salto de orden cero, h² = 2(1 + 2A) B₀ t / N²
[verificar]): 11.3 m sin arrastre (A = 0) y 13.3 m con A = 0.2; CROCO y pyOCEAN dan 12.5 m (k) y 13 m (T).

**Pasa** en cinco de seis casos con errores ≤ 1 % del rango (el ciclo diurno ≤ 0.4 %, la capa oscilante ≤ 5 % en k
y ε a media corrida, en las fases de k pequeño). **FORCED_EKBBL no pasa en la capa del fondo** (clase a, abajo); lejos
del fondo (u, v, ν) coincide a < 1 %.

### 3. σ contra ALE (criterio 3)

max |ALE − σ| / max |σ| en toda la corrida:

| caso | coord. | T | u | v | k | ε | ν | κ_t |
|---|---|---|---|---|---|---|---|---|
| kato_phillips | z* | 8.5e-13 | 2.9e-12 | 0 | 5.2e-12 | 1.3e-12 | 2.2e-10 | 3.7e-10 |
| willis_deardorff | z* | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diurnal_cycle | z* | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| forced_ekbbl | z* | 0 | 9.8e-14 | 2.0e-13 | 4.9e-14 | 7.4e-14 | 0 | 3.4e-13 |
| forced_dbleek | z* | 3.7e-15 | 2.6e-15 | 4.1e-15 | 2.8e-15 | 2.1e-15 | 8.3e-15 | 6.9e-15 |
| forced_nonrotbbl | z* | 1.9e-12 | 0 | 1.1e-12 | 3.0e-12 | 2.2e-12 | 1.8e-10 | 4.0e-10 |
| forced_oscnonrotbbl | z* | 8.3e-13 | 0 | 5.6e-12 | 1.1e-11 | 5.1e-12 | 1.5e-09 | 3.0e-09 |
| kato_phillips | híbrido | 1.9e-4 | 1.9e-3 | 0 | 1.6e-2 | 5.7e-3 | 4.5e-2 | 4.0e-2 |
| willis_deardorff | híbrido | 3.0e-3 | 0 | 0 | 1.2e-1 | 1.1e-1 | 1.8e-1 | 2.5e-1 |
| diurnal_cycle | híbrido | 1.5e-3 | 0 | 0 | 1.9e-1 | 9.7e-2 | 3.8e-1 | 4.8e-1 |

- **z* con capas fijas = σ a redondeo** (≤ 3e-9 relativo en κ_t, que hereda el redondeo de k²/ε cerca del piso; el plan
  acepta hasta 1e-8). **Pasa.**
- **Híbrido (capas que se mueven):** k ≥ 1e-10 y ε ≥ 1e-12 en todas las horas (pisos de CROCO), nada no finito. Los
  máximos del 10-50 % en k, ν, κ_t están en la base de la capa de mezcla, donde las interfaces colapsan: es el máximo
  sobre una interpolación a niveles fijos de un perfil con un salto. Las magnitudes integrales: D_m en Kato a 0.1 % de σ,
  Willis −0.36 m (k) / +1 m (T, una celda), ciclo diurno D_m 06 h día 8 26.42 contra 26.67 m. RMS de D_m híbrido − σ
  en el ciclo diurno 0.72 m (8 días); la mayor diferencia, −5.1 m a las 08 h del día 8, cuando la convección nocturna
  se apaga: en ese momento hay una capa de 1.0 mm justo en la base de la capa de mezcla y ν cae a 1.9e-4 bajo ella
  (σ: 9.3e-3). Las capas de 1 mm–2 cm aparecen en 102 de 193 horas del ciclo diurno (hasta 4 a la vez). Es el umbral
  de fusión de capas delgadas de la GLS (1 mm, `H_MASSLESS`): una capa de exactamente `h_min` = 1 mm no se fusiona.
  Ver "Qué debe mejorar".
- La conservación de ∫k en el remapeo está probada a 1e-12 en `tests/test_gls_ale.py`; aquí no hay un diagnóstico
  aparte de mezcla numérica de k (anti-objetivo 3): la diferencia híbrido − σ en D_m (≤ 0.1 % en Kato con regrid) es la
  cota disponible.

### 4. KPP (decisión 8: contra la teoría; CROCO-KPP solo como referencia, diferencias de constantes = clase c)

| caso | corrida | h_bl final [m] | D_m(T) final [m] | SST final [°C] |
|---|---|---|---|---|
| kato_phillips (Price 34.51 m a 30 h) | pyOCEAN σ / z* KPP | 33.23 | 32.50 | 15.222 |
| | CROCO-KPP (preset) | 41.36 | 43.50 | 15.014 |
| | CROCO-GLS | — | 36.00 | 15.264 |
| willis_deardorff (teoría 11.3–13.3 m) | pyOCEAN σ / z* KPP | 13.36 | 14.00 | 20.855 |
| | CROCO-KPP (corregido) | 15.35 | 16.00 | 20.772 |
| | CROCO-GLS | — | 13.00 | 20.852 |
| diurnal_cycle (fin = medianoche del día 8) | pyOCEAN σ / z* KPP | 19.85 | 27.00 | 21.857 |
| | CROCO-KPP (corregido, Ric 0.15) | 37.46 | 37.00 | 21.809 |
| | CROCO-GLS | — | 28.00 | 21.851 |

Kato–Phillips h_bl a 6/12/24/30 h: pyOCEAN 16.6 / 22.2 / 30.1 / 33.2 m (Price 15.4 / 21.8 / 30.9 / 34.5): **−3.7 % de
Price a 30 h, pasa ±5 %**. CROCO-KPP 19.7 / 27.6 / 36.9 / 41.4 (+20 %; `kppsfc` igual, 43.9). Ciclo diurno, día 8 a
las 06 h: pyOCEAN KPP h_bl 27.3 m (= GLS 27 m), CROCO-KPP 37.4 m; rango de la SST 0.346 (pyOCEAN KPP), 0.314
(CROCO-KPP), 0.340 K (GLS). σ = z* en KPP (las tablas son idénticas a los decimales mostrados; la prueba de 1e-10 está en
`tests/test_kpp_model.py`).

### 5. `gls_production="energy"` (Burchard 2002) — recomendación para Esteban

En Kato–Phillips (Δz = 0.5 m, Δt = 30 s) `energy` y `croco` dan el mismo D_m a < 1 cm en σ y z* (0.1 m en el
híbrido); los campos difieren ≤ 0.4 % (k), 0.12 % (ν), 2e-5 (T). El caso no distingue las dos formas: con capas finas
y Δt corto la energía que quita la viscosidad implícita y ν S² son casi iguales. **Recomendación:** dejar `croco` por
defecto (conserva el bit a bit de la matriz dorada, criterio 6) y no cambiarlo por este caso; volver a decidir con
Lorenz y DOME (capas gruesas, Δt de 600 s), donde Burchard (2002) espera la diferencia.

### 6. Tiempos de reloj (columna periódica 5 × 5, un proceso, ≤ 5 hilos Numba)

| corrida | pasos | σ GLS | z* GLS | híbrido GLS | σ KPP | z* KPP |
|---|---|---|---|---|---|---|
| kato_phillips (N = 100) | 3600 | 35 s (9.8 ms/paso) | 54 s (15.0) | 54 s (15.0) | 54 s (15.0) | 69 s (19.1) |
| willis_deardorff (50) | 8640 | 30 s (3.4) | 53 s (6.1) | 50 s (5.8) | 51 s (5.9) | 68 s (7.9) |
| diurnal_cycle (150) | 23040 | 250 s (10.9) | 379 s (16.4) | 375 s (16.3) | 377 s (16.4) | 509 s (22.1) |
| forced_ekbbl (40) | 2880 | 19 s (6.5) | 25 s (8.7) | — | — | — |
| forced_dbleek (25) | 2880 | 12 s (4.3) | 17 s (6.0) | — | — | — |
| forced_nonrotbbl (50) | 3600 | 21 s (5.8) | 31 s (8.5) | — | — | — |
| forced_oscnonrotbbl (50) | 7200 | 41 s (5.7) | 60 s (8.4) | — | — | — |

CROCO: 0–19 s por caso (un núcleo). pyOCEAN es 10-25 veces más lento en estas columnas; buena parte es el halo de 13
celdas de la advección periódica (31 × 31 celdas para una columna 5 × 5).

## Clasificación de las diferencias

| diferencia | clase | evidencia |
|---|---|---|
| FORCED_EKBBL: ángulo de la capa del fondo 62.8° contra 75.8°, k máx −7 %, κ_t 4 % | **(a) nuestro** | El arrastre log de pyOCEAN usa la velocidad de la propia componente en cada cara (C_d \|u\| u, C_d \|v\| v), CROCO usa \|U\| = √(u² + v²) (nota del ledger, tarea 4). En la capa del fondo CROCO tiene (u, v) = (0.60, 2.36) m/s, pyOCEAN (1.18, 2.31): con \|u\| = 1.18 en vez de \|U\| ≈ 2.4 el esfuerzo sobre u sale a la mitad y u del fondo al doble. En DBLEEK (u ≈ v pequeños) el efecto es 0.3°, en los casos sin rotación (solo v) cero. No se cambia el modelo en esta tarea. |
| CROCO-KPP con EOS lineal: B₀ × ρ₀ | **(d) CROCO** | `alfabeta.F:133` vs `:116-118`; corregido localmente (arriba). |
| KPP de pyOCEAN contra CROCO-KPP: h_bl −20 % (Kato), −13 % (Willis), −27 % (ciclo diurno 06 h) | **(c)** | CVMix (Ri_c = 0.3, sin núcleo Kern) contra el KPP 2005 "if-less" de CROCO (Ric = 0.45, 0.15 diurno, `lmd_skpp2005.F:95-117`) y el preset con Ri interior y convección. pyOCEAN queda más cerca de Price (−3.7 %) y de la teoría de Willis (13.4 contra 11.3–13.3 m). |
| Híbrido − σ: D_m −5.1 m en una hora del ciclo diurno; ν, κ_t en la base de la capa de mezcla | **(c)**, con un punto abierto | Malla distinta (capas que colapsan en la base). La capa de 1 mm sin fusionar justo en la base es el umbral de 1 mm de la tarea 5. |
| FORCED_OSCNONROTBBL: k, ε 3-5 % del rango a media corrida | **(c)** | En las fases de inversión del flujo, k pequeño; la amplitud y fase de v̄ coinciden a 0.4 % y 0°. |
| FORCED_NONROTBBL v̄ 0.7 % menor que CROCO | **(c)** | Cambia la discretización temporal del esfuerzo de fondo (σ de pyOCEAN: C_d \|u*\| linealizado sobre u + Δt G) — no investigado más; está dentro del 1 %. |

## Qué debe mejorar pyOCEAN

1. **Arrastre log con la rapidez completa** √(u² + v²) en cada cara (como CROCO y MOM6) en σ y ALE: hoy en una capa de
   Ekman de fondo el ángulo de giro sale 13° menor. Afecta Lorenz/DOME/regional con rotación. (clase a)
2. **Umbral de fusión de capas delgadas de la GLS** (1 mm): en el híbrido las capas de 1 mm–2 cm aparecen la mitad del
   tiempo en la base de la capa de mezcla diurna, y una de 1.0 mm sin fusionar cortó la mezcla 5 m durante una hora.
   Evaluar el umbral relativo/dependiente de Δt de MOM6 (≈ 8 mm) antes de DOME híbrido.
3. **Costo de las columnas periódicas:** 10-25 × CROCO; un camino sin gradientes horizontales (o un halo mínimo) para
   columnas haría baratas las calibraciones y las columnas de MOM6 (tarea 9).
4. **Fuerza de cuerpo y flujo de calor de fondo como entradas del paso** (hoy en el arnés): las necesitan las columnas y
   cualquier caso idealizado con gradiente de presión impuesto.
5. **Diagnóstico de mezcla numérica de k y ε en el remapeo** (anti-objetivo 3): no existe todavía; aquí solo la cota
   indirecta de D_m.

## Archivos

`cases.py` (definición de los casos y `build`), `run_pyocean.py` (corridas, fotos horarias), `compare.py` (métricas y
tablas), `croco/build_croco.sh`, `croco/run_croco.sh`; test `tests/test_turbulence_columns.py` (casos = CROCO, fuerza
de cuerpo, definición de D_m, Kato–Phillips 6 h a ±10 % de Price).
