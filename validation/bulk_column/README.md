# Columna ERA5: pyOCEAN contra MITgcm (LY04) y CROCO (COARE 3.0)

N2b, tarea 7: el criterio 5 del spec `docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md`.
Una columna periódica de la plataforma noreste de EE. UU. se fuerza 45 días con ERA5 horario. La comparación es con
dos modelos de referencia, cada uno con su propia fórmula bulk: MITgcm `pkg/exf` (LY04, contra `'ncar_ly04'`) y CROCO
`BULK_FLUX` (COARE 3.0, contra `'coare'`). Los tres usan KPP. Todos los valores compartidos están en `cases.py`, y
`tests/test_bulk_column.py` comprueba que los insumos de las referencias los repiten. No se ajustó nada.

## Cómo correr

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
python -m validation.bulk_column.forcing --days 45            # serie ERA5 en el punto, insumos MITgcm y CROCO
validation/bulk_column/mitgcm/run_mitgcm.sh build && validation/bulk_column/mitgcm/run_mitgcm.sh run 45   # ≈ 1 min
validation/bulk_column/croco/build_croco.sh && validation/bulk_column/croco/run_croco.sh 45               # ≈ 10 s
python -m validation.bulk_column.run_pyocean --all --days 45  # ly04, coare, coare_croco, fms: ≈ 7.5 min cada una
for t in flux_mitgcm flux_croco coare_bg flux_croco_bg; do python -m validation.bulk_column.run_pyocean $t; done
python -m validation.bulk_column.compare --figs
pytest tests/test_bulk_column.py -q
```

Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column/` (nunca al repo): `series_45d.npz`, `mitgcm_45d/`,
`croco_45d/`, `pyocean_45d/*.npz` y `figs/`. Las compilaciones van a `~/pyOCEAN_refs/build/bulk_column/mitgcm` y
`~/pyOCEAN_refs/croco_runs/bulk_column`.

## 1. Punto, periodo y estado inicial

- **Punto P = 40.24 °N, 73.16 °W**, a 1.3 km al sur-suroeste de la boya NDBC 44025 (40.251 °N, 73.164 °W, 36.3 m
  [verificar]).
  - ERA5 tiene valores sobre tierra y `fill_land` no los quita (revisión de la tarea 3, m4). Por eso el punto se
    elige para que el estencil bilineal de ERA5 sean los cuatro nodos de mar 40.00/40.25 °N × 73.25/73.00 °W.
  - Prueba sin descargar la máscara: el rango diario medio de t2m en julio de 2017 es de 1.55–1.71 K en esos
    nodos y de 1.7–1.8 K en mar abierto (70 °W). Los nodos que tocan Long Island (40.50 °N, 73.25 °W: 1.91 K) o
    Nueva Jersey (74.00 °W: 4–7 K) tienen más.
  - La posición de la boya (40.251 °N) daría un peso de 0.4 % a la fila de 40.50 °N, que es costera.
  - No se pidió nada nuevo a CDS.
- **Columna:** H = 36 m (la profundidad de 44025), 36 capas uniformes de 1 m en los tres modelos.
  - pyOCEAN: σ, caja 5 × 5 doblemente periódica, Δx = 2 km.
  - CROCO: `SINGLE_COLUMN`, 5 × 5 periódica, σ uniforme.
  - MITgcm: 1 × 1 periódica, como `verification/1D_ocean_ice_column`.
  - f = 2Ω sin φ = 9.42e-5 s⁻¹. Δt = 120 s.
- **Periodo:** 2017-07-01 00:00 → 2017-08-15 00:00 UTC (45 días).
- **Estado inicial:** reposo, con T y S de WOA23 julio (decav, 0.25°, `t_an`/`s_an`) en la celda 40.0–40.25 °N ×
  73.25–73.0 °W que contiene a P, interpolados linealmente a los centros de capa.
  - Fuente: OPeNDAP
    `https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/decav/0.25/woa23_decav_t07_04.nc`
    (y `salinity/…s07…`), leída el 2026-10-31; el subconjunto está en `RUNS/woa/`.
  - Perfil: 22.1 °C / 30.8 g/kg en superficie y 6.8 °C / 32.5 a 40 m. Es la "piscina fría" del verano.

## 2. Configuración compartida y elecciones

| | pyOCEAN | MITgcm (git 27256b2) | CROCO (git ab63b58) |
|---|---|---|---|
| bulk | `'ncar_ly04'` (+ `'ncar'` FMS informativo) / `'coare'` | `exf` con `ALLOW_BULK_LARGEYEAGER04`, `EXF_CALC_ATMRHO` | `bulk_flux.F` por defecto (COARE 3.0, `BULK_GUSTINESS`, `BULK_LW`) |
| viento | relativo a la corriente de la capa superior | `useRelativeWind` | absoluto (CROCO) |
| T, q, presión | 2 m, 2 m; msl en `'ncar_ly04'`/`'ncar'`, 1e5 Pa constante en `'coare'` (como CROCO, `surface_fluxes.py` `PSURF`) | 2 m (`ht`), `apressure` = msl | 2 m, `rhum` = q en g/kg (regla m2 de la tarea 2), p = 1e5 Pa constante |
| onda corta | (1 − 0.066) SW↓, Jerlov III | `exf_albedo` 0.066, `swfrac.F` jwtype = 5 | `radsw` neta = (1 − 0.066) SW↓, `lmd_swfrac.F` lmd_Jwt = 5 |
| onda larga | 0.97 (LW↓ − σT⁴) (LY04) / LW↓ − 0.985 σT⁴ (COARE) | `EXF_LWDOWN_WITH_EMISSIVITY`, 0.97 | `BULK_LW`, 0.985 σT⁴ sin reflexión |
| agua dulce | real, fw = (P − E)/ρ₀ | sal virtual (`convertFW2Salt` = −1) | sal virtual (E − P)·S |
| EOS | lineal, ρ₀ = 1025, α = 2.0e-4, β = 7.5e-4, T₀ = 15, S₀ = 31.5 | `LINEAR`, los mismos | lineal, R0 = 25, Tcoef = 0.205, Scoef = 0.76875; `RESET_RHO0` cambia ρ₀ a 1025.41 |
| c_p, g | 3985, 9.81 | 3985, 9.81 | 3985, 9.81 |
| KPP | CVMix (`KPPParams()`), sin fondo | `pkg/kpp`, difm0 = difs0 = dift0 = dif*con = 0, `LimitHblStable` = F, sin fondo | `LMD_SKPP` + `LMD_NONLOCAL`; interior ν = 1e-4, κ = 1e-5 fijo en `lmd_vmix.F:116-117` |
| fondo | ley del muro, z₀ = 1 mm | `zRoughBot` = 1 mm (κ = 0.4) | Zob = 1 mm (κ = 0.41) |

Elecciones del constructor:

- **Referencia NCAR = MITgcm.** MOM6 no calcula el bulk. En la configuración acoplada lo hace `surface_flux` de
  FMScoupler dentro del intercambio con el hielo (SIS2), y SIS2 no está en disco. El `solo_driver` solo lee flujos ya
  formados (`docs/design/n2b_refs/mom6_fms_bulk.md` §0). Por eso la referencia es MITgcm `exf`, la misma LY04 que
  `'ncar_ly04'` (`validation/bulk_points/README.md`: 4.7e-3 relativo punto a punto con W ≥ 0.5 m/s). `'ncar'` (FMS)
  se corre también, solo como información.
- **Jerlov III en los tres modelos.** MITgcm fija IA y CROCO fija I, y las dos copias de compilación se parchan.
  Con el tipo I, el 8.8 % de la onda corta llega al fondo a 36 m, y los modelos lo tratan distinto: pyOCEAN lo
  reparte en la columna como MOM6, MITgcm y CROCO lo dejan en la capa del fondo. La plataforma en verano es agua
  costera [verificar]. Aun con III, el 0.26 % llega al fondo (0.6 W m⁻²), y MITgcm calienta la capa del fondo
  0.6 K en 45 días.
- **KPP solo de superficie y sin fondo.** Es la forma por defecto de pyOCEAN (anti-objetivo del spec).
  - MITgcm se configura igual.
  - CROCO no puede: con `LMD_MIXING`, `lmd_vmix.F` siempre suma ν = 1e-4 y κ = 1e-5 bajo la capa límite, y
    `Akv_bak`/`Akt_bak` de `croco.in` no se leen. El efecto se separa con las corridas `*_bg` (§4).
- **Fondo:** la ley del muro con z₀ = 1 mm da el mismo C_d en pyOCEAN y CROCO, y en MITgcm hay diferencias de
  κ y de forma, ~3 %.
- **Parches de CROCO** (todos en copias de compilación, documentados en `croco/build_croco.sh`): la clave de caso
  `BULK_COLUMN` (N, profundidad, f, perfil inicial y flujo de calor del fondo nulo), `lmd_Jwt`, y la corrección de
  `alfabeta.F` de N3 (clase d: α = Tcoef sin 1/ρ₀ con EOS lineal).
- **El archivo bulk de CROCO lleva un registro extra** al final (copia del último en t + 1 h). `get_bulk.F` pide el
  registro siguiente al último paso y se detiene si no existe; ese registro tiene peso 0 antes de t_end.
- **Mismo ERA5 en los tres modelos.** Es la serie del proveedor de pyOCEAN en sus propios registros horarios
  (`cases.point_series`). La interpolación es lineal en el tiempo en los tres. La onda corta media coincide a < 1e-3
  entre los tres, y la prueba lo exige.

## 3. Resultados (45 días)

### Flujos medios (+ hacia el océano; entre paréntesis, la diferencia relativa con la referencia)

| flux | MITgcm exf LY04 | pyOCEAN 'ncar_ly04' | CROCO COARE 3.0 | pyOCEAN 'coare' | pyOCEAN 'coare', CROCO coupling | pyOCEAN 'ncar' (FMS) |
|---|---|---|---|---|---|---|
| Q_sw [W m⁻²] | 238.15 | 238.15 (-0.0 %) | 238.15 | 238.15 (+0.0 %) | 238.15 (+0.0 %) | 238.15 (-0.0 %) |
| Q_lw [W m⁻²] | -61.74 | -60.24 (+2.4 %) | -52.79 | -54.75 (-3.7 %) | -53.80 (-1.9 %) | -60.38 (+2.2 %) |
| Q_sens [W m⁻²] | -13.06 | -10.96 (+16.1 %) | -7.19 | -10.35 (-44.1 %) | -8.99 (-25.2 %) | -11.33 (+13.3 %) |
| Q_lat [W m⁻²] | -89.68 | -82.92 (+7.5 %) | -76.18 | -85.55 (-12.3 %) | -81.56 (-7.1 %) | -82.29 (+8.2 %) |
| Q_net [W m⁻²] | 73.67 | 84.03 (+14.1 %) | 102.00 | 87.51 (-14.2 %) | 93.79 (-8.0 %) | 84.15 (+14.2 %) |
| τ_x [N m⁻²] | 0.0029 | 0.0029 (-1.6 %) | 0.0027 | 0.0024 (-9.8 %) | 0.0025 (-7.3 %) | 0.0029 (-1.5 %) |
| τ_y [N m⁻²] | 0.0104 | 0.0103 (-1.0 %) | 0.0106 | 0.0104 (-1.8 %) | 0.0105 (-1.4 %) | 0.0103 (-0.9 %) |
| |τ| [N m⁻²] | 0.0406 | 0.0403 (-0.7 %) | 0.0439 | 0.0433 (-1.4 %) | 0.0440 (+0.2 %) | 0.0403 (-0.6 %) |
| E [mm d⁻¹] | 3.099 | 2.866 (-7.5 %) | 2.691 | 3.025 (+12.4 %) | 2.884 (+7.1 %) | 2.844 (-8.2 %) |
| P [mm d⁻¹] | 3.770 | 3.770 (-0.0 %) | 3.770 | 3.770 (+0.0 %) | 3.770 (+0.0 %) | 3.770 (-0.0 %) |

### SST y capa de mezcla (Δ = pyOCEAN − referencia; "end" = la última foto común, 1079 h para MITgcm)

| run | SST start | SST end | ΔSST end | ΔSST last day | RMS ΔSST | max ΔSST | MLD mean | ΔMLD RMS | hbl mean | SSS end |
|---|---|---|---|---|---|---|---|---|---|---|
| MITgcm exf LY04 | 21.998 | 25.300 | — | — | — | — | 7.33 | — | 8.41 | 31.070 |
| pyOCEAN 'ncar_ly04' | 21.998 | 24.945 | -0.300 | -0.353 | 0.288 | -0.476 | 7.76 | 1.01 | 8.58 | 31.145 |
| CROCO COARE 3.0 | 21.998 | 24.566 | — | — | — | — | 6.39 | — | 7.20 | 31.283 |
| pyOCEAN 'coare' | 21.998 | 24.889 | +0.323 | +0.316 | 0.597 | +1.442 | 7.99 | 2.25 | 8.72 | 31.194 |
| pyOCEAN 'coare', CROCO coupling | 21.998 | 24.715 | +0.149 | +0.131 | 0.496 | +1.304 | 8.09 | 2.38 | 8.91 | 31.231 |
| pyOCEAN 'ncar' (FMS) | 21.998 | 24.956 | -0.289 | -0.343 | 0.272 | -0.476 | 7.75 | 1.00 | 8.56 | 31.143 |
| pyOCEAN, MITgcm fluxes | 21.998 | 24.000 | -1.256 | -1.288 | 0.775 | -1.359 | 8.13 | 1.24 | 8.92 | 31.188 |
| pyOCEAN, CROCO fluxes | 21.998 | 25.504 | +0.938 | +0.910 | 0.803 | +1.222 | 7.94 | 2.43 | 8.82 | 31.194 |
| pyOCEAN, CROCO fluxes + CROCO bg | 21.998 | 25.246 | +0.680 | +0.608 | 0.420 | +0.712 | 6.09 | 1.11 | 6.60 | 31.259 |
| pyOCEAN 'coare' + CROCO bg | 21.998 | 24.761 | +0.195 | +0.128 | 0.320 | +0.970 | 6.02 | 1.09 | 6.49 | 31.269 |

### Separación fórmula / estado (`compare.split`)

Cada esquema de pyOCEAN se evalúa fuera de línea sobre la SST y la corriente de superficie del otro modelo. "Fórmula
en el estado de la referencia" es la diferencia de la fórmula sola. "Estado" es lo que cambia porque la SST y la
corriente de pyOCEAN son otras. "Aprox." es el error de la evaluación fuera de línea, que usa SST y corriente a
medio intervalo: ≤ 0.16 W m⁻².

| run | flux | total py − ref | formula at ref state | state (SST, current) | offline approx. |
|---|---|---|---|---|---|
| ly04 | Q_lw | +1.500 | -0.029 | +1.529 | -0.000 |
| ly04 | Q_sens | +2.099 | +0.093 | +2.033 | +0.026 |
| ly04 | Q_lat | +6.760 | +0.270 | +6.617 | +0.127 |
| ly04 | Q_net | +10.359 | +0.334 | +10.179 | +0.153 |
| ly04 | |τ| | -0.0003 | -0.0001 | -0.0003 | -0.0001 |
| ly04 | E | -0.234 | -0.009 | -0.229 | -0.004 |
| coare | Q_lw | -1.955 | -0.001 | -1.955 | -0.001 |
| coare | Q_sens | -3.166 | +0.087 | -3.234 | +0.019 |
| coare | Q_lat | -9.369 | +0.656 | -9.886 | +0.138 |
| coare | Q_net | -14.489 | +0.743 | -15.075 | +0.157 |
| coare | |τ| | -0.0006 | -0.0012 | +0.0004 | -0.0001 |
| coare | E | +0.334 | -0.022 | +0.351 | -0.005 |
| coare_croco | Q_lw | -1.008 | -0.001 | -1.008 | -0.001 |
| coare_croco | Q_sens | -1.807 | +0.016 | -1.806 | +0.016 |
| coare_croco | Q_lat | -5.386 | +0.124 | -5.381 | +0.129 |
| coare_croco | Q_net | -8.201 | +0.140 | -8.196 | +0.145 |
| coare_croco | |τ| | +0.0001 | -0.0001 | +0.0001 | -0.0001 |
| coare_croco | E | +0.192 | -0.003 | +0.191 | -0.005 |
| fms | Q_lw | +1.355 | -0.029 | +1.383 | -0.000 |
| fms | Q_sens | +1.734 | -0.099 | +1.859 | +0.026 |
| fms | Q_lat | +7.395 | +1.283 | +6.249 | +0.137 |
| fms | Q_net | +10.483 | +1.154 | +9.491 | +0.163 |
| fms | |τ| | -0.0003 | -0.0001 | -0.0002 | -0.0001 |
| fms | E | -0.256 | -0.044 | -0.216 | -0.005 |

## 4. Diferencias clasificadas

**Criterio 5: los flujos medios difieren más del 2 % y la SST final más de 0.2 °C, salvo `'coare'` con el
acoplamiento de CROCO. Las diferencias quedan explicadas: con el mismo estado, las fórmulas coinciden dentro del 2 %;
el resto viene del modelo oceánico (KPP).**

| # | Diferencia | Evidencia | Clase |
|---|---|---|---|
| 1 | **Fórmula LY04 de pyOCEAN contra `exf`**, con la misma SST y corriente: Q_net +0.33 W m⁻² (0.45 %), Q_lat +0.27 (0.3 %), Q_sens +0.09 (0.7 %), Q_lw −0.03, \|τ\| −0.25 %, E −0.3 % | separación, fila `ly04`, "fórmula"; dentro del 2 % en todos los flujos. Son las elecciones de constantes de `validation/bulk_points/README.md` (c_p 1000.5/1005, Γ 0.0098/0.010, viento mínimo en los flujos, ρ_a de θ o de T) | c |
| 2 | **Fórmula COARE de pyOCEAN contra `bulk_flux.F`**, con el estado de CROCO y el viento absoluto de CROCO: Q_net +0.14 W m⁻² (0.14 %), Q_lat +0.12, \|τ\| −0.2 % | separación, fila `coare_croco`; el port es idéntico al compilado (tarea 2). Lo que queda es la evaluación fuera de línea (aprox. +0.15) | — (a redondeo de la evaluación) |
| 3 | **Viento relativo** (pyOCEAN por defecto) contra el viento absoluto de CROCO: \|τ\| −2.7 % en el estado de CROCO, τ_x −10 %; la SST final pasa de +0.32 a +0.15 °C contra CROCO | `coare` contra `coare_croco`, fila "fórmula" de \|τ\| (−0.0012 contra −0.0001 N m⁻²). Con `h_sfc` = 1 m = la capa superior de 1 m, el promedio de SST no cambia nada aquí | c (elección de CROCO; LY04 y MOM6 usan viento relativo) |
| 4 | **SST de pyOCEAN 0.30 °C bajo MITgcm** al final (RMS 0.29) y, por eso, flujos medios mayores hacia el océano: Q_net +14 %, Q_lat +7.5 %, Q_sens +16 %, E −7.5 % | La separación pone 98 % de la diferencia de Q_net en el estado. Con los **flujos horarios de MITgcm prescritos** (`flux_mitgcm`), el océano de pyOCEAN acaba **1.26 °C** bajo MITgcm: mezcla más hondo (MLD 8.1 contra 7.3 m, hbl 8.9 contra 8.4 m). Con el bulk acoplado, la realimentación de la SST reduce la brecha a 0.30 °C. Los presupuestos de calor de los dos cierran (ΔHC/∫Q_net = 1.000 y 1.004). Es la diferencia entre el KPP de CVMix (pyOCEAN, = MOM6 a 0.3 % en N3) y el de MITgcm (búsqueda de h, V_t², interpolación de Ri_b, `KPP_SMOOTH_*`), amplificada por la piscina fría: una diferencia de 0.8 m en la capa de mezcla sobre un salto de 10 K en 10 m. **Horquilla:** con los flujos de MITgcm, pyOCEAN mezcla más hondo que el KPP de MITgcm (−1.26 °C, MLD 8.1 contra 7.3 m); con los flujos de CROCO y su fondo, mezcla menos que el KPP de CROCO (+0.68 °C, `flux_croco_bg`). El KPP de pyOCEAN queda entre dos implementaciones publicadas. Cuál está bien no lo decide este caso | **c [hipótesis]** hasta la mejora 1 (columna MOM6/CVMix): si MOM6 coincide con pyOCEAN es c; si no, a. No se tocó ningún parámetro |
| 5 | **SST de pyOCEAN 0.32 °C sobre CROCO** (máx. +1.4 °C) y flujos medios menores hacia el océano: Q_net −14 %, Q_lat −12 %, Q_sens −44 % (el flujo sensible medio es pequeño, −7 W m⁻²), E +12 % | La separación pone el 100 % en el estado. La diferencia nace en el temporal del 24–25 de julio (\|τ\| 0.18 N m⁻²): CROCO mezcla hasta la piscina fría y su SST cae 1.5 °C más que la de pyOCEAN, y después recibe más calor porque está más fría (Q_net 102 contra 88 W m⁻²). (i) CROCO suma siempre ν = 1e-4 y κ = 1e-5 bajo la capa límite (`lmd_vmix.F:116-117`): con ese fondo, `coare_bg` acaba a +0.20 °C (último día +0.13) y la MLD media baja de 8.0 a 6.0 m (CROCO 6.4). (ii) Ri_c = 0.45 y la forma "sin ifs" del KPP 2005 de CROCO contra CVMix (N3: h_bl de CROCO-KPP 13–27 % distinta). (iii) Viento relativo (fila 3). Con los flujos de CROCO prescritos, pyOCEAN acaba +0.94 °C (+0.68 con el fondo de CROCO): el resto es la diferencia de KPP | c (elecciones de CROCO: fondo fijo, viento absoluto, su KPP) |
| 6 | **Onda larga** −3.7 % / +2.4 % | Solo estado: la fórmula difiere < 0.05 W m⁻². σT⁴ sigue a la SST (0.3 °C ≈ 1.7 W m⁻²) | c (consecuencia de 4 y 5) |
| 7 | **E en volumen**: CROCO a ρ₀ = 1025, pyOCEAN masa/ρ₀, MITgcm a 1000. `compare` lo compara como flujo de masa | E y Q_lat difieren en el mismo porcentaje en todas las corridas | — (convención resuelta) |
| 8 | **Agua dulce real** en pyOCEAN contra la sal virtual de MITgcm y CROCO: SSS final 31.15 contra 31.07 (MITgcm) y 31.19 contra 31.28 (CROCO); P − E ≈ +0.7–1.1 mm d⁻¹ | η de pyOCEAN sube 3–4 cm en 45 días; la diferencia de SSS sigue a la de la MLD (más mezcla con agua salada de abajo) | c |
| 9 | **CROCO `RESET_RHO0`**: con EOS lineal, CROCO cambia ρ₀ a la media inicial (1025.41) | flujos cinemáticos 0.04 % distintos | c (d leve: el ρ₀ de `croco.in` no se usa) |
| 10 | **Onda corta en el fondo**: MITgcm y CROCO dejan el remanente en la capa del fondo y pyOCEAN lo reparte (MOM6). Capa del fondo de MITgcm +0.6 K en 45 días, la de pyOCEAN +0.09 K | 0.26 % de SW con Jerlov III; no afecta a la SST | c |
| 11 | **Hallazgo (a), corregido: el adaptador ERA5 perdía el segundo mes.** Con archivos mensuales (instant y accum de julio y agosto), `xr.merge(compat="override")` guardaba el primer mes de cada variable y dejaba NaN en agosto; CROCO estalló el día 31 | `ERA5Source._combine` concatena en el tiempo por grupo de variables y luego une los grupos. Commit c79c1cf, con la prueba `test_time_chunks_are_concatenated` (falla antes, pasa después) | **a** |
| 12 | Columna 5 × 5: las columnas se separan desde el redondeo (§5) | semilla en η de 3e-21 m; KPP la amplifica en saltos (≈ ×10⁶ por evento); con cierre constante no crece | arnés (`h_mix`); propiedad abierta del KPP de pyOCEAN en 3-D (mejora 4) |

**Contra el criterio:**

- **NCAR:** la fórmula pasa dentro del 2 % en todos los flujos con el mismo estado (máx. 0.7 %).
  - Los flujos medios acoplados no: Q_net +14 %, Q_lat +7.5 %, Q_sens +16 %.
  - La SST final tampoco: −0.30 °C.
  - Las dos diferencias vienen del océano (KPP), como muestra `flux_mitgcm`; clase c [hipótesis] hasta la mejora 1
    (el KPP de pyOCEAN queda entre el de MITgcm y el de CROCO, fila 4).
- **COARE:** la fórmula pasa a 0.14 % con el estado de CROCO.
  - Con el acoplamiento de CROCO, la SST final pasa: +0.15 °C, último día +0.13.
  - `'coare'` (viento relativo) acaba a +0.32 °C.
  - Los flujos medios acoplados difieren 7–25 % por el estado (clase c: el fondo de mezcla fijo, Ri_c y el viento
    absoluto de CROCO).
  - `coare_bg` (viento relativo y el fondo de CROCO) acaba a +0.20 °C.

## 5. Columna 5 × 5: crecimiento de diferencias de redondeo (hallazgo)

La corrida `coare_croco` original (viento absoluto, `h_sfc` = 0, sin mezcla horizontal) terminó con 3.1 °C de
diferencia entre las 25 columnas. El controlador lo marcó como posible error (a).

- **No es el código de flujos.**
  - Bisección (100 h): con viento absoluto la dispersión llega a 0.1 °C con `h_sfc` = 0 y también con `h_sfc` = 1.
  - El camino `h_sfc` = 0 es idéntico al oráculo de la tarea 4 (`tests/test_surface_average.py`, 36 pruebas).
- **Semilla:** tras el primer paso, η difiere en 3.4e-21 m entre columnas. Es el redondeo de la superficie libre, en
  el último bit de 5.7e-6 m. T, S, u y v son idénticos. CROCO no tiene semilla (su caja queda idéntica bit a bit);
  MITgcm es 1 × 1.
- **KPP es el amplificador.** El control del revisor (`spread_exp.py`, 100 h, viento absoluto, sin `h_mix`):
  - con KPP, la dispersión de T va 0 → 4e-13 (h 12) → 4e-10 (h 30) → 1.5e-7 (h 54) → 7e-3 (h 78) → 0.137 °C (h 84),
    y queda plana después (0.13 a h 96);
  - con un cierre constante (κ = 1e-3, la misma atmósfera), queda en el último bit de T (≤ 5e-13) las 100 h.
- **El crecimiento llega en saltos, no como una inestabilidad exponencial.**
  - Los saltos coinciden con la profundización de la capa límite (aquí, horas 24–27, 50–52 y 72–78) y son de 10³ a
    10⁶ por evento (1e-8 → 7e-3 en uno solo).
  - Entre eventos la dispersión queda plana o decae. La dinámica 3-D sola no la hace crecer (control con cierre
    constante).
  - [hipótesis] El salto viene de la selección de la profundidad de la capa límite: el índice del primer cruce de
    Ri_c o el cambio del índice de la capa de superficie (εh) caen en pasos distintos según la columna.
- **El viento relativo no evita los eventos.** Las corridas con viento relativo y sin `h_mix` (`ly04`, `coare`,
  `fms`) llegan a 1.4–4.4e-6 °C al día 45, unas 10¹⁵ veces la semilla. El revisor ve el mismo salto ×10³ en h 72–78.
  Lo que hace el viento relativo es amortiguar la diferencia *entre* eventos: la tasa ρ_a C_d |ΔU|/(ρ₀ h) ≈
  1.4e-5 s⁻¹ actúa sobre la corriente de superficie de cada columna. Con viento absoluto o esfuerzo prescrito
  (`coare_croco`, `flux_*`) no hay ese amortiguamiento, y los saltos se acumulan.
- **La solución 1-D no es caótica** a la escala del criterio. Control del revisor (`sens_exp.py`): +1e-9 °C uniforme en
  la capa superior no cambia la SST de la columna (0, 0) en 1e-6 °C en 100 h. Las columnas solo divergen por
  estructura horizontal.
- **Arreglo en el arnés:** `h_mix` = 100 m² s⁻¹ (viscosidad = difusividad horizontal) en todas las corridas sin
  realimentación del viento (`coare_croco`, `flux_*`, `coare_bg`). Es nula sobre una columna uniforme.
  - 90 h de `'coare'` con y sin `h_mix` = 10: la columna (0, 0) difiere ≤ 1.1e-9 °C.
  - La SST final de `coare_croco` cambió 1e-3 °C entre `h_mix` = 10 y 100 (24.714 → 24.715).
  - La primera ronda usó 10 en `flux_*` y `coare_bg` y 100 en `coare_croco` (con 10, `coare_croco` quedaba en
    2.1e-2 °C). La ronda de correcciones 1 usa 100 en todas y las corrió de nuevo: dispersión final `flux_mitgcm` 7.9e-9, `flux_croco` 6.3e-9 (antes 1.0e-4), `flux_croco_bg` 2.9e-11, `coare_bg` 4.5e-9 °C; las tablas no cambian en ningún dígito mostrado.
  - La prueba exige < 1e-3 °C en todas las corridas.
- **Clase:** el arnés queda resuelto (c). La ganancia de KPP en 3-D es una propiedad abierta de pyOCEAN (mejora 4).
  En una corrida regional con mezcla lateral débil convertiría perturbaciones de 1e-8 en ruido de escala de malla en
  hbl y SST.

## 5b. Máscara tierra-mar de ERA5 (R7b, 2026-10-31)

Sin máscara, ERA5 trae valores de tierra y la interpolación bilineal los mezcla en las celdas costeras. La práctica de
referencia de CROCO enmascara siempre: `lsm` está en la petición por defecto (`era5_crocotools_param.py:48`), cualquier
lsm ≠ 0 es tierra (`make_ERA5.m:78-81`) y `get_missing_val` rellena todos los campos (`interp_ERA5.m`). MITgcm `exf` no
enmascara (`exf_set_fld.F:236-237, 293-294`). pyOCEAN ahora aplica la máscara si hay `lsm` (`lsm_threshold` = 0 por
defecto, como CROCO) y `fetch_era5.py` la baja por defecto.

- **Máscara del punto:** `era5_nwa_shelf_lsm.nc`, misma rejilla 0.25° que los datos (latitud y longitud idénticas, 21 × 29).
  Los cuatro nodos del estencil bilineal de P (40.24 °N, 73.16 °W) valen **lsm = 0.000000 exacto**: 40.25 °N × 73.25 °W,
  40.25 °N × 73.00 °W, 40.00 °N × 73.25 °W, 40.00 °N × 73.00 °W. Alrededor: 40.25 °N × 73.50 °W 0.0033; 40.50 °N × 73.25 °W
  0.173; 40.50 °N × 73.00 °W 0.020; 40.50 °N × 72.75 °W 0.0002; 40.75 °N 0.37–0.55 (Long Island).
- **Resultado:** el campo de P con la máscara (umbral 0 o 0.5) es **idéntico bit a bit** (muestreado cada 6 h, umbrales 0 y 0.5) al de sin máscara en los ocho
  campos durante los 45 días. Las tres corridas enmascaradas (`pyocean_45d_lsm/`, `compare_45d_lsm.txt`) coinciden con las
  originales en SST final (ly04 24.945, coare 24.889, fms 24.956 °C), Q_net medio y cada flujo: diferencia máxima horaria
  0.0 en q_sens, q_lat, q_lw, q_sw, τ y E. Haber elegido P para que el estencil fuera todo mar (§1) ya hacía lo que hace la máscara.
- **Titulares:** no cambian. Los números de §3 y de la fila N2b de CLAUDE.md quedan igual.
- **Entradas de las referencias:** `forcing.py` escribe la serie de P del proveedor pyOCEAN (`cases.point_series`), no
  la cadena `make_ERA5` de CROCO, y esa serie salió **sin máscara**. Esto difiere del preprocesado estándar de CROCO
  (que enmascara y rellena con `get_missing_val`), pero como la serie enmascarada es idéntica, no hay efecto numérico: clase
  (c) (elección de arnés) sin consecuencia en este punto. MITgcm no enmascara de todos modos. Ninguna referencia se re-corrió.
- **Cuál configuración coincide mejor:** ninguna; las dos son la misma. La máscara importa en puntos costeros (40.5 °N y
  más al norte aquí), no en P.

## 6. Qué debe mejorar pyOCEAN

1. **Comparar el KPP de pyOCEAN con MOM6/CVMix en esta columna ERA5** (MOM6 `single_column` con los flujos de MITgcm
   o de pyOCEAN prescritos). La diferencia de 1.26 °C con el KPP de MITgcm (`flux_mitgcm`) es la mayor del
   benchmark. CVMix es la implementación de referencia de pyOCEAN (N3: = MOM6 a 0.3 % en casos idealizados), pero
   no se ha comprobado con forzamiento realista, piscina fría y ondas inerciales de 0.7 m/s. Si MOM6 da lo mismo
   que pyOCEAN, la diferencia es de MITgcm (c); si no, es nuestra (a).
2. **Una opción de mezcla interior de fondo** documentada para KPP (ya existe `KPPParams.kappa_bg/nu_bg`) y una nota
   de que CROCO la impone siempre. La SST de pyOCEAN contra CROCO es muy sensible a ese fondo (+0.32 → +0.20 °C),
   y una columna de plataforma sin mezcla interior ni de Ri deja la piscina fría intacta. El valor por defecto no
   cambia (anti-objetivo del spec); la decisión es de Esteban, con observaciones (44025) y no con un modelo.
3. **Mezcla por cizalla interior en las columnas KPP** (Jackson ya existe; `shear_mixing='jackson'` con KPP). Bajo
   la capa límite, la cizalla de las ondas inerciales (corrientes de hasta 0.7–0.9 m/s) puede bajar Ri por
   debajo de 0.25, y ni pyOCEAN ni MITgcm la mezclan aquí [hipótesis: no se diagnosticó Ri bajo h]. Repetir con
   Jackson y comparar con 44025.
4. **La ganancia de KPP en 3-D** (§5): ≈ ×10⁶ por evento de profundización entre columnas vecinas casi iguales.
   - Buscar el paso de `kpp_column` que la produce. Candidatos: el índice o salto del primer cruce de Ri_c, y el
     cambio del índice de la capa de superficie (εh).
   - Registrar el índice de la capa límite por columna en h 72–78 del caso con viento absoluto.
   - Comparar con MOM6/CVMix, que interpola Ri_b de forma cuadrática.
   - Con mezcla lateral débil, esta ganancia daría ruido de escala de malla en hbl y SST en las corridas regionales.
   - La semilla en η (3e-21 m) no es el problema: cualquier corrida real tiene perturbaciones mucho mayores.
5. **Validar contra la boya NDBC 44025** (SST horaria, julio–agosto 2017) en el proyecto 3: es la verdad que decide
   entre KPP, fondo y fórmula. Aquí solo se compara entre modelos.
6. **Adaptador ERA5**: la prueba de archivos mensuales ya está (hallazgo 11). Añadir a `tests/test_atmosphere.py`
   una prueba `needs_era5` con julio y agosto reales.
7. Las ráfagas de OM4 (`gust`) siguen fuera de `bulk_ncar_fms` (tarea 1). Aquí no importan: la fórmula `'ncar'`
   difiere de exf un 1.6 % en Q_net, por las elecciones de la tabla de `bulk_points`.

## 7. Figuras (`compare.py --figs`, en `RUNS/figs/`)

- `sst.png`: SST de cada modelo y diferencias contra la referencia, con la banda de ±0.2 °C.
- `mld.png`: profundidad de la capa de mezcla (media diaria móvil).
- `flux_bars.png`: flujos medios por modelo, con la banda de ±2 % de cada referencia.
- `sst_separation.png`: SST de la corrida bulk contra el mismo océano forzado con los flujos de la referencia.

## Referencias

- Large & Yeager (2004), NCAR Tech. Note TN-460+STR; Fairall et al. (2003), J. Climate 16, 571–591 (COARE 3.0).
- Brodeau et al. (2017), J. Phys. Oceanogr. 47, doi:10.1175/JPO-D-16-0169.1: con la misma atmósfera, COARE y
  NCAR difieren en los flujos turbulentos en un orden comparable al de aquí. Brunke et al. (2003), J. Climate 16,
  doi:10.1175/1520-0442(2003)016<0619:WBAAAL>2.0.CO;2; Zeng et al. (1998), J. Climate 11,
  doi:10.1175/1520-0442(1998)011<2628:IOBAAF>2.0.CO;2. Están en `docs/literature/reading_list.md` (N2b).
- Paulson & Simpson (1977) (Jerlov); Large et al. (1994) (KPP; interior ν = 1e-4, κ = 1e-5 de CROCO);
  de Boyer Montégut et al. (2004) (criterio de MLD de 0.03 kg m⁻³) [verificar].
