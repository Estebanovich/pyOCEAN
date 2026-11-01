# TEOS-10 en el núcleo 3D: casos de validación (criterios 4, 5, 8, 9, 10 y 11)

Task 9 del plan `docs/superpowers/plans/2026-10-31-teos10-3d.md`; spec
`docs/superpowers/specs/2026-10-31-teos10-3d-design.md`. Definiciones en `cases.py`, corridas en `run_pyocean.py`,
tablas en `compare.py`, pruebas rápidas en `tests/test_teos10_validation.py`. Sin ajuste de parámetros a ningún caso
(CLAUDE.md, Validation Strategy).

## Cómo correr

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=5
python validation/teos10/woa_gulfstream_profile.py          # una vez: tests/data/woa23_gulfstream_profile.npz
python -m validation.teos10.run_pyocean --all               # 33 corridas, una a la vez (~45 min en 5 núcleos P)
python -m validation.bulk_column.run_pyocean fms --eos wright_full
python -m validation.bulk_column.run_pyocean fms --eos teos10
python -m validation.teos10.compare                         # escribe las tablas entre los marcadores
pytest tests/test_teos10_validation.py                      # versiones rápidas (~45 s)
```

Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10/` (y `bulk_column/pyocean_45d_<eos>/`), nunca en el repo.

## 1. Perfil y casos

**Perfil WOA23** (`woa_gulfstream_profile.py`): climatología anual, todas las décadas, 1°, `t_an` (temperatura
in situ) y `s_an` (salinidad práctica). 40 °N 70 °W es la esquina común de cuatro celdas de 1°, todas con agua; están
a la misma distancia, y el empate se rompe por el dato más profundo: **celda 39–40 °N × 71–70 °W (centro 39.5 °N,
70.5 °W), 75 niveles, 0–2800 m**. Conversión: S_A = (35.16504/35) S_P (regla de MOM6, sin δS_A), p de
`gsw.p_from_z`, θ = `gsw.pt_from_t(S_A, t, p, 0)`. θ va de 16.08 °C a 2.20 °C y S_P de 34.00 a 34.91; σ₀ es
monótona. Debajo de 2800 m (la montaña llega a 4500 m) θ y S_P conservan su valor de 2800 m: un abismo neutro en
densidad potencial, elección del caso y no dato.

**Ecuación lineal equivalente** (criterio 4; `cases.linear_equivalent`): mínimos cuadrados de ρ_Wright(θ, S_P, 0)
sobre los 75 niveles del perfil, y luego S₀ se mueve para que la ordenada sea ρ₀ = 1035 (α y β no cambian):

```python
p = load_profile()
th, sp = p["theta"], p["SP"]
rho = wright_full.density(th, sp, np.zeros_like(th))
t0, s0 = th.mean(), sp.mean()
A = np.column_stack([np.ones_like(th), -(th - t0), sp - s0])
(c, alpha, beta), *_ = np.linalg.lstsq(A, rho, rcond=None)
s_ref = s0 - (c - RHO0) / beta            # ρ_lin = 1035 − α (θ − θ₀) + β (S_P − s_ref)
```

| Caso | Malla | Δt | Física | Métrica |
|---|---|---|---|---|
| `rest_case("seamount", eos, coord)` | Beckmann & Haidvogel (1993): 64 × 48 × 20, Δx 6 km, H₀ 4500 m, montaña 4050 m, L 25 km, f 1e-4 s⁻¹ | 600 s | ν_v 1e-4, sin ν_h ni K_h, sin arrastre | max\|u\|, \|v\| en 10 días; \|PGF\| inicial |
| `rest_case("bayiw_step", eos, coord)` | sección 2-D 80 × 1 × 50, Δx 1 km, escalón 600 → 164 m (boca de BayIW), f 0 | 120 s | igual | igual |
| `cabbeling_case(eos)` | sección 2-D 64 × 1 × 20, Δx 500 m, H 200 m, z\* | 60 s | K_h 10 m² s⁻¹, κ_v 1e-5, ν_h 1, ν_v 1e-4 | volumen de agua más densa que las dos fuentes (ρ a p = 0, umbral 1e-6 kg m⁻³), altura media ponderada por el exceso de densidad |
| `thermobaric_case(eos, 100 \| 4000)` | 5 × 5 doblemente periódico, z\*, capas de 10 m (100 m) o 50 m (4000 m) | 60 s | GLS k-ε estratificado, sin forzamiento | fracción del calor del agua de fondo que cruza la interfaz inicial |

Coordenadas del reposo: z\* con niveles uniformes H_max/N; híbrido HYCOM1 con pisos z\* geométricos de 2 m en la
superficie hasta el fondo (`cases.hybrid_floors`; un piso que no llega al fondo deja el agua profunda homogénea en
una sola capa, aviso del constructor) y objetivos `rho_target` del perfil: σ₂ + 1000 con EOS no lineal (meta D), y
con `linear` la densidad lineal del perfil **a las profundidades de los objetivos de `wright_full`**, para que la
corrida lineal empiece sobre las mismas capas (sección 3, diferencia 2).

Aguas de cabbeling: A = (5 °C, 34.0) y B = (15 °C, S_B), S_B por Newton con la EOS del caso para que
ρ(B, 0) = ρ(A, 0): S_B = 36.16198 (`wright_full`, en θ/S_P) y 36.19457 (`teos10`, en Θ/S_A). Control lineal: α de
Wright en el punto medio, β = α ΔT/ΔS (las dos aguas con la misma densidad exacta). Columna termobárica: las aguas
de la Task 6 (0 °C, 34.5 sobre 4 °C, 34.95; 200 m cálidos en el fondo), los mismos números en las dos EOS (sus
propias variables). Control lineal: α y β de Wright en la media del par a p = 0 (estable a toda profundidad).

## Tablas (`compare.py`, corridas de producción)

<!-- teos10-tables:start -->
Equivalent linear EOS (``cases.linear_equivalent``): α = 0.15178 kg m⁻³ °C⁻¹, β = 0.90924 kg m⁻³ PSU⁻¹, θ₀ = 7.5153 °C, S₀ = 43.5611, ρ₀ = 1035; rms misfit to ρ_Wright(θ, S_P, 0) on the profile 0.0471 kg m⁻³.

### Criterion 4: rest over steep topography (10 days)

| case | coord | ρ_ref | EOS | max\|PGF\| t=0 [m s⁻²] | max\|u\| 10 d [m s⁻¹] | final [m s⁻¹] | ratio to linear | ≤ 2× |
|---|---|---|---|---|---|---|---|---|
| seamount | zstar | auto | linear | 1.88e-07 | 2.78e-04 | 2.66e-04 | 1.00 | — |
| seamount | zstar | auto | wright_full | 1.43e-07 | 2.61e-04 | 1.87e-04 | 0.94 | yes |
| seamount | zstar | auto | teos10 | 1.46e-07 | 2.63e-04 | 1.90e-04 | 0.95 | yes |
| seamount | zstar | none | linear | 6.13e-05 | 4.25e-02 | 3.38e-02 | 1.00 | — |
| seamount | zstar | none | wright_full | 6.08e-05 | 4.61e-02 | 3.66e-02 | 1.09 | — |
| seamount | zstar | none | teos10 | 6.12e-05 | 4.62e-02 | 3.69e-02 | 1.09 | — |
| seamount | hybrid | auto | linear | 2.14e-05 | 1.02e-01 | 9.12e-02 | 1.00 | — |
| seamount | hybrid | auto | wright_full | 2.43e-05 | 1.08e-01 | 1.08e-01 | 1.06 | yes |
| seamount | hybrid | auto | teos10 | 2.44e-05 | 1.15e-01 | 1.15e-01 | 1.13 | yes |
| seamount | hybrid | none | linear | 3.70e-05 | 2.56e-01 | 1.32e-01 | 1.00 | — |
| seamount | hybrid | none | wright_full | 3.66e-05 | 2.28e-01 | 1.31e-01 | 0.89 | — |
| seamount | hybrid | none | teos10 | 3.68e-05 | 2.28e-01 | 1.33e-01 | 0.89 | — |
| bayiw_step | zstar | auto | linear | 3.56e-15 | 4.00e-10 | 4.00e-10 | 1.00 | — |
| bayiw_step | zstar | auto | wright_full | 9.73e-17 | 2.37e-10 | 2.26e-10 | 0.59 | yes |
| bayiw_step | zstar | auto | teos10 | 6.31e-16 | 2.34e-10 | 2.17e-10 | 0.59 | yes |
| bayiw_step | zstar | none | linear | 1.21e-03 | 2.10e-03 | 7.71e-04 | 1.00 | — |
| bayiw_step | zstar | none | wright_full | 1.18e-03 | 2.06e-03 | 8.09e-04 | 0.98 | — |
| bayiw_step | zstar | none | teos10 | 1.20e-03 | 2.09e-03 | 8.18e-04 | 0.99 | — |
| bayiw_step | hybrid | auto | linear | 3.11e-06 | 1.36e-04 | 3.66e-05 | 1.00 | — |
| bayiw_step | hybrid | auto | wright_full | 3.03e-06 | 1.69e-04 | 3.75e-05 | 1.24 | yes |
| bayiw_step | hybrid | auto | teos10 | 3.07e-06 | 1.70e-04 | 3.72e-05 | 1.25 | yes |
| bayiw_step | hybrid | none | linear | 1.18e-03 | 6.47e-03 | 2.39e-03 | 1.00 | — |
| bayiw_step | hybrid | none | wright_full | 1.16e-03 | 7.30e-03 | 3.00e-03 | 1.13 | — |
| bayiw_step | hybrid | none | teos10 | 1.18e-03 | 7.40e-03 | 3.04e-03 | 1.14 | — |

### Criterion 5a: cabbeling lock exchange (10 days)

| EOS | first denser water [h] | volume 2 h / 6 h / 1 d [10⁹ m³] | domain [10⁹ m³] | lowest excess-weighted height [m] (at [h]) | max\|u\| [m s⁻¹] |
|---|---|---|---|---|---|
| linear | never | 0.00 / 0.00 / 0.00 | 3.2 | — | 3.69e-09 |
| wright_full | 1 | 0.52 / 1.29 / 3.20 | 3.2 | -147.4 (15) | 9.68e-01 |
| teos10 | 1 | 0.52 / 1.30 / 3.20 | 3.2 | -147.5 (15) | 9.75e-01 |

### Criterion 5b: thermobaric column (10 days)

| EOS | interface [m] | rise 1 h | rise 6 h | rise 10 d | bed T final [°C] | overturn |
|---|---|---|---|---|---|---|
| linear | 100 | 0.000 | 0.000 | 0.000 | 4.000 | no |
| linear | 4000 | 0.000 | 0.000 | 0.000 | 4.000 | no |
| wright_full | 100 | 0.000 | 0.000 | 0.000 | 4.000 | no |
| wright_full | 4000 | 0.631 | 0.921 | 0.950 | 0.199 | yes |
| teos10 | 100 | 0.000 | 0.000 | 0.000 | 4.000 | no |
| teos10 | 4000 | 0.632 | 0.921 | 0.951 | 0.198 | yes |

### Criterion 10: N2b ERA5 column with the three EOS (45 days, run 'fms')

| EOS | c_p [J kg⁻¹ K⁻¹] | final SST θ [°C] | mean SST last 5 d [°C] | Δ vs linear [°C] | Δ teos10 − wright_full [°C] | final hbl [m] |
|---|---|---|---|---|---|---|
| linear | 3985.00 | 24.956 | 24.456 | +0.000 |  | 5.2 |
| wright_full | 3991.87 | 25.054 | 24.525 | +0.099 |  | 4.3 |
| teos10 | 3991.87 | 25.046 | 24.518 | +0.090 | -0.009 | 4.3 |

<!-- teos10-tables:end -->

Lectura de las tablas: "ratio to linear" = max\|u\| de la EOS / max\|u\| de la lineal equivalente en el mismo caso,
coordenada y modo de referencia. "rise" = fracción del calor del agua de fondo (respecto a T_fría) por encima de la
interfaz inicial. Volúmenes de cabbeling: agua con ρ(T, S, 0) mayor que la de las dos fuentes por más de
1e-6 kg m⁻³; el dominio entero (3.2e9 m³) queda más denso en ~1 día.

## 2. Criterios

- **Criterio 4 (reposo), parte del cociente 2×: cumplida** (la fuerza cero a redondeo no, punto siguiente). Con `rho_ref="auto"`, `wright_full`/`teos10` quedan entre 0.59× y 1.25× de la
  lineal equivalente en los ocho casos (límite 2×). En z\* las corrientes son 2.6e-4 m s⁻¹ en la montaña (el error
  conocido de z\* en la montaña, 0.08 cm/s en CLAUDE.md con el perfil analítico) y 2e-10 m s⁻¹ en el escalón.
  En el híbrido son 0.10–0.12 m s⁻¹ en la montaña y 1.4–1.7e-4 m s⁻¹ en el escalón, iguales en las tres EOS
  (sección 3, diferencia 1).
- **Criterio 4, fuerza cero a redondeo con el perfil sin variación horizontal: NO cumplido** (sí la parte del
  cociente 2×). El spec la pide en z\* y en híbrido, también sobre la montaña y el escalón. Medido, PGF inicial:
  - z\*, caras entre celdas completas: 2e-18 (`wright_full`), 5e-17 (`teos10`), 4e-16 (lineal) m s⁻², a redondeo;
    escalón 1e-16–4e-15, a redondeo;
  - z\*, montaña, junto a celdas parciales de fondo: 1.4e-7 m s⁻² (lineal 1.9e-7), que deja 2.6e-4 m s⁻¹ a los
    10 días: interpolación del perfil de referencia en 400 clases (el centro de una celda parcial cae entre los
    nodos), clase (c), independiente de la EOS;
  - híbrido: 2e-5 m s⁻² en la montaña y 3e-6 en el escalón, iguales con la lineal: truncamiento de las capas
    gruesas, (c), baja 51× de 20 a 50 capas (diferencia 1).

  El residuo no es termobárico (es igual con la lineal), pero no es de redondeo: abierto, sección 3 diferencias 1 y
  3, sección 4 puntos 1 y 2.
- **Criterio 4, modo MOM6 (`rho_ref=None`), sin límite:** montaña z\* 4.2–4.6 cm/s, híbrido 23–26 cm/s; escalón z\*
  2.1 mm/s, híbrido 6.5–7.4 mm/s; las no lineales 0.89–1.14× la lineal. **No se corrió MOM6 en el mismo caso**
  (aplazado a la cadena regional / NWA12, donde MOM6 corre en el mismo recorte; decisión del controlador del plan, 2026-11-01).
- **Criterio 5a (cabbeling): cumplido.** Con `wright_full` y `teos10` aparece agua más densa que las dos fuentes en
  la primera hora, su volumen crece (0.52e9 m³ a 2 h, 1.3e9 a 6 h, el dominio a 1 día) y su altura media ponderada por
  el exceso de densidad baja hasta −147 m (H = 200 m) a las 15 h, antes de que el dominio se mezcle entero. Con la
  lineal el volumen es 0 y max\|u\| = 3.7e-9 m s⁻¹. Las dos EOS no lineales coinciden a 1 %.
- **Criterio 5b (termobaricidad): cumplido.** La columna con la interfaz a 4000 m se invierte (0.63 del calor del
  fondo sobre la interfaz en 1 h, 0.95 a los 10 días; T del fondo 4.00 → 0.20 °C); con la interfaz a 100 m no
  (0.000); el control lineal no se invierte a ninguna profundidad. El signo es el de ∂²ρ/∂T∂p: el N² local de la
  Task 6 es +2e-6 s⁻² a 100 m y −1.9e-5 s⁻² a 4000 m.
- **Criterio 8 (conservación):** `test_variance_budget_closes` (σ y z\*, las tres EOS, con viento): presupuesto de
  varianza de S y T ≤ 1e-12 cada paso; volumen y sal ≤ 1e-13. El contenido de calor con bulk y calor prescrito lo
  cubre la Task 7 (`tests/test_eos_surface.py`).
- **Criterio 9:** `test_restart_and_threads_bitwise`: 4 pasos en 5 hilos = 2 pasos en 1 hilo + restart en un modelo
  nuevo + 2 pasos, bit a bit, `wright_full` y `teos10` en σ, z\* e híbrido, con GLS estratificado y viento.
- **Criterio 10 (columna bulk):** `teos10` − `wright_full` = −0.009 °C de SST final (θ), dentro de la diferencia
  física esperada; las dos no lineales quedan +0.09/+0.10 °C sobre la lineal. Explicación:
  - **α:** a 25 °C y 31.5 la α no lineal es 2.92e-4 K⁻¹ (Wright y `gsw` coinciden) frente a 2.0e-4 de la lineal
    del caso (46 % más). El calentamiento de verano da más flotabilidad por vatio, la columna se estratifica más y la
    capa límite de KPP es más somera (media 7.8 frente a 8.6 m; final 4.3 frente a 5.2 m), así que el calor queda
    más arriba: SST más alta y media de la columna más baja (θ̄ final 16.31 frente a 16.45 °C, con más pérdida por
    onda larga y calor latente en la superficie más caliente).
  - **c_p:** 3991.87 frente a 3985 (+0.17 %): con el mismo flujo, el calentamiento es 0.17 % menor; sobre los
    +3 °C de la SST (22.0 → 25.0 °C) son ~−0.005 °C, un orden menos que el efecto de α.
  - **`teos10` − `wright_full` (−0.009 °C):** sobre todo el c_p. El contenido de calor de verdad es c_p⁰ Θ
    (potential enthalpy), y dθ = (c_p⁰ / c_p(S, θ, 0)) dΘ. `teos10` recibe el flujo en Θ con c_p⁰ y lo convierte a θ
    exacto; `wright_full` lo pone directamente en θ con c_p⁰ = 3991.87, mientras que el c_p real en la superficie
    de esta corrida es 4016.4–4019.4 J kg⁻¹ K⁻¹ (`gsw.cp_t_exact`, media 4017.7; media de la columna 4009.8). Así
    `wright_full` calienta θ un 0.65 % de más: sobre los +3.05 °C de la SST (21.998 → 25.05) son +0.020 °C, y la
    estimación sin realimentación da `teos10` − `wright_full` ≈ −0.020 °C. La SST más caliente de `wright_full`
    pierde más calor: el flujo neto integrado es 3.073e8 J m⁻² (`wright_full`) frente a 3.084e8 (`teos10`), 1.1e6
    J m⁻² de diferencia ≈ 0.0075 °C sobre los 36 m, lo que lleva la estimación a ≈ −0.012 °C frente a −0.009
    medido. El resto (~0.003 °C) es Roquet frente a Wright en α y en la capa límite [hipótesis, no separado]. (La
    diferencia Θ − θ en sí, 0.114 °C a 22 °C y 0.131 °C a 25 °C con S_P 31.5, no es la escala: la SST se compara en
    θ en las dos.) Esto es una elección de NWA12/MOM6 (θ con c_p constante), clase (c) frente a TEOS-10.

- **Criterio 11 (costo por paso): cumplido con `wright_full` (1.37×) y con `teos10` (1.87×)** tras la
  optimización exacta del 2026-11-01 (bit a bit con 8a2bc3a; antes 1.90× y 3.16×, registro
  `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10/cost/eos_cost_2026-11-01.log`). `scripts/eos_cost.py` (modelo de
  `scripts/bulk_cost.py`): BayIW cut z\*, 50 capas, Δt 120 s, `salinity=True` (S = 35) en los tres modos, sin
  forzamiento, `rho_ref="auto"`, 5 hilos, un proceso por modo, alternados (linear, wright_full, teos10) × 3 rondas,
  mediana de 50 pasos tras 10 de calentamiento, máquina sin otra carga. Registro:
  `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10/cost/eos_cost_2026-11-01_opt.log`.

  | EOS | s/paso (mediana) | cociente a `linear` mín/mediana/máx | ≤ 2.5× |
  |---|---|---|---|
  | linear | 0.390 | 1 | — |
  | wright_full | 0.534 | 1.364 / 1.369 / 1.375 | sí |
  | teos10 | 0.729 | 1.866 / 1.872 / 1.873 | sí |

  Desglose por fase del driver (ms por paso, mediana de las rondas; `Driver._ale_<fase>` cronometradas):

  | fase | linear | wright_full | teos10 |
  |---|---|---|---|
  | explicit_momentum (gradiente de presión) | 64.3 | 190.5 (+126.2) | 377.6 (+313.3) |
  | tracers | 123.0 | 143.6 (+20.7) | 148.6 (+25.6) |
  | vertical_coordinate | 26.6 | 25.8 | 29.5 (+2.8) |
  | las demás (prepare, time_combine, implicit_and_free_surface, boundary_velocity, layer_transport, finalize) | 170.5 | 167.6 | 168.8 |
  | suma | 384.3 | 527.5 | 724.4 |

  Optimizaciones (todas exactas: mismas operaciones en el mismo orden; `fv_pgf_nonlinear` idéntico bit a bit
  con 8a2bc3a en las 2 EOS × perfil sí/no × 3 pesos de masa × 1 y 5 hilos sobre la entrada BayIW; gradiente de
  presión `teos10` medido aislado, 5 hilos): (1) `inline="always"` en los núcleos escalares de
  `physics/eos/roquet_rho.py` y `wright_full.py` — `_rho_parts` se llamaba fuera de línea por la GOT en cada
  punto, devolviendo la tupla por memoria (Roquet 13.7 → 5.3 ns por evaluación, `scripts/eos_microbench.py`;
  836 → 688 ms); (2) en el modo con perfil, un solo índice del perfil por punto, caminado desde el del punto
  anterior, para T_ref y S_ref (antes dos bisecciones por punto, ~17 ns cada una; 688 → 345 ms); (3) sin
  asignación por columna en el PLM y sin EOS en columnas inactivas del paso de columnas (345 → 344 ms). El
  resto del exceso es aritmética de la EOS (~70 evaluaciones por celda y capa con el perfil: 2 × (5 + 2 × 15)).
  Los +21–26 ms de `tracers` son la densidad de los cierres y la difusión sobre T − T_ref, S − S_ref
  (`ReferenceProfile.__call__` 15.6 frente a 3.7 ms por paso en lineal, cProfile; el resto repartido).

## 3. Diferencias clasificadas

1. **Corrientes del híbrido sobre la montaña (0.1 m s⁻¹ con perfil; 0.23 sin él): (c) error de truncamiento de
   capas gruesas, comprobado con la resolución.** Las capas gruesas (hasta 2500 m) cortadas por la topografía llevan
   la media de capa de un perfil curvo; la reconstrucción PLM de T − T_ref no reproduce la curvatura, así que la
   integral de ρ de una capa cortada difiere de la de sus vecinas. Es el mismo con la lineal (1.06–1.13×) y no
   cambia con un perfil de referencia fino (2.92e-5 frente a 2.99e-5 m s⁻² en la versión rápida). Prueba de
   resolución (misma montaña, `wright_full`, `rho_ref="auto"`, 10 días; pisos y objetivos siguen a N,
   `run_pyocean rest seamount wright_full hybrid N`):

   | N capas | max\|PGF\| t=0 [m s⁻²] | max\|u\| 10 d [m s⁻¹] | max\|u\| 1 d |
   |---|---|---|---|
   | 10 | 2.3e-5 | 0.64 | 0.29 |
   | 20 | 2.4e-5 | 0.108 | 0.049 |
   | 30 | 5.5e-6 | 9.5e-3 | 6.4e-3 |
   | 50 | 3.2e-6 | 2.1e-3 | 1.6e-3 |
   | 50, lineal | 3.9e-6 | 2.9e-3 | 2.5e-3 |
   | z\* 20 / 50 | 1.4e-7 / 4.8e-8 | 2.6e-4 / 1.4e-4 | 1.7e-4 / 1.3e-4 |

   Baja 51× de 20 a 50 capas (más rápido que N⁻²; el PGF inicial baja como N⁻²·²), como un error de truncamiento
   de la reconstrucción, sin truncaciones CFL; con 50 capas el híbrido queda 15× sobre z\* y la no lineal 0.73× la
   lineal. No apunta a un error de la inicialización ni del PGF del híbrido; se mantiene (c).
2. **Objetivos del híbrido elegidos por separado en cada EOS: (c) configuración del caso, corregido en el caso.** Con
   objetivos equiespaciados en la densidad de cada EOS, la versión rápida del escalón dio 4.8× (`wright_full`) frente
   a la lineal: las capas iniciales eran distintas (una capa de 1 m en el borde del escalón con `wright_full`). Con
   los objetivos lineales puestos a las profundidades de los de `wright_full` (mismas capas) el cociente es 0.5–0.7×
   en la versión rápida y 1.24× en la de producción. El criterio compara EOS, no mallas.
3. **Residuo de PGF inicial en z\* junto a celdas parciales (1.4–1.9e-7 m s⁻²): (c).** Perfil de referencia en 400
   clases interpolado linealmente; igual en las tres EOS.
4. **Cabbeling con movimiento fuerte (0.97 m s⁻¹): física, no error.** Las dos aguas tienen la misma ρ a p = 0 pero no
   a 200 m (la fría es 0.049 kg m⁻³ más densa por termobaricidad); el agua mezclada es hasta 0.14 kg m⁻³ más densa
   (cabbeling) y forma corrientes de gravedad de g′H ≈ (0.5 m s⁻¹)².
5. **Inversión termobárica por GLS en una columna 1-D: (b) física ausente por diseño (hidrostático).** Sin
   movimiento (max\|u\| = 0): la convección es la mezcla del cierre con N² < 0, no plumas resueltas.
6. **Criterio 4 modo MOM6 sin la corrida de MOM6: pendiente** (no es diferencia; falta la referencia).
7. **Fuerza cero a redondeo del criterio 4 no alcanzada** en z\* sobre la montaña (celdas parciales, diferencia 3,
   (c)) ni en el híbrido (diferencia 1): abierto, independiente de la EOS.
8. **Costo de `teos10` 3.16× (límite 2.5×): (a) resuelto el 2026-11-01** con optimizaciones exactas (núcleos de
   la EOS no inlineados, dos bisecciones del perfil por punto): `teos10` 1.87×, `wright_full` 1.37× (criterio 11).

## 4. Qué debe mejorar pyOCEAN

1. **Reconstrucción de mayor orden de T y S en el gradiente de presión del híbrido** (PPM, como la opción
   `RECONSTRUCT_FOR_PRESSURE`/PPM de MOM6 [verificar el nombre del parámetro]): con 20 capas las capas gruesas
   cortadas por la topografía dan 0.1 m s⁻¹ espurios en la montaña con cualquier EOS; con 50 capas 2e-3 m s⁻¹
   (diferencia 1). Una reconstrucción de mayor orden debería acercar el híbrido de pocas capas a z\*.
2. **Perfil de referencia de `rho_ref="auto"` exacto en celdas parciales:** muestrear el perfil de los datos
   (h_src) en lugar de clases de los centros de capa; quitaría el residuo de 1e-7 m s⁻² (diferencia 3).
3. **Correr MOM6 en el caso de reposo (montaña, z\* y ρ, WRIGHT_FULL)** para poner las cifras del modo
   `rho_ref=None` junto a las suyas, como pide el criterio 4.
4. **Validar cabbeling/termobaricidad contra observaciones o LES** (no hay verdad en esta tarea: solo signo y
   aparición). Candidatos: la formación de agua por cabbeling en frentes (McDougall 1987 [verificar]).
5. **Columna bulk contra la boya 44025** con `teos10`: la diferencia lineal/no lineal (+0.1 °C) es del orden de la
   que el benchmark puede resolver; decidir con observaciones, no con el modelo (proyecto 3 de la cadena).
6. **Costo del gradiente de presión no lineal** (criterio 11 cumplido: `teos10` 1.87×, `wright_full` 1.37×): si
   hiciera falta más, opciones exactas restantes: vectorizar la EOS por lotes de puntos (SIMD sin cambiar la
   aritmética por punto) y equilibrar la carga de las filas con tierra; opciones no exactas: 3 subcolumnas por cara,
   PLM de la anomalía de densidad, siempre con MOM6 como oráculo en el modo sin perfil.
7. **BBL con el N²_bot de MOM6** (`find_N2`, `MOM_set_diffusivity.F90:1202-1258`; clase (a), aplazado hasta el OK
   de Esteban porque cambia las corridas BBL lineales y las cifras de DOME; efecto medido en Kd ≤ 1.6 % en columnas
   suaves, 2–37 % aleatorias, 100 % en una; `docs/design/teos10_refs/mom6_eos.md` §5).
8. **Proyección de dos trazadores de Jackson** (MOM6 proyecta T y S por separado dentro de los subpasos; aquí solo el
   N² inicial es igual a MOM6, diferencia de segundo orden, (c)).
9. **Comprobar la cota de ajuste de Roquet contra el artículo** (`ROQUET_FIT_MAX` = 1.1 × 5.355e-2 kg m⁻³ medido,
   [verificar Roquet 2015]; el artículo sigue ⬜ en `docs/literature/reading_list.md`).
