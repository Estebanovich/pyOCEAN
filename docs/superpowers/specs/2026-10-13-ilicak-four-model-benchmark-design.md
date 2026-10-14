# pyOCEAN contra MITgcm, MOM6 y CROCO: mezcla espuria, gradiente de presión y fronteras abiertas

**Fecha:** 2026-10-13. **Estado:** diseño aprobado por Esteban en la conversación por secciones; falta revisar el texto escrito.
**Ramas:** `ilicak-benchmark` (parte A) y, después, `obc-benchmark` (parte B), cada una desde `main` y con su propio plan.

**Motivo:**
- El benchmark de MOM6 en BayIW mostró que pyOCEAN cuesta lo mismo que MOM6 con el mismo Δt y que nuestra híbrida de dos capas tiene un problema propio. Pero BayIW es un solo caso y solo se compararon dos modelos con remapeo.
- Esteban pidió un experimento en el que se puedan comparar los cuatro modelos (pyOCEAN, MITgcm, MOM6, CROCO), con HYCOM solo para costo, en costo y en resultados, con corridas considerablemente baratas.
- Los casos idealizados de Ilicak et al. (2012) son el estándar para la mezcla espuria y son 2-D. Su conclusión central, que la mezcla la controla el número de Reynolds de malla Re_Δ = UΔx/ν_h y el cierre del momento, toca directamente una decisión de pyOCEAN: el momento vector-invariante con KE Arakawa no tiene disipación propia.
- La lectura de Cailleau et al. (2008), Oddo & Pinardi (2008), Mason et al. (2010), Herzfeld et al. (2011) y Klingbeil et al. (2018) dio casos baratos para medir las fronteras abiertas con la misma idea (parte B).

**Reglas que valen para todo el benchmark:**
- Toda diferencia se clasifica de (a) a (d); solo (a) y (b) cambian el modelo.
- El benchmark solo mide: no se cambia pyOCEAN durante los planes. Un error claro de clase (a) se reporta y se decide con Esteban si se arregla antes de seguir.
- No se modifica el código fuente de MOM6, FMS, CROCO, MITgcm ni HYCOM. Los archivos de configuración de cada caso (`MOM_input`, `cppdefs.h`, `param.h`, `SIZE.h`, `data*`, `blkdat.input`) se generan en directorios de corrida.
- Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/` y `.../obc_benchmark/`, nunca al repo.
- Una corrida pesada a la vez, con ≤ 5 núcleos P (`NUMBA_NUM_THREADS=5`, 5 procesos MPI como máximo).

## Parte A: mezcla espuria y gradiente de presión

### A1. Casos

Física común: EOS lineal ρ = ρ₀ − α(T − T_ref) con ρ₀ = 1000 kg m⁻³ y α = 0.2 kg m⁻³ °C⁻¹, S constante, g = 9.81 m s⁻², sin rotación (salvo el seamount), sin difusión explícita de trazadores ni ajuste convectivo, ν_v = 1e-4 m² s⁻¹, sin fricción de fondo, paredes cerradas con deslizamiento libre.

| Caso | Definición | Malla | Δt de resultados | Duración | ν_h [m² s⁻¹] |
|---|---|---|---|---|---|
| Lock exchange | 64 km × 20 m; T = 5 °C para x < 32 km y 30 °C para x ≥ 32 km (Δρ = 5) | 128 × 20 (Δx 500 m, Δz 1 m) | 10 s | 17 h | 0.1, 10, 100 |
| Overflow | 200 km; fondo d(x) = d₁ + ½(d₂ − d₁)[1 + tanh((x − x₀)/r)] con d₁ = 500 m, d₂ = 2000 m, x₀ = 40 km, r = 7 km; T = 10 °C para x < 20 km y 20 °C para x ≥ 20 km (Δρ = 2) | 200 × 100 (Δx 1 km, Δz 20 m) | 10 s | 40 h | 1, 100, 1000 |
| Onda interna | 250 km × 500 m, fondo plano; T₀(z) = T_bot + (T_top − T_bot)(z_bot − z)/z_bot con T_bot = 10.1 °C, T_top = 20.1 °C, z_bot = −487.5 m; perturbación T′ = −A cos(π(x − x₀)/(2L)) sin(π(z + Δz/2)/(z_bot + Δz/2)) para \|x − x₀\| < L, con A = 2 °C, L = 50 km, x₀ = 125 km, Δz = 25 m | 50 × 20 (Δx 5 km, Δz 25 m) | 300 s | 200 días | 0.01, 15, 150 |
| Seamount | Beckmann & Haidvogel (1993), la definición de `validation/seamount` (384 × 288 km, H₀ = 4500 m, monte de 4050 m y L = 25 km, f₀ = 1e-4, ρ(z) = ρ₀ + 3(1 − e^{z/1000})) | 64 × 48 × 20 | el de `validation/seamount` | 10 días | el de `validation/seamount` |

**Fuentes y decisiones:**
- **Lock exchange:** Ilicak et al. (2012, §3.1, ec. 3–4) escribe T = 5/35 °C con Δρ = 5 kg m⁻³ y ρ entre 1022 y 1027. Petersen et al. (2015, ec. 21–22) y el ejemplo `lock_exchange` de MOM6 (ρ = 1000 − 0.2T + 0.8S) usan 5/30 °C con α = 0.2. Se usa 5/30: es lo único consistente con Δρ = 5 y α = 0.2; el 35 de Ilicak se toma como errata. La velocidad de Benjamin (1968) es u_f = ½√(g′H) = 0.496 m s⁻¹ (Petersen: frente derecho en 62.4 km a las 17 h).
- **Δt del lock exchange:** Ilicak usa 1 s y Petersen 120 s (60 y 30 s con ν_h = 100 y 200); se usa 10 s, el de las corridas de pyOCEAN contra CROCO (`validation/lock_exchange`). El número de Courant viscoso ν_hΔt/Δx² ≤ 0.004 en los tres valores.
- **Overflow:** Ilicak (§4) no da la plataforma ni el largo; se usan las ec. 24–25 y la tabla 5 de Petersen. Ninguno de los modelos trae este caso: el `OVERFLOW` de CROCO es el de Haidvogel–Beckmann (40 m de fondo) y el `flow_downslope` de MOM6 es DOME2D (800 km, 4000 m, salinidad).
- **Onda interna:** perfil y perturbación de Petersen (ec. 26–27), confirmados en Chen et al. (2024, ec. 15). Con α = 0.2 da N = 6.34e-3 s⁻¹; Ilicak dice 6.26e-3, que sale de 10 °C en 500 m. Se usa la definición de Petersen y se anota la diferencia. **Solo la versión de alta amplitud (A = 2 °C):** la amplitud baja de Ilicak (Fig. 11a) solo está en una figura y no se puede reproducir.
- **Remolinos baroclínicos:** fuera, por ser 3-D (más de 1e9 celdas × pasos).

### A2. Configuraciones

| Modelo | Coordenadas | Esquemas |
|---|---|---|
| pyOCEAN | z\*, σ e híbrida (ALE) | los por defecto: momento vector-invariante con KE Arakawa, trazadores WENO5-Z, remapeo PPM_H4, continuidad PPM, PGF de volumen finito |
| MOM6 | z\* y ρ (`REGRIDDING_COORDINATE_MODE = "RHO"`) | remapeo PPM_H4, trazadores PPM:H3, `KE_ARAKAWA` |
| CROCO | σ | los de `cppdefs.h` de un caso regional; se registran en el README |
| MITgcm | z\* con celdas parciales | trazadores `tempAdvScheme = 7` (OS7MP, el que usó Ilicak) |

- La híbrida de pyOCEAN y MOM6 ρ usan los mismos objetivos de densidad, repartidos de forma uniforme sobre el rango del caso, con el mismo número de capas que los niveles z del caso.
- **Seamount:** σ en pyOCEAN, CROCO y MOM6, y z\* en los cuatro. No se barre ν_h.
- **Corridas de resultados:** 3 casos × 3 ν_h × 7 configuraciones = 63, más ~7 de seamount.
- **Costo:** por caso y configuración, una corrida con el Δt más grande que el modelo admita sin inestabilidad (lo que haría un usuario), en s por hora simulada y, donde el modelo lo da, repartido por componente. Son ~28 corridas.

### A3. HYCOM, solo costo

- Caso: el overflow, con 100 capas híbridas, EOS de 7 términos con S constante y paredes cerradas.
- **Compuerta:** si en ~1 h de trabajo HYCOM 2.3 (`HYCOM-src`, `ARCH=generic-gnu-relo`, `TYPE=mpi`, sin `-mavx`) no compila y corre un caso trivial (una cuenca en reposo, 10 pasos, `mpiexec -n 5`), se para y se reporta a Esteban.
- Se mide el s por hora simulada y el reparto de sus temporizadores (`-DTIMER`).

### A4. Métricas

Todas en `metrics.py`, idénticas para todos los modelos, sobre una serie común por corrida (tiempo, x, el grosor de cada celda h(t, k, x), T y u).
- **RPE** (Winters et al. 1995; Ilicak 2012, ec. 1–2) con `diagnostics/mixing.py`, sobre el volumen real de cada celda (celdas parciales, capas ALE, σ). Convención única: ρ completa y z medida desde el fondo, porque RPE₀ depende del origen de z (Ilicak, apéndice A). Se reportan (RPE − RPE₀)/RPE₀ en el tiempo y dRPE/dt: a las 17 h en el lock exchange (Ilicak), promediada entre 13 y 37 h en el overflow y entre los días 10 y 100 en la onda interna (Petersen, tabla 5).
- **Rango de T:** min y max de T en el tiempo, para detectar extremos nuevos. Una RPE baja con extremos nuevos no cuenta como poca mezcla (Ilicak, Fig. 7).
- **Física de cada caso:**
  - lock exchange: posición de los frentes denso y ligero contra Benjamin, medida donde la columna densa (o ligera) pasa de 1 m (punta) y 5 m (cuerpo), como en `validation/mom6_lock_exchange`;
  - overflow: profundidad y distancia del agua densa (centro de masa del agua con T < 15 °C);
  - onda interna: amplitud y fase de la perturbación de la isoterma central en x₀;
  - seamount: max\|u\| en el tiempo.

### A5. Criterios de lectura

No hay umbral de aprobación; la comparación es entre modelos y contra la física conocida.
- Lock exchange: frente dentro de ±10 % de Benjamin.
- La RPE de cada coordenada de pyOCEAN se compara con la de los otros modelos que la usan: z\* contra MITgcm y MOM6 z\*; σ contra CROCO; híbrida contra MOM6 ρ.
- Cada modelo debe mostrar la caída de dRPE/dt al bajar Re_Δ (Ilicak: un orden de magnitud entre Re_Δ > 100 y Re_Δ < 5). Si pyOCEAN no la muestra, es un hallazgo.
- Ninguna corrida debe salir del rango de T inicial más allá del redondeo.
- Onda interna: la que más pierde amplitud es la que más disipa o mezcla.
- Seamount: max\|u\| a 10 días contra ~1 cm s⁻¹ (Beckmann & Haidvogel) y contra los otros modelos.
- Costo: se señala toda pieza donde pyOCEAN cuesta más del doble que la misma pieza de otro modelo.
- **Clasificación:** si pyOCEAN se separa de todos los modelos de su coordenada, se trata como (a) o (b) y se diagnostica; si los modelos difieren entre sí y pyOCEAN cae dentro del rango, es (c).
- **Hipótesis previas,** para revisar con los resultados y no para afinar: pyOCEAN mezclará más que CROCO a Re_Δ alto (CROCO usa momento upwind de tercer orden; Ilicak §7); la híbrida y MOM6 ρ mezclarán mucho menos que z\* en el lock exchange y el overflow.

## Parte B: fronteras abiertas

**Idea (Herzfeld et al. 2011):** cada modelo se compara contra sí mismo en un dominio grande o periódico, antes de que regrese cualquier onda reflejada desde sus paredes. Las diferencias entre modelos quedan en la calidad de su frontera.

| Caso | Configuración | Referencia | Duración |
|---|---|---|---|
| B1. Anomalía de SSH que sale | Cuenca plana de 200 × 200 km × 100 m, Δx 2 km, homogénea, sin rotación; una gaussiana de η en el centro que sale por las 4 fronteras | el mismo modelo en 600 × 600 km | ~6 h |
| B2. Viento impulsivo en un canal | Canal con plataforma, f = 1e-4, fricción lineal de fondo, viento a lo largo de la costa constante desde t = 0; los dos extremos abiertos | el mismo canal periódico | ~5 días |
| B3. Onda interna de modo 1 | El canal plano de `tests/test_sigma_ale.py`: modo 1 forzado en el oeste, frontera radiante en el este | el mismo canal 3 veces más largo | ~10 períodos |

- **Fronteras de cada modelo:** su configuración radiante usual con exterior en reposo: pyOCEAN Flather + Orlanski, con y sin esponja; MOM6 `FLATHER,ORLANSKI`; CROCO `M2FLATHER`, `M3ORLANSKI` y `TORLANSKI`; MITgcm OBCS con Orlanski (con superficie libre lineal no admite η prescrita, `obcs_check.F:561-569`: diferencia de clase c).
- **Coordenadas:** z\* en pyOCEAN, MOM6 y MITgcm; σ en CROCO y en pyOCEAN.
- **Métricas:**
  - reflexión: energía que queda en el dominio pequeño frente a la referencia al mismo tiempo (Mason et al. 2010, Fig. 3: ≤ 1 % en B1);
  - deriva de η̄ (el "vaciado de la cuenca" de Herzfeld);
  - índice de continuidad de Cailleau et al. (2008, ec. 7) para u y η frente a la referencia, en una franja junto a la frontera;
  - corriente de borde: índice de Mason (ec. 10), en B2;
  - B3: amplitud reflejada, del ajuste de una onda incidente más una reflejada;
  - costo.
- **Corridas:** 3 casos × 6 configuraciones × 2 dominios ≈ 36, todas cortas.
- **Fuera:** las mejoras de frontera de la lectura (corrección de volumen de Mason ec. 9, Flather generalizado de Oddo & Pinardi ec. 14, τ_in = Δx/U_max) se evalúan con estos casos en un plan aparte.
- La geometría exacta de B2 (ancho del canal y de la plataforma, viento, fricción) y la de la gaussiana de B1 se fijan en el plan de la parte B, con la condición de que la referencia no reciba reflexiones de sus propias paredes antes del final.

## Piezas

**`validation/ilicak_benchmark/`** (parte A):
- `cases.py`: cada caso como función que devuelve `dx`, `dz`, H(x), T₀(x, z), EOS, ν_h, ν_v, Δt, duración e intervalo de salida. Es la única fuente de verdad. El seamount reutiliza `validation/seamount`.
- `mom6.py`: supergrid, `topog.nc`, T inicial, `MOM_input`, `diag_table` e `input.nml` para z\* y ρ. La sección 2-D se rellena con tierra hasta 10 columnas (FMS compara todos los halos con el tamaño en x, `mpp_domains_define.inc:899-906`, como en `validation/mom6_bayiw`). `G_EARTH = 9.81`.
- `croco.py`: `croco_grd.nc`, `croco_ini.nc`, `croco.in`, `cppdefs.h` y `param.h` en el directorio de la corrida; un build por malla (CROCO fija el tamaño al compilar). Los casos 2-D llevan unas pocas filas en y, como el `OVERFLOW` de CROCO.
- `mitgcm.py`: batimetría y T en binario, `data`, `data.pkg`, `SIZE.h` y `CPP_OPTIONS.h` en el directorio de `-mods`; un build por malla, con el `optfile` `darwin_arm64_gfortran_esteban` como en `validation/internal_wave/run_mitgcm.sh`. Ny = 1.
- `run_pyocean.py`: construye `SigmaBaroclinic` en ALE desde `cases.py`.
- `read_<modelo>.py` (o un solo `readers.py`): cada salida a la serie común `series.npz`.
- `metrics.py` y `compare.py`: métricas, tablas por caso y por ν_h con los modelos lado a lado, y figuras de T al tiempo final.
- `run_<modelo>.sh`: `build | run <caso> <coord> <ν> | cost <caso>`. Cada script mata solo sus propios procesos; si una corrida falla, termina con FAILED sin procesar la salida.
- `hycom/`: build, compuerta y costo del overflow.
- `README.md`.

**`validation/obc_benchmark/`** (parte B): la misma estructura, reutilizando los escritores y lectores de la parte A.

## Pruebas (sin simulaciones)

- Los escritores reproducen `cases.py` en esquinas, centro y en la compuerta o el frente.
- Cada lector reconstruye un campo conocido de un archivo sintético con el formato de su modelo.
- La RPE no depende de cómo se reparte la columna: la misma ρ(z) dada en niveles z, en σ y en capas ALE da la misma RPE.
- Los `MOM_input`, `data` y `cppdefs.h` generados contienen los parámetros pedidos (coordenada, capas, Δt, EOS, ν, duración).
- `cases.py`: Δρ del lock exchange = 5 y del overflow = 2; la T de la onda interna en x₀ y en los bordes; la batimetría del overflow en 0, x₀ y 200 km.

## Entregables

- `README.md` de cada parte con tablas por caso, figuras, la clasificación (a)–(d) y la lista de qué mejorar en pyOCEAN.
- Filas nuevas en el marcador (§7 de `docs/design/vision_roadmap.md`): mezcla espuria frente a Re_Δ, onda interna, overflow, seamount multimodelo, fronteras abiertas y costo de cada modelo.
- `CLAUDE.md` actualizado.
- Los artículos usados, citados en el README; los nuevos, en `docs/literature/reading_list.md`.

## Presupuesto de cómputo (estimado)

| Corrida | Estimado |
|---|---|
| Builds: CROCO (4 mallas) y MITgcm (4 mallas) | ~1–2 min cada uno |
| Lock exchange: 21 corridas de 17 h | < 1 min cada una |
| Overflow: 21 corridas de 40 h | ~2–5 min cada una |
| Onda interna: 21 corridas de 200 días | ~1–5 min cada una |
| Seamount: ~7 corridas de 10 días | ~2–5 min cada una |
| Costo (parte A): ~28 corridas cortas | ~15 min en total |
| HYCOM: build, compuerta y costo | ≤ 1 h de trabajo |
| Parte B: ~36 corridas cortas | ~1–3 min cada una |

Si una corrida pasa de 3 veces su estimado, se reporta antes de seguir.

## Referencias

- Ilicak, M., Adcroft, A. J., Griffies, S. M., Hallberg, R. W. (2012). Spurious dianeutral mixing and the role of momentum closure. Ocean Modell. 45–46, 37–58. [10.1016/j.ocemod.2011.10.003](https://doi.org/10.1016/j.ocemod.2011.10.003)
- Petersen, M. R. et al. (2015). Evaluation of the arbitrary Lagrangian–Eulerian vertical coordinate method in the MPAS-Ocean model. Ocean Modell. 86, 93–113.
- Benjamin, T. B. (1968). Gravity currents and related phenomena. J. Fluid Mech. 31.
- Beckmann, A., Haidvogel, D. B. (1993). Numerical simulation of flow around a tall isolated seamount. J. Phys. Oceanogr. 23.
- Winters, K. B. et al. (1995). Available potential energy and mixing in density-stratified fluids. J. Fluid Mech. 289.
- Herzfeld, M. et al. (2011). Realistic test cases for limited area ocean modelling. Ocean Modell. 37. [10.1016/j.ocemod.2010.12.008](https://doi.org/10.1016/j.ocemod.2010.12.008)
- Cailleau, S. et al. (2008). Comparison of different numerical methods used to handle the open boundary of a regional ocean circulation model of the Bay of Biscay. Ocean Modell. 25. [10.1016/j.ocemod.2008.05.009](https://doi.org/10.1016/j.ocemod.2008.05.009)
- Mason, E. et al. (2010). Procedures for offline grid nesting in regional ocean models. Ocean Modell. 35. [10.1016/j.ocemod.2010.05.007](https://doi.org/10.1016/j.ocemod.2010.05.007)
- Oddo, P., Pinardi, N. (2008). Lateral open boundary conditions for nested limited area models: a scale selective approach. Ocean Modell. 20. [10.1016/j.ocemod.2007.08.001](https://doi.org/10.1016/j.ocemod.2007.08.001)
- Klingbeil, K. et al. (2018). The numerics of hydrostatic structured-grid coastal ocean models: state of the art and future perspectives. Ocean Modell. 125. [10.1016/j.ocemod.2018.01.007](https://doi.org/10.1016/j.ocemod.2018.01.007)
- Chen, et al. (2024). Gradient-tracking generalized vertical coordinate for ocean models (ROMS). [verificar]
