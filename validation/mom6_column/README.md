# Validación de nivel 1: kit de columna ALE contra el Fortran de MOM6

Compara las rutinas por columna de `core/vertical/` (remapeo y reglas de malla) con las
rutinas originales de MOM6, compiladas en pequeños programas Fortran. Es el nivel 1 de la
matriz de validación de la especificación
(`docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` §8).

## Compilación

Requisitos (macOS, Homebrew): `gfortran` 15, Open MPI (`mpif90`, `mpicc`),
`netcdf` + `netcdf-fortran` (`nc-config`, `nf-config`), `autoconf`, `automake` y `libtool`.

Fuentes:
- MOM6 en `~/pyOCEAN_refs/MOM6` (commit 0e9a561), con los submódulos
  `pkg/GSW-Fortran` y `pkg/CVMix-src` inicializados (`git submodule update --init`).
- FMS 2025.02.01 en `~/pyOCEAN_refs/FMS`. Se clona localmente con `FMS_URL=file://…`.

```bash
bash validation/mom6_column/build_reference.sh
```

El script usa el Makefile de `MOM6/.testing` (`make -j5 build/symmetric/MOM6`). Después
enlaza cada `*_driver.F90` de esta carpeta con los objetos de MOM6 (sin los drivers
`MOM_driver`, `ocean_solo` y `MOM_sum_driver`), `libFMS.a`, `libgsw.a` y `libcvmix.a`.
Los ejecutables quedan en `~/pyOCEAN_refs/build/column/`.

Para esta máquina hubo que adaptar el plan en tres puntos:

| Adaptación | Motivo |
|---|---|
| `FC=mpif90 MPIFC=mpif90 MPICC=mpicc` en la llamada a `make` | Sin `FC`, el configure de FMS cae en `f77`. Con `FC=gfortran` no encuentra `mpi.mod`. |
| `DEPS = .testing/build/deps` | Es la ruta real de `$(BUILD)/deps` en el Makefile. |
| `$(nc-config --libs)` al enlazar | `nf-config --flibs` no da la ruta de la librería C `-lnetcdf`. |

Los tests de pytest (`tests/test_vertical_vs_mom6.py`) se saltan cuando falta un ejecutable.

## Qué se compara

| Driver | Rutina de MOM6 | Rutina de pyOCEAN | Tolerancia |
|---|---|---|---|
| `remap_driver` | `MOM_remapping::remapping_core_h` (`initialize_remapping` con `boundary_extrapolation=.false.`, `om4_remap_via_sub_cells=.false.`, `answer_date=99991231`, `h_neglect=1e-30`, `h_neglect_edge=1e-10`) | `remap.remap_column` | 1e-11 |
| `regrid_driver` | `coord_zlike::build_zstar_column`, `coord_sigma::build_sigma_column` (min_thickness 1e-3), `regrid_interp::build_and_interpolate_grid` (`P1M_H2`, sin extrapolación, `answer_date=99991231`) | `regrid.zstar_column`, `regrid.sigma_column`, `regrid._p1m_h2_positions` | 1e-10 (z\*, σ), 1e-9 (P1M) |
| `kappa_shear_driver` | `MOM_kappa_shear::kappa_shear_column` (privada, extraída; `nzc = nz`, EOS lineal en T, Boussinesq, valores por defecto) | `physics.mixing.kappa_shear.kappa_shear_column` | 1e-10 × max de la columna (κ y Q) |
| `bbl_driver lotw` | `MOM_set_diffusivity::add_LOTW_BBL_diffusivity` (extraída con `set_diffusivity_CS`; Boussinesq, valores OM4) | `physics.mixing.bbl_mixing.lotw_bbl_column` | 1e-10 × max Kd de la columna |
| `bbl_driver tke` | `MOM_set_diffusivity::set_BBL_TKE` (extraída) | factor contra C_d\|U\|³ y √C_d\|U\| | 1e-12 relativo |

Las columnas de prueba se generan en `columns.py`:
- semilla 2026;
- 15 % de capas de origen vacías;
- Σh igual en origen y destino;
- tamaños 10→10, 25→40, 50→50 y 4→7.

Las columnas de regrid (semilla 7) son:
- z\* y σ con los `delR` de 10 niveles (Σ = 201 m) para H = 201, 164, 40 y 11 m, con η de −0.2 a 0.3 m;
- 6 perfiles de densidad de 12 capas con 13 densidades objetivo, varias fuera del rango de la columna.

## Resultados

### Remapeo (`remapping_core_h`)

| Esquema | 10→10 | 25→40 | 50→50 | 4→7 |
|---|---|---|---|---|
| PCM | 1.2e-14 | 9.6e-14 | 1.2e-12 | 1.6e-14 |
| PLM | 1.4e-13 | 6.0e-14 | 4.7e-12 | 1.1e-14 |
| PPM_H4 | 5.2e-14 | 1.6e-13 | 8.5e-13 | 5.3e-15 |

La tabla da max |u_pyOCEAN − u_MOM6| para valores de orden 10 con rangos de 1 a 10. La
concordancia es de redondeo en los tres esquemas, también con capas de origen vacías y en
la columna de 4 celdas, donde PPM baja a PLM.

**Extremos degenerados** (añadidos tras la revisión final). Son 108 columnas de 12 capas, 36 por esquema:
4 capas gruesas en el interior y, en cada extremo, una de 6 combinaciones de capas vacías,
de 1e-10 m y de 1 mm junto a una capa gruesa. El max |Δ| es 1.3e-13 (PCM), 5.5e-14 (PLM) y
9.1e-14 (PPM_H4). La primera versión resolvía el sistema 4×4 de los valores de extremo H4
por eliminación directa. Con estos extremos fallaba (división por cero, o errores de hasta
0.3). Ahora se usa la forma cerrada de MOM6 `end_value_h4`
(`regrid_edge_values.F90:636-749`), con sus guardas `min_frac`.

**Diferencia de diseño, no comparada.** pyOCEAN baja el orden según el número de capas
*activas* (1 → PCM, ≤ 3 → PLM). MOM6 cuenta todas las capas
(`MOM_remapping.F90:453-460`). Una columna z\* con ≤ 3 capas activas entre capas vacías
usa PLM en pyOCEAN y PPM_H4 en MOM6, y puede diferir hasta ~0.4 en valores de orden 10. En
una columna seca (sin grosor), pyOCEAN conserva los valores.

### Reglas de malla

| Caso | max \|Δz\| (m) |
|---|---|
| z\*, H = 201 / 164 / 40 / 11 m | 1.1e-14 / 2.8e-14 / 0 / 2.2e-16 |
| σ, H = 201 / 164 / 40 / 11 m | 1.7e-14 / 1.7e-14 / 0 / 2.0e-15 |
| P1M_H2, 6 perfiles (posiciones interiores no triviales, p. ej. 9 de 13 en el primero) | 0 (bit a bit) |

En H = 40 m, z\* colapsa los niveles nominales de 30, 50 y 70 m al grosor mínimo en el
fondo, igual que MOM6.

Diferencias de diseño, no cubiertas por estos casos:
- `build_sigma_column` de MOM6 usa el grosor mínimo tal cual (`coord_sigma.F90:76-81`).
  `regrid.sigma_column` usa min(h_min, D/N), como `build_zstar_column` (`coord_zlike.F90:90`)
  y como pide la especificación.
- pyOCEAN fija la interfaz superior en η y luego separa las interfaces desde arriba y desde
  el fondo (`enforce_min_thickness`). MOM6 solo infla desde el fondo. La versión anterior
  hacía el barrido desde el fondo y después fijaba η, lo que dejaba la capa superficial
  negativa con σ estirada en aguas muy someras (−2.3 cm con D = 5 cm, Song–Haidvogel θ_s = 5,
  50 capas). La revisión final lo encontró.

Falta comparar `build_hycom1_column` completo (piso z\* y grosor máximo) contra MOM6. Solo se
comparó su interpolación P1M_H2.

## Regla híbrida contra `hybgen` de HYCOM (puerto de MOM6)

Esta comparación es cuantitativa, no de redondeo: HYCOM1 (`regrid.hycom1_column`) y
`hybgen` son algoritmos distintos.
- **HYCOM1** invierte una reconstrucción P1M de la densidad para cada densidad objetivo, con
  un piso z\* para las interfaces.
- **hybgen** mueve interfaces arrastrando agua de las capas vecinas, con grosores mínimos
  que pasan por la función de amortiguamiento `cushn` (Bleck 2002).

**Driver.** Las rutinas de columna de hybgen son privadas en `MOM_hybgen_regrid.F90`.
`build_reference.sh` extrae `hybgen_column_init`, `cushn` y `hybgen_column_regrid` del fuente
local de MOM6 a un módulo con una copia pública de los 7 campos del tipo de control que usan.
`hybgen_driver.F90` repite la preparación de `init_hybgen_regrid`
(`MOM_hybgen_regrid.F90:224-239`) y la dilatación z\* de `hybgen_regrid` (`:505-512`), con los
valores por defecto de MOM6.

**Montaje** (`compare_hybgen.py`), común a las dos reglas:
- 50 capas;
- malla inicial z\* sobre los `delR` de BayIW;
- densidades de BayIW (EOS lineal) o perfiles idealizados del mismo rango;
- grosor mínimo de 1 mm;
- hybgen con `nsigma = 0`, `isotop = 0`, `thkbot = 0` y `qhybrlx = 1`.

Un mismo perfil dz sirve como `dp0k` de hybgen y como piso z\* de HYCOM1. Se mide un paso de
regrid y el estado tras 10 ciclos de regrid + remapeo PPM_H4 de la densidad (el mismo remapeo
para las dos reglas).

| Configuración | dz | Objetivos de densidad |
|---|---|---|
| A | `delR` (Σ = 599.9 m) | uniformes en el rango del perfil BayIW |
| B | `delR` en las 15 capas superiores (31 m) y 1 mm debajo, al estilo HYCOM | igual que A |
| C | igual que B | las 15 capas superiores más ligeras que cualquier agua (estilo OM4), el resto uniformes |

Columnas:
- estratificada de 600 m (exponencial);
- mezclada de 600 m;
- somera de 30 m (pyOCEAN con transición a σ, d_σ = 50 m, d_hyb = 100 m);
- 164 m con los 10 m de fondo 0.5 kg m⁻³ más ligeros;
- frontera de BayIW de 600 m.

`/h` es max |Δz| dividido por el grosor de la capa de hybgen en ese punto.

```
Configuración A (dz = delR)
columna                  | 1 paso: max|Δz| m   medio   k     /h | 10 ciclos: max|Δz|   medio   k     /h | capas>2h_min hybgen/pyOCEAN
-------------------------------------------------------------------------------------------------------------------------------------
estratificada 600 m      |           163.996  62.754  46   4.51 |            163.996  62.754  46   4.51 |  50 /  46
mezclada 600 m           |           517.976 169.545  26  68.15 |            517.976 169.545  26  68.15 |  50 /  26
somera 30 m              |            20.366   9.525  16   8.99 |             20.366   9.525  16   8.99 |  16 /  50
164 m, fondo ligero      |           136.265  55.916  15  42.58 |            125.768  51.578  18  30.68 |  35 /  18
BayIW frontera 600 m     |            88.466  61.465  16  25.28 |             88.466  61.465  16  25.28 |  50 /  50

Configuración B (z en 31 m superiores)
columna                  | 1 paso: max|Δz| m   medio   k     /h | 10 ciclos: max|Δz|   medio   k     /h | capas>2h_min hybgen/pyOCEAN
-------------------------------------------------------------------------------------------------------------------------------------
estratificada 600 m      |           127.597  51.142  46   3.25 |             46.398  12.426  15   0.86 |  43 /  46
mezclada 600 m           |           510.377 162.820  26  34.72 |            406.890  73.540  26   3.44 |  32 /  16
somera 30 m              |            20.366   9.525  16   8.99 |             20.366   9.525  16   8.99 |  17 /  50
164 m, fondo ligero      |           136.265  53.201  15  20.34 |             84.886  28.693  12  32.65 |  26 /  18
BayIW frontera 600 m     |            88.186  56.390  15  13.16 |             88.186  30.188  15   1.63 |  47 /  50

Configuración C (B + objetivos ligeros arriba)
columna                  | 1 paso: max|Δz| m   medio   k     /h | 10 ciclos: max|Δz|   medio   k     /h | capas>2h_min hybgen/pyOCEAN
-------------------------------------------------------------------------------------------------------------------------------------
estratificada 600 m      |            82.522   4.772  47   1.94 |             17.770   1.461  47   0.19 |  44 /  44
mezclada 600 m           |           449.783 104.278  33  16.36 |            449.783  28.751  33   2.04 |  31 /  16
somera 30 m              |            20.366   9.525  16   8.99 |             20.366   9.525  16   8.99 |  17 /  50
164 m, fondo ligero      |            75.548  13.685  25   9.94 |             37.337   3.507  28   3.60 |  33 /  28
BayIW frontera 600 m     |            41.186  11.418  48   0.54 |             41.182   4.076  48   0.38 |  48 /  50
```

**Interpretación**

- **A no sirve para comparar.** Con `dp0k = delR` y Σ`delR` igual a la profundidad, todas las
  capas de hybgen están en su grosor mínimo y ninguna interfaz puede moverse
  (`MOM_hybgen_regrid.F90:775-790`). HYCOM1 usa dz solo como piso de las interfaces, así que
  sí mueve las profundas. Las diferencias grandes de A vienen del montaje, no de los algoritmos.
- **B: diferencias de 13 a 25 capas locales, por una decisión de diseño.**
  - En hybgen, las capas k ≤ `fixlay` son z fijas.
  - En HYCOM1 el piso z\* solo impide que una interfaz suba. Con objetivos que empiezan en la
    densidad superficial, la primera interfaz baja hasta la isopicna de 999.84 (39 m en BayIW)
    y toda la capa de mezcla queda en una sola capa.
  - Hybgen, por su parte, deja capas vacías alternadas en el interior (137.9, 137.9; 150.1,
    150.1…). Es el "thin-thick-thin null mode" que el propio código de MOM6 reconoce
    (`MOM_hybgen_regrid.F90:806-813`).
- **C: coinciden en el interior.** Con objetivos ligeros arriba, como en OM4, HYCOM1 mantiene
  z en las capas superiores. Tras 10 ciclos, la diferencia media es de 4.1 m en BayIW y 1.5 m
  en la columna estratificada, con un máximo de 0.4 y 0.2 grosores de capa locales (antes del
  Plan B2c: 4.0 y 0.8 m). Los máximos
  absolutos (41 y 15 m) están en las capas 47–48, junto al fondo, donde hybgen aplica
  `dp_far_from_bot` y deja capas vacías.
- **Mezclada y fondo ligero.** En una columna homogénea no hay respuesta única: las capas son
  vacías en las dos reglas, solo cambia dónde quedan. En el fondo ligero, HYCOM1 monotoniza el
  perfil y hybgen arrastra agua capa a capa, y la diferencia media es de 6 m.
- **Somera.** pyOCEAN pasa a σ por diseño; hybgen con `nsigma = 0` sigue en z.

**Conclusión para la especificación (§3).** Las diferencias mayores que una capa no vienen de
un defecto de HYCOM1. Vienen de dos rasgos de hybgen:
- capas z superiores fijas;
- capas vacías alternadas, que son un defecto conocido.

**Plan B2 (2026-10-12).** La tabla de arriba ya incluye dos cambios de la regla:
- las capas sin masa toman la densidad de la capa con masa de arriba, como el relleno de hybgen
  (`hybgen.F90:904-925`); la columna «164 m, fondo ligero» quedó más cerca de hybgen (Δz medio a
  10 ciclos 55.8 → 51.6 m en A, 33.0 → 28.7 m en B, 6.2 → 3.5 m en C);
- `keep_isopycnal` (B2c, por defecto): con `keep_isopycnal` las capas que ya están en su rango
  de densidad no se mueven; las diferencias con hybgen que quedan son las de las capas
  inconsistentes. En la columna estratificada de C la diferencia media a 10 ciclos sube de 0.8 a
  1.5 m, porque las capas que ya están en su rango dejan de acercarse a la densidad objetivo
  exacta; es el precio de no mezclar en flujo adiabático (lock exchange híbrido: ganancia de RPE
  35 veces menor que ALE-σ).

No hace falta portar hybgen. Para el Plan B, la regla híbrida necesita objetivos más ligeros
que el agua superficial en las capas de arriba (estilo OM4), o grosores máximos
(`max_h_top`), para que la capa de mezcla quede resuelta en z.

## Mezcla por cizalla de Jackson et al. (2008) (`kappa_shear_column` de MOM6)

Compara el oráculo NumPy `physics/mixing/kappa_shear.py` (Jackson, tarea 1) con la rutina
de columna de MOM6 compilada (`MOM_kappa_shear.F90`, git 0e9a561).

**Driver.** `kappa_shear_column`, `calculate_projected_state` y `find_kappa_tke` son
privadas (`MOM_kappa_shear.F90:866-2069`). `build_reference.sh` las copia tal cual a
`~/pyOCEAN_refs/build/column/kappa_shear_column_mom6.F90`, junto con el tipo
`Kappa_shear_CS` (`:36-128`, hecho público) y las líneas `use` del módulo (`:8-21`). En la
copia se insertan cuatro contadores enteros, que no cambian ninguna operación de punto flotante:
- iteraciones de Newton (tras `:1825`);
- eventos de la rampa de truncamiento de Picard (tras `:1802` y `:1817`);
- abortos de Newton (tras `:2035`);
- iteraciones κ–Q (tras `:1701`).

`kappa_shear_driver.F90` repite la preparación de `Calculate_kappa_shear` (`:299-327`):
- sin capas sin masa, así que `nzc = nz`;
- κ inicial = `KD_SEED_KAPPA_SHEAR` (`:325`);
- Boussinesq, escalas de unidades 1, sin presión superficial;
- `Kappa_shear_CS` con los valores de `JacksonParams()` (= valores por defecto de MOM6),
  `KAPPA_SHEAR_ITER_BUG = False`, `VEL_UNDERFLOW = 0`, `USE_RESTRICTIVE_TOLERANCE_CHECK = False`,
  `KD_TRUNC_KAPPA_SHEAR = 0.01 κ0`.

**Flotabilidad.** `tv%T` no está asociado, así que MOM6 usa `dbuoy_dT = −g/ρ₀` y
`dbuoy_dS = 0` (`:1148-1149`); sin temperatura, MOM6 mismo pasa la densidad como "T"
(`:307-310`). Con `g = ρ₀ = 1` y `T = −b`, `dbuoy_dT·(T(k−1) − T(k)) = b(k−1) − b(k)` sin
redondeo: es una EOS lineal en T, y la diferencia de flotabilidad es la del oráculo.

**Compilación.** El driver se compila con `-ffp-contract=off`. gfortran contrae `a*b + c` en
FMA por defecto en arm64, y NumPy no lo hace.

**Casos** (`columns.kappa_shear_cases`):

| # | Columna | Δt (s) | κ0 (m² s⁻¹) |
|---|---|---|---|
| 1 | capa de cizalla tanh, Ri_min 0.1, N0 = 0.01 s⁻¹, d = 5 m, 100 × 2 m | 600 | 1e-7 |
| 2 | chorro de fondo tipo overflow: u máx 0.6 m/s a 100 m del fondo, v que gira, interfaz densa (Δb 1e-3 m s⁻²) a 130 m, N² de fondo 1e-6, 70 capas de 2 a 10 m (κ llega al fondo: rama no-slip) | 600 | 1e-7 |
| 3 | subcrítica en toda la columna (Ri = 4) | 600 | 1e-7 |
| 4 | capa de cizalla del caso 1 | 3600 | 1e-7 |
| 5 | cizalla uniforme, Ri = 0.22, 30 × 1.67 m (Newton) | 60 | 1e-7 |
| 6 | cizalla uniforme, Ri = 0.15, 30 × 6.67 m (Newton) | 3600 | 1e-7 |
| 7–18 | 12 columnas aleatorias (semilla 2008): N = 8–60, dz 0.5–15 m, cizalla aleatoria, N² ≥ 0 con ruido, f² ≤ 2e-8 | 300 / 1800 / 7200 | 1e-7 / 1e-6 / 1e-5 |

**Resultados.** Se da max |Δκ| / max κ y max |ΔQ| por columna.

| Caso | max κ (m² s⁻¹) | Δκ sin FMA | ΔQ sin FMA | Δκ/max κ con FMA | Newton / rampa / abortos / iteraciones (MOM6) |
|---|---|---|---|---|---|
| 1 capa de cizalla, 600 s | 2.54e-2 | 0 | 0 | 2.7e-16 | 0 / 9 / 5 / 56 |
| 2 chorro de fondo | 0.987 | 0 | 0 | 2.5e-15 | 0 / 25 / 18 / 206 |
| 3 subcrítica | 0 | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
| 4 capa de cizalla, 3600 s | 4.29e-3 | 0 | 0 | 3.0e-16 | 0 / 5 / 5 / 45 |
| 5 cizalla uniforme, Ri 0.22 | 0.121 | 0 | 0 | 2.0e-15 | 4 / 0 / 1 / 10 |
| 6 cizalla uniforme, Ri 0.15 | 0.936 | 0 | 0 | 2.1e-13 | 2 / 49 / 12 / 293 |
| 7–18 aleatorias | 1.3e-3 – 7.6e-2 | 0 | 0 (2.6e-26 m² s⁻² antes de la tarea 3) | ≤ 1.1e-13 | 0 / 2–135 / 3–18 / 19–198 |

Sin FMA, el oráculo reproduce MOM6 bit a bit en los 18 casos. Hasta la tarea 3 quedaba un
2.6e-26 m² s⁻² en Q de una columna aleatoria: el oráculo elevaba al cuadrado con `x ** 2` de
NumPy/Python, que llama a `pow()` de libm y difiere de `x*x` en 1 ulp para ~0.1 % de los
argumentos; gfortran compila `x**2` como `x*x`. El oráculo escribe ahora los cuadrados como
productos. Con la contracción FMA (compilación de prueba, no guardada), la diferencia sube a
≤ 2.1e-13 relativa, sin cambiar ninguna decisión discreta.

**Tolerancia del test.** |Δκ| ≤ 1e-10 max κ y |ΔQ| ≤ 1e-10 max Q, por columna. El esquema
itera hasta `KAPPA_SHEAR_TOL_ERR = 0.1` y tiene ramas discretas:
- truncamiento en κ_trunc;
- reducción del subpaso a la mitad;
- cambio a Newton.

Un error de traducción, o una rama tomada de otra forma, da diferencias ≥ 1e-3 de κ. Por
ejemplo, cambiar `FRI_CURVATURE` de −0.97 a −0.9701 da de 7e-5 a 2e-3 relativo. 1e-10 está
muy por encima del redondeo (también con FMA) y nueve órdenes por debajo de la tolerancia
del esquema. El test también exige `n_newton > 0` en los casos 5 y 6, κ ≡ 0 en el caso 3
y al menos un evento de rampa.

**Rama de Newton.** La tarea 1 encontró que, en el oráculo, Newton no se usaba nunca
(0 de 7303 iteraciones). Aquí, MOM6 compilado lo confirma en las columnas con mezcla
localizada (casos 1–4 y 7–18: 0 iteraciones de Newton). La causa es la prueba de
convergencia (`:2003-2021`):
- `do_Newton` se pone en `.true.` antes del bucle;
- la primera interfaz con |Δκ| > 0.2 κ̄ marca `abort_Newton`, aunque la iteración fuera de Picard;
- eso baja `Newton_err` de 0.2 a 0.1 = `tol_err`, y ya no se cambia a Newton en esa llamada.

Una interfaz con κ = 0 vuelve a empezar en Q = Q_min = 1e-20 (`:1673-1677`), y su Q cambia en
órdenes de magnitud en la primera iteración. Por eso, Newton solo se alcanza en llamadas con
arranque caliente (corrector o subpasos posteriores, con `K_Q` del paso anterior) en una
columna turbulenta en todas sus interfaces interiores. Un barrido de 432 columnas de cizalla
uniforme lo confirma: Newton aparece en 26, todas con κ > 0 en toda la columna. Los casos 5
y 6 lo activan (4 y 2 iteraciones de Newton), y ahí el oráculo coincide bit a bit. La rama
de Newton con la derivada corregida (`:1860-1866`) y la rampa de truncamiento de Picard
(`:1801-1803`, `:1816-1818`) quedan cubiertas.

**Clasificación.** No queda ninguna diferencia que clasificar: el oráculo coincide con MOM6
al redondeo. Que MOM6 aborte Newton en la primera iteración de casi toda llamada es un
rasgo de MOM6, clase (d) como posible defecto de la referencia: hace más iteraciones de
Picard, pero no cambia el resultado convergido. El oráculo lo reproduce a propósito.

## Mezcla BBL por arrastre (`add_LOTW_BBL_diffusivity` y `set_BBL_TKE` de MOM6)

Compara el oráculo NumPy `physics/mixing/bbl_mixing.py` (BBL, tarea 1) con MOM6 compilado
(`MOM_set_diffusivity.F90`, git 0e9a561; SD abajo). Es el criterio 1 de la especificación
`docs/superpowers/specs/2026-10-30-bbl-drag-mixing-design.md`.

**Driver.** `add_LOTW_BBL_diffusivity` es privada; `set_BBL_TKE` es pública, pero exige un
`set_diffusivity_CS` inicializado, cuyo tipo es privado (sus campos no se pueden fijar
desde fuera sin `set_diffusivity_init` y un archivo de parámetros). `build_reference.sh` copia tal cual a
`~/pyOCEAN_refs/build/column/bbl_column_mom6.F90`:
- las líneas `use` del módulo (SD:8-45);
- el tipo `set_diffusivity_CS` (SD:62-203), hecho público;
- `add_LOTW_BBL_diffusivity` (SD:1606-1789) y `set_BBL_TKE` (SD:1947-2152), sin cambiar
  ninguna línea.

Los demás tipos que reciben (`ocean_grid_type`, `verticalGrid_type`, `unit_scale_type`,
`thermo_var_ptrs`, `forcing`, `vertvisc_type`) son públicos con campos públicos.
`bbl_driver.F90` construye a mano los mínimos, sin `MOM_grid_init` ni archivos de
parámetros. Es más simple que extraer el cuerpo y reescribir sus argumentos:
- malla de 3 × 3 celdas (celda de cómputo i = j = 1, un halo, memoria simétrica) con
  celdas de 1 km × 1 km;
- Boussinesq y escalas de unidades 1 (H = Z = L = m);
- `GV%dZ_subroundoff` con la fórmula de `MOM_verticalGrid.F90:220`;
- sin `Ray_u`/`Ray_v`, sin `ustar_tidal` ni `BBL_tidal_dis` (pyOCEAN no los tiene);
- `LOTW_BBL_ANSWER_DATE = 99991231`.

Se compila con `-ffp-contract=off`, igual que `kappa_shear_driver`.

**Entradas de `lotw`.** MOM6 recibe `visc%ustar_BBL = u*` y
`visc%BBL_meanKE_loss = tke / BBL_EFFIC`. Así, su `TKE_column = BBL_EFFIC × BBL_meanKE_loss`
(SD:1709) es la `tke` de pyOCEAN, salvo 1 ulp, y se compara el núcleo con la misma energía.
Las cuatro f de las esquinas son iguales, de modo que `absf` (SD:1680-1681) es |f| exacto.
Ω es el `OMEGA` de pyOCEAN. Ninguna columna fusiona capas delgadas (decisión 6 de la
especificación).

**Casos** (`columns.bbl_cases`; parámetros OM4 de `BBLParams()` salvo donde se indica):

| # | Columna | N | max Kd (m² s⁻¹) | max \|ΔKd\| / max Kd | interfaces con Kd > 0 |
|---|---|---|---|---|---|
| 1 | estratificada, 40 × 5 m, N² = 1e-5 e^{z/100 m}, u\* 1 cm/s, \|f\| 1e-4, tke 5e-6 m³ s⁻³ | 40 | 3.56e-2 | 0 | 3 |
| 2 | 1 con N² = −2e-6 y 0 a 5 y 10 m del fondo (el piso Ω² mantiene W > 0) | 40 | 5.00e-2 | 0 | 4 |
| 3 | 2 sin piso Ω² (`use_omega=False`): las dos interfaces toman KD_MAX = 0.1 (SD:1769-1772) | 40 | 0.100 | 0 | 4 |
| 4 | 1 con f = 0 (escala de decaimiento = 200 m) | 40 | 3.90e-2 | 0 | 3 |
| 5 | 1 con capas de 1 mm en el fondo y 3 capas más arriba | 40 | 3.55e-2 | 0 | 5 |
| 6 | u\* grande: 5 cm/s, \|f\| 1e-4 (u\*/\|f\| = 500 m, tope de 200 m), 50 × 20 m, tke 1e-3 | 50 | 1.99 | 0 | 8 |
| 7 | energía sobrada, N² = 1e-8, 20 × 2 m, u\* 2 cm/s: Kd = Kd_wall sin recorte (SD:1756-1757) | 20 | 7.81e-2 | 0 | 19 |
| 8–15 | 8 aleatorias (semilla 2006): N = 5–60, dz 0.2–30 m, N² de −1e-6 a 1e-4, u\* 0.1–5 cm/s, \|f\| 0–1.4e-4, tke 1e-9–1e-4; la mitad sin piso Ω² | 13–60 | 6.3e-6 – 8.3e-2 | ≤ 1.5e-17 | 1–2 |

**Resultado.** El oráculo coincide con MOM6 bit a bit en 14 de 15 columnas. La otra
(aleatoria 0) difiere en 1.5e-17 relativo: es el ulp de `tke / BBL_EFFIC × BBL_EFFIC`. No
hizo falta corregir el oráculo. El caso 7 también se compara con la fórmula cerrada
κ u\*² z (D − z)/(u\* D + |f| z (D − z)) a 1e-12.

**Tolerancia del test.** |ΔKd| ≤ 1e-10 max Kd por columna, la de la especificación. El
núcleo es un solo barrido sin iteración. Sus ramas discretas son el recorte por energía
(SD:1765) y la rama de KD_MAX (SD:1769). Un error de traducción mueve Kd en O(1). El test
exige también que la rama de KD_MAX aparezca (dos interfaces del caso 3).

**Diferencia conocida, no comparada.** Con u\* = 0 y energía > 0, MOM6 da KD_MAX en todas
las interfaces interiores. El oráculo devuelve 0 (docstring de `bbl_mixing.py`). Ningún caso
usa u\* = 0.

### El factor de `set_BBL_TKE`

`bbl_driver tke` corre `set_BBL_TKE` compilado (extraído tal cual, sin transcribir) para un
flujo uniforme en x: u = U en todas las capas, v = 0. Sus entradas por cara son las que da
`set_viscous_BBL` (`MOM_set_viscosity.F90`, SV), escritas en el driver:
- u\*_f = √C_d √(u² + v_at_u² + U_bg²) (SV:687-689, 711). En las caras v, √(v² + u_at_v² + U_bg²)
  vale lo mismo, así que las cuatro caras llevan el mismo u\*_f;
- kv_bbl = (√C_d u\*_f) bbl_thick (SV:1115) y max(KV_BBL_MIN, kv_bbl) (SV:1142), con
  `CORRECT_BBL_BOUNDS = False` (valor por defecto de MOM6 y de DOME).

`set_viscous_BBL` no se corre: necesita `set_visc_init`, la EOS y la ecuación de
Killworth–Edwards del espesor. El espesor no cambia el factor, porque U es uniforme.

| Caso | C_d | U (m/s) | U_bg (m/s) | bbl_thick (m) | ustar_BBL / (√C_d \|U_b\|) | BBL_meanKE_loss / (C_d \|U_b\| U²) |
|---|---|---|---|---|---|---|
| 1 | 3e-3 | 0.2 | 0 | 10 | 1.414213562373095 | 1.9999999999999993 |
| 2 | 2e-3 | 0.5 | 0 | 25 | 1.414213562373095 | 2.0 |
| 3 | 3e-3 | 0.05 | 0.1 | 7 | 1.4142135623730951 | 2.0 |
| 4 | 2e-3 | 0.002 | 0 | 10, KV_BBL_MIN 1e-4 | 3.5355339059327378 = √2 × 2.5 | 5.0 = 2 × 2.5 |

con |U_b| = √(U² + U_bg²).

**Confirmado:** `ustar_BBL` = √2 √C_d|U_b| y `BBL_meanKE_loss` = 2 C_d|U_b| U², es decir,
2 C_d|U|³ sin U_bg. Las causas:
- SD:2132 toma ½ Σ de u\*_f² sobre las **cuatro** caras, y las cuatro llevan u\*_f;
- SD:2134-2137 suma las dos caras u **sin** ½ (las caras v no tienen velocidad normal).

Es clase (d), un rasgo de la referencia. pyOCEAN usa la energía física C_d|U|³ y
u\* = √(|τ_b|/ρ₀) (decisión de la especificación). Por eso, γ = BBL_EFFIC = 0.2 en pyOCEAN
equivale a 0.1 en MOM6/OM4, y la u\* de MOM6 es √2 mayor. Hay que decirlo al comparar.

**KV_BBL_MIN** (la duda de la lectura, confirmada). Con `CORRECT_BBL_BOUNDS = False`, SV:1142
sube kv_bbl a KV_BBL_MIN sin cambiar bbl_thick. Entonces `set_BBL_TKE` recupera
u\*_f = KV_BBL_MIN/(√C_d bbl_thick), mayor que √C_d|U_b|. Pasa cuando
C_d|U_b| h_bbl < KV_BBL_MIN, por ejemplo con |U| < 5 mm/s para C_d = 2e-3, h_bbl = 10 m y
KV_BBL_MIN = 1e-4 (DOME). Los dos factores crecen en KV_BBL_MIN/(C_d|U_b| h_bbl): 2.5 en el
caso 4. Con `CORRECT_BBL_BOUNDS = True`, SV:1103-1110 recalcula bbl_thick y el u\* queda
consistente. OM4 usa KV_BBL_MIN = 0. Solo afecta a flujos muy débiles, donde la energía es
pequeña de todos modos.
