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
