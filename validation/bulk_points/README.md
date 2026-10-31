# Flujos bulk: pyOCEAN contra FMS (MOM6), MITgcm `pkg/exf` y CROCO, punto a punto

Criterio 1 de N2b (parte NCAR, spec `docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md`):
con la misma SST, la misma corriente y la misma atmósfera, `forcing/bulk.py::bulk_ncar_fms` da los flujos
de la rutina NCAR del acoplador de FMS que usa MOM6 (OM4). Lectura de la fuente: `docs/design/n2b_refs/mom6_fms_bulk.md` §2.

## Archivos

| Archivo | Qué hace |
|---|---|
| `points.py` | `POINTS`: 250 estados (viento 0.1–30 m/s, SST −1.8–32 °C, T_a − SST −10…+5 K, HR 50–95 %, corriente 0.3–1.5 m/s en la mitad); `LY04_VS_FMS_MAX_REL` |
| `fms_ncar_driver.F90` | llama a `surface_flux` de FMScoupler (sin cambios) sobre los puntos, todo agua |
| `mitgcm_exf_driver.F` | llama a `EXF_BULKFORMULAE` de MITgcm (sin cambios), rellenando los bloques comunes |
| `croco_bulk_driver.F90` | llama a `bulk_point`, el bloque escalar de `bulk_flux.F` de CROCO extraído por `build_reference.sh` |
| `build_reference.sh` | compila los drivers, los corre y escribe `tests/data/bulk_fms_ncar_reference.npz` (47 kB) y `tests/data/bulk_croco_coare_reference.npz` (33 kB; `build_reference.sh coare` hace solo esta, sin FMS) |
| `compare_mitgcm.py` | compara `bulk_ncar` con la salida de MITgcm |
| `fetch_era5.py` | descarga ERA5 horario con `cdsapi` (u10, v10, t2m, d2m, msl, ssrd, strd, tp y los flujos medios `avg_sdswrf`/`avg_sdlwrf`), un archivo por mes y grupo (`instant`, `accum`, `mean`), a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/forcing/era5/`; por defecto julio–agosto de 2017, caja 38–43° N, 75–68° W (criterio 5 de N2b). Lo usan `forcing/atmosphere.py` y `tests/test_atmosphere.py` (tarea 3) |

## Fuentes y compilación

- **FMScoupler** `~/pyOCEAN_refs/FMScoupler`, NOAA-GFDL, **tag 2025.02, git 5b700a4fb585b4c8f47c479d9431178a1707c33a**
  (2025-04-28): es el release que acompaña a FMS 2025.02 (añade las llamadas a `generic_exchange`, que entró en FMS 2025.02); el
  FMS local es 2025.02.01 (parche). La rutina está en `shared/surface_flux.F90`, no en `full/`.
- **ice_param** `~/pyOCEAN_refs/ice_param`, git 1d19a57 (2024-06-12; sin tags 2025): solo `ocean_rough.F90`, porque
  `surface_flux.F90:120` lo usa; no se llama con `do_iter_monin_obukhov = .false.`. También se leyó `ocean_albedo.F90` (opción 2).
- **MITgcm** `~/pyOCEAN_refs/MITgcm`, git 27256b2: `pkg/exf/exf_bulkformulae.F`.
- FMS: `libFMS.a` y módulos que construye `validation/mom6_column/build_reference.sh` (correrlo antes).

```bash
bash validation/bulk_points/build_reference.sh
python -m validation.bulk_points.compare_mitgcm
pytest tests/test_bulk_fms.py -q
```

Decisiones del driver (ninguna cambia el algoritmo):

| Decisión | Motivo |
|---|---|
| `-fdefault-real-8 -fdefault-double-8` | como GFDL y como `libFMS.a`: los literales de `surface_flux.F90` (`0.5`, `2.7/u10`…) son `real` por defecto |
| `-ffp-contract=off` | gfortran fusiona a·b + c en arm64 con `-O2`; la referencia es el algoritmo, que NumPy reproduce sin FMA (como `validation/mom6_column/`) |
| namelist de OM4 (`OM4_025/input.nml:108-116`): `ncar_ocean_flux`, `raoult_sat_vap`, tablas de `sat_vapor_pres` | la configuración de MOM6 acoplado |
| `p_atm = p_surf = p` | OM4 `data_table`: `p_bot` y `p_surf` son `psl` ⇒ θ_a = T_a |
| `gust = 0` | `bulk_ncar_fms` no tiene argumento de ráfagas; OM4 usa 1e-4 m/s, efecto ≤ 5e-7 relativo a 0.1 m/s |
| rugosidad 5.8e-5 m | solo alimenta `mo_drag`, que `ncar_ocean_fluxes` sobrescribe sobre agua (`surface_flux.F90:386-405`) |
| tabla `z2` con `ncar_ocean_flux_multilevel`, `bulk_zt = bulk_zq = 2` | la ruta de ERA5 (T y q a 2 m), `:905-1044` |

## Resultado: `bulk_ncar_fms` = FMS compilado

Sobre los 250 puntos, en las dos tablas (`z10`: todo a 10 m con `ncar_ocean_fluxes`; `z2`: T y q a 2 m con
`ncar_ocean_fluxes_multilevel`), τ_x, τ_y, flujo sensible, flujo de vapor, C_d, C_h, C_e y u\* de `bulk_ncar_fms` son
**iguales bit a bit** a los del Fortran (diferencia máxima 0). El test usa 1e-10 relativo, como pide el spec, para no depender
de la libm. Esto incluye la tabla de presión de vapor de FMS (`esat_fms`: Goff–Gratch, mezcla con hielo bajo 0.01 °C,
interpolación cuadrática cada 0.1 K), reconstruida en Python.

## LY04 (`bulk_ncar`, forma MITgcm) contra FMS

`bulk_ncar` y `bulk_ncar_fms` en `POINTS`, todo a 10 m, cada uno con su ρ_a y su θ. Medida: max|Δ| / max|FMS| por campo
(τ = |τ|). Diferencia total: τ 7.67e-4, Q_S 3.68e-2, Q_L y E 5.88e-2 (`points.LY04_VS_FMS_MAX_REL`, que el test exige).
Cada diferencia sola (un modelo escalonado que activa las nueve reproduce `bulk_ncar` desde FMS a 6e-16, así que la lista es completa):

| # | Diferencia | FMS | `bulk_ncar` | Efecto (max/max) | Clase |
|---|---|---|---|---|---|
| 1 | viento en los flujos | W = \|ΔU\| (`:423-425`); el piso de 0.5 m/s solo en los coeficientes (`:846`) | max(W, 0.5) también en los flujos | Q_S 3.8e-2, Q_L 4.8e-2, τ 2.5e-4 (factor 5 en W = 0.1 m/s) | c |
| 2 | q_sat | tabla Goff–Gratch de FMS, ε e/(p − (1−ε)e) × 0.98 (`:308`, `:320`, `:331`) | LY04 eq. 5, 640380/ρ e^{−5107.4/T} × 0.98 | q_s +0.04 … +2.2 % (32 °C); Q_L 5.8e-2 | c |
| 3 | θ del aire | T_a (p_atm = p_surf) | T + 0.0098 z_t | Q_S 1.0e-2 | c |
| 4 | c_p del aire | 1004.64 (R_d/κ) | 1000.5 | Q_S 4.1e-3 | c |
| 5 | piso de U₁₀N | no (`:878`) | 0.5 m/s | Q_L 3.6e-4 | c |
| 6 | pendiente de C_dN | 0.0764 (`:849`, como CESM) | 1/13.09 (LY04 impreso) | τ 6.8e-5 | c |
| 7 | g | 9.80 | 9.81 | ≤ 4.3e-5 | c |
| 8 | ρ_a | p/(R_d T(1 + 0.6078 q)) (`:343`, `:428`) | p/(R_d θ(1 + 0.608 q)) | ≤ 4.5e-6 | c |
| 9 | rama estable en ζ = 0 | ζ ≥ 0 | ζ > 0 | 0 en estos puntos | c |

Ninguna es un error de un código (a) ni física ausente (b); son elecciones entre formas publicadas. Punto a punto la diferencia 2 se
amplifica donde q_a − q_s es pequeño: en el punto de 30 m/s, SST 32 °C, ΔT +5 K (q_a 0.0257, q_s 0.0294) Q_L difiere un 29 % aunque
q_s solo difiera un 2.2 % y C_e sea igual a 3e-4.

## MITgcm `pkg/exf` (paso 7)

`EXF_BULKFORMULAE` compilado sin cambios con `ALLOW_BULK_LARGEYEAGER04` y `EXF_CALC_ATMRHO`, constantes por defecto de
`exf_readparms.F:330-363` salvo `hu = ht = hq = 10 m` y `rhoConstFresh = 1000`, `wspeed` y `sh` como `exf_wind.F:129-149, 267`
con viento relativo. La rutina lee y escribe en bloques comunes (`EXF_FIELDS.h`, `DYNVARS.h`, `EXF_PARAM.h`), así que el driver los
rellena a mano; con eso el driver mínimo fue posible (un punto cada dos filas para que la corriente en malla C de cada punto sea suya).
No hay test obligatorio: la tabla depende de una configuración de CPP elegida aquí. `compare_mitgcm.py`, max|Δ|/max|MITgcm|:

| | τ | Q_S | Q_L, E |
|---|---|---|---|
| `bulk_ncar` tal cual | 4.5e-4 | 3.3e-2 | 4.3e-2 |
| idem, solo puntos con W ≥ 0.5 m/s | 4.5e-4 | 4.7e-3 | 5.8e-4 |
| `bulk_ncar` con c_p 1005 y Γ 0.010 de MITgcm, W ≥ 0.5 | 4.6e-4 | 4.2e-4 | 5.9e-4 |

Clasificación (todo c):
- **Viento débil**: MITgcm, como FMS, usa W sin piso en los flujos (`tau = atmrho_loc·rd·wspeed`, `exf_bulkformulae.F:456`) y el piso solo
  en los coeficientes (`sh`, `:309`, `:432`); `bulk_ncar` lo aplica también a los flujos. El docstring de `bulk_ncar` dice que es "el
  algoritmo de MITgcm"; en esto no lo es (sigue la forma de LY04/CESM [verificar CESM `shr_flux_atmOcn`]).
- **Constantes**: c_p 1005 (MITgcm) / 1000.5 (pyOCEAN), Γ 0.010 / 0.0098 K/m: Q_S 4.7e-3 → 4.2e-4 al igualarlas.
- **Resto ≤ 6e-4**, por lectura del código (no separadas una a una): ρ_a de T_a en MITgcm y de θ en `bulk_ncar` (0.1 K de 280 K ≈ 3.5e-4),
  factor de humedad 0.606 / 0.608, pendiente de C_dN 0.0764 / 1/13.09 y U₁₀N con W sin piso en el numerador (`:423`).

Hallazgo del arnés: el `EXF_OPTIONS.h` por defecto de MITgcm tiene `#undef ALLOW_BULK_LARGEYEAGER04` (rama Large & Pond 1981) y sin
`EXF_CALC_ATMRHO` usa ρ_a = 1.2 constante; la primera compilación tomó ese archivo por la regla de búsqueda de `#include "…"` (el
directorio del fuente va primero) y daba diferencias de 10 %, casi todas por ρ_a. Toda comparación con MITgcm `exf` debe declarar
estas dos opciones.

## Qué debe mejorar pyOCEAN

1. Elegir por defecto la forma de FMS (`'ncar_fms'`) para comparar con MOM6 (spec, criterio 1); `bulk_ncar` queda como opción LY04/MITgcm.
2. Corregir el docstring de `bulk_ncar`: el piso de viento en los flujos no es el de MITgcm ni el de FMS (la conducta no se cambia en esta tarea).
3. Las ráfagas de OM4 (`gust` 1e-4 m/s) no están en `bulk_ncar_fms`; añadir el argumento si se quiere el bit a bit con un `data_table` de OM4.
4. La tabla de FMS mezcla hielo bajo 0.01 °C (−0.16 % en e_s a −1.8 °C): elección de FMS (d leve) que `bulk_ncar_fms` copia a propósito.

## COARE 3.0: `forcing/bulk_coare.py::bulk_coare_croco` = CROCO compilado (Task 2)

Criterio 1 de N2b, parte COARE. Lectura de la fuente: `docs/design/n2b_refs/croco_bulk.md` (§2 algoritmo y peculiaridades, §5
pseudocódigo). Fuente: `~/pyOCEAN_refs/croco`, **git ab63b58**, `OCEAN/bulk_flux.F`, opciones por defecto
(`cppdefs.h:178-185`: `BULK_FLUX`, `BULK_GUSTINESS`, `BULK_LW`), más `SALINITY` para la evaporación.

Arnés (`build_reference.sh`, primera sección; salida en `~/pyOCEAN_refs/build/croco_bulk/`, fuera del repo):

| Decisión | Motivo |
|---|---|
| `bulk_point` = declaraciones `:44-116` + bloque escalar `:199-659` (con su guarda `# ifndef ABL1D` de `:198` y un `# endif` que cierra la rama `ABL1D` de `:593`, como `:669`) + funciones `:758-974`, copiados con `sed` | `bulk_flux_tile` depende de bloques comunes y teselas; el bloque escalar no (`croco_bulk.md` §6) |
| sustituciones `sed` de arrays `(i,j)` por escalares (`wspd`, `tair`, `t(…,itemp)`, `radlw`, `aer`, `cer`, `stflx`, …) | sin cambio de aritmética |
| **un cambio**: `:207` `Q = spec_hum(RH,psurf,TairC)` → `Q = RH`, con RH = q_air | pyOCEAN recibe q; `spec_hum` interpreta RH ≥ 2 como g/kg y RH < 2 como fracción, así que no puede llevar q < 2 g/kg (hay puntos con q = 0.75 g/kg) |
| g = 9.81, C_p = 3985, κ = 0.41 (`scalars.h:633,640,643`), ρ₀ = 1025 (`croco.in:61`); `radlw = LW↓/(ρ₀ C_p)` (`get_bulk.F:384`); `wspd = √(u² + v²)` (`get_bulk.F:888`) | como el modelo |
| τ en el punto: `:706-712` con los dos vecinos iguales al punto (½(c + c) = c exacto) | el promedio a caras u/v es aritmético y lo hace el modelo, no el algoritmo |
| `-O2 -fdefault-real-8 -fdefault-double-8 -std=legacy` (`jobcomp:249`) + `-ffp-contract=off` | doble precisión como CROCO; sin fusión a·b + c en arm64, como la tabla FMS |

Entradas: `points.COARE_POINTS`, los 250 estados de `POINTS` en las unidades de CROCO (T en °C = `POINTS` − 273.15, viento
absoluto sin la corriente, q tal cual, p = 1e5 Pa constante de CROCO, no `POINTS["p"]`) más una onda larga descendente que
`POINTS` no tenía: (0.70 … 0.95) σ T_a⁴, determinista, ciclando con el índice. `POINTS` y la tabla FMS no cambian.

**Resultado.** Los intermedios (u\*, θ\*, q\*, ΔU final con ráfagas, q_s, Charnock, p a 2 m, ρ_a) y C_d y E son **iguales bit a bit**
al Fortran. τ, Q_S, Q_L difieren ≤ 2.4e-16 relativo y Q_LW ≤ 8.9e-15, solo porque la tabla guarda los flujos cinemáticos de CROCO
(K m/s, m²/s²) y el test los pasa a W/m² y N/m² con ρ₀ C_p (Q_LW: cancelación entre LW↓/(ρ₀C_p) y 0.985σT⁴/(ρ₀C_p)). El test
exige 1e-10 (`tests/test_bulk_coare.py`). Las peculiaridades de CROCO que el port copia a propósito (docstring de
`forcing/bulk_coare.py`): CtoK = 273.16; q_sat del mar con 1e5 Pa y factor 0.98; Charnock fijado con el viento inicial; ψ_t(0) =
−0.004524; u\* de la última iteración sin recalcular con el último ΔU; corrección de Webb; 0.985 σ SST⁴ sin reflexión;
E = −C_p hflat/L_v (volumen a ρ₀, no a 1000 kg/m³); ζ sin recorte.

Adaptador `forcing.bulk._SCHEMES["coare"]` (`bulk_coare`): T del aire en K → °C con 273.15, viento relativo a la corriente
(u_ocean = 0 da el viento absoluto de CROCO), `msl` no se usa (psurf = 1e5 de CROCO; `p_sfc=` lo cambia), devuelve solo la parte
turbulenta con C_h = C_e.

Qué debe mejorar pyOCEAN (COARE):
1. Si se quiere `READ_PATM` (presión de ERA5) habrá que pasarla como `p_sfc` por punto; hoy `p_sfc` es un escalar, como CROCO.
2. CROCO no tiene viento relativo, piel fría ni capa cálida; pyOCEAN con `'coare'` las hereda. Son elecciones de CROCO (c);
   el benchmark de la Task 7 las separa con `relative_wind=False`.
