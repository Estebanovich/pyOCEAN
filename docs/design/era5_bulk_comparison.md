# Fórmulas bulk aire–mar: MITgcm vs CROCO vs HYCOM vs MOM6

Documento de diseño para `forcing/era5.py` (Fase 2, paso 2).
Fecha: 2026-10-10.

Fuentes leídas (solo lectura, para reimplementar algoritmos con su cita; no se copia código):

- MITgcm `pkg/exf` (licencia tipo MIT) — `~/pyOCEAN_refs/MITgcm`
- CROCO v2.1.3 `OCEAN/bulk_flux.F` (CeCILL-C) — `~/pyOCEAN_refs/croco`
- HYCOM `thermf.F90`, `mod_momtum.F90` — `HYCOM-src/`
- MOM6 `config_src/drivers` (LGPL-3) — `~/pyOCEAN_refs/MOM6`

---

## 1. Resumen comparativo

| Aspecto | MITgcm (exf) | CROCO (bulk_flux) | HYCOM (flxflg=6) | MOM6 |
|---|---|---|---|---|
| Esquema | Large & Pond 81/82 con coef. de Large & Yeager 2004 | COARE 3.0 (Fairall et al. 2003) | Ajuste polinomial a COARE 3.0 (Kara et al. 2005), sin iteración | **No calcula bulk**: recibe flujos del acoplador (FMS/CMEPS) o de archivos |
| Iteraciones Monin–Obukhov | 2 fijas, sin test de convergencia | 3 fijas, sin test | 0 (fórmula cerrada); 3 en la opción CORE (flxflg=5) | — |
| ψ_m, ψ_h | Businger–Dyer (16ζ), estable −5ζ | Kansas+convectivo (Grachev 2000), estable Beljaars–Holtslag 1991 | Implícito en el ajuste | — |
| Rugosidad | Implícita en C_dN(U) | Charnock (0.011→0.018) + flujo liso; z0T de Liu et al. 1979 | Implícita | — |
| Rachas (gustiness) | No; viento mínimo 0.5 m/s | Sí: β=1.2, z_i=600 m | No; viento en [1.2, 34] m/s | Gust constante en u* |
| Alturas 2 m / 10 m | Coeficientes desplazados a h_t=2, h_u=10 | Nativo: ζ_T = ζ_U·zT/zW | **Ninguna**: T y q a 2 m tratados como de 10 m | — |
| q_sat | 0.98·640380·exp(−5107.4/T)/ρ_a (sin presión) | Fórmula SURFEX con p; ×0.98 | Polinomio de Lowe con p; factor por SSS | — |
| ρ_a | Constante 1.2 (opción p/(R_d T_v)) | p(1+q)/(R_d T(1+0.622q)) — **error, ver §4** | p/(R_d T(1+0.608r)) | — |
| L_v | 2.5e6 constante | 2.5008e6 − 2371.9·SST | 2.47e6 constante | — |
| Onda larga | εσT_s⁴ − ε·LW↓, ε=0.97 (consistente) | LW↓ − 0.985σT_s⁴ (no refleja) | LW↓ − σT_s⁴ (ε=1) | — |
| Onda corta | (1−α)SW↓, α=0.1 | Lee SW **neto** (albedo en preproceso) | α=0.09 o Large–Yeager(φ) | — |
| Viento relativo a corriente | Opción (`useRelativeWind`) | Opción CFB, Renault et al. 2016/2020 | Sí por defecto en wndflg=4 | — |
| Límite de esfuerzo | ±2 N/m² | No | Cd acotado en 34 m/s (Sraj 2013) | — |
| Malla | Todo en puntos T; τ promediado a caras u/v | Cd y ρ_a·U en ρ, promediados a u/v | Viento leído en malla p | — |
| Signos | Q_net positivo **hacia arriba** | Positivo hacia el océano | Positivo hacia el océano | — |
| T0 (K) | 273.15 | 273.16 (y 273.15 en ONLINE) | 273.16 | — |

---

## 2. Lecciones de diseño

1. **Separar el cálculo de flujos de su aplicación (lección de MOM6).**
   pyOCEAN ya tiene `surface_tendency()` en `forcing/surface.py`, que recibe flujos.
   El cálculo bulk debe ser una función pura sobre arreglos: recibe atmósfera + SST + corriente y devuelve (τx, τy, Q_net, Q_sw, E−P).
   Esto permite cambiar de esquema, o en el futuro acoplar WW3 o un atmosférico, sin tocar el núcleo.

2. **Ningún modelo verifica la convergencia** de la iteración Monin–Obukhov.
   pyOCEAN debe iterar con tolerancia y máximo de iteraciones (parámetro `n_iter`, por defecto 2 para reproducir MITgcm), y tener un test que compare contra 10 iteraciones.

3. **Alturas nativas 2 m / 10 m** (enfoque de MITgcm y CROCO), no el de HYCOM.
   ERA5 da T y Td a 2 m; tratarlos como de 10 m (HYCOM) introduce sesgo en condiciones estables.

4. **Usar la presión real (msl)** en q, q_sat y ρ_a. Los tres modelos la ignoran o la fijan por defecto.

5. **Onda larga con emisividad consistente:** Q_LW = ε(LW↓ − σT_s⁴).
   Es la forma de MITgcm; la de CROCO y HYCOM no refleja la parte 1−ε de LW↓.

6. **No usar la evaporación de ERA5** (`e`); E se calcula del flujo latente para que calor y agua sean consistentes. Coinciden los tres modelos.

7. **Viento relativo a la corriente** como opción. Apagado para comparar con MITgcm; encendido para la operación en el Golfo (como en HYCOM-IOA).
   Renault et al. (2016) muestran que amortigua la energía de mesoescala.

8. **Huracanes en el Golfo:** la C_dN de Large–Yeager 2004 crece sin límite. Incluir el tope de LY09 (C_d = 2.34e-3 para U > 33 m/s) como opción.

---

## 3. Propuesta para pyOCEAN

### Archivos
- `forcing/bulk.py` — algoritmos (funciones puras, NumPy, luego Numba):
  - `qsat`, `specific_humidity_from_dewpoint`, `air_density`, `psi_m`, `psi_h`
  - `bulk_ncar` (Large & Yeager 2004/2009, reproduce MITgcm)
  - `bulk_coare30` (Fairall et al. 2003, iteración con convergencia, rachas)
  - más adelante, `bulk_kara` (Kara et al. 2005, reproduce HYCOM; útil en Fase 3)
- `forcing/era5.py` — lectura de NetCDF ERA5, conversión de unidades, interpolación temporal y espacial a la malla del modelo, llamada al esquema bulk, y entrega de flujos a `surface_tendency`.
- `tests/test_bulk.py`, `tests/test_era5.py`.

### Orden de implementación (respetando la validación contra MITgcm)
1. `bulk_ncar` primero: es el esquema de MITgcm, así que la validación de Fase 2 es directa.
2. `bulk_coare30` después: es el más completo físicamente; será el default operacional.
3. `bulk_kara` solo cuando lleguemos a la comparación con HYCOM.

### Conversiones ERA5

| ERA5 | A pyOCEAN | Nota |
|---|---|---|
| u10, v10 [m/s] | U a 10 m | Viento real, no u10n |
| t2m [K] | T_a a 2 m | Temperatura potencial: + Γ·z con Γ = 0.0098 K/m |
| d2m [K] + msl [Pa] | q [kg/kg] | e = e_s(Td) (Tetens del IFS); q = εe/(p − (1−ε)e), ε = 0.621981 |
| msl [Pa] | p | Para q, q_sat, ρ_a |
| ssrd [J/m²] | SW↓ [W/m²] | ÷3600 (horario); acumulación de la hora anterior: asignar a t − 30 min; recortar negativos |
| strd [J/m²] | LW↓ [W/m²] | Igual que ssrd |
| tp [m] | P [m/s] | ÷3600 |
| — | E | Calculada del flujo latente |

### Constantes (una sola fuente: `core/constants.py` o encabezado de `bulk.py`)
T0 = 273.15 K; R_d = 287.04; R_v = 461.5; c_pa = 1004.7; κ = 0.4; σ = 5.670374e-8; ε_LW = 0.97; g = 9.81.

### Tests propuestos
- Límite neutro: con ΔT = Δq = 0, los coeficientes deben igualar los C_dN, C_hN y C_eN analíticos.
- Convergencia: el resultado con 2 iteraciones difiere menos de 1 % del de 10 iteraciones para U > 3 m/s.
- Valores de referencia publicados (casos de prueba de COARE 3.0 de Fairall).
- Signos: aire frío y seco sobre mar cálido da Q_net < 0 y E > 0.
- Unidades ERA5: una acumulación conocida de J/m² da el W/m² esperado.
- Equivalencia con MITgcm: reproducir punto por punto la salida de `exf_bulkformulae` para un caso sintético.

---

## 4. Detalles a NO copiar (encontrados en el código)

- **CROCO** (`bulk_flux.F:233`): ρ_a = p(1+q)/(R_d T(1+0.622q)). Lo correcto es 1 + q/0.622 en el denominador.
  Sobreestima ρ_a, y con ello todos los flujos turbulentos y el esfuerzo, en ~2 % cuando q = 0.02.
- **CROCO**: el ciclo ONLINE de ERA convierte q→RH con 1020 hPa y luego RH→q con 1000 hPa (~2 % de error en q).
- **CROCO**: las llaves cpp del caso COASTAL (`ECUMEv0`, `GUSTINESS`) no existen, así que se ignoran en silencio.
- **MITgcm**: atemp = 0 se usa como máscara de tierra. Mezcla velocidad recortada y sin recortar (el propio comentario dice "strange").
  Los comentarios dicen °C para T_s, pero está en K.
- **HYCOM**: la humedad es razón de mezcla en flxflg 1/2/4/6 y humedad específica en flxflg 5.
  El factor de salinidad usa (S − 0.03). T0 = 273.16.
- **Los tres**: σ_SB y otras constantes con valores distintos entre archivos del mismo modelo.

---

## 5. Referencias

- Beljaars, A.C.M. & Holtslag, A.A.M. (1991). *J. Appl. Meteor.* 30, 327–341.
- Fairall, C.W. et al. (1996). *J. Geophys. Res.* 101(C2), 3747–3764.
- Fairall, C.W., Bradley, E.F., Hare, J.E., Grachev, A.A. & Edson, J.B. (2003). *J. Climate* 16, 571–591.
- Grachev, A.A., Fairall, C.W. & Bradley, E.F. (2000). *Bound.-Layer Meteor.* 94, 495–515.
- Kara, A.B., Hurlburt, H.E. & Wallcraft, A.J. (2005). *J. Atmos. Oceanic Technol.* 22, 1080–1094.
- Large, W.G. & Pond, S. (1981). *J. Phys. Oceanogr.* 11, 324–336.
- Large, W.G. & Pond, S. (1982). *J. Phys. Oceanogr.* 12, 464–482.
- Large, W.G. & Yeager, S.G. (2004). NCAR/TN-460+STR.
- Large, W.G. & Yeager, S.G. (2009). *Clim. Dyn.* 33, 341–364.
- Liu, W.T., Katsaros, K.B. & Businger, J.A. (1979). *J. Atmos. Sci.* 36, 1722–1735.
- Renault, L. et al. (2016). *J. Phys. Oceanogr.* 46, 1685–1704.
- Renault, L. et al. (2020). *J. Adv. Model. Earth Syst.* 12, e2019MS001715.
- Sraj, I. et al. (2013). *Mon. Wea. Rev.* 141.
- Webb, E.K., Pearman, G.I. & Leuning, R. (1980). *Q. J. R. Meteorol. Soc.* 106, 85–100.

---

## 6. Configuración real de HYCOM-IOA (`~/HYCOM/blkdat.input`)

Encabezado: *"PAC_MEX corrida libre Abril 2015 - restart GOFS3.1 + forzamiento WRF; Sigma-2; WOA23 relax; KPP; nested GOFS3.1 ab nativo"*, malla 1600×1125, 36 capas (16 sigma).

| Flag | Valor | Significado → implicación para pyOCEAN |
|---|---|---|
| `flxflg` | 6 | Kara et al. 2005 + MSLP + q_sat con SSS + viento relativo → `bulk_kara` es necesario para reproducir HYCOM-IOA |
| `wndflg` | 4 | Se lee U10/V10; τ = ρ_a C_d(COARE fit, con p) \|U−u_o\|(U−u_o) |
| `ocnscl` | 1.0 | Viento relativo completo (corriente de los 3 m superiores, `thkcdw` = 3) |
| `ustflg` | 4 | u* a partir de τ |
| `empflg` | 6 | E calculada con el mismo esquema que el calor latente |
| `empbal` | 1 | Balance de E−P en el dominio por offset (objetivo 0) |
| `lwflag` | −1 | Se lee LW↓; Q_LW = LW↓ − σT_s⁴ (ε = 1) |
| `albflg` | 2 | Albedo de Large & Yeager: 0.069 − 0.011·cos(2φ) |
| `dswflg` | 0 | Sin corrección diurna (WRF ya resuelve el ciclo diurno) |
| `mslprf` | 1 | Presión al nivel del mar como forzamiento |
| `jerlv0` | −1 | Penetración de onda corta basada en clorofila (Lee et al. 2005) |
| `sssflg` | 1 | Relajación de SSS a climatología (WOA23), `sefold` = 30 d, `thkmls` = 15 m |
| `sstflg` | 0 | Sin relajación de SST |
| `priver` | 1 | Ríos como precipitación ficticia ("bogus") |
| `lbflag` | 2 | Fronteras barotrópicas por anidamiento (no Flather) |
| `relax` | 1 | Nudging lateral (GOFS 3.1 nativo `.a/.b`) |
| `tidflg` | 0 | **Sin mareas** (los parámetros de marea están preparados, `tidcon` = 8 constituyentes, pero apagados) |
| `cb` | −1 | Fricción de fondo cuadrática espacialmente variable (`cb.a/.b`) |
| `mlflag` | 1 | KPP (Large et al. 1994), `difmiw` = 0.3e-4, `difsiw` = 0.1e-4 |

**Conclusión:** para comparar pyOCEAN con HYCOM-IOA punto a punto hace falta `bulk_kara` (flxflg=6) con viento relativo. El orden propuesto sigue siendo NCAR primero (validación contra MITgcm, Fase 2), pero `bulk_kara` sube de prioridad: conviene implementarlo antes de la comparación con HYCOM y no dejarlo hasta la Fase 3.
