# Benchmark DOME: overflow 3-D con rotación (Legg et al. 2006, Caso 1)

- **Spec:** `docs/superpowers/specs/2026-10-14-dome-overflow-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-14-dome-overflow.md`

**Nota de vigencia (2026-11-03).** Dos correcciones de clase (a) del núcleo cambiaron las corridas de DOME después
de la mayoría de las tablas de abajo: la del flujo de PV del momento vector-invariante (`e1ffbbb`, 2026-11-02: el
transporte de PV se arma con uⁿ y hⁿ en cada paso; arregla la inestabilidad de θ_bc = ½, punto 5 y punto 1 de la
lista) y la de la viscosidad horizontal en las caras abiertas (`ca36b59`, 2026-11-03: la cara de entrada `'clamped'`
ya no es una pared en la tensión de Smagorinsky ni en la laplaciana de A_h). Sólo se repitieron las corridas de la
sección del θ_bc (abajo, punto 5) y la principal de la BBL (sección de la BBL). **Las demás tablas (GLS,
Smagorinsky, Jackson, controles de la BBL, z\* y σ) son de antes de `e1ffbbb`** y quedan como registro; repetirlas es
trabajo pendiente antes de cualquier lectura nueva.

## Qué es

El primer caso 3-D con rotación, talud y una entrada de agua densa. Responde dos preguntas:
1. ¿pyOCEAN baja y mezcla la pluma como los modelos de referencia, en z\*, σ e híbrida?
2. ¿Aparecen en un overflow realista los pendientes de capas delgadas de la híbrida (README de Ilicak, 4b y 4d)?

El caso es el Caso 1 de Legg, Hallberg & Girton (2006) en la geometría de MOM6 (`DOME_initialization.F90`). Se define una sola vez en `cases.py`; cada modelo tiene su escritor y su lector, y todos se miden con las mismas funciones (`metrics.py`).

| | |
|---|---|
| Dominio | 1500 × 700 km; plano f, f = 1e-4 s⁻¹ |
| Batimetría | talud de 1 % entre y = 300 y 600 km (3600 → 600 m); canal de entrada de 100 km (1000 < x < 1100 km) y 100 km de largo |
| Mallas | 10 km (150 × 70) y 5 km (300 × 140); 25 capas o niveles |
| Estratificación | N² lineal, Δρ = 2 kg/m³ (1030 → 1032); EOS lineal ρ = 1031 − 0.2 T, g = 9.8 |
| Entrada | Legg ec. 3–8: h = h₀ e^(−x_w/L_ρ), transición F con Ri_m = 1/3; 8.38 Sv a 10 km y 8.57 a 5 km (Legg: Δb₀h₀²/2f = 8.55); tinte s = 1 en el agua más densa que el ambiente |
| Esponjas | las de MOM6 DOME: 10 día⁻¹ a menos de 100 km de las paredes oeste y este, rampa lineal hasta 0 a 200 km; T hacia el perfil inicial y s hacia 0 |
| Duración | 50 días; métricas promediadas en los días 30–50 |

**Física común** (el MITgcm de Legg a 10 km):
- ν_h laplaciana 50 m²/s a 10 km y 25 a 5 km (∝ Δx);
- ν_v = 0.5 m²/s;
- sin difusión explícita de trazadores y sin mezcla por Ri (sin Jackson);
- fricción cuadrática C_d = 2e-3 sobre los `hbbl` = 10 m de fondo en pyOCEAN y MOM6; MITgcm la aplica en la celda de fondo.

**Modelos:**

| Modelo | Coordenada | Detalles |
|---|---|---|
| pyOCEAN | z\* (25 × 144 m), σ (25), híbrida (objetivos = clases de MOM6, piso z\* hasta el fondo) | defaults del plan de robustez (vector-invariante + KE Arakawa, WENO5-Z + FCT, PPM_H4, continuidad PPM, PGF de volumen finito, θ_bc = ½); entrada `clamped` con su transporte exacto (`OBData.transport`), trazador pasivo, esponja interior y fricción cuadrática nuevos de este plan; 5 hilos |
| MOM6 (`repro -O2`, 1 proceso) | ρ en modo capa (`USE_REGRIDDING = False`), 25 clases R_k = 1030 + (k − 1)·2/24 | el `MOM_input` de `ocean_only/DOME` con la física común (`LAPLACIAN`, `KH`, `KV = 0.5`, `KD = 0`, sin Jackson ni Smagorinsky, `CDRAG = 0.002`, `HBBL = 10`); entrada `SIMPLE` |
| MITgcm (1 proceso) | z con celdas parciales, z\* (`select_rStar = 2`, `nonlinFreeSurf = 4`) | `obcs` (entrada prescrita), `ptracers` (tinte), `rbcs` (esponjas); T y s con limitador de flujo (`77`, Pietrzak 1998, como Legg); variante "legg" (fondo no-slip, ec. 10 de Legg) y "common" (free-slip + fricción) |

**Diferencias de configuración, todas (c) salvo que se diga:**
- **Entrada por clase.** MOM6 reparte el transporte por capa con su propio perfil (`DOME_set_OBC_data`): 85 % en la clase más densa. En z\* y en MITgcm las capas de 144 m promedian el perfil de Legg, así que la clase más densa recibe 21–44 % menos.
- **Tinte de MOM6 (d).** `DOME_set_OBC_data` llena `segment%field(1)%buffer_src`, pero `register_segment_tracer` sin `OBC_scalar` deja el valor del segmento en 0 (`MOM_open_boundary.F90:5390-5400`), así que el tinte nunca entra. Tampoco entra con `OBC_SEGMENT_001_DATA = "TR_D1=value:1.0"`. En su lugar, `mom6.read` usa un proxy: el grosor ganado en las dos clases más densas, s = (h − h⁰)/h donde la ganancia pasa de 1 m. Sirve en los primeros días (centroide a 2 días: 550 m, contra 510 m de pyOCEAN z\* y 500 m de MITgcm), pero no después (punto 3).
- **Transporte en la cara de MOM6.** v·h en la cara norte no es el transporte `SIMPLE` (3.2 Sv). El volumen sí lo es: η sube como Σ tr_k = 8.80 Sv.
- **El nivel del mar sube ~40 m en 50 días** en los tres modelos: la entrada no tiene salida. La hipsometría del RPE llega hasta +60 m.
- **Ancla de Legg.** Ver el punto 1: canal de 100 km (Legg usó 50 en sus corridas z), pared oeste cerrada con esponja (Legg: Orlanski abierto) y malla C (Legg: C–D a 10 km).

## Cómo reproducirlo

```bash
validation/dome/run_mitgcm.sh build 10; validation/dome/run_mitgcm.sh build 5
NUMBA_NUM_THREADS=5 python validation/dome/run_pyocean.py 10 zstar              # zstar | sigma | hybrid; Δt 600 s por defecto
bash validation/dome/run_mom6.sh 10 common
validation/dome/run_mitgcm.sh run 10 common                                     # legg | common
python validation/dome/compare.py 10                                            # tabla → RUNS/compare_10km.txt
python validation/dome/figures.py 10                                            # figuras → RUNS/figs/
```

- Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/<res>km/<modelo>_<coord>_<variante>/series.npz`, en el formato común (capas desde la superficie).
- **Figuras** (fuera del repo):
  - `RUNS/figs/dome_10km_bottom_dye.png` y `dome_5km_bottom_dye.png`: tinte de fondo a los días 10, 20, 30 y 50;
  - `RUNS/figs/dome_10km_path_transport.png` y `dome_5km_path_transport.png`: Y′(x′) con los puntos de Legg, y T(x′).
- Una corrida de 50 días tarda 4 min (MITgcm, 10 km), 10–12 min (MOM6 y pyOCEAN, 10 km) y 28–77 min a 5 km.

**Métricas** (x′ = x − 1050 km, el centro de la entrada; y′ = y − 600 km, el borde del talud; como Legg):
- **Trayectoria media** Y(x) = ∫ y s dy dz / ∫ s dy dz (Legg ec. 18).
- **Transporte de la pluma** T(x) = ∫_A (−u) dy dz con A = {s > 0.01} (ec. 11).
- **Entrainment** α_E = (dT/dx)/(L Ū) (ec. 12–13), medio entre x′ = 0 y −300 km.
- **Densidad y flotabilidad final** del agua con tinte al oeste de x = 800 km (Fig. 15), b_f = −g(ρ_f − 1030)/ρ₀.
- **Centroide** del tinte (profundidad media) a los días 10, 20, 30 y 50.
- **ΔRPE/RPE₀:** domina la masa que entra (2.5e-2 en todos los modelos), así que no mide la mezcla aquí.
- **Rangos** de T y s; truncaciones CFL y `n_limited`; max\|u\| en caras con un lado de menos de 1 cm contra caras gruesas (4b).
- **Entrada:** volumen (d/dt ∫η dA) y transporte por clase en la fila de entrada.
- **Costo:** s por día simulado, 1 proceso (pyOCEAN 5 hilos).

## Resultados

### 10 km (días 30–50)

| corrida | Y′(x′ = −200/−300/−500/−800 km) [km] | T(x′) [Sv] | α_E | ρ_f / b_f [1e-3 m s⁻²] | centroide 10/20/30/50 d [m] | \|u\| delgadas/gruesas [m/s] | trunc / lim | Δt [s] | s por día |
|---|---|---|---|---|---|---|---|---|---|
| MITgcm z legg | −52 / −72 / −107 / −204 | 20.4 / 22.1 / 25.9 / 21.0 | 1.94e-3 | 1031.142 / −10.85 | 883 / 1176 / 1337 / 1433 | — / 2.04 | — | 600 | 4.7 |
| MITgcm z common | −40 / −55 / −75 / −91 | 21.4 / 24.8 / 27.8 / 32.1 | 1.83e-3 | 1031.077 / −10.23 | 858 / 1088 / 1079 / 1064 | — / 2.24 | — | 600 | 4.6 |
| **pyOCEAN z\*** | −49 / −72 / −99 / −126 | 22.6 / 26.4 / 30.4 / 33.3 | 2.69e-3 | 1031.060 / −10.07 | 835 / 1088 / 1151 / 1174 | — / 2.04 | 0 / 0 | 600 | 13.8 |
| **pyOCEAN σ** | −89 / −117 / −150 / −202 | 23.8 / 27.6 / 33.5 / 40.3 | 2.14e-3 | 1031.391 / −13.22 | 1221 / 1615 / 1798 / 1717 | — / 2.63 | 0 / 0 | 600 | 13.8 |
| **pyOCEAN híbrida** | −115 / −150 / −192 / −224 | 11.2 / 15.4 / 19.7 / 19.4 | 1.37e-3 | 1031.702 / −16.18 | 1309 / 1823 / 2080 / 2179 | 3.10 / 3.21 | 0 / 0 | 600 | 14.9 |
| MOM6 ρ (proxy) | −416 / −417 / −437 / −446 | 0.0 / 0.0 / −0.2 / −0.4 | — | 1031.917 / −18.22 | 1438 / 2099 / 2473 / 2808 | 0.46 / 2.03 | — | 1200 | 12.0 |

Las filas de pyOCEAN a 10 km son posteriores a la corrección de la fricción de la revisión final (2026-10-15: el esfuerzo se reparte sobre los `hbbl` de fondo; antes solo actuaba en la capa 0, a veces evanescente). σ no cambia. En z\* la trayectoria se movió ≤ 13 km, el transporte ≤ 4 % y b_f 1.5 % (−9.92 → −10.07); en la híbrida, ≤ 2 km, ≤ 4 % y b_f igual.

Legg 2006 a 10 km (Figs. 15–17), Y′ en x′ = −200 / −300 / −500 / −700 / −900 km: z −80 / −105 / −150 / −170 / −250; isopicnal −90 / −95 / −100 / −120; isopicnal con ν_v = 0.5 −95 / −110 / −130 / −145. b_f: z −11.7, isopicnal −11.0, isopicnal con ν_v = 0.5 −12.0 (1e-3 m s⁻²).

### 5 km (días 30–50)

Las filas de pyOCEAN a 5 km son anteriores a la corrección de la fricción; por lo medido a 10 km, el cambio esperado es del mismo orden (≤ 13 km en la trayectoria, ≤ 4 % en el transporte, ~1.5 % en b_f de z\*).

| corrida | Y′(x′ = −200/−300/−500/−800 km) [km] | T(x′) [Sv] | α_E | ρ_f / b_f [1e-3 m s⁻²] | centroide 10/20/30/50 d [m] | \|u\| delgadas/gruesas [m/s] | trunc / lim | Δt [s] | s por día |
|---|---|---|---|---|---|---|---|---|---|
| MITgcm z common | −58 / −80 / −101 / −108 | 21.8 / 26.0 / 29.4 / 31.7 | 1.81e-3 | 1031.135 / −10.79 | 946 / 1210 / 1242 / 1277 | — / 2.63 | — | 300 | 33.7 |
| **pyOCEAN z\*** | −67 / −83 / −105 / −141 | 25.7 / 28.3 / 31.8 / 34.2 | 2.34e-3 | 1031.104 / −10.49 | 922 / 1246 / 1289 / 1309 | — / 2.72 | 0 / 0 | 600 | 62.7 |
| **pyOCEAN σ** | −95 / −123 / −156 / −205 | 22.7 / 26.1 / 32.9 / 38.9 | 2.29e-3 | 1031.455 / −13.83 | 1313 / 1711 / 1875 / 1807 | — / 2.99 | 0 / 0 | 600 | 60.8 |
| **pyOCEAN híbrida** | −138 / −175 / −211 / −243 | 11.2 / 14.7 / 17.0 / 21.2 | 1.32e-3 | 1031.838 / −17.47 | 1497 / 2098 / 2327 / 2322 | 2.75 / 3.00 | 0 / 2323 | 600 | 64.1 |
| MOM6 ρ (proxy) | −419 / −417 / −436 / −447 | 0.0 / 0.0 / −0.3 / −0.4 | — | 1031.917 / −18.22 | 1409 / 2062 / 2450 / 2801 | 0.87 / 2.60 | — | 600 | 92.8 |

### Entrada (criterio 6)

| | volumen, días 30–50 (10 / 5 km) | dos clases más densas en la fila de entrada (10 / 5 km) |
|---|---|---|
| prescrito | 8.382 / 8.572 Sv | MOM6: 7.53 Sv |
| pyOCEAN z\* | exacto / exacto | 5.96 / 5.23 Sv (promedio en capas de 144 m) |
| pyOCEAN σ | exacto / exacto | 7.70 / 7.93 Sv (+2.2 / +5.3 %) |
| pyOCEAN híbrida | exacto / exacto | **5.34** / 7.35 Sv (−29 / −2.4 %; −53 % antes de la corrección de la fricción), ver punto 6 |
| MITgcm z | exacto / exacto | 4.25 / 5.33 Sv (promedio en niveles de 144 m) |
| MOM6 ρ | 8.797 Sv (su Σ tr_k, +2.6–5 % sobre Legg) | — (v·h en la cara no es el transporte `SIMPLE`) |

"Exacto" quiere decir el transporte prescrito a 1e-4 %. La medida en la fila de entrada usa el grosor de la celda, no el de la cara, y por eso la suma total difiere hasta 1.6 %. El flujo en la cara es exacto por construcción (`flather_q`).

## Lectura y clasificación

1. **El caso está bien definido: ancla de Legg ✅ (criterio 1, punto de decisión 1).** MITgcm "legg" a 10 km queda a 7 % de Legg en la flotabilidad final (−10.85 contra −11.7e-3 m s⁻²) y a 5 % en la trayectoria lejana (−204 km en x′ = −800 contra ≈ −215). Cerca de la entrada la pluma baja 25–35 % menos (−52 / −72 / −107 contra −80 / −105 / −150 km). Las causas probables son de configuración, todas (c): el canal de 100 km (Legg usó 50 en sus corridas z), la pared oeste cerrada con esponja (Legg: Orlanski abierto) y la malla C (Legg: C–D a 10 km). Esteban eligió seguir con esta diferencia documentada (2026-10-14).

2. **pyOCEAN z\* ≈ MITgcm z (criterio 2): ✅ en T y b_f, 🔶 en la trayectoria lejana.**
   - **Flotabilidad final:** a 1.6 % a 10 km (−10.07 contra −10.23) y a 3 % a 5 km (−10.49 contra −10.79).
   - **Transporte de la pluma:** a ≤ 9 % a 10 km y a ≤ 18 % a 5 km (solo x′ = −200 pasa de 15 %).
   - **Trayectoria:** pyOCEAN baja 9–35 km más a 10 km (1–3.5 celdas) y 3–33 km más a 5 km. En x′ = −300 y −500 queda a 4 % con 5 km; en x′ = −800, a 31–38 % en las dos resoluciones.
   - **Entrainment:** α_E es 29–47 % mayor, pero la pluma no sale más ligera.
   - **Clase (c) probable:** el esquema de trazadores (WENO5 + FCT contra el limitador `77`, más compresivo) y la fricción (sobre 10 m de fondo contra la celda de fondo de 144 m). La parte (a) de la fricción, que no llegaba al agua sobre capas evanescentes, se corrigió en la revisión final: acercó b_f y el transporte a MITgcm, pero no la trayectoria. El resto no se separó; queda como estudio (punto 5 de la lista).
   - **Corrección del arnés (a), 2026-10-15:** `rbcs` de MITgcm no relajaba el tinte. Faltaba `relaxMaskFile(3)` (1 = T, 2 = S, 3 = ptracer 1; MITgcm avisaba "No relaxation for pTr = 1"). El tinte se acumulaba en la esponja oeste (10.7e12 m³) y volvía al este por el fondo. La trayectoria de MITgcm common en x′ = −500 pasó de −117 a −75 km, y su transporte, que decaía aguas abajo (15.6 / 7.7 Sv), ahora crece como el de pyOCEAN. Prueba: `test_mitgcm_sponge_relaxes_the_dye`.

3. **Híbrida ≈ MOM6 ρ (criterio 3): no evaluable con el tinte de MOM6 (d + c).**
   - **Sin pluma en el talud.** El proxy de MOM6 no ve pluma en el talud: las dos clases densas no pasan de y ≈ 535 km (grosor < 1 m entre y = 495 y 375 km en todo x).
   - **La cuenca profunda gana grosor desde el día 5 en todo x** (1–5 m a los 5 días, 25–30 m a los 40). Eso no puede ser transporte del penacho; es el ajuste por ondas y la transferencia diapicna de las esponjas en modo capa, que relajan las alturas de las interfaces.
   - **En el talud engordan las capas ambientales** k = 11–19 (300–500 m).
   - **Por eso las métricas del proxy de MOM6 no son comparables:** trayectoria −416 km, transporte ≈ 0, y b_f ≡ la densidad de las dos clases por construcción.
   - **Lo único comparable es el centroide de las clases densas:** la híbrida queda a 7 % a los 10 días, a 12 % a los 20 y a 16–21 % después.
   - **Contra el isopicnal de Legg** (HIM con la mezcla por Ri de Hallberg 2000, ν_v = 0.5), la híbrida baja 22–48 % más (−116 / −151 / −192 contra −95 / −110 / −130 km) y sale más densa (b_f −16.2 contra −12.0). Sin mezcla por Ri, la capa densa casi no se diluye: el tinte de fondo es 1 en casi todo el talud (figura). Esto es **(b): falta la mezcla por cizalla dependiente de Ri** (punto 3 de la lista).

4. **Tendencia con la resolución (criterio 4): ✅, en la dirección de Legg.** De 10 a 5 km, todas las coordenadas y MITgcm dan una pluma más densa y más abajo del talud:
   - z\*: b_f −10.07 → −10.49; Y′ en x′ = −200, −49 → −67 km;
   - σ: −13.22 → −13.83; −89 → −95 km;
   - híbrida: −16.18 → −17.47; −115 → −138 km;
   - MITgcm: −10.23 → −10.79; −40 → −58 km.

   El entrainment baja en z\* (2.69 → 2.34e-3) y en la híbrida. En σ sube un poco (2.14 → 2.29e-3).

5. **Rangos y robustez (criterio 5): ✅ con Δt = 600 s; ❌ con 1200 s a 10 km.**
   - **Rangos:** s ∈ [−5.5e-19, 1] y T dentro de sus fuentes (−5 °C de la entrada; 4.8–4.99 °C del perfil inicial o del objetivo de la esponja según la coordenada) en todas las corridas. Sin truncaciones; `n_limited` 0 a 10 km y 2323 en la híbrida a 5 km.
   - **Pendientes de capas delgadas:** las caras delgadas de la híbrida van más lentas que las gruesas (3.10 / 3.21 y 2.75 / 3.00 m/s), así que 4b no aparece en DOME. La lámina de 4d tampoco aparece.
   - **Con Δt = 1200 s (el de MOM6) a 10 km, las tres coordenadas se detienen** en el día 36 por truncaciones CFL. Un modo de la parte alta de la cuenca crece desde el día ~28: η con desviación de 0.19 m contra 0.07 y \|u\| de superficie rms de 1 m/s. Se diagnosticó con dos corridas en lugar de semillas, porque la falla es determinista: con 600 s, o con θ_bc = 1 a 1200 s, llega estable a 50 y 40 días.
   - **Causa:** el transporte baroclínico centrado (θ_bc = ½, neutro) deja crecer el modo con fΔt = 0.12. A 5 km con Δt = 600 s (mismo Courant, fΔt = 0.06) no aparece. **(a), punto de decisión 3:** cambiar θ_bc reabre el amortiguamiento de la onda interna de Ilicak, así que va como plan aparte (punto 1 de la lista).
   - **Corregido el 2026-11-02 (`e1ffbbb`, clase a; `docs/design/theta_bc_inertial/root_cause.md`):** la causa no era
     θ_bc sino el transporte del flujo de PV, que se tomaba del paso anterior (Q de la continuidad) y no de uⁿ, hⁿ;
     MOM6 sólo usa el transporte retrasado en su predictor. Después de la corrección, a 10 km y 50 días (días 30–50),
     **Δt = 1200 s llega sin truncaciones en las tres coordenadas**: z\* b_f −10.26e-3 (0 eventos del limitador),
     σ −13.23 (0), híbrida −15.84 (14 eventos del limitador, 0 con 600 s); a 600 s z\* −10.25 y híbrida −15.44. La
     onda interna de Ilicak queda igual (0.736 / 0.579 / 0.261). Corridas en `RUNS/10km/pyocean_*_thetafix_dt{600,1200}`.
     Contra las líneas base del 2026-10-15 a 600 s, z\* cambia 1.8 % en b_f y la híbrida hasta 20 km en la trayectoria;
     ese cambio junta la corrección con otros cambios del código desde entonces y no se separó.
   - **La variante `kv_low` (ν_v = 1e-4) se descartó para todos los modelos:** MOM6 se detiene en el día 3 (truncaciones) y MITgcm en el día ~4 (cg2d). Sin la viscosidad Smagorinsky biarmónica ni la mezcla de Jackson del DOME propio de MOM6, el caso no está bien planteado (c/b).

6. **Entrada (criterio 6): ✅ en volumen; por clase ✅ en σ y en la híbrida a 5 km, (a) en la híbrida a 10 km.**
   - **Volumen:** exacto en las tres coordenadas.
   - **z\* y MITgcm:** pierden 21–44 % en la clase más densa por el promedio en capas de 144 m (c).
   - **Híbrida a 10 km:** la fila de entrada alterna entre una capa de fondo gruesa y mezclada (327 m a −3.4 °C el día 40; 355 m a −2.5 °C antes de la corrección de la fricción) y una a −5.0 °C (día 50). La entrada trae −5. El perfil exterior se remapea a las capas interiores actuales, así que cuando la capa interior de fondo es gruesa promedia el agua de entrada con la de arriba y la clase más densa se pierde en la primera celda. La clase densa pierde 29 % en promedio (53 % antes de la corrección). Con 5 km la capa de fondo es de 177 m a −5.0 °C.
   - **Consecuencia:** b_f de la híbrida es −16.2 a 10 km contra −17.5 a 5 km. **(a)**, punto 2 de la lista.

7. **σ queda entre z\* y la híbrida:** b_f −13.2, y la trayectoria más cercana a la z de Legg (−89 / −117 / −150 contra −80 / −105 / −150 km). Tiene más remolinos en la pluma (figura) y el mayor transporte aguas abajo (40 Sv en x′ = −800). Sin referencia σ en esta campaña, queda como observación (c).

8. **Costo (criterio 8).** Al Δt de la campaña, 1 proceso (pyOCEAN 5 hilos), en s por día simulado:

   | | pyOCEAN (z\* / σ / híbrida) | MITgcm z | MOM6 ρ |
   |---|---|---|---|
   | 10 km | 13.8 / 13.8 / 14.9 (Δt 600 s, el máximo estable) | 4.6 (Δt 600 s) | 12.0 (Δt 1200 s) |
   | 5 km | 62.7 / 60.8 / 64.1 (Δt 600 s) | 33.7 (Δt 300 s) | 92.8 (Δt 600 s) |

   - **pyOCEAN contra MITgcm:** a 10 km es 3 veces más caro con 5 veces los núcleos, con el mismo Δt; a 5 km, 1.9 veces, con el doble de Δt.
   - **pyOCEAN contra MOM6:** a 5 km es 1.5 veces más barato. A 10 km es 1.2 veces más caro, porque necesita la mitad del Δt (punto 5).
   - **Escaleras de Δt:** no se corrieron para MITgcm ni MOM6. Sus Δt son los de Legg y del ejemplo de MOM6.

9. **Validaciones existentes (criterio 9): ✅.** Con `n_tracers = 0`, sin esponja interior y sin fricción, las huellas (`lock_defaults`, `openbox_defaults` y las demás) y la suite quedan bit a bit iguales.

## Qué debe mejorar pyOCEAN

1. ~~**(a) Δt máximo a 10 km: modo inestable con θ_bc = ½ y fΔt = 0.12** (punto 5).~~ **Hecho el 2026-11-02 (`e1ffbbb`): 1200 s estable en las tres coordenadas (punto 5).** MOM6 corre DOME con 1200 s; pyOCEAN necesita 600. Candidatos:
   - descentrar un poco θ_bc (0.55–0.6, como el BE = 0.6 de MOM6);
   - analizar la estabilidad del acoplamiento AB3 + trapecio con Coriolis.

   Hay que verificarlo contra la onda interna de Ilicak (0.74 / 0.58 / 0.26). Plan aparte (punto de decisión 3).
2. **(a) Entrada por frontera abierta en la híbrida** (punto 6): el exterior se promedia sobre las capas interiores. Con una capa de entrada gruesa la clase más densa no sobrevive a la primera celda. Candidatos:
   - regridar las columnas de entrada con la densidad del exterior;
   - imponer en las caras de entrada la estructura de capas del exterior (`h_ext`), como el transporte por capa de MOM6 en modo capa o el anidado de interfaces de HYCOM.
3. **(b) Mezcla por cizalla dependiente de Ri** (Hallberg 2000; Jackson et al. 2008), para la híbrida. Sin ella la pluma isopicna sale con b_f −16 a −17.5 contra −11 / −12 de Legg (punto 3). Es la parametrización que MOM6 activa en su DOME (`USE_JACKSON_PARAM`).
4. **(b) Viscosidad Smagorinsky biarmónica,** para que ν_v bajo sea un caso bien planteado (punto 5, `kv_low`). El DOME de MOM6 la usa (`SMAG_BI_CONST = 0.06`).
5. **Estudio (c): entrainment de z\* contra MITgcm z** (punto 2). α_E es 29–47 % mayor y la pluma va 1–3 celdas más abajo del talud. Separar el efecto del esquema de trazadores (WENO5 + FCT contra el limitador `77`) y el de la fricción (10 m de fondo contra la celda de fondo).
6. **Rendimiento del paso ALE 3-D:** 3 veces MITgcm con 5 veces los núcleos al mismo Δt (punto 8). Reducir el trabajo fijo por paso; los kernels están limitados por memoria (`validation/bayiw_cut/README.md`).

**Para Esteban, fuera de pyOCEAN:**
- **Tinte de MOM6 (d).** Para comparar la híbrida con MOM6 ρ hace falta que el tinte entre. Hay dos opciones: compilar una copia parcheada de MOM6 (registrar el trazador del segmento con su `OBC_scalar`), como se hizo con la copia de HYCOM, o reportarlo a los desarrolladores. No se tocó el código de MOM6.
- **Esponjas de MOM6 en modo capa.** Mueven masa entre clases. Con un tinte real esto dejaría de importar para las métricas; si no, una corrida de MOM6 en ALE ρ con esponja ALE aislaría el efecto.

## Híbrida + GLS (N3 Task 10, criterio 5 del spec N3)

`--closure gls` cambia solo el cierre respecto a la línea base (decisión 10 del spec N3): GLS k-ε estratificado (`closure='gls'`, `gls_stratified=True`) para ν y κ_t; el arrastre cuadrático C_d = 2e-3 sobre `hbbl` = 10 m queda igual y su u*² es la condición de fondo del GLS; `kv` = 0, así que κ_t tiene el piso del GLS (1e-6 m²/s, `nuws` de CROCO) y ν el suyo (1e-5). Las salidas van a `RUNS/10km/pyocean_hybrid_gls/`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/dome/run_pyocean.py 10 hybrid --closure gls          # 50 días, Δt 600 s, 5 hilos
python validation/dome/run_pyocean.py 10 hybrid --closure gls --days 1 --cost   # costo, no escribe nada
```

Costo (10 km, 1 día, Δt 600 s, 5 hilos): 23.0 s por día con GLS contra 15.7 de la línea base (constante); 50 días ≈ 19 min. Con la ruta NumPy anterior de las capas delgadas (todas las columnas por la fusión) eran 26.0 s por día; ahora solo las columnas con capas < 1 mm se fusionan, en un kernel de Numba (`physics/mixing/gls_merged.py`). En el primer día: sin truncamientos, T en el rango inicial.

### Resultados (Task 10b, 10 km, días 30–50)

Corrida del 2026-10-29/30, 50 días, Δt 600 s, 5 hilos; métricas de `metrics.py` con las mismas funciones que la tabla de arriba (`compare.row`).

| corrida | Y′(x′ = −200/−300/−500/−800 km) [km] | T(x′) [Sv] | T máx. en x [Sv] | α_E | ρ_f / b_f [1e-3 m s⁻²] | centroide 10/20/30/50 d [m] | \|u\| delgadas/gruesas [m/s] | trunc / lim | s por día |
|---|---|---|---|---|---|---|---|---|---|
| MITgcm z legg | −52 / −72 / −107 / −204 | 20.4 / 22.1 / 25.9 / 21.0 | 36.8 (x′ = −665) | 1.94e-3 | 1031.142 / −10.85 | 883 / 1176 / 1337 / 1433 | — / 2.04 | — | 4.7 |
| pyOCEAN z\* | −49 / −72 / −99 / −126 | 22.6 / 26.4 / 30.4 / 33.3 | 35.1 (−765) | 2.69e-3 | 1031.060 / −10.07 | 835 / 1088 / 1151 / 1174 | — / 2.04 | 0 / 0 | 13.8 |
| pyOCEAN híbrida, ν_v = 0.5 | −115 / −150 / −192 / −224 | 11.2 / 15.4 / 19.7 / 19.4 | 22.5 (−695) | 1.37e-3 | 1031.702 / −16.18 | 1309 / 1823 / 2080 / 2179 | 3.10 / 3.21 | 0 / 0 | 14.9 |
| **pyOCEAN híbrida + GLS** | −70 / −83 / −111 / −156 | 17.8 / 18.4 / 19.9 / 22.0 | 22.2 (−745) | 2.29e-3 | 1031.367 / **−12.99** | 1129 / 1449 / 1503 / 1527 | 5.66 / 5.52 | **1** / 1072 | 20.9 |

Legg 2006 (10 km): b_f z −11.7, isopicnal −11.0, isopicnal con ν_v = 0.5 −12.0; Y′ (x′ = −200 / −300 / −500 / −700) z −80 / −105 / −150 / −170, isopicnal con ν_v = 0.5 −95 / −110 / −130 / −145 km.

**Rango de T: exacto.** T ∈ [−4.99963, 4.98611] °C en todos los registros. Las fuentes son la entrada (−5 °C) y el perfil exterior y de la esponja (máx. 4.98611 °C; la línea base llega a 4.986111111, el GLS no lo pasa). El trazador s ∈ [−2.8e-27, 1].

**Figuras** (fuera del repo): `RUNS/figs/n3/dome10_n3_bottom_dye.png` (también `.gif` y `.mp4`): tinte de fondo de las cuatro corridas; `RUNS/figs/n3/dome10_n3_path_transport.png`: Y′(x′) con los puntos de Legg, y T(x′).

#### Lectura y clasificación

1. **b_f: de −16.18 a −12.99 ✅ (criterio 5).** El GLS mueve la flotabilidad final 76 % del camino hacia el isopicnal de Legg con ν_v = 0.5 (−12.0): queda 8 % más densa que ese y 11 % más que la z de Legg (−11.7). Era la (b) del punto 3 de la lista de arriba (falta mezcla por cizalla): un cierre que mezcla donde hay cizalla diluye la pluma. Lo que queda es **(c)**: el k-ε de CROCO (Canuto A) no es la mezcla por Ri de Hallberg (2000) que usa el isopicnal de Legg, más la entrada de la híbrida a 10 km, que pierde 29 % de la clase más densa en la primera celda (punto 6 de arriba, (a), abierto).
2. **Trayectoria y centroide: la pluma ya no se hunde de más.** Y′ −70 / −83 / −111 / −156 km, entre MITgcm legg (−52 / −72 / −107 / −204) y la z de Legg (−80 / −105 / −150); antes, −115 / −150 / −192 / −224. El centroide del día 50 es 1527 m: 6 % más hondo que MITgcm (1434), contra 52 % con ν_v constante (2179). (c), el mismo cierre.
3. **Transporte de la pluma: plano en ~18–22 Sv.** MITgcm y z\* llegan a 35–37 Sv en x′ ≈ −665 a −765 km; la híbrida, con o sin GLS, no pasa de 22.5. α_E (x′ = 0 a −300 km) sube de 1.37e-3 a 2.29e-3, más que MITgcm (1.94e-3): el GLS diluye la pluma cerca de la entrada (17.8 Sv en x′ = −200 contra 11.2), y aguas abajo casi no entra más agua. Hipótesis (c): en la híbrida la pluma es una o dos capas gruesas, y el entrainment a través de la interfaz depende del κ_t del GLS en una sola interfaz, mientras que en z la mezcla numérica de la advección vertical sigue diluyendo; además la clase densa que pierde la entrada (punto 6) deja menos contraste. No separado (punto 4 de la lista de abajo).
4. **Velocidades más altas y el limitador activo (c/b).** max\|u\| sube de 3.2 a 5.5–5.7 m/s, y las caras delgadas ahora van un poco más rápido que las gruesas (máximos sobre los 51 registros: 5.66 contra 5.52 m/s; el de 5.66 es del día 43 en una cara evanescente de la capa 5 sobre el talud profundo, H = 3350 m; 4b en el límite). `n_limited` = 1072 en 50 días (línea base 0): el limitador de salida (`limit_outflow`) recorta a menudo. Hipótesis: el GLS deja ν cerca de su piso (1e-5 m²/s) fuera de la pluma (no medido), contra ν_v = 0.5 de la línea base, y no hay viscosidad horizontal de Smagorinsky: (b), punto 4 de la lista de arriba.
5. **Un truncamiento CFL (criterio 5: "sin truncamientos nuevos").** La serie guarda solo el total, así que la corrida se repitió en memoria (sin escribir nada; script de diagnóstico no versionado) registrando cada truncamiento; reproduce n_trunc = 1 y n_limited = 1072 exactamente, así que es determinista y el cambio de `sigma_wd.py` (punto 8) no la altera. **Dónde y cuándo:** día 31.8 (paso ≈ 4584), cara v (j = 51, i = 80; x = 800 km, y = 510 km, en el talud, H = 1450–1550 m), capa 17 contando desde el fondo (la 8.ª desde la superficie), v = 8.96 m/s contra el umbral 0.5 Δx/Δt = 8.33 m/s. La cara tiene 1.3 cm: es una capa casi evanescente entre dos capas gruesas de la columna (235 m arriba, 297 m abajo); el umbral de conteo es 1 cm, así que contó por 3 mm. Es un solo evento en 7200 pasos, y en el resto de la corrida max\|v\| se queda en 2.1–5.5 m/s. `n_limited` crece en el mismo periodo: 22 hasta el día 12, 77 el día 28 y 1072 el día 50 (≈ 85 % después del día 30), así que el truncamiento es parte del mismo régimen de capas delgadas rápidas en la pluma madura. Clase **(c/b)**: una capa evanescente sin viscosidad que la frene, el respaldo de MOM6 (`vertvisc_limit_vel`) actuó como debe; no hay un error (a) visible. Criterio 5 estricto ("sin truncamientos nuevos"): **no cumplido por un evento**; la línea base tenía 0. Se reporta, no se ajustó nada.
6. **El control con ν_v = 1e-5 constante (abortado, decisión del controlador).** Se planeó una corrida con ν_v constante igual al piso del GLS para separar "menos viscosidad" de "mezcla donde hay cizalla". Se detuvo a los 38 s de reloj (los primeros días simulados) con 1034 truncamientos CFL (> `max_truncations` = 1000): capa 3, cara u (j = 57, i = 96), u = −8.67 m/s, cara de 1.8 m (`RUNS/10km/hybrid_nu1e-5.log`). No se reemplazó. Lo que muestra: con ν bajo en todas partes el caso no está bien planteado (como `kv_low`, punto 5 de arriba), y el GLS no mejora b_f por bajar ν, sino por subirlo donde hay cizalla (la pluma y su interfaz). Clase (c) del caso; refuerza la (b) de Smagorinsky.
7. **Costo:** 20.9 s por día contra 14.9 de la línea base (+40 %); 17 min los 50 días.
8. **Aviso `invalid value encountered in divide` (`core/dynamics/sigma_wd.py`), corregido en Task 10b.** Salía en todas las corridas DOME de pyOCEAN desde el 2026-10-15 (z\*, σ e híbrida; `RUNS/log_py10_*.txt`), con o sin GLS. La corrección δ de la media vertical de `column_step` dividía 0/0 en las caras de las celdas de tierra (Σ dz_mean = 0 y sin tasa de fondo). El NaN no llegaba al estado: `WetDryFreeSurface.step` lo enmascara (`np.where(wu, …)`) y u = 0 en caras secas. Ahora δ = 0 donde el denominador es 0; en las caras mojadas el resultado es el mismo bit a bit (matriz dorada de N1 y tests de N3 sin cambio). Prueba: `tests/test_sigma_ale.py::test_ale_step_next_to_land_emits_no_runtime_warning`. Clase (a), inocua.

#### Qué debe mejorar pyOCEAN (DOME + GLS)

1. **(a) Entrada de la híbrida a 10 km** (punto 2 de la lista de arriba): sigue igual con el GLS. **[Hipótesis]** que sea la mayor parte de la diferencia restante de b_f con Legg: el efecto de la entrada y el del cierre no se separaron (ver el punto 4).
2. **(b) Viscosidad horizontal de Smagorinsky** (punto 4 de arriba): con el GLS, ν_v baja a su piso fuera de la pluma; las velocidades suben a 5.7 m/s, el limitador actúa 1072 veces y aparece un truncamiento. Con ν_v = 1e-5 constante el caso revienta.
3. **Mezcla por Ri (Jackson et al. 2008) como opción frente al GLS** en la híbrida, para comparar con el cierre del isopicnal de Legg y del DOME de MOM6.
4. **Estudio (c): transporte plano de la pluma** en la híbrida (punto 3). Separar el efecto de la entrada (punto 1) del del cierre en una sola interfaz, p. ej. con la entrada de 5 km (que no pierde la clase densa) y GLS.
5. **Diagnóstico de truncamientos:** el runner solo guarda el total. Registrar el paso, la cara y la velocidad de cada truncamiento en el `series.npz` (aquí hubo que repetir la corrida en memoria).

## Smagorinsky (spec 2026-10-30, criterios 5 y 6)

`--smag C` agrega la viscosidad horizontal de Smagorinsky laplaciana (`smagorinsky=C`, `core/dynamics/hor_visc.py`; ν = max(ah, C Δ² |S|) con el ah = 50 m²/s de Legg como fondo y la cota de estabilidad de MOM6 reescalada a AB3). Con C > 0 cambian **dos** cosas respecto a la corrida de referencia: se agrega C Δ² |S|, y el fondo ah también pasa del laplaciano simple (sin peso por grosor) al operador de tensor pesado por grosor de capa (revisión final M2). La corrida de **control de operador** con C = 1e-9 (C ≈ 0: el fondo ah en el operador de tensor, sin término de Smagorinsky; `n_visc_limited` = 0 y ν máx. = 50 m²/s) separa los dos cambios: corrida base → control = cambio de operador; control → C = 0.15 = Smagorinsky. Las salidas van a `RUNS/10km/pyocean_hybrid_gls_smag<C>/` y `RUNS/10km/pyocean_hybrid_kv_low_smag<C>/`. Las bitácoras están en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/smag_logs/` (`*_v2.log`, `dome_gls_smag1e-9.log`).

**Corridas anteriores a la corrección 47f2730 (2026-10-30).** Hasta esa corrección, en z\*/híbrida el operador usaba el grosor de cara recortado en el fondo menos hondo; junto a cada escalón de la batimetría h_min/h ≈ 0, y ν (incluido el fondo ah) caía a ≈ 0 en esas celdas. Clase (a), encontrada en la revisión final (I1). Ahora usa la media sin recortar ½(h_L + h_R) de MOM6 (`MOM_hor_visc.F90:769-782`). Todos los números de esta sección son de las corridas nuevas; las anteriores quedan como `*_preI1` junto a cada corrida y `compare.py` las marca "(antes de la corrección I1)".

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/dome/run_pyocean.py 10 hybrid --closure gls --smag 0.15   # criterio 5 (C = 0.5: sensibilidad)
python validation/dome/run_pyocean.py 10 hybrid --closure gls --smag 1e-9      # control de operador, C ≈ 0
python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15           # criterio 6: ν_v = 1e-4 constante
python validation/dome/compare.py 10 --only hybrid                             # tabla (columna trunc / lim / visc)
python validation/dome/smag_offline.py pyocean_hybrid_gls_smag0.15             # ν de Smagorinsky fuera de línea
python validation/dome/figures.py 10 --runs mitgcm_z_legg,pyocean_hybrid,pyocean_hybrid_gls,pyocean_hybrid_gls_smag1e-09,pyocean_hybrid_gls_smag0.15,pyocean_hybrid_gls_smag0.5,pyocean_hybrid_kv_low_smag0.15 --sub smag
```

### Resultados (2026-10-30, después de la corrección I1; 10 km, 50 días, Δt 600 s, 5 hilos; métricas de los días 30–50)

| corrida | Y′(x′ = −200/−300/−500/−800 km) [km] | T(x′) [Sv] | α_E | ρ_f / b_f [1e-3 m s⁻²] | centroide 10/20/30/50 d [m] | \|u\| delgadas/gruesas [m/s] | máx. diario de \|u\| (días 30–50, media) | trunc / lim / visc | s por día |
|---|---|---|---|---|---|---|---|---|---|
| MITgcm z legg | −52 / −72 / −107 / −204 | 20.4 / 22.1 / 25.9 / 21.0 | 1.94e-3 | 1031.142 / −10.85 | 883 / 1176 / 1337 / 1433 | — / 2.04 | 2.02 | — | 4.7 |
| pyOCEAN híbrida + GLS | −70 / −83 / −111 / −156 | 17.8 / 18.4 / 19.9 / 22.0 | 2.29e-3 | 1031.367 / −12.99 | 1129 / 1449 / 1503 / 1527 | 5.66 / 5.52 | 3.16 | 1 / 1072 / — | 20.9 |
| + control de operador, C ≈ 0 (1e-9) | −66 / −83 / −110 / −156 | 17.6 / 18.8 / 19.7 / 21.6 | 2.46e-3 | 1031.373 / −13.05 | 1147 / 1444 / 1497 / 1524 | 4.30 / 3.56 | 2.94 | 1 / 641 / 0 | 21.8 |
| **+ Smagorinsky C = 0.15** | −74 / −92 / −118 / −159 | 19.1 / 18.1 / 20.6 / 19.2 | 1.83e-3 | 1031.400 / −13.31 | 1146 / 1524 / 1601 / 1539 | 3.10 / 3.54 | 2.86 | **0 / 1** / 101 921 | 21.4 |
| + Smagorinsky C = 0.5 (sensibilidad) | −79 / −98 / −125 / −168 | 14.8 / 16.8 / 18.9 / 16.7 | 2.21e-3 | 1031.432 / −13.61 | 1166 / 1607 / 1684 / 1536 | 2.70 / 3.11 | 2.63 | 0 / 0 / 8 667 220 | 22.0 |
| híbrida `kv_low` (ν_v = 1e-4) + C = 0.15 | −101 / −128 / −163 / −230 | 10.1 / 9.2 / 9.3 / 11.8 | 1.01e-3 | 1031.885 / −17.91 | 1346 / 1868 / 1971 / 2090 | 5.81 / 7.61 | 5.99 | 61 / 239 337 / 1 563 566 | 16.9 |

"\|u\| delgadas/gruesas": máximo sobre los 51 registros de \|u\| en caras u con el lado más delgado < 1 cm / con los dos lados ≥ 1 m (`metrics.thin_vs_thick`). "Máx. diario de \|u\|": el máximo de \|u\| y \|v\| en caras con las dos celdas de ≥ 1 m en cada registro diario, promediado en los días 30–50 (cálculo sobre los `series.npz`, no versionado). El rango de T es exacto en todas las corridas: mínimo −5.00000 a −4.99956 °C y máximo 4.98611 °C (el de la esponja). El trazador s ∈ [−3.8e-22, 1].

**Figuras** (fuera del repo): `RUNS/figs/smag/dome_10km_path_transport.png` (Y′(x′) con los puntos de Legg, y T(x′)) y `RUNS/figs/smag/dome_10km_bottom_dye.png` (también `.gif` y `.mp4`), con el control de operador.

### Lectura y clasificación

Cada cambio se atribuye con el control: (operador) = de la corrida base al control C ≈ 0; (Smagorinsky) = del control a C = 0.15.

1. **Criterio 5: cero truncamientos ✅, por el término de Smagorinsky.** Con C = 0.15 no hay truncamientos (antes 1) y el limitador de salida actúa 1 vez (antes 1072). El control C ≈ 0 conserva el truncamiento y 641 eventos del limitador: el cambio de operador quita 40 % de los eventos (operador) y el término de Smagorinsky quita el truncamiento y el resto (Smagorinsky). C = 0.5 también da 0 / 0. Es un solo evento en la base, así que "cero" es evidencia débil de robustez (c).
2. **max\|u\| más cerca de MITgcm ✅, sobre todo por el operador.**
   - Caras gruesas: 5.52 → 3.56 (operador) → 3.54 m/s (Smagorinsky; MITgcm 2.04). El 3.56 del control es del día 10 y el 3.54 de C = 0.15 del día 44.
   - Caras delgadas: 5.66 → 4.30 (operador) → 3.10 m/s (Smagorinsky).
   - Media del máximo diario, días 30–50: 3.16 → 2.94 (operador) → 2.86 (Smagorinsky); MITgcm 2.02. Lo que falta (≈ 0.8 m/s) es (c): la coordenada y el cierre, no la viscosidad horizontal (la híbrida con ν_v = 0.5 constante da 3.21 en caras gruesas).
   - **[Hipótesis]** El operador de tensor pesa la viscosidad por h, así que una capa gruesa que corre junto a capas delgadas del vecino se frena más que con el laplaciano simple de u; no se aisló.
3. **b_f, entrainment y transporte: un poco menos de mezcla, por Smagorinsky (c).**
   - b_f: −12.99 → −13.05 (operador) → −13.31 (Smagorinsky), 2.5 % más densa en total. Se aleja un poco de Legg (isopicnal con ν_v = 0.5: −12.0; z: −11.7).
   - α_E: 2.29e-3 → 2.46e-3 (operador, +7 %) → 1.83e-3 (Smagorinsky, −26 %), ahora 6 % por debajo de MITgcm (1.94e-3).
   - T(x′): ±1.5 Sv sin signo fijo entre x′ = −200 y −500 km; −2.4 Sv en −800 (Smagorinsky; el control queda a ≤ 0.4 Sv de la base).
   - **[Hipótesis]** Smagorinsky frena la cizalla lateral y las velocidades de escala de malla de la pluma. Con menos cizalla, el GLS produce menos turbulencia y diluye menos. No se midió la producción por cizalla del GLS. Es una elección del cierre de momento (c). No se ajustó nada.
4. **Trayectoria y centroide: algo más abajo con Smagorinsky (c).**
   - Y′: el control queda a ≤ 4 km de la base; C = 0.15 lleva la pluma 3–9 km más al sur (−74 / −92 / −118 / −159). Se acerca a la z de Legg (−80 / −105 / −150) y se aleja de MITgcm legg.
   - Centroide: el control queda a ≤ 18 m de la base; con C = 0.15 la pluma está 80–104 m más honda que el control los días 20 y 30 y 15 m el día 50 (1539 m; MITgcm 1433).
5. **Cuánto actúa Smagorinsky [aproximado].** Se reconstruyó C Δ² |S| fuera de línea en los registros de los días 10, 30 y 50 con `validation/dome/smag_offline.py`. El script usa `smagorinsky_numpy` con la media sin recortar de las dos celdas (la del modelo desde 47f2730) pero con máscaras de tierra aproximadas. En los puntos h de las capas de ≥ 1 m, con C = 0.15:
   - ν_S supera el fondo de 50 m²/s en el 13.5 % de los puntos el día 10, el 88 % el día 30 y el 99.7 % el día 50 [aproximado];
   - la mediana es 50 / 152 / 419 m²/s y el percentil 99 es 626–1304 m²/s; la cota no actúa en ningún punto de esas capas.

   Al final de la corrida, Smagorinsky fija ν casi en todo el dominio, mediana ≈ 8 veces el fondo de Legg. Eso también separa este caso de la física común (ν_h = 50 constante): (c).
6. **`n_visc_limited`: 101 921 en 7200 pasos (≈ 14 puntos h + q por paso) con C = 0.15.** Antes de la corrección I1 eran 27.9 M (≈ 3900 por paso): casi todos eran las celdas junto a caras recortadas en el fondo, donde ν (incluido ah) caía a ≈ 0. Con la media sin recortar, h_min/h ≥ ½ junto a una capa evanescente, así que lo que queda son los puntos de capas delgadas donde la cota × h_min/h corta a C Δ² |S| o a ah; en las capas de ≥ 1 m la reconstrucción no encuentra ninguno. `nu_t_max` = 5120 m²/s con C = 0.15: entre la mitad (4545) y la cota completa (9091 m²/s a Δx = 10 km, Δt = 600 s). Con C = 0.5 (8.7 M, ≈ 1200 por paso) y en `kv_low` (1.56 M) `nu_t_max` llega a la cota completa; con C = 0.5 la reconstrucción encuentra 25–330 puntos limitados por registro en las capas de ≥ 1 m.
7. **C = 0.5 (sensibilidad, no arreglo).** También tiene 0 truncamientos y 0 eventos del limitador.
   - La pluma baja más: Y′ 9–15 km más al sur que la base, centroide del día 30 a 1684 m.
   - b_f −13.61, α_E 2.21e-3, transporte 14.8–18.9 Sv (menos que con C = 0.15).
   - Es la tendencia esperada de más viscosidad lateral, sin mejora clara contra MITgcm o Legg. Se queda C = 0.15, el coeficiente laplaciano de OM4p5 (Adcroft et al. 2019, §2.2.5: "Laplacian viscosity taken as the maximum of a Smagorinsky dynamic viscosity with coefficient 0.15"; OM4p25 solo usa la biarmónica, 0.06, copia local `Adcroft_2019 - The GFDL Global Ocean and Sea Ice Model OM4.0.md`, l. 676; `SMAG_LAP_CONST = 0.15` en `~/pyOCEAN_refs/MOM6-examples/ice_ocean_SIS2/OM_1deg/MOM_input:499`).
8. **Criterio 6, `kv_low` (ν_v = 1e-4 constante, híbrida, ah = 50 + C = 0.15): completa los 50 días (7200 pasos), pero no es robusto 🔶.** MOM6 y MITgcm se detienen en los días 3–4 sin Smagorinsky.
   - **61 truncamientos CFL** (antes de I1: 1) y 239 337 eventos del limitador (antes: 31 900). No llegó a `max_truncations` = 1000, así que la corrida terminó. La bitácora (`dome_kvlow_smag0.15_v2.log`) solo tiene el aviso del piso z\* de la híbrida, y el runner guarda solo los totales, así que el paso y la cara de cada truncamiento no se conocen (punto 2 de la lista de abajo).
   - **Dónde (de los registros diarios):** entre los días 25 y 36 el máximo diario de \|u\| se queda en 3.9–4.5 m/s. Desde el día 37 crece en la cuenca profunda (H = 3600 m, y ≈ 85–265 km), en capas interiores (capas 10–23 desde el fondo, caras con lados de 0.6 a 125 m): 4.99 el día 37, 6.86 el día 40, 7.73 el día 44 y 8.29 m/s el día 50 en una cara v de la capa 19 desde el fondo (lado de 0.6 m contra 125 m). Es el 99.5 % del umbral de truncamiento (0.5 Δx/Δt = 8.33 m/s). **[Hipótesis]** los 61 truncamientos son de los días ≥ 40, cuando el máximo diario pasa de 6.9 m/s; no verificado.
   - **Por qué:** la media vertical ū no pasa de 0.93 m/s en esos días; es cizalla entre capas que ν_v = 1e-4 no acopla (lo mismo que antes de I1, ahora antes y más fuerte). Clase (b): falta mezcla por cizalla (Jackson et al. 2008, o el GLS) con ν_v bajo. **[Hipótesis sin verificar]** que el cambio respecto a la corrida anterior sea de trayectoria (el caso no tiene freno vertical, así que pequeñas diferencias crecen), no un efecto sistemático de la media sin recortar.
   - La pluma casi no mezcla: b_f −17.91, 9–12 Sv, α_E 1.01e-3.
9. **Costo:** 21.4 s por día con C = 0.15 contra 20.9 sin Smagorinsky (+2.4 %), 21.8 el control C ≈ 0 (+4.3 %) y 22.0 con C = 0.5 (+5.3 %). Son tiempos de las corridas de producción, una a la vez (`wall_s` del meta). Las diferencias entre las tres corridas con el operador son del orden del ruido; quedan dentro del objetivo de ≤ 10 % del criterio 9. Por paso (`scripts/smag_cost.py`, después de la corrección): 0.992× en DOME.
10. **Nota sobre el spec.** El spec dice "max\|u\| de 5.7 m/s contra 3.2 de MITgcm". Los 3.2 m/s son de la híbrida de pyOCEAN con ν_v = 0.5. El max\|u\| de MITgcm legg es 2.04 m/s (tabla de N3 de arriba).

#### Qué debe mejorar pyOCEAN (Smagorinsky en DOME)

1. **Contador `n_visc_limited` útil.** Contar solo los puntos donde C Δ² |S| > cota, o separar los dos casos. Después de la corrección I1 quedan ≈ 14 puntos por paso con C = 0.15, en capas delgadas.
2. **Diagnóstico de truncamientos en el `series.npz`** (punto 5 de N3, sigue abierto): paso, cara y velocidad de cada evento. Hace falta para ubicar los 61 truncamientos de `kv_low`.
3. **Mezcla por cizalla con ν_v bajo (b).** `kv_low` termina, pero la cuenca profunda se acelera sin freno vertical hasta el umbral de truncamiento. Una mezcla por Ri (Jackson et al. 2008) es lo que usa el DOME de MOM6 con su ν_v bajo.
4. **Estudio (c): por qué el operador de tensor pesado por grosor baja max\|u\|** (punto 2): 5.52 → 3.56 m/s en caras gruesas sin término de Smagorinsky.

## Mezcla por cizalla de Jackson et al. (2008) contra la verdad no hidrostática de Legg (spec 2026-10-30, criterios 4, 5 y 7)

`--shear jackson` agrega la mezcla interior por cizalla de Jackson, Hallberg & Legg (2008), el `kappa_shear` de MOM6 (`shear_mixing='jackson'`, `shear_prandtl` = `--shear-pr`, por defecto 1), al cierre constante. El kernel coincide con `kappa_shear_column` de MOM6 compilado (`validation/mom6_column/README.md`, criterio 1). Solo cambia el cierre: κ_shear se suma a la difusividad de los trazadores, que en DOME es 0 (física común, sin difusión de trazadores), y ν_shear = Pr κ_shear se suma a ν_v con un paso de retraso. Con kv = 0, κ0 = 1e-7 m²/s (MOM6: max(KD, 1e-7)). La corrida principal es la híbrida `kv_low` (ν_v = 1e-4) + Jackson Pr = 1 + Smagorinsky C = 0.15: es el equivalente a 10 km del isopicnal de Legg (ν_v = 1e-4 + mezcla por Ri de Hallberg 2000 + Smagorinsky; Legg 2006 p. 77). Las salidas van a `RUNS/10km/pyocean_<coord>_<variante>_jackson[pr0]/`. Las bitácoras están en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/shear_logs/` (cadena `chain.sh`, 8 corridas, rc = 0, 2026-10-30 19:36–21:19).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson                 # la principal
python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson --shear-pr 0      # sensibilidad Pr = 0
python validation/dome/run_pyocean.py 10 hybrid --shear jackson                                        # ν_v = 0.5 + Jackson
python validation/dome/run_pyocean.py 10 zstar --kv-low --smag 0.15 [--shear jackson]                  # también sigma
python validation/dome/compare.py 10 --only "pyocean_hybrid,pyocean_zstar_kv,pyocean_sigma_kv,mitgcm_z_legg"
python validation/dome/figures.py 10 --no-anim --sub jackson --runs mitgcm_z_legg,pyocean_hybrid_kv_low_smag0.15,pyocean_hybrid_kv_low_smag0.15_jackson,pyocean_hybrid_kv_low_smag0.15_jacksonpr0,pyocean_hybrid,pyocean_hybrid_jackson,pyocean_hybrid_gls,pyocean_zstar_kv_low_smag0.15,pyocean_zstar_kv_low_smag0.15_jackson,pyocean_sigma_kv_low_smag0.15,pyocean_sigma_kv_low_smag0.15_jackson
python validation/dome/figures.py 10 --before-after pyocean_hybrid_kv_low_smag0.15,pyocean_hybrid_kv_low_smag0.15_jackson,mitgcm_z_legg --sub jackson
```

### Resultados (10 km, 50 días, Δt 600 s, 5 hilos; E, T, α_E, b_f, trayectoria en los días 30–50; trazador por clase en la segunda mitad, días 25–50)

**Contra la verdad.** "dig." = leído de las figuras de Legg et al. (2006) (`legg2006_nonhydro.json`, con figura, página e incertidumbre: ±5 % en E, ±0.15e-3 en b_f, ±0.5×10⁶ m² en el color de la Fig. 4). El 8.6e-4 de E no hidrostático sale del texto (p. 79) y a 0.5 km; a 10 km, las referencias equivalentes de E son las corridas de 10 km de Legg (z 2.19e-4, isopicnal 9.4e-4), porque S crece con la resolución. Además, E cambia ~×2 con la definición (`compare.py`): diferencias de E menores que ×2 no son significativas.

| corrida | E esquina, U_rms pluma / todo | T_out, T_in [Sv] | T(x′ = −200 / −300) [Sv] | α_E (−180…−50 km) | b_f [1e-3 m s⁻²] | b medio del trazador en x′ = −300 [1e-3 m s⁻²] | clase con más trazador x′ = −300 / −100 |
|---|---|---|---|---|---|---|---|
| Legg no hidrostático 0.5 km (verdad) | 8.6e-4 (texto) | — | 22.5 / 20.6 (dig.) | 1.03e-3 (dig.) | −8.75 (dig.) | −8.99 (dig.) | −10…−9 (dig.) |
| Legg 10 km z (ν_v 0.5) | 2.19e-4 (dig.) | — | — | — | −11.67 (dig.) | — | — |
| Legg 10 km isopicnal (ν_v 1e-4 + Ri) | 9.4e-4 (dig.) | — | — | — | −11.07 (dig.); ν_v 0.5: −12.09 | — | — |
| MITgcm z legg | 1.04e-3 / 1.96e-3 | 14.83, 12.66 | 20.4 / 22.1 | 6.14e-4 | −10.85 | −10.79 | −14…−13 / −16…−15 |
| híbrida `kv_low` + Smag 0.15 (línea base) | −1.02e-4 / −2.40e-4 | 8.29, 9.26 | 10.1 / 9.2 | 1.92e-5 | −17.91 | −17.94 | −19…−18 / −19…−18 |
| **+ Jackson Pr = 1 (principal)** | **4.88e-4** / 1.13e-3 | 12.46, 9.83 | **15.9 / 17.9** | **4.77e-4** | **−12.84** | **−12.98** | **−17…−16** / −19…−18 |
| + Jackson Pr = 0 (sensibilidad) | 1.14e-3 / 1.50e-3 | 16.49, 12.73 | 12.3 / 16.3 | 1.56e-3 | −8.29 | −8.18 | −9…−8 / −8…−7 |
| híbrida ν_v = 0.5 | −4.48e-5 / −1.55e-4 | 8.24, 8.49 | 11.2 / 15.4 | 1.70e-4 | −16.18 | −16.27 | −19…−18 / −19…−18 |
| + Jackson Pr = 1 | 1.96e-4 / 6.53e-4 | 10.21, 9.12 | 15.5 / 18.4 | 2.96e-4 | −16.27 | −16.44 | −19…−18 / −19…−18 |
| híbrida + GLS (referencia, N3) | 5.09e-4 / 1.28e-3 | 13.90, 10.94 | 17.8 / 18.4 | 3.69e-4 | −12.99 | −13.16 | −18…−17 / −18…−17 |
| z\* `kv_low` + Smag 0.15 | 8.41e-4 / 1.18e-3 | 13.75, 10.49 | 19.2 / 20.2 | 8.46e-4 | −8.81 | −9.27 | −12…−11 / −18…−17 |
| + Jackson Pr = 1 | 1.10e-3 / 1.59e-3 | 13.82, 10.38 | 18.4 / 19.7 | 1.28e-3 | −8.98 | −9.41 | −12…−11 / −18…−17 |
| σ `kv_low` + Smag 0.15 | 1.48e-4 / 2.19e-4 | 10.34, 9.72 | 17.7 / 20.7 | 1.21e-3 | −11.84 | −12.04 | −14…−13 / −19…−18 |
| + Jackson Pr = 1 | 5.70e-4 / 8.57e-4 | 11.76, 9.51 | 19.7 / 20.8 | 1.41e-3 | −11.31 | −11.34 | −13…−12 / −18…−17 |

"b medio del trazador": primer momento de la distribución del trazador por clase en la columna de x′ = −300 km (`figures.mean_class_b`); el de Legg sale del mismo cálculo sobre su Fig. 4 digitalizada.

**Trayectoria, robustez y costo.**

| corrida | Y′(x′ = −200/−300/−500/−800 km) [km] | centroide día 50 [m] | \|u\| delgadas / gruesas [m/s] | máx. diario de \|u\|, días 30–50 (media) / día 50 [m/s] | trunc / lim / visc | s por día (reloj de producción) |
|---|---|---|---|---|---|---|
| MITgcm z legg | −52 / −72 / −107 / −204 | 1433 | — / 2.04 | 2.02 / — | — | 4.7 |
| híbrida `kv_low` + Smag 0.15 | −101 / −128 / −163 / −230 | 2090 | 5.81 / 7.61 | 5.99 / 8.23 | 61 / 239 337 / 1 563 566 | 16.9 |
| **+ Jackson Pr = 1** | −69 / −86 / −105 / −152 | 1428 | 2.70 / 2.64 | 2.45 / 2.87 | **0 / 0** / 57 993 | 21.1 |
| + Jackson Pr = 0 | −50 / −56 / −69 / −85 | 902 | 4.07 / 3.59 | 2.47 / 2.79 | 0 / 0 / 105 001 | 18.6 |
| híbrida ν_v = 0.5 | −115 / −150 / −192 / −224 | 2179 | 3.10 / 3.21 | 2.57 / 3.21 | 0 / 0 / — | 14.9 |
| + Jackson Pr = 1 | −125 / −160 / −199 / −238 | 2321 | 2.25 / 2.76 | 2.34 / 2.42 | 0 / 0 / — | 16.2 |
| z\* `kv_low` + Smag 0.15 | −40 / −51 / −67 / −116 | 928 | — / 5.66 | 2.72 / 5.82 | 0 / 0 / 598 | 14.8 |
| + Jackson Pr = 1 | −41 / −52 / −73 / −116 | 962 | — / 3.41 | 2.23 / 3.27 | 0 / 0 / 1 893 | 16.9 |
| σ `kv_low` + Smag 0.15 | −71 / −95 / −130 / −161 | 1459 | — / 7.34 | 3.85 / 6.55 | 0 / 0 / 0 | 14.9 |
| + Jackson Pr = 1 | −71 / −91 / −122 / −166 | 1328 | — / 3.24 | 2.60 / 3.94 | 0 / 0 / 0 | 17.7 |

Legg 10 km: Y′(x′ = −200 / −300 / −500 / −700) z −80 / −105 / −150 / −170, isopicnal −90 / −95 / −100 / −120 km. "Máx. diario de \|u\|": máximo de \|u\| y \|v\| en caras con las dos celdas ≥ 1 m en cada registro diario (cálculo sobre los `series.npz`, no versionado; con la cara delgada incluida el máximo de la línea base el día 50 es 8.29 m/s, sección de Smagorinsky). El rango de T es exacto en todas las corridas (−5.000 a 4.986 °C híbrida, 4.967 σ, 4.800 z\*) y s ∈ [−2.2e-20, 1] en la principal.

**Figuras** (fuera del repo, `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/figs/jackson/`):
- `dome_10km_before_after.png`: la principal frente a su línea base: tinte de fondo los días 10/20/30/50, trayectoria, transporte y trazador por clase en x′ = −300 km con la verdad digitalizada;
- `dome_10km_bottom_dye_before_after.gif` (y `.mp4`): animación diaria del tinte de fondo, línea base, principal y MITgcm legg;
- `dome_10km_legg_scorecard.png`: b_f, E, T(x′ = −200), α_E y b medio del trazador de las 11 corridas contra Legg;
- `dome_10km_legg_truth.png` (T(x′) y α_E(x′) contra la Fig. 3), `dome_10km_tracer_by_class.png` (Fig. 4 de cada corrida junto a la de Legg), `dome_10km_bottom_dye.png`, `dome_10km_path_transport.png`.

### Lectura y clasificación

1. **¿Jackson acerca pyOCEAN a la verdad no hidrostática? Sí, en todas las métricas contra la verdad no hidrostática de la corrida principal (criterio 4, dirección ✅); la magnitud queda abierta.** La trayectoria contra el isopicnal de Legg a 10 km no mejora en todo punto: en x′ = −200 km se aleja (de 11 a 21 km). Contra su línea base:
   - **b_f:** −17.91 → −12.84 (verdad −8.75 dig.): 55 % del camino. Queda 16 % más densa que el isopicnal de Legg a 10 km (−11.07) y 6 % más densa que su isopicnal con ν_v = 0.5 (−12.09).
   - **E de la esquina:** −1.0e-4 (no entraba agua) → 4.88e-4. Está a ×1.9 debajo del isopicnal de Legg a 10 km (9.4e-4) y ×2.2 encima de su z (2.19e-4): dentro de la banda de ×2 de la definición, así que igual a sus 10 km dentro de lo que la métrica distingue.
   - **Transporte:** T(x′ = −200) 10.1 → 15.9 Sv (verdad 22.5 dig.); α_E medio −180…−50 km 1.9e-5 → 4.77e-4 (verdad 1.03e-3 dig.).
   - **Trazador por clase:** la clase con más trazador en x′ = −300 km pasa de la del agua de entrada sin mezclar (−19…−18) a −17…−16; el b medio, de −17.94 a −12.98 (verdad −8.99 dig., con el máximo en −10…−9). La distribución se ensancha entre −17 y −8 (figura `before_after`), pero no tiene el pico angosto de Legg.
   - **Trayectoria:** −101 / −128 / −163 → −69 / −86 / −105 km en x′ = −200 / −300 / −500, a ≤ 21 km (2 celdas) del isopicnal de Legg a 10 km (−90 / −95 / −100). En x′ = −300 y −500 se acerca (33 → 9 km y 63 → 5 km); en x′ = −200 se aleja (11 → 21 km). El centroide del día 50 baja de 2090 a 1428 m (MITgcm legg 1433).

   **Lo que falta, clasificado:**
   - **(c) Resolución.** Las corridas de 10 km del propio Legg están igual de lejos de su no hidrostático: b_f −11.07 (iso) y −11.67 (z) contra −8.75, y la nuestra −12.84. A 10 km la interfaz de la pluma tiene una o dos capas; la verdad no hidrostática resuelve la mezcla de Kelvin–Helmholtz a 30 m.
   - **(a) Entrada de la híbrida por frontera abierta (abierto desde 2026-10-15, punto 2 de "Qué debe mejorar").** La columna de x′ = −100 km sigue con el máximo en la clase de entrada (−19…−18) con y sin Jackson: el agua más densa entra sin mezclar en una capa gruesa. **[Hipótesis]** que sea una parte grande del b_f restante; no separado.
   - **(c) Configuración de Jackson.** El DOME de MOM6 usa `PRANDTL_TURB = 0` y `KD = 1e-4` (`~/pyOCEAN_refs/MOM6-examples/ocean_only/DOME/MOM_input:228, 373`), así que su κ0 = 1e-4 (aquí 1e-7, porque la física común no tiene difusión de trazadores). Con Pr = 0 (siguiente punto) b_f llega a −8.29.
   - **(c) Geometría** de MOM6 (canal de 100 km, esponja, arrastre cuadrático) contra la de Legg (50 km, Orlanski, no-slip), como en el resto del benchmark. T_in = 9.83 Sv contra 8.4 Sv de la entrada: 1.4 Sv de agua con s > 0.01 se mezcla ya en el canal (MITgcm legg: 12.66).
   - **El transporte no hidrostático (dig.) tiene su máximo en x′ = −150 km (24.6 Sv), se queda en 20–23 Sv hasta x′ = −350 km y baja al oeste de ahí** (1.55 Sv en x′ = −650); ningún modelo a 10 km lo hace (MITgcm sigue creciendo hasta 36.8 Sv). **[Hipótesis]** es el criterio s > 0.01 con la pluma muy diluida y el canal de 50 km de Legg (c); por eso T se compara en x′ = −200 / −300 km.
2. **Pr = 0 (sensibilidad, no calibración): la más cercana a la verdad en la mezcla, la más lejana en la trayectoria.** b_f −8.29 (verdad −8.75), máximo de trazador en −9…−8 en x′ = −300 (verdad −10…−9), b medio −8.18 (−8.99), E 1.14e-3, α_E 1.56e-3 (verdad 1.03e-3). Pero la pluma casi no baja: Y′ −50 / −56 / −69 / −85 km (isopicnal de Legg −90 / −95 / −100 / −120) y centroide 902 m. **Por qué:** con Pr = 1, ν_shear quita la cizalla que produce la mezcla y el número de Richardson vuelve a ~¼; con Pr = 0 la cizalla queda y κ crece (κ_shear máximo en el dominio, calculado fuera de línea con el kernel de pyOCEAN sobre los registros de los días 30/40/50: 14 / 39 / 87 m²/s con Pr = 0 contra 2.1 / 4.7 / 12.7 con Pr = 1 [aproximado: un registro diario, no el paso del modelo]). Jackson et al. (2008) usan Pr = 1; MOM6 usa Pr = 0 en su DOME. Es una elección de configuración (c), no un error; no se cambió el valor por defecto.
3. **Con ν_v = 0.5, Jackson casi no actúa (c, abierto).** b_f −16.18 → −16.27 y la clase de entrada sigue dominando; E sube de ≈ 0 a 1.96e-4. Fuera de línea (kernel de pyOCEAN sobre los registros de los días 30, 40 y 50, mismo Δt, κ0 = 1e-7; script no versionado) κ_shear > 1e-4 m²/s en **0 %** de las interfaces de la pluma (s > 0.01 a un lado), contra 13–19 % en la principal: ν_v = 0.5 m²/s alisa la cizalla y Ri queda arriba de ¼. El isopicnal de Legg con ν_v = 0.5 y la mezcla por Ri de Hallberg (2000) llega a −12.09. **[Hipótesis]** la diferencia está en la forma de la mezcla por Ri del HIM (Ri de par de capas sobre capas gruesas) contra el κ de Jackson en interfaces con la cizalla resuelta; no se probó.
4. **Criterio 5 ✅: la principal termina sin truncamientos ni eventos del limitador.** 61 → 0 truncamientos y 239 337 → 0 eventos; `n_visc_limited` 1 563 566 → 57 993. La cuenca profunda ya no se acelera: el máximo diario de \|u\| (caras gruesas) promedia 2.45 m/s en los días 30–50 (línea base 5.99) y es 2.87 m/s el día 50 (8.23); MITgcm 2.02. Era la (b) de la sección de Smagorinsky (punto 3 de su lista): con Pr = 1, ν_shear acopla las capas interiores. Con Pr = 0 tampoco hay truncamientos (máx. diario 2.47 m/s): los acopla la mezcla de densidad, que baja la cizalla al quitar la estratificación de la interfaz.
5. **Dependencia con la coordenada.**
   - **z\*:** Jackson casi no cambia b_f (−8.81 → −8.98) ni la trayectoria, sube E (8.4e-4 → 1.1e-3) y α_E (8.5e-4 → 1.28e-3), y baja el máximo de \|u\| del día 50 de 5.82 a 3.27 m/s. **[Hipótesis]** que z\* con ν_v = 1e-4 esté cerca de la verdad en b_f (−8.81 contra −8.75) por mezcla numérica (c): la z de Legg a 10 km da −11.67, MITgcm legg −10.85 y z\* con ν_v = 0.5 −10.07, y con ν_v bajo la pluma de escalones mezclaría más. No está medido: lo separaría el balance de varianza (`variance_budget=True`, `num_adv` contra `phy_v`; punto 4 de la lista de abajo).
   - **σ:** b_f −11.84 → −11.31, E 1.5e-4 → 5.7e-4, máximo de \|u\| del día 50 6.55 → 3.94 m/s.
   - En las tres coordenadas Jackson sube E y baja las velocidades interiores; el efecto sobre b_f es grande solo donde no había otra mezcla (híbrida, isopicnal en el interior).
6. **Contra el GLS (N3).** La principal y la híbrida + GLS quedan juntas en b_f (−12.84 / −12.99), E (4.88e-4 / 5.09e-4) y trayectoria; Jackson + Smagorinsky lo hace sin truncamientos ni eventos del limitador (GLS: 1 / 1072) y con max\|u\| 2.64 contra 5.52 m/s en caras gruesas. Dos cierres distintos dan la misma pluma: **[hipótesis]** lo que falta para la verdad no depende de elegir GLS o Jackson con Pr = 1; sí puede depender de la configuración de Jackson (punto 1, causa "(c) configuración de Jackson"): con Pr = 0 b_f pasa de −12.84 a −8.29.
7. **Costo (criterio 7).** Por paso (`scripts/shear_cost.py`, subsección siguiente): BayIW cut z\* 1.275×, DOME 1.096×. En producción (reloj de la corrida completa, incluye la salida y una trayectoria distinta): principal 1055 s contra 845 (1.25×), Pr = 0 930 s (1.10×), híbrida ν_v = 0.5 808 contra 747 s (1.08×), z\* 843 contra 742 (1.14×), σ 883 contra 743 (1.19×).

### Qué debe mejorar pyOCEAN (Jackson en DOME)

1. **(a) Entrada de la híbrida a 10 km** (punto 2 de la primera lista, sigue abierto): la clase de entrada sin mezclar domina en x′ = −100 km con cualquier cierre. Es lo primero para acercar b_f y el trazador por clase.
2. **Variante con la configuración de Jackson del DOME de MOM6** (KD = 1e-4, así κ0 = 1e-4, y Pr = 0) como caso de referencia definido en `cases.py`, no como ajuste: separa la configuración (c) de lo que hace el esquema.
3. **Verdad con más resolución.** DOME a 10 km no distingue la física de mezcla de la resolución (las corridas de 10 km de Legg están igual de lejos). Siguiente paso condicionado del spec: Xu et al. (2006) con Nek5000 como verdad, o DOME a 5 km con Jackson para ver la tendencia con la resolución (Legg Figs. 14–15 tienen 2.5 km).
4. **κ_shear y ν_shear en la salida** (`kd_shear`, `kv_shear` están en el registro pero no en la serie ni en CF/Zarr; pendiente menor de la tarea 4) y **el balance de varianza** (`variance_budget=True`, término `phy_v`) en DOME e Ilicak, para medir en línea la mezcla física contra la numérica en vez del cálculo fuera de línea de aquí.
5. **Estudio (c): por qué Jackson no actúa con ν_v = 0.5** (punto 3) cuando la mezcla por Ri de Hallberg (2000) sí lo hace en el HIM de Legg.

### Costo de la mezcla por cizalla de Jackson et al. (2008) (`shear_mixing='jackson'`)

`scripts/shear_cost.py`: mediana de s/paso de 50 pasos tras 10 de calentamiento, 5 hilos, cada configuración en su propio proceso y alternando sin/con (3 rondas); `shear_mixing='jackson'` (Pr = 1) se pasa al constructor. El script verifica que el término esté activo (κ_shear máximo > 0, `update` llamado en los 50 pasos), mide el `update` del cierre aparte y suma las resoluciones implícitas verticales (`vertical_implicit`, ν y κ). Los dos casos ya llevan en su línea base lo que los distingue (BayIW cut z\*: `validation/bayiw_cut/cut.py::build_model("2layer", "bay", "zstar", 120.0)`; DOME 10 km híbrida: `kv_low` ν_v = 1e-4 y Smagorinsky 0.15), de modo que la razón aísla el efecto de Jackson.

| Caso | sin Jackson [s/paso] | con Jackson [s/paso] | razón mín / mediana / máx (3 rondas) | `update` del cierre | resoluciones implícitas |
|------|------|------|------|------|------|
| BayIW cut z\*, Δt 120 s | 0.335 | 0.428 | 1.270 / **1.275** / 1.280 | 81.0 ms (18.9 % del paso) | 12.8 → 12.9 ms |
| DOME 10 km híbrida, `kv_low` + Smagorinsky 0.15, Δt 600 s | 0.117 | 0.128 | 1.046 / **1.096** / 1.157 | 7.6 ms (6.0 %) | 1.7 → 3.1 ms |

- **1.30 es el umbral del spec (criterio 7) para perfilar, no un límite de aprobación.** BayIW queda justo debajo (mediana 1.275). Una primera medición de 2 rondas dio 1.314 y 1.275 (mediana 1.295), así que el umbral se tocó en una ronda; por eso se perfiló.
- **Perfil parcial.** El `update` del cierre (el kernel de columna de Jackson) explica ≈ 82 % del aumento en BayIW (81.0 de ≈ 92 ms) y ≈ 65 % en DOME (7.6 de ≈ 11.7 ms). Las resoluciones implícitas verticales no cambian en BayIW (12.8 → 12.9 ms) y suben 1.4 ms en DOME (ν no uniforme). **[Hipótesis]** los ≈ 11 ms restantes de BayIW (y ≈ 2.7 ms de DOME) son trabajo del controlador alrededor del `update` (densidad de capa, promedios a los centros, ν_shear en las caras); no se midieron por separado.
- κ_shear máximo: 1.1e-2 m²/s (BayIW) y 0.50 m²/s (DOME).
- Humo (`scripts/n1_smoke.py` contra `main`): 9/9 hashes idénticos con la opción apagada (por defecto).

## Mezcla diapicna de la capa límite de fondo por la fricción (spec 2026-10-30-bbl-drag-mixing, criterios 4, 5, 6 y 7)

- **Spec:** `docs/superpowers/specs/2026-10-30-bbl-drag-mixing-design.md`; lecturas de referencia `docs/design/bbl_refs/mom6_bbl_mixing.md` (MOM6, Legg Ap. A) y `docs/design/bbl_refs/entrainment_theory.md` (teoría de entrainment).
- **Qué agrega.** `--bbl lotw` (`bbl_mixing='lotw'`) suma al cierre una difusividad de capa de fondo Kd_BBL, el esquema de OM4 `add_LOTW_BBL_diffusivity` de MOM6 (`MOM_set_diffusivity.F90:1606-1789`): ley de la pared κ u\* z limitada por la rotación y por la energía que la fricción de fondo de pyOCEAN quita en el paso, γ × Σ_k r_k u_k² con γ = `bbl_effic` = 0.2. Solo difusividad de trazadores; no hay viscosidad ni número de Prandtl (MOM6 tampoco). El kernel coincide con MOM6 compilado (criterio 1, `validation/mom6_column/README.md`).
- **Convención de energía (decisión de Esteban, 2026-10-30).** pyOCEAN usa el trabajo físico de la fricción, ≈ C_d|U|³. `set_BBL_TKE` de MOM6 da 2 C_d|U|³ y √2 u\* para flujo uniforme (medido en MOM6 compilado: `validation/mom6_column/README.md`, "El factor de `set_BBL_TKE`"), clase (d). Por eso **γ = 0.2 aquí equivale a γ = 0.1 en MOM6/OM4**. Con energía de sobra no hay tope `kd_max` en el Kd de la pared (como MOM6, SD:1769-1772): en DOME Kd_BBL llega a 4.1 m²/s (prueba de costo).
- **Diseño de las corridas** (decisiones 1 y 4 del spec). La referencia (Legg y el DOME de MOM6) usa `DRAG_BG_VEL` = 0.05 m/s; la línea base de pyOCEAN, 0. Por eso las corridas con BBL usan `--drag-bg-vel 0.05`, y cada una tiene un **control con la misma U_bg y sin BBL**: la dirección del criterio 4 se juzga contra ese control. La principal de Jackson (U_bg = 0) se reporta también, para separar el efecto de U_bg.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05            # control
python validation/dome/run_pyocean.py 10 hybrid --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05 --bbl lotw   # la principal
python validation/dome/run_pyocean.py 10 zstar  --kv-low --smag 0.15 --shear jackson --drag-bg-vel 0.05 [--bbl lotw] # también sigma
python validation/dome/compare.py 10 --theory --only "_jackson,mitgcm_z_legg"
python validation/dome/figures.py 10 --before-after pyocean_hybrid_kv_low_smag0.15_jackson_ubg0.05,pyocean_hybrid_kv_low_smag0.15_jackson_ubg0.05_bbl,pyocean_hybrid_kv_low_smag0.15_jackson,mitgcm_z_legg --sub bbl
python validation/dome/figures.py 10 --no-anim --sub bbl --runs mitgcm_z_legg,pyocean_hybrid_kv_low_smag0.15_jackson,pyocean_hybrid_kv_low_smag0.15_jackson_ubg0.05,pyocean_hybrid_kv_low_smag0.15_jackson_ubg0.05_bbl,pyocean_zstar_kv_low_smag0.15_jackson_ubg0.05,pyocean_zstar_kv_low_smag0.15_jackson_ubg0.05_bbl,pyocean_sigma_kv_low_smag0.15_jackson_ubg0.05,pyocean_sigma_kv_low_smag0.15_jackson_ubg0.05_bbl
```

Salidas en `RUNS/10km/pyocean_<coord>_kv_low_smag0.15_jackson_ubg0.05[_bbl]/`; bitácoras en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bbl_logs/` (cadena `chain.sh`, 6 corridas, rc = 0, 2026-10-30 23:57 – 2026-10-31 01:30, en `c8207f4`; las tres con BBL se repitieron tras la corrección I1 con `chain_I1.sh`, rc = 0, 2026-10-31 01:58 – 02:44, en `323c43b`; las salidas anteriores quedan en `RUNS/10km/*_bbl_preI1/`, sus bitácoras en `*_preI1.log` y sus figuras en `RUNS/figs/bbl_preI1/`).

### Resultados (10 km, 50 días, Δt 600 s, 5 hilos; métricas de los días 30–50; trazador por clase en la segunda mitad, días 25–50)

**Números después de la corrección I1 de la revisión final (clase a, 2026-10-31):** N² en las interfaces de capas fusionadas sale ahora de las densidades fusionadas (como GLS y Jackson); antes se reusaba el N² del modelo a través de una capa de 1 mm, que lo duplicaba. Movimiento: híbrida + BBL b_f −11.54 → −11.86, eventos del limitador 32 → 0, T(x′ = −300) 20.8 → 18.0 Sv, E de la esquina 1.87e-4 → 2.66e-4; z\* + BBL cambia < 0.02 en b_f; σ + BBL es idéntica bit a bit (en σ no hay capas que fusionar). Los controles no llevan BBL y no cambian.

**Números después de la corrección del N² de fondo (clase a, 2026-11-01, aprobada por Esteban).** La BBL toma ahora en las dos interfaces interiores más bajas el N²_bot de bulto de `find_N2` de MOM6 (`n2_bottom_bulk`, igual a MOM6 compilado; `validation/mom6_column/README.md`). Las tres corridas con BBL se repitieron con `bbl_logs/chain_n2bot.sh` (rc = 0, 2026-11-01 09:22 – 10:08, en `9a53be2`); las salidas anteriores quedan en `RUNS/10km/*_bbl_preN2bot/` y sus bitácoras en `*_preN2bot.log`. **Las tablas y la lectura de abajo son de antes de esta corrección.** El cambio:

| corrida | b_f [1e-3 m s⁻²] | E esquina | T(x′ −200 / −300) [Sv] | α_E ec. 12 | clase con más trazador x′ −300 / −100 | Y′(−200 / −300 / −500 / −800) [km] | trunc / lim |
|---|---|---|---|---|---|---|---|
| híbrida + BBL, antes | −11.86 | 2.66e-4 | 16.1 / 18.0 | 6.40e-4 | −14…−13 / −17…−16 | −62 / −81 / −105 / −122 | 0 / 0 |
| **híbrida + BBL, después** | **−12.47** | 2.34e-4 | **17.4 / 19.2** | **7.99e-4** | −15…−14 / −17…−16 | −58 / −77 / −99 / −133 | 0 / 0 |
| z\* + BBL, antes → después | −8.76 → −8.93 | 1.55e-3 → 1.54e-3 | 19.1 / 20.7 → 18.7 / 20.0 | 1.48e-3 → 1.45e-3 | −11…−10 → −12…−11 / −15…−14 | −41 / −52 / −68 / −127 → −120 | 0 / 0 |
| σ + BBL, antes → después | −10.18 → −10.28 | 5.82e-4 → 4.79e-4 | 20.5 / 21.4 → 20.8 / 21.0 | 1.38e-3 → 1.38e-3 | −12…−11 → −11…−10 / −14…−13 | −70 / −85 / −101 / −131 → −68 / −86 / −106 / −117 | 0 / 0 |

- **El efecto de la BBL sobre b_f es menor en las tres coordenadas.** En la híbrida, la BBL cierra ahora 0.23 de los 1.63 × 10⁻³ m s⁻² entre el control (−12.70) y la isopicna de Legg (−11.07): 14 % (antes 52 %). La clase con más trazador en x′ = −300 km vuelve un escalón hacia lo denso.
- **El transporte y α_E se acercan a la verdad.** T(x′ = −200 / −300) 16.1 / 18.0 → 17.4 / 19.2 Sv (no hidrostática 22.5 / 20.6 dig.; el control 17.7 / 19.0) y α_E 6.40e-4 → 7.99e-4 (1.03e-3 dig.). Antes la BBL alejaba el transporte; ahora casi no lo mueve.
- **[Hipótesis, no probada]** En una pluma bien mezclada cerca del fondo el N² local de la interfaz de fondo es pequeño; el bulto lo sube con el salto de la interfaz de arriba, la ley de la pared gasta su energía más abajo y queda menos Kd para la parte alta de la pluma.
- **Clasificación:** el cambio es la corrección de un error nuestro (a), no un ajuste: pyOCEAN hace ahora lo que hace MOM6. La distancia restante a −11.07 sigue repartida como en la lectura de abajo (esquema y energía c, resolución de la interfaz c, entrada de la híbrida a, geometría c), ahora con más peso.
- Robustez sin cambio: 0 truncamientos y 0 eventos del limitador en las tres; costo 19.6 / 16.7 / 17.6 s por día simulado (antes 19.1 / 17.0 / 18.1).

**Números después de las correcciones de θ_bc (`e1ffbbb`) y de la cara abierta (`ca36b59`), 2026-11-03.** La principal
(híbrida + Jackson + U_bg 0.05 + BBL) se repitió dos veces el 2026-11-03 (`smag-openface-report.md` del registro de la
cadena regional; salidas `RUNS/10km/pyocean_hybrid_kv_low_smag0.15_jackson_ubg0.05_bbl_head2f80d5a/` y `…_bbl_openface/`):

| corrida | b_f [1e-3 m s⁻²] | E esquina (pluma / todo) | T(x′ −200 / −300) [Sv] | α_E ec. 12 | clase x′ −300 / −100 | Y′(−200 / −300 / −500 / −800) [km] | trunc / lim | s/día |
|---|---|---|---|---|---|---|---|---|
| tabla de arriba (`9a53be2`, antes de `e1ffbbb`) | −12.47 | 2.34e-4 / 4.29e-4 | 17.4 / 19.2 | 7.99e-4 | −15…−14 / −17…−16 | −58 / −77 / −99 / −133 | 0 / 0 | 19.6 |
| `2f80d5a` (con la corrección de θ_bc) | −12.94 | 3.49e-4 / 8.59e-4 | 17.1 / 18.4 | 4.29e-4 | −15…−14 / −17…−16 | −65 / −89 / −131 / −153 | 0 / 0 | 19.2 |
| **`ca36b59` (+ cara abierta; vigente)** | **−12.79** | 3.70e-4 / 9.06e-4 | 17.3 / **19.2** | **4.15e-4** | −15…−14 / −16…−15 | −68 / −91 / −144 / −162 | 0 / 0 | 18.4 |

- La corrección de θ_bc alejó la principal de Legg (−12.47 → −12.94; α_E 7.99e-4 → 4.29e-4; trayectoria 7–20 km más
  abajo del talud); la de la cara abierta la acerca un poco (−12.79, 8 % de la distancia a −11.07; T(x′ −300) +0.8 Sv
  hacia 20.6). Las dos son correcciones de errores nuestros (a), no ajustes.
- **Los controles (sin BBL) y las corridas z\* y σ no se repitieron**: el efecto de la BBL contra su control (la
  lectura de abajo) es de antes de `e1ffbbb` y no se puede releer con estos números.

**Contra la isopicna de Legg a 10 km (referencia principal) y la no hidrostática (secundaria).** "dig." = leído de las figuras de Legg et al. (2006) (`legg2006_nonhydro.json`). La isopicna de 10 km de Legg tiene la misma física que la principal: ν_v = 1e-4, mezcla por Ri de Hallberg (2000), Smagorinsky y mezcla BBL por arrastre (Ap. A, γ = 0.2, U_bg = 5 cm/s; Legg Ap. A y p. 77; `bbl_refs/mom6_bbl_mixing.md` §3). E de la esquina es la ec. 14 y α_E la ec. 12 (sección de Jackson); diferencias de E menores que ×2 no son significativas.

| corrida | b_f [1e-3 m s⁻²] | E esquina ec. 14, U_rms pluma / todo | T_out, T_in [Sv] | T(x′ = −200 / −300) [Sv] | α_E ec. 12 (−180…−50 km) | b medio del trazador x′ = −300 / −100 [1e-3 m s⁻²] | clase con más trazador x′ = −300 / −100 |
|---|---|---|---|---|---|---|---|
| Legg 10 km isopicnal (ν_v 1e-4 + Ri + BBL; **referencia principal**) | **−11.07** (dig.) | 9.4e-4 (dig.) | — | — | — | — | — |
| Legg no hidrostático 0.5 km (secundaria) | −8.75 (dig.) | 8.6e-4 (texto, 0.5 km) | — | 22.5 / 20.6 (dig.) | 1.03e-3 (dig.) | −8.99 (dig.) / — | −10…−9 (dig.) / — |
| MITgcm z legg | −10.85 | 1.04e-3 / 1.96e-3 | 14.83, 12.66 | 20.4 / 22.1 | 6.14e-4 | — | −14…−13 / −16…−15 |
| híbrida + Jackson, U_bg 0 (principal de Jackson) | −12.84 | 4.88e-4 / 1.13e-3 | 12.46, 9.83 | 15.9 / 17.9 | 4.77e-4 | −12.98 / −13.75 | −17…−16 / −19…−18 |
| híbrida + Jackson, U_bg 0.05 (**control**) | −12.70 | 3.51e-4 / 7.42e-4 | 11.80, 9.96 | 17.7 / 19.0 | 9.95e-4 | −12.75 / −13.57 | −17…−16 / −19…−18 |
| **+ BBL (principal)** | **−11.86** | 2.66e-4 / 4.94e-4 | 12.59, 11.38 | 16.1 / 18.0 | 6.40e-4 | **−12.17** / −13.36 | **−14…−13** / −17…−16 |
| z\* + Jackson, U_bg 0.05 (control) | −9.04 | 1.12e-3 / 1.64e-3 | 13.86, 10.39 | 18.6 / 20.2 | 1.28e-3 | −9.40 / −10.68 | −12…−11 / −18…−17 |
| + BBL | −8.76 | 1.55e-3 / 2.37e-3 | 14.17, 10.63 | 19.1 / 20.7 | 1.48e-3 | −9.35 / −10.42 | −11…−10 / −15…−14 |
| σ + Jackson, U_bg 0.05 (control) | −11.39 | 5.39e-4 / 8.17e-4 | 11.67, 9.53 | 19.0 / 21.0 | 1.23e-3 | −11.48 / −12.45 | −14…−13 / −17…−16 |
| + BBL | −10.18 | 5.82e-4 / 8.22e-4 | 11.83, 9.79 | 20.5 / 21.4 | 1.38e-3 | −10.57 / −11.95 | −12…−11 / −14…−13 |

Todas con `kv_low` (ν_v = 1e-4), Smagorinsky 0.15 y Jackson Pr = 1. La entrada es de 8.38 Sv. "b medio del trazador": primer momento de la distribución por clase en la columna (`figures.mean_class_b`).

**Trayectoria, robustez y costo.**

| corrida | Y′(x′ = −200 / −300 / −500 / −700 / −800 km) [km] | centroide día 50 [m] | máx. diario de \|u\| en caras gruesas, días 30–50 (media) / día 50 [m/s] | trunc / lim / visc | reloj [s] (s por día) |
|---|---|---|---|---|---|
| Legg 10 km isopicnal | −90 / −95 / −100 / −120 / — (dig.) | — | — | — | — |
| MITgcm z legg | −52 / −72 / −107 / — / −204 | 1433 | 2.02 / — | — | (4.7) |
| híbrida + Jackson, U_bg 0 | −69 / −86 / −105 / −137 / −152 | 1428 | 2.45 / 2.87 | 0 / 0 / 57 993 | 1055 (21.1) |
| híbrida control U_bg 0.05 | −71 / −87 / −108 / −135 / −154 | 1423 | 2.52 / 2.74 | 0 / 0 / 67 507 | 1031 (20.6) |
| **+ BBL** | −62 / −81 / −105 / −127 / −122 | 1388 | 2.27 / 2.74 | **0 / 0** / 88 474 | 957 (19.1) |
| z\* control | −41 / −53 / −70 / −93 / −118 | 970 | 2.21 / 3.27 | 0 / 0 / 1 113 | 844 (16.9) |
| + BBL | −41 / −52 / −68 / −80 / −127 | 958 | 2.11 / 3.23 | 0 / 0 / 281 | 852 (17.0) |
| σ control | −71 / −91 / −119 / −140 / −167 | 1337 | 2.63 / 3.64 | 0 / 0 / 0 | 865 (17.3) |
| + BBL | −70 / −85 / −101 / −121 / −131 | 1262 | 2.49 / 3.87 | 0 / 0 / 0 | 903 (18.1) |

"Máx. diario de \|u\|": máximo de \|u\| y \|v\| en caras con las dos celdas ≥ 1 m en cada registro diario, como en la sección de Jackson (cálculo sobre los `series.npz`, script no versionado). El rango de T queda dentro del de las fuentes en todas (la principal −4.986 a 4.986 °C: con BBL no queda agua de entrada de −5 °C sin mezclar en ninguna celda; σ + BBL −4.972, z\* + BBL −4.976); s ∈ [−8e-26, 1] en la principal y s máx. 0.998 en z\* + BBL.

**Figuras** (fuera del repo, `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/figs/bbl/`):
- `dome_10km_before_after.png`: la principal frente a su control: tinte de fondo los días 10/20/30/50, trayectoria, transporte y trazador por clase en x′ = −300 km, con la principal de Jackson (U_bg 0), MITgcm legg, la isopicna de 10 km de Legg y la verdad no hidrostática digitalizada;
- `dome_10km_bottom_dye_before_after.gif` (y `.mp4`): animación diaria del tinte de fondo de control, principal, Jackson U_bg 0 y MITgcm legg;
- `dome_10km_legg_scorecard.png`, `dome_10km_legg_truth.png`, `dome_10km_tracer_by_class.png`, `dome_10km_path_transport.png`, `dome_10km_bottom_dye.png` de las 8 corridas;
- `dome_10km_plume_theory.png`: el diagnóstico teórico (criterio 5) de las 8 corridas.

### Lectura y clasificación

1. **Criterio 4 (principal contra la isopicna de Legg a 10 km): dirección ✅, magnitud 52 % del camino en b_f; robustez ✅ tras la corrección I1 (0 truncamientos, 0 eventos del limitador); trazador por clase y transporte, mezclados.**
   - **b_f:** control −12.70 → principal −11.86; la isopicna de Legg da −11.07. La BBL cierra 0.84 de los 1.63 × 10⁻³ m s⁻² que faltaban (52 %; antes de I1, 71 %); queda 0.79 (7 %) más densa. Contra la no hidrostática (−8.75 dig.) cierra 21 % (3.95 → 3.11).
   - **U_bg sola** (Jackson U_bg 0 → control): b_f −12.84 → −12.70, T(x′ = −200) 15.9 → 17.7 Sv y α_E 4.77e-4 → 9.95e-4. El efecto sobre b_f es pequeño; el de α_E no: U_bg aumenta la fricción en las zonas lentas de la cuña de entrada [inferencia, no aislado por x].
   - **Trazador por clase:** la clase con más trazador en x′ = −300 km pasa de −17…−16 a −14…−13 (no hidrostática −10…−9 dig.; antes de I1 −13…−12) y el b medio de −12.75 a −12.17 (−8.99 dig.). La distribución tiene un pico angosto (figura `before_after`), como la de Legg, corrido 4 × 10⁻³ m s⁻² hacia lo denso. **En x′ = −100 km la clase de entrada sin mezclar (−19…−18) deja de dominar** (−17…−16; antes solo lo lograba Jackson Pr = 0, cuya pluma no baja el talud): la BBL mezcla el agua de la entrada ya en el canal (T_in 9.96 → 11.38 Sv contra 8.38 Sv que entran).
   - **Transporte: se aleja.** T(x′ = −200 / −300) 17.7 / 19.0 → 16.1 / 18.0 Sv (no hidrostática 22.5 / 20.6 dig.; antes de I1 19.0 / 20.8, que coincidía con la verdad en −300 km). α_E (ec. 12) baja 9.95e-4 → 6.40e-4 (verdad 1.03e-3 dig.). Con la BBL la pluma es más liviana pero lleva menos agua aguas abajo de la esquina; **[hipótesis]** la mezcla en el canal (T_in sube 1.4 Sv) reduce Δb en la esquina y con él el entrainment de interfaz que hace Jackson (Ri_B cerca 5.63 → 8.82, tabla de teoría). No se probó.
   - **E de la esquina: 3.51e-4 → 2.66e-4** (isopicna de Legg 9.4e-4; ×3.5 debajo, fuera de la banda ×2). T_out sube (11.80 → 12.59 Sv) y T_in sube más (9.96 → 11.38), porque parte de la dilución pasa al canal, aguas arriba de la ventana de la ec. 14 (T_out − T_in baja de 1.84 a 1.21 Sv). Clase (c), definición de la métrica con un canal de 100 km (el de Legg mide 50 km); **[hipótesis]** que la BBL en el canal de Legg diluya menos, porque su entrada es más corta. No se probó.
   - **Trayectoria** contra la isopicna de Legg (−90 / −95 / −100 / −120 km en x′ = −200 / −300 / −500 / −700): control a 19 / 8 / 8 / 15 km, principal a 28 / 14 / 5 / 7 km. Se acerca aguas abajo (x′ ≤ −500) y se aleja cerca de la esquina; la pluma de la principal baja menos en los primeros 300 km (centroide del día 50 1423 → 1388 m; MITgcm legg 1433).
   - **Robustez (cambió con I1):** 0 truncamientos y **0 eventos del limitador** (antes de I1: 32), `n_visc_limited` 67 507 → 88 474 (antes 105 522). El máximo diario de \|u\| en caras gruesas promedia 2.27 m/s en los días 30–50 (control 2.52) y 2.5–2.8 m/s en los días 46–50 (control 2.5–2.7); las capas de fondo de 3.3–3.9 m/s de la cuenca profunda que se veían antes de I1 desaparecieron. **[Inferencia]** venían del N² duplicado en las interfaces fusionadas sobre capas de fondo delgadas de la híbrida (menos Kd_BBL ahí donde la capa delgada estaba encima de una gruesa, más donde no) — la corrida repetida es la única evidencia. El criterio "sin nuevos eventos del limitador" del spec **se cumple**.

   **Lo que queda contra −11.07, con el diagnóstico (punto 3):**
   - **(c) Esquema y energía.** Legg pone toda la energía en la segunda capa desde el fondo con escala 0.4 u\*/f (A.2); LOTW la reparte con la ley de la pared y escala u\*/f hasta 200 m (`bbl_refs/mom6_bbl_mixing.md` §3). La E_BBL de Legg (A.1) se escribe ρ₀ c_D|u|³ y el texto dice "= ρ₀ u\*³", lo que no es consistente con u\* = √c_D|u| **[verificar en el PDF]**; con la convención física de pyOCEAN, γ = 0.2 es la mitad de la energía del γ = 0.2 de MOM6. No se cambia γ (anti-meta: no se ajusta).
   - **(c) Resolución de la interfaz aguas abajo.** En −500…−200 km la principal tiene 1.1 interfaces con Ri < 0.25 de 7.9 en la banda de la pluma, menos que los 2–4 puntos que pide Jackson (§6) (control 2.1 de 11.8).
   - **(a) Entrada de la híbrida.** Mejor que antes en x′ = −100 km (punto anterior), pero el agua de la entrada sigue siendo la más densa de la columna; no separado.
   - **(c) Geometría** de MOM6 (canal de 100 km, esponja) contra la de Legg (50 km, Orlanski), como en el resto del benchmark.
2. **Criterio 7 (z\* y σ).**
   - **z\*:** la BBL casi no cambia b_f (−9.04 → −8.76) ni la trayectoria, y sube E de la esquina (1.12e-3 → 1.55e-3) y α_E (1.28e-3 → 1.48e-3). La z\* ya era más liviana que la isopicna de Legg (por mezcla numérica, (c) **[hipótesis]** de la sección de Jackson, no medida). Con niveles de 144 m en la cuenca solo 1–2 interfaces caen dentro de la escala de decaimiento min(u\*/|f|, 200 m) **[inferencia]**, así que la BBL tiene poco donde actuar. s máx. baja a 0.998: no queda agua de entrada sin mezclar ni en la celda de entrada. Clase (c).
   - **σ:** b_f −11.39 → −10.18, pasa de −11.07 y queda 0.89 × 10⁻³ m s⁻² más liviana que la isopicna de Legg (hacia la no hidrostática). E 5.39e-4 → 5.82e-4, T(x′ = −200) 19.0 → 20.5 Sv, clase con más trazador en x′ = −300 −14…−13 → −12…−11; la trayectoria se acerca a la isopicna de Legg aguas abajo (−119 → −101 en x′ = −500, −140 → −121 en −700). En σ la capa de fondo tiene grosor fijo D/25 (≤ 144 m) en toda la cuenca, de modo que la BBL alcanza siempre a la pluma completa **[inferencia]**. Clase (c).
   - En las tres coordenadas la BBL aligera la pluma; el tamaño del efecto (z\* 0.28, híbrida 0.84, σ 1.21 × 10⁻³ m s⁻²) sigue a cuántas interfaces caen dentro de la capa de fondo **[inferencia]**. No hay truncamientos ni eventos del limitador en z\* y σ.
3. **Qué dice la teoría** (criterio 5; `theory.py`, tabla de `compare.py --theory`; ventanas cerca −180…−50 km / lejos −500…−200 km). Ri_B y Fr dependen del umbral de la pluma y solo se comparan entre corridas, no con el 0.4–0.5 de Legg (pie de la tabla de `compare.py`).

   | corrida | h [m] | Ri_B | Fr | celdas Ri < 0.25 / banda (cerca ; lejos) | Ri_g mediana | Ekman de fondo / bajada de la pluma | u_pluma/(g′s/f) |
   |---|---|---|---|---|---|---|---|
   | MITgcm z legg | 428 / 468 | 9.07 / 18.92 | 0.34 / 0.24 | 0.6 / 2.9 ; 0.6 / 3.5 | 3.08 / 6.82 | 0.04 / 0.02 | 0.71 / 0.73 |
   | híbrida + Jackson, U_bg 0 | 447 / 372 | 6.53 / 4.37 | 0.40 / 0.49 | 5.1 / 16.6 ; 2.3 / 12.1 | 0.36 / 0.47 | 0.11 / 0.20 | 0.81 / 0.72 |
   | híbrida control U_bg 0.05 | 399 / 373 | 5.63 / 5.25 | 0.44 / 0.45 | 4.7 / 15.9 ; 2.1 / 11.8 | 0.37 / 0.53 | 0.08 / 0.04 | 0.82 / 0.67 |
   | + BBL | 408 / 414 | 8.82 / 7.75 | 0.34 / 0.37 | 2.9 / 11.6 ; 1.1 / 7.9 | 0.51 / 0.72 | 0.08 / — | 1.00 / 0.72 |
   | z\* control / + BBL | 669 / 717 → 614 / 711 | 10.51 / 8.54 → 9.13 / 9.09 | 0.31 / 0.35 → 0.33 / 0.34 | 0.1 / 4.4 ; 0.2 / 4.9 → 0.1 / 4.1 ; 0.5 / 4.8 | 2.10 / 1.79 → 2.20 / 1.73 | 0.04 / 0.08 → 0.06 / 0.03 | 0.84 / 0.59 → 0.93 / 0.63 |
   | σ control / + BBL | 457 / 572 → 416 / 576 | 3.56 / 4.67 → 3.72 / 5.20 | 0.54 / 0.48 → 0.52 / 0.45 | 1.4 / 9.8 ; 1.2 / 8.8 → 2.2 / 9.0 ; 1.9 / 9.1 | 0.53 / 0.49 → 0.56 / 0.50 | 0.07 / 0.06 → 0.06 / 0.03 | 0.92 / 0.64 → 0.87 / 0.64 |

   - **Las leyes de laboratorio no predicen DOME.** Ellison–Turner (Xu et al. 2006, ecs. 3–4) da E = 0 en todas estas corridas, porque Ri_B ≥ 0.8; la ley de Xu (ec. 16) con la mediana de Ri_g también da 0 (Ri_g ≥ 0.25). El α_E medido es 1e-4–1.5e-3. Legg §3.3 (p. 85) dice que el entrainment rotante es "considerablemente menor" que el de laboratorio (`bbl_refs/entrainment_theory.md`, "Qué dicen para DOME"). Ninguna ley local incluye f **[inferencia]**.
   - **La pluma es subcrítica y geostrófica, con y sin BBL.** Fr 0.34–0.45 en la híbrida (Ezer: 0.4–0.6 aguas abajo, `entrainment_theory.md` "Qué dicen para DOME"); la velocidad a lo largo del talud sigue a Nof (1983), u_t = g′s/f (Ezer ec. 2): 0.67–1.00 de u_t. La BBL no cambia el régimen.
   - **La BBL quita trabajo a Jackson en la interfaz.** Las interfaces con Ri < 0.25 bajan de 4.7 a 2.9 (cerca) y de 2.1 a 1.1 (lejos), la mediana de Ri_g sube de 0.37 a 0.51 y Ri_B cerca de 5.63 a 8.82 (ΔU 0.62 → 0.50 m/s). **[Inferencia]** al mezclar la pluma desde abajo cae el número de interfaces de la banda (15.9 → 11.6) y la cizalla de la interfaz; aguas abajo quedan menos de los 2–4 puntos de Ri bajo que pide Jackson (§6; `entrainment_theory.md` Leyes 3), así que ahí la mezcla de interfaz no está resuelta (c).
   - **Ekman de fondo.** El exceso talud abajo en los 10 m de fondo queda en 8 % del transporte talud abajo de la pluma (cerca), igual que el control (antes de I1 bajaba a 4 %); lejos `theory.py` no da valor ('—'). Ezer da 22 % (`entrainment_theory.md`, "Qué dicen para DOME"); MITgcm legg 2–4 %. Tras I1 la BBL ya no cambia esta fracción; con esta medida de una sola corrida (`theory.py`, docstring de `bottom_ekman_transport`) es una cota inferior.
   - **Lo que la teoría no explica.** Ninguna de las magnitudes de la tabla separa la BBL de la isopicna de Legg: a 10 km su Ri_B, Fr y Nof no están publicados. El diagnóstico explica por qué las leyes locales no sirven como meta y señala la interfaz no resuelta aguas abajo; la diferencia restante en b_f (0.79) queda atribuida a esquema y energía (c) y resolución (c), sin medir por separado.
4. **Criterio 6 (costo, subsección siguiente): ✅ tras la corrección de costo del 2026-10-31, DOME 1.003×, BayIW cut 1.038× (antes 1.027× y 1.071× ❌).** En producción (reloj de la corrida completa, incluye la salida y una trayectoria distinta) la principal tardó 957 s contra 1031 del control (0.93×), z\* 1.01× y σ 1.04×.

### DOME 5 km híbrida `kv_low` + Smagorinsky 0.15 (con y sin Jackson; 2026-10-30)

Corridas lanzadas durante el plan BBL, sin BBL y con U_bg = 0, Δt 600 s (el mismo que a 10 km; salida en `RUNS/5km/pyocean_hybrid_kv_low_smag0.15[_jackson]/`):
- **Sin Jackson aborta** a los ~2 min de corrida con 1004 truncamientos CFL (capa 1, u = −4.21 m/s en una cara de 4.7 m). A 10 km la misma configuración completaba con 61 truncamientos. Clase (b), falta de mezcla por cizalla, como a 10 km.
- **Con Jackson completa los 50 días, pero no es robusta:** 469 truncamientos, 232 331 eventos del limitador y `n_visc_limited` 4 230 005. b_f −12.28e-3 (10 km −12.84; isopicna de Legg a 10 km −11.07), E de la esquina 3.64e-4 / 6.09e-4, T(x′ = −200 / −300) 17.8 / 20.9 Sv, α_E (ec. 12) 6.48e-4, clase con más trazador en x′ = −300 / −100 −13…−12 / −18…−17, Y′ −81 / −105 / −138 / −170 km en x′ = −200 / −300 / −500 / −800; 98 s por día (10 km: 21). La tendencia con la resolución va en la dirección de Legg (b_f más liviano a 5 km), pero los truncamientos la invalidan como resultado. **[Hipótesis]** Δt 600 s a 5 km deja la mitad del margen CFL de 10 km; no se probó con Δt 300 s. Abierto, sin clasificar.

### Qué debe mejorar pyOCEAN (BBL en DOME)

1. **(resuelto por la corrección I1) Los 32 eventos del limitador y las capas de fondo de 3–4 m/s de la principal en los días 46–50**: desaparecieron al calcular N² con las densidades fusionadas (clase a). Queda abierto, para el criterio 4, **por qué la BBL baja el transporte aguas abajo** (T(x′ = −300) 19.0 → 18.0 Sv, α_E 9.95e-4 → 6.40e-4) mientras aligera la pluma: medir Δb y ΔU en la esquina con y sin BBL.
2. **Kd_BBL en la salida** (`kd_bbl` está en el registro pero no en la serie ni en CF/Zarr, como `kd_shear`) y el **balance de varianza** (`variance_budget=True`, `phy_v`) para medir en línea cuánto de la mezcla de la pluma es BBL, Jackson o numérica.
3. **(resuelto 2026-10-31) Costo en BayIW (criterio 6, 1.071× → 1.038×):** la causa no era sobre todo la fase 7 sino el pegamento NumPy alrededor del kernel; ver la subsección de costo. Queda la fase 7 en sí (espesor húmedo y densidad, ≈ 4 ms en BayIW), que podría reutilizar la densidad que ya calcula el paso.
4. **E de la esquina con la dilución del canal** (punto 1): reportar también T_in contra la entrada nominal (8.38 Sv), o medir E desde la boca del canal, para que la métrica no esconda la mezcla que la BBL hace aguas arriba.
5. **DOME 5 km robusto** (subsección anterior): la misma configuración con Δt 300 s y con BBL, para la tendencia con la resolución (Legg Figs. 14–15) sin truncamientos.
6. **Variante con la forma de una capa de Legg (A.2)** o `add_drag_diffusivity` de MOM6, solo si un caso lo pide (no meta del spec): separaría la parte (c) de esquema del resultado.

### Costo de la mezcla de la capa límite de fondo por la fricción (`bbl_mixing='lotw'`)

`scripts/bbl_cost.py` (plan 2026-10-30-bbl-drag-mixing, tarea 5; criterio 6 del spec: razón ≤ 1.05): mediana de s/paso de 50 pasos tras 10 de calentamiento, 5 hilos, cada configuración en su propio proceso y alternando sin/con (3 rondas). BayIW cut z\* lleva `bottom_drag='quadratic'` en ambos modos (la mezcla lo exige); DOME 10 km híbrida lleva `kv_low`, Smagorinsky 0.15 y Jackson en ambos modos, de modo que la razón aísla `bbl_mixing='lotw'`. El script verifica `BBLAddedClosure` y Kd_BBL máximo > 0, y mide el `update` del cierre (la parte BBL = `update` externo menos el del cierre envuelto) y `Driver._ale_bed_energy`.

| Caso | sin BBL [s/paso] | con BBL [s/paso] | razón mín / mediana / máx¹ | parte BBL del `update` | energía de la fricción | Kd_BBL máx |
|------|------|------|------|------|------|------|
| BayIW cut z\*, Δt 120 s | 0.348 | 0.360 | 1.034 / **1.038** / 1.043 | 5.0 ms (1.4 %) | 2.9 ms (0.8 %) | 6e-18 m²/s |
| DOME 10 km híbrida, `kv_low` + Smag 0.15 + Jackson, Δt 600 s | 0.1189 | 0.1199 | 0.982 / **1.003** / 1.018 | 1.2 ms (1.0 %) | 0.4 ms (0.3 %) | 4.1 m²/s |

¹ La razón reportada es la mediana de las razones por ronda (cada ronda: s/paso con BBL entre s/paso sin BBL), no la razón de las medianas de s/paso; la razón de las medianas da 0.3604/0.3484 = 1.034 en BayIW y 0.1199/0.1189 = 1.008 en DOME; las dos cumplen igual. Medido tras la corrección de costo del 2026-10-31 (registro `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bbl_logs/cost_fix1.log`; antes, tras I1: BayIW 0.350 / 0.375 s, 1.071×, parte BBL 13.7 ms; DOME 1.027×, 2.9 ms; `cost_postI1.log`).

- **Criterio 6 ✅ en los dos casos tras la corrección de costo del 2026-10-31 (rama `bbl-cost`).** Perfil (cProfile, 30 pasos con y sin BBL, diferencia por paso) antes de la corrección, ≈ 25 ms en BayIW: kernel 5.9 ms; en el `update` del envoltorio ≈ 7.7 ms de NumPy (copia de las columnas húmedas de dz y ρ con `[:, wet]` ≈ 4.6 ms, ceros y devolución ≈ 1.2 ms); en la fase 7 del controlador `_wet_thickness` 2.5, densidad 1.8 y las velocidades en los centros ≈ 3 ms que el cierre constante no lee; energía de la fricción 2.9; κ como arreglo 1.1. La hipótesis de la fase 7 era cierta solo en parte. Corrección, bit a bit (hash de 30 pasos de BayIW y DOME con BBL igual a `main`): (i) el kernel recibe la malla completa y salta las columnas secas antes de leer ρ (un ρ no finito solo se rechaza en una columna húmeda, `test_dry_columns_ignore_rho`); (ii) los arreglos de trabajo del kernel se reservan una vez por bloque de 64 columnas, no por columna; (iii) el controlador arma u, v en los centros solo si el cierre los lee (`Closure.reads_velocity`, `test_velocity_only_for_closures_that_read_it`). Con eso la parte BBL del `update` baja de 13.7 a 5.0 ms (BayIW) y de 2.9 a 1.2 ms (DOME). Historia: tras I1 (2026-10-31; antes 1.092 y 1.031): la diferencia en BayIW es ≈ 25 ms/paso, de los que el `update` de la BBL es ≈ 14 ms (antes ≈ 31 y 17 ms). Perfil anterior a I1 (cProfile, 30 pasos, con y sin BBL, diferencia por paso en tiempo acumulado): el `update` de `BBLAddedClosure` +16.6 ms (kernel `lotw_bbl_columns` 4.5, N² de la columna 2.9, el resto NumPy del envoltorio), `_wet_thickness` +2.5, densidad `eos.density` +1.8, energía de la fricción +2.9, `tracer_diffusivity` +1.2. **[Hipótesis, con el perfil a favor]** con el cierre constante y sin cizalla el controlador no ejecutaba la fase 7 (densidad, N², espesor húmedo); `bbl_mixing` la activa, y ese costo fijo no depende de la magnitud de Kd_BBL. En DOME esa fase ya corre por Jackson y el costo es solo la parte BBL (razón 1.03).
- **BayIW no ejercita el término**: en la corrida corta de la prueba la fricción casi no hace trabajo (Kd_BBL máx 6e-18 m²/s, > 0 solo en sentido estricto); el costo medido es el de la ruta completa pero el kernel no ve mezcla real. DOME sí lo ejercita (4.1 m²/s).
- Humo (`scripts/n1_smoke.py` contra `main`): 9/9 hashes idénticos con la opción apagada (por defecto).

## Referencias

- Legg, S., Hallberg, R. W., Girton, J. B. (2006). Comparison of entrainment in overflows simulated by z-coordinate, isopycnal and non-hydrostatic models. *Ocean Modell.* 11, 69–97. [10.1016/j.ocemod.2004.11.006](https://doi.org/10.1016/j.ocemod.2004.11.006)
- Hallberg, R. (2000). Time integration of diapycnal diffusion and Richardson number–dependent mixing in isopycnal coordinate ocean models. *Mon. Wea. Rev.* 128, 1402–1419. [verificar]
- Jackson, L., Hallberg, R., Legg, S. (2008). A parameterization of shear-driven turbulence for ocean climate models. *J. Phys. Oceanogr.* 38, 1033–1053. [10.1175/2007JPO3779.1](https://doi.org/10.1175/2007JPO3779.1)
- Xu, X., Chang, Y. S., Peters, H., Özgökmen, T. M., Chassignet, E. P. (2006). Parameterization of gravity current entrainment for ocean circulation models using a high-order 3D nonhydrostatic spectral element model. *Ocean Modell.* 14, 19–44. [10.1016/j.ocemod.2006.02.006](https://doi.org/10.1016/j.ocemod.2006.02.006)
- Pietrzak, J. (1998). The use of TVD limiters for forward-in-time upstream-biased advection schemes in ocean modeling. *Mon. Wea. Rev.* 126, 812–830. [verificar]
- Ezer, T., Mellor, G. L. (2004). A generalized coordinate ocean model and a comparison of the bottom boundary layer dynamics in terrain-following and in z-level grids. *Ocean Modell.* 6, 379–403.
- Ezer, T. (2005). Entrainment, diapycnal mixing and transport in three-dimensional bottom gravity current simulations using the Mellor–Yamada turbulence scheme. *Ocean Modell.* 9, 151–168. [10.1016/j.ocemod.2004.06.001](https://doi.org/10.1016/j.ocemod.2004.06.001) (en la biblioteca como `Ezer_2004`)
- Nof, D. (1983). The translation of isolated cold eddies on a sloping bottom. *Deep-Sea Res.* 30A, 171–182. [verificar]
- Adcroft, A., et al. (2019). The GFDL global ocean and sea ice model OM4.0: model description and simulation features. *J. Adv. Model. Earth Syst.* 11, 3167–3211. [10.1029/2019MS001726](https://doi.org/10.1029/2019ms001726) (§2.2.4: mezcla BBL de OM4)
- MOM6 `src/parameterizations/vertical/MOM_set_diffusivity.F90` (`add_LOTW_BBL_diffusivity`, `set_BBL_TKE`), lectura en `docs/design/bbl_refs/mom6_bbl_mixing.md`.
- MOM6 `src/user/DOME_initialization.F90` y `MOM6-examples/ocean_only/DOME/MOM_input` (`~/pyOCEAN_refs/`).
