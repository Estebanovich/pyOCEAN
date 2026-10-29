# Benchmark DOME: overflow 3-D con rotación (Legg et al. 2006, Caso 1)

- **Spec:** `docs/superpowers/specs/2026-10-14-dome-overflow-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-14-dome-overflow.md`

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

1. **(a) Δt máximo a 10 km: modo inestable con θ_bc = ½ y fΔt = 0.12** (punto 5). MOM6 corre DOME con 1200 s; pyOCEAN necesita 600. Candidatos:
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

Costo (10 km, 1 día, Δt 600 s, 5 hilos): 23.0 s por día con GLS contra 15.7 de la línea base (constante); 50 días ≈ 19 min. Con la ruta NumPy anterior de las capas delgadas (todas las columnas por la fusión) eran 26.0 s por día; ahora solo las columnas con capas < 1 mm se fusionan, en un kernel de Numba (`physics/mixing/gls_merged.py`). En el primer día: sin truncamientos, T en el rango inicial. Resultados: Task 10b.

## Referencias

- Legg, S., Hallberg, R. W., Girton, J. B. (2006). Comparison of entrainment in overflows simulated by z-coordinate, isopycnal and non-hydrostatic models. *Ocean Modell.* 11, 69–97. [10.1016/j.ocemod.2004.11.006](https://doi.org/10.1016/j.ocemod.2004.11.006)
- Hallberg, R. (2000). Time integration of diapycnal diffusion and Richardson number–dependent mixing in isopycnal coordinate ocean models. *Mon. Wea. Rev.* 128, 1402–1419. [verificar]
- Jackson, L., Hallberg, R., Legg, S. (2008). A parameterization of shear-driven turbulence for ocean climate models. *J. Phys. Oceanogr.* 38, 1033–1053. [verificar]
- Pietrzak, J. (1998). The use of TVD limiters for forward-in-time upstream-biased advection schemes in ocean modeling. *Mon. Wea. Rev.* 126, 812–830. [verificar]
- Ezer, T., Mellor, G. L. (2004). A generalized coordinate ocean model and a comparison of the bottom boundary layer dynamics in terrain-following and in z-level grids. *Ocean Modell.* 6, 379–403.
- MOM6 `src/user/DOME_initialization.F90` y `MOM6-examples/ocean_only/DOME/MOM_input` (`~/pyOCEAN_refs/`).
