# Lock exchange: velocidad del frente y mezcla espuria (pyOCEAN contra CROCO `GRAV_ADJ` y la teoría)

Caso de Ilicak et al. (2012), en la versión `GRAV_ADJ` de CROCO:
- canal de 64 km × 20 m, malla 128 × 40 (Δx = 500 m), f = 0, cerrado y sin fricción de fondo;
- T = 5/α a la izquierda y 10/α a la derecha (Δρ = 5 kg/m³, EOS lineal ρ = ρ₀ − αT, α = 0.28);
- Δt = 10 s, 17 h, viscosidad vertical 1e-6 m²/s, sin difusión explícita de trazadores.

**Verdad:** el frente de Benjamin (1968), U = ½√(g′H) = 0.495 m/s, una vez pasada la aceleración inicial.
**Mezcla espuria:** se mide con la RPE (Winters et al. 1995; Ilicak et al. 2012) y con el decaimiento de varianza en línea (Burchard & Rennau 2008). Ambos están en `diagnostics/mixing.py`.

## Cómo reproducirlo

```bash
python validation/lock_exchange/run_lock_exchange.py --croco ~/pyOCEAN_refs/croco_runs/grav_adj/gravadj_his.nc --ah 0 1 10 100
```

**CROCO:** el caso viene en `~/pyOCEAN_refs/croco_runs/grav_adj`; `cppdefs.h` lleva `#define GRAV_ADJ` y `#undef REGIONAL`.
- Usa WENO5-Z en trazadores y momento, horizontal y vertical, y no tiene viscosidad explícita.
- Hay una variante con capas σ uniformes (θ_s = 0) en `grav_adj_uniform`. Da la misma RPE (2.19e-5), así que el estiramiento vertical de CROCO no influye.

## Resultados (17 h)

| Corrida | Velocidad del frente 2–8 h, denso / ligero [m/s] | ΔRPE/RPE₀ | Rango de T |
|---|---|---|---|
| Benjamin | 0.495 | — | — |
| CROCO (WENO5-Z) | 0.463 / 0.486 | 2.19e-5 | 5–10 |
| **pyOCEAN, forma de flujo + WENO5-Z, A_h = 0** | **0.463 / 0.486** | 3.22e-5 | 5–10 |
| pyOCEAN, WENO5-JS, A_h = 0 | 0.463 / 0.486 | 2.92e-5 | 5–10 |
| pyOCEAN, A_h = 10 (Re_Δ = 25) | 0.463 / 0.486 | 2.86e-5 | 5–10 |
| pyOCEAN, A_h = 100 (Re_Δ = 2.5) | 0.509 / 0.509 | 1.26e-5 | 5–10 |
| pyOCEAN, momento advectivo upwind3 (antes) | 0.42 | — | — |
| pyOCEAN, trazadores UP3 (antes) | — | 9.2e-6 (incluye "desmezcla" por sobreoscilaciones) | 1.95–13.2 |

- Las posiciones del frente coinciden con CROCO dentro de una celda: 3.25 / 6.25 / 9.75 / 13.25 km a las 2 / 4 / 6 / 8 h, contra 2.75 / 6.25 / 9.75 / 12.75 de CROCO.
- La RPE se comporta como en Ilicak et al. (2012): la mezcla espuria baja cuando Re_Δ < 10.
- θ (0.5 / 0.6 / 1) y Δt (5 / 10 s) no cambian la RPE.

## Qué mejoró pyOCEAN con este caso

1. **Diagnósticos de mezcla espuria** (`diagnostics/mixing.py`): la RPE, con hipsometría, y el decaimiento numérico de varianza en línea. Comprobado que suma exactamente la pérdida global de varianza.
2. **Modelo σ estratificado** (`core/dynamics/sigma_baroclinic.py`):
   - trazadores en forma de flujo, con transportes por capa que suman exactamente el transporte de la superficie libre (constancia y conservación a 1e-12);
   - ω a partir de la continuidad;
   - gradiente de presión baroclínico;
   - AB3 en las tendencias explícitas;
   - SSP-RK3 en los trazadores.
3. **WENO5 (Z y JS) en los trazadores.** UP3 se pasaba ~50 % del rango en la nariz del frente, lo que creaba agua más densa y más ligera que cualquiera inicial y hacía bajar la RPE.
4. **Advección de momento en forma de flujo** (como ROMS/CROCO). La forma advectiva daba el frente 16 % lento. La velocidad del frente sale del balance de flujo de momento (Borden & Meiburg 2013; Ungarish et al. 2018).

## Pendiente: pyOCEAN mezcla 1.3–1.5 veces más que CROCO

**Descartado,** con la misma malla y casi el mismo resultado de RPE en cada caso:
- el estiramiento vertical de CROCO: θ_s = 0 da la misma RPE, 2.19e-5;
- θ (0.5 / 0.6 / 1) y Δt (5 / 10 s);
- la variante de WENO: Z da 3.22e-5 y JS 2.92e-5;
- el tratamiento de bordes, idéntico al de CROCO (primer orden, WENO3, WENO5);
- el avance en el tiempo de los trazadores: el predictor-corrector al estilo CROCO (`tracer_time="pc"`, predictor centrado de 6º orden y corrector WENO) da 3.21e-5, igual que SSP-RK3.

**Lo que sí difiere:**

| A 6 h / 17 h | CROCO | pyOCEAN |
|---|---|---|
| w rms | 4.1e-4 m/s | 4.6e-4 m/s (+12 %) |
| Rugosidad de w en x (2ª diferencia / rms) | 1.32 | 1.42 |
| Grosor de la interfaz (10–90 %) a 17 h, lejos del centro | 3.5–4 m | 4.5–5 m |
| Fracción de agua mezclada a 17 h | 23 % | 27 % |

La mezcla de más es vertical a lo largo de la interfaz, y la acompaña una velocidad vertical algo más ruidosa a escala de malla. Los candidatos están en el lado del momento:
1. AB3 en lugar del LF-AM3 de CROCO, que amortigua distinto las altas frecuencias;
2. el acoplamiento barotrópico/baroclínico: superficie libre θ-implícita con corrección uniforme, contra el split-explicit filtrado de Shchepetkin & McWilliams (2005);
3. la vertical de la advección de momento: ω en las caras u como promedio de las celdas.

**Siguiente prueba:** medir la RPE con un filtro de momento o LF-AM3. Falta también el barrido de A_h en CROCO para la curva RPE contra Re_Δ.
