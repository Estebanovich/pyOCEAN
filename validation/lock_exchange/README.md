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

## Pendiente

- **pyOCEAN mezcla 1.3–1.5 veces más que CROCO**, con la misma malla, el mismo WENO y el mismo tratamiento de bordes. Lo que queda distinto es el avance en el tiempo de los trazadores:
  - CROCO usa LF-AM3 con un predictor centrado de 6º orden (no disipativo) y WENO solo en el corrector;
  - pyOCEAN usa SSP-RK3, con WENO en las tres etapas.

  Candidatos: un predictor-corrector al estilo de Shchepetkin & McWilliams (2005), o RK3 con WENO solo en la última etapa. Hay que medir el efecto contra la RPE.
- **Barrido de A_h en CROCO** (UV_VIS2 con distintas viscosidades) para comparar la curva RPE contra Re_Δ entre modelos, y el barrido completo de Ilicak (A_h = 0.01–200).
