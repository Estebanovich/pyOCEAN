# Inestabilidad casi inercial con θ_bc = ½: causa raíz

**Fecha:** 2026-11-02. **Rama:** `regional-chain`. **Clase:** (a).

**Origen:** se encontró en la corrida de 123 días del recorte NWA12. Ahí un modo casi inercial crece desde el día ~40 y llega a 5.5–6.5 m/s hacia el día 75 (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/figs/gs_diag/README.md`). Es la misma inestabilidad de DOME a 10 km con Δt = 1200 s. Los scripts del análisis están en el scratchpad de la sesión (`lsa.py`, `lsa_lag.py`, `repro.py`, `table2.py`, `nobound.py`).

## Causa

**Dónde está el desfase.** En el modo ALE, la fuerza de Coriolis vector-invariante q·(vh, −uh) del paso n usa los transportes de capa Qx, Qy de la continuidad del paso anterior.

**Por qué desfasa a Coriolis.**
- Esos transportes llevan la velocidad θ_bc uⁿ + (1−θ_bc) uⁿ⁻¹.
- AB3 los extrapola como si estuvieran en tⁿ.
- En consecuencia, Coriolis actúa en n + θ_bc − ½ en lugar de n + ½. Con θ_bc = ½ eso es Coriolis de Euler hacia adelante, que amplifica el movimiento casi inercial en (1−θ_bc)(fΔt)² por paso.

**Dónde está en el código.**
- `core/components/momentum.py:194` usa `st.Qx, st.Qy`; L u h̄ del estado actual solo se usa en el primer paso, `:195-197`.
- `core/model/driver.py:1001` guarda esos transportes.
- `core/vertical/ale.py:790-805` construye la velocidad de transporte con θ_bc.

**Tasa de crecimiento.** σ = (1−θ_bc) f² Δt:

| Caso | σ (por día) |
|---|---|
| DOME, Δt 1200 s | 0.52 |
| NWA12, Δt 600 s | 0.21 |
| NWA12, Δt 300 s | 0.10 |
| Cualquier caso con θ_bc = 1 | 0 |

**Por qué σ y ALE `momentum='flux'` no la tienen:** aplican Coriolis a uⁿ directamente.

**Por qué las columnas uniformes no la detectan:** BOUND_CORIOLIS recorta el término a f·vⁿ cuando el flujo es horizontalmente uniforme.

## Evidencia

**Caja periódica en el plano f** (16×16×4, ALE z\*, 1 hilo), crecimiento por día:

| fΔt | θ_bc 0.5 | 0.55 | 0.6 | 1.0 | ½ con Coriolis(uⁿ) |
|---|---|---|---|---|---|
| 0.027 | +0.095 | +0.086 | +0.077 | −0.004 | −0.001 |
| 0.054 | +0.163 | +0.151 | +0.137 | −0.007 | −0.001 |
| 0.12 | +0.22 (satura) | +0.225 | +0.228 | −0.019 | −0.006 |

**Sin BOUND_CORIOLIS**, la parte del crecimiento que se debe al desfase coincide con el análisis lineal a 5–20 %. Sin el desfase, AB3 es estable para todo θ_bc, fΔt y ck/f probados.

## Comparación con MOM6

`MOM_dynamics_split_RK2.F90`, git 0e9a561:
- **Predictor:** usa el `uh` del paso anterior (`:566`, `:1103`).
- **Corrector:** recalcula Coriolis con los transportes predichos (continuidad de `up` en `:793`, `CorAdCalc` en `:903`, actualización en `:968`). Así Coriolis queda en n + BE.
- **BE:** vale 0.6 (`:1472-1478`) y amortigua.

pyOCEAN copió el transporte desfasado del predictor pero no tiene corrector. El docstring de `momentum.py:189-191` es incorrecto, porque lo que dice vale solo para el predictor de MOM6.

## Corrección propuesta

**El cambio.** En `Momentum.vector_invariant`, construir en cada paso el transporte del flujo de PV desde el estado actual: Q = L·uⁿ·h̄ⁿ_cara, con `mean_face_thickness(hⁿ)`. Es la rama del primer paso, usada siempre.

**Qué no cambia.**
- θ_bc = ½ sigue en la continuidad.
- No hay estado nuevo ni cambio en el restart.

**Variante.** Q por PPM sobre hⁿ.

**Opciones descartadas.**
- θ_bc = 0.55–0.6: solo quita el 10–20 % del crecimiento.
- θ_bc = 1: vuelve el amortiguamiento de la onda interna de Ilicak.
- Predictor–corrector completo: es un cambio mayor que no hace falta.

**Plan de prueba.**
1. Prueba nueva de la caja rotante: crecimiento ≤ 0, más el radio espectral del análisis lineal ≤ 1.
2. DOME a 10 km con Δt 1200 s por 50 días, en z\*, σ e híbrida: sin truncamientos y con desviación estándar de η de unos 0.07 m.
3. NWA12 a Δt 600 s y a Δt 300 s desde el restart del día 60.5: max|u| ≤ 2.5 m/s.
4. Onda interna de Ilicak: 0.74 / 0.58 / 0.26 ± 0.01. Allí f₀ = 0, así que solo cambia el término ζ·Q.
5. Casos rotantes con momento vector-invariante, dentro de sus tolerancias.
6. Goldens:
   - σ y flujo: bit a bit;
   - vector-invariante: se regeneran a propósito;
   - restart y 1 = 5 hilos: bit a bit.

**Relacionado, aparte.** Remapear la historia de AB3 tras el regrid, como hace MOM6.
