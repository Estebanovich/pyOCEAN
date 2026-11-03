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

## Resultado

**Corrección aplicada** (2026-11-02, commit e1ffbbb). Se aplicó tal como se propuso: en `Momentum.vector_invariant` (`core/components/momentum.py`) el transporte del flujo de PV es Q = L uⁿ h̄ⁿ en cada paso.
- `st.Qx, st.Qy` conservan su significado para los trazadores y la forma de flujo.
- No hay estado nuevo ni cambio en el restart.
- El docstring cita el predictor de MOM6 (`:566`, `:1103`) y su corrector (`:793`, `:903`, `:968`).

**Otros consumidores del momento.** Se revisaron y ninguno tiene el desfase:
- ζ, q, K (Arakawa/Godunov) y Smagorinsky usan uⁿ, vⁿ, hⁿ;
- las cotas de BOUND_CORIOLIS, (f+ζ)·u y (f+ζ)·v, también usan uⁿ y vⁿ;
- la advección en forma de flujo usa `st.Qx` a propósito: es el transporte de la continuidad, consistente con su volumen. Esa ruta no cambia.

**Prueba nueva** (`tests/test_inertial_stability.py`, 9 pruebas, ~10 s en 1 hilo). Crecimiento de la caja rotante en 8 días con θ_bc = ½:

| fΔt | antes | después |
|---|---|---|
| 0.12 | +0.185 /d | −0.162 /d |
| 0.054 | +0.027 /d | −0.156 /d |

Las demás pruebas:
- La forma de flujo da −0.162 /d.
- Los términos vector-invariantes ya no dependen de `st.Qx/Qy`.
- **Radio espectral lineal con Coriolis(uⁿ):** ≤ 1 + 1e-12 en F ∈ [0, 0.3], a ∈ [0, 1] y θ_bc ∈ {½, 0.6, 1}. Con a ∈ [0, 2] ningún esquema es estable, porque el límite CFL de la onda de gravedad del paso adelante–atrás está en a ≈ 1.05–1.3; por eso el rango se limitó a a ≤ 1.
- **Variante desfasada:** su radio es > 1, lo que documenta el error.

**Goldens.**
- σ (14 casos) y `zstar_flux`: bit a bit.
- Los 17 casos ALE vector-invariantes y `zstar_reuse` se regeneraron. El |Δ| máximo en el paso 10 es u ≤ 9.4e-5 m/s, η ≤ 4.2e-5 m y T ≤ 6e-5 °C.
- También se regeneraron `n1_restart_main.nc`, las huellas `openbox` de `ale_perf_reference.json` y `off3d` de `task5_base_reference.npz`. Esta última se regeneró en 71e848c con la corrección aplicada; su cambio es ≤ 4e-17.

**Suite.** 1 hilo: pasan todas excepto 7 pruebas que necesitan 5 hilos de Numba; esas 7 pasan con 5 hilos. Restart bit a bit y 1 = 5 hilos: ✅.

**Ilicak, z\*.** Onda interna con los ν_h del caso (0.01 / 15 / 150): 0.736 / 0.579 / 0.261, igual que antes a 3 cifras. Estos casos tienen f₀ = 0 y ny = 1. El lock exchange (ν 10) y el overflow (ν 100) no cambian: frentes, RPE, centroides y dRPE/dt son iguales a 4–6 cifras.

**DOME 10 km, 50 días, días 30–50.** El Δt 1200 s, que antes se detenía en el día 36, ya corre:

| corrida | trunc / lim | η std [m] | b_f [1e-3] | Y′(−200/−300/−500/−800) [km] | T [Sv] | s/día |
|---|---|---|---|---|---|---|
| z\* 600 (base 2026-10-15) | 0 / 0 | 0.064 | −10.07 | −49 / −72 / −99 / −126 | 22.6 / 26.4 / 30.4 / 33.3 | 13.8 |
| z\* 600 corregida | 0 / 0 | 0.070 | −10.25 | −52 / −79 / −105 / −150 | 22.8 / 26.8 / 32.1 / 32.0 | 14.2 |
| z\* 1200 corregida | 0 / 0 | 0.070 | −10.26 | −52 / −77 / −103 / −149 | 22.5 / 26.6 / 31.7 / 31.9 | 7.1 |
| σ 600 (base) | 0 / 0 | 0.067 | −13.22 | −89 / −117 / −150 / −202 | 23.8 / 27.6 / 33.5 / 40.3 | 13.8 |
| σ 1200 corregida | 0 / 0 | 0.073 | −13.23 | −89 / −118 / −153 / −198 | 24.4 / 28.4 / 34.3 / 40.6 | 7.0 |
| híbrida 600 (base) | 0 / 0 | 0.064 | −16.18 | −115 / −150 / −192 / −224 | 11.2 / 15.4 / 19.7 / 19.4 | 14.9 |
| híbrida 600 corregida | 0 / 0 | 0.078 | −15.44 | −134 / −161 / −193 / −218 | 13.0 / 16.3 / 23.5 / 16.2 | 14.9 |
| híbrida 1200 corregida | 0 / 14 | 0.080 | −15.84 | −135 / −165 / −196 / −238 | 14.0 / 17.1 / 21.9 / 18.4 | 7.5 |

**Lectura de la tabla.**
- **Efecto del Δt, con el código corregido:** en z\*, 600 y 1200 s dan lo mismo (b_f 0.1 %, trayectoria ≤ 2 km). En la híbrida, b_f cambia 2.6 %, la trayectoria hasta 20 km en x′ = −800, y aparecen 14 eventos del limitador.
- **Contra la base del 2026-10-15:** las diferencias aparecen ya a 600 s. En z\*, la trayectoria se desplaza 24 km a x′ = −800 y b_f cambia 1.8 %. En la híbrida, Y′ cambia 19 km a x′ = −200 y T sube 16 % ahí.
- **Origen de esas diferencias:** son la corrección más los cambios del código desde la base (remapeo exacto en contenido, etc.); esta campaña no los separa.
- **σ** no cambia más allá del ruido: b_f 0.1 %.

**Costo** (`scripts/n1_smoke.py`, contra el padre 212953b para aislar la corrección):
- `bayiw_cut_zstar` 1.023×, `bayiw_cut_hybrid` 1.012× y `ilicak_overflow_zstar` 1.072× (0.5 ms por paso, por el `mean_face_thickness` adicional).
- Cambian los hashes de `bayiw_cut_zstar`, `bayiw_cut_hybrid` y `dome_10km_zstar` (vector-invariante, 2-D/3-D con f ≠ 0).
- Los 6 restantes son idénticos: σ, y lock, overflow e IW de Ilicak con f₀ = 0 y ny = 1.
- DOME a 1200 s cuesta la mitad por día simulado.

**Pendiente.** Las verificaciones de NWA12 a Δt 600 y 300 s desde el restart del día 60.5 las hace el controlador.
