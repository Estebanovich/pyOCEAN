# Lock exchange híbrido: pyOCEAN contra MOM6

**Configuración** (Ilıcak et al. 2012, `MOM6-examples/ocean_only/lock_exchange`):
- canal de 64 km × 20 m, Δx = 500 m;
- Δρ = 5 kg m⁻³ (agua densa a la izquierda de la compuerta en x = 32 km);
- sin rotación, 6 h;
- velocidad de Benjamin U = ½ √(g′H) = 0.495 m/s.

MOM6 usa 21 capas, con 19 intermedias vacías, KV = 1e-4 y KH = 0.01 m²/s. pyOCEAN usa 10 capas con la regla híbrida de `tests/test_sigma_ale.py`.

```bash
validation/mom6_lock_exchange/run_mom6.sh layer     # capas isopicnas puras
validation/mom6_lock_exchange/run_mom6.sh rho       # isopicna continua: regrid a las densidades de capa cada paso
NUMBA_NUM_THREADS=5 python validation/mom6_lock_exchange/compare.py
```

Las salidas de MOM6 van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_lock_exchange/<modo>/`. MOM6 tarda 4–6 s. MPICH se cuelga en `MPI_Finalize`, así que el script detiene el proceso al ver la tabla final de tiempos.

## Resultados (2026-10-12)

Velocidad de los frentes entre 2 y 6 h, medida donde la columna de agua densa (o ligera) supera 1 m (punta) y 5 m (cuerpo de la corriente de 10 m de Benjamin):

| Caso | Frente 1 m (denso, ligero) | Cuerpo 5 m (denso, ligero) | Ganancia de RPE en 6 h [J por 500 m de ancho] | max\|u\| [m/s] |
|---|---|---|---|---|
| MOM6, capas (21) | 1.05, 1.05 U | 1.05, 1.05 U | 0.04 | 0.56 |
| MOM6, RHO (21, P1M_H4 + PPM_H4) | 1.05, 0.98 U | 0.98, 0.98 U | 4.5e7 | 0.56 |
| pyOCEAN híbrida, `keep_isopycnal` (10) | 1.96, 1.61 U | 0.84, 0.14 U | 9.0e6 | 2.06 |
| pyOCEAN híbrida, P1M en todas las capas (10) | 1.26, 1.61 U | 1.05, 0.98 U | 2.9e8 | 0.89 |
| pyOCEAN ALE-σ (10) | 0.84, 0.84 U | 0.84, 0.84 U | 3.2e8 | 0.76 |

**Lectura:**
- **MOM6 forma la cabeza de Benjamin a esta resolución, en los dos modos.** Hay una corriente limpia de 7–12 m de grosor y no hay películas por delante. Con capas isopicnas, la coordenada no es el problema.
- **pyOCEAN mezcla menos que MOM6 RHO con `keep_isopycnal`** (9.0e6 contra 4.5e7 J), **pero su dinámica es mala.** Las velocidades llegan al doble o cuádruple que en MOM6 (max|u| 0.9–2.1 contra 0.56 m/s). Las películas de 1 m se adelantan, y con `keep_isopycnal` el cuerpo ligero casi no avanza.
- **Con ν = 1e-4, la viscosidad de MOM6, no se corrige** (prueba aparte: frentes a 1 m de 1.47 U, max|u| 1.11 m/s).
- **Clasificación: (a), un defecto de pyOCEAN.** La dinámica de las capas que se adelgazan (gradiente de presión, momento o transporte en caras con una capa desvanecida) acelera de más el agua.

## Qué debe mejorar pyOCEAN

1. **Encontrar de dónde sale la velocidad de más.** Comparar campo por campo con MOM6 en la primera media hora (u por capa, h y la fuerza de presión en la compuerta). MOM6 escribe `CAu`, `PFu` y `diffu` en `available_diags` si se agregan al `diag_table`.
2. **Candidatos, según las diferencias de diseño con MOM6:**
   - momento en forma de flujo con WENO5 frente a la forma vectorial invariante de MOM6 (`KE_GUDONOV`, PV);
   - gradiente de presión de volumen finito en capas desvanecidas: ponderación por masa solo junto a ellas;
   - transporte donor-cell frente a PPM con ajuste barotrópico (B2b no lo arregló por sí solo);
   - z\* floor de 0.01 m por capa en la regla híbrida, que MOM6 no tiene.
3. **Repetir la tabla** cuando se corrija, y pasar a ✅ los xfail de `tests/test_sigma_ale.py`.
