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

## Diagnóstico (2026-10-12)

`run_mom6.sh budget` guarda los términos de momento de MOM6 en cada paso durante la primera hora.

- **Primeros 30 minutos:** los dos modelos coinciden. Velocidades máximas de 0.51–0.54 m/s; el caudal denso en la compuerta es 10–15 % mayor en pyOCEAN.
- **Desde la primera hora, pyOCEAN se separa.**
  - La capa densa de fondo se adelgaza en una película de 0.4–2 m que va a 1.1–1.8 m/s, unas 3.5 veces √(g′h): es supercrítica.
  - A las 4 h la punta está en 13 km y el cuerpo en 4.8 km. En MOM6, los dos están en 7.3 km.
- **En la película, el gradiente de presión es razonable** (~1e-4 m/s², el g′∂h/∂x de una corriente). **La advección de momento en forma de flujo empuja la punta** con +3 a +14 × 10⁻⁴ m/s². El grosor de la capa muestra ondulaciones de malla.
- **Experimento: advección de momento en forma advectiva** (`adv="upwind3"`, sin dividir por el volumen de la celda):

  | Frente | Forma de flujo (ahora) | Forma advectiva |
  |---|---|---|
  | 1 m (denso / ligero) | 1.96 / 1.61 U | 1.05 / 1.12 U |
  | 5 m (denso / ligero) | 0.84 / 0.14 U | 1.12 / 1.12 U |

  La ganancia de RPE queda en 7.3e6 J. Así que la causa principal es la advección de momento en forma de flujo con capas que se adelgazan. Es el problema que Bleck & Smith (1990, apéndice B) y Higdon (2005) describen para capas de grosor muy variable, y la razón por la que MOM6 y HYCOM usan la forma vectorial invariante.
- **Queda una capa intermedia de 8.8 cm a 3.15 m/s.** Las capas delgadas necesitan además el acoplamiento viscoso o el truncamiento de MOM6.

## Resultado con momento vectorial invariante (2026-10-12)

Plan `docs/superpowers/plans/2026-10-12-ale-vector-invariant-momentum.md`. En ALE, el momento pasa a la forma de MOM6: flujo de vorticidad potencial de Sadourny con `BOUND_CORIOLIS`, más el gradiente de energía cinética Godunov (aguas arriba), y truncamiento CFL como red de seguridad. La corrida híbrida usa la viscosidad de MOM6 (KV = 1e-4, KH = 0.01).

| Caso | Frente 1 m (denso, ligero) | Cuerpo 5 m (denso, ligero) | Ganancia de RPE en 6 h [J por 500 m] | max\|u\| [m/s] |
|---|---|---|---|---|
| MOM6, capas (21) | 1.05, 1.05 U | 1.05, 1.05 U | 0.04 | 0.56 |
| MOM6, RHO (21) | 1.05, 0.98 U | 0.98, 0.98 U | 4.5e7 | 0.56 |
| pyOCEAN híbrida, `keep_isopycnal` (10) | 1.12, 1.05 U | 1.05, 1.05 U | 3.0e6 | 0.94 |
| pyOCEAN híbrida, `thickness_flux='ppm'` (21, `validation/mom6_ke_schemes/`) | 1.05, 0.98 U | 1.05, 0.98 U | 8.8e5 | 0.74 |
| pyOCEAN híbrida, P1M en todas las capas (10) | 0.77, 1.05 U | 0.84, 1.05 U | 8.4e7 | 0.56 |
| pyOCEAN ALE-σ (10) | 0.91, 0.91 U | 0.91, 0.84 U | 2.1e8 | 0.67 |

- **Con `keep_isopycnal` el cuerpo de la corriente coincide con MOM6** (1.05 U), y antes iba a 0.84 / 0.14 U. La mezcla es 9 veces menor que la de MOM6 RHO.
- **Queda abierto:**
  - la punta densa medida a 1 m va a 1.12 U (criterio ±10 %);
  - max|u| llega a 0.94 m/s dentro de la corriente densa (MOM6: 0.56 m/s). Se observa en capas de 0.6 a 1.2 m.
- **Energía cinética:** con la de Arakawa, el lock exchange ALE-σ (20 capas) da 0.91 / 0.84 U; con Godunov, 0.98 / 0.98 U. Por eso Godunov es el valor por defecto, como en el `lock_exchange` de MOM6.
- **Sin regresiones:** escalón z\* 3.5e-12 m/s; monte submarino z\* + volumen finito + ρ_ref 0.07 cm/s el día 10 (antes 0.11).
- **Costo:** 0.362 s/paso contra 0.356 con la forma de flujo (160 × 100 × 30, 5 hilos).

## Qué debe mejorar pyOCEAN

1. ~~**Momento en forma vectorial invariante para el modo ALE**~~ ✅ (2026-10-12): el cuerpo ya coincide con MOM6.
2. **Velocidad interna de la corriente densa** (0.94 contra 0.56 m/s) **y punta a 1.12 U.**
   - Este caso no tiene rotación (f = 0, como el de MOM6) y es una sola fila (ny = 1), así que ζ = 0 y q = 0. El flujo de vorticidad potencial no actúa, y `CORIOLIS_EN_DIS` (primer paso del plan B del spec) no cambiaría nada aquí.
   - Lo que sí puede actuar:
     - el esquema de energía cinética;
     - la viscosidad vertical con grosor aguas arriba/armónico en la capa límite de fondo (`hvel` de MOM6);
     - la resolución: 21 capas como MOM6, en lugar de 10.
3. **Pasar a ✅ el xfail** de `tests/test_sigma_ale.py` cuando se resuelva.
