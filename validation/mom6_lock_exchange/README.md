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

## Qué debe mejorar pyOCEAN

1. **Momento en forma vectorial invariante para el modo ALE**, como MOM6 y HYCOM. Es el gradiente de energía cinética más el flujo de vorticidad potencial ponderada por grosor. Está en la etapa B del roadmap.
2. **Velocidades en capas delgadas:** acoplamiento viscoso con un grosor despreciable (MOM6 `MOM_vert_friction`) o truncamiento CFL (`vertvisc_limit_vel`, `:2555`).
3. **Repetir la tabla** cuando se corrija, y pasar a ✅ los xfail de `tests/test_sigma_ale.py`.
