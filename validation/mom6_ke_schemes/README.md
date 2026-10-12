# Energía cinética Arakawa o Godunov en el momento vectorial invariante (2026-10-12)

**Pregunta:** en el modo ALE de pyOCEAN, ¿qué esquema conviene por defecto para la energía cinética del momento vectorial invariante?
- **KE_GUDONOV**, aguas arriba, el que usa el `lock_exchange` de MOM6;
- **KE_ARAKAWA**, que conserva la energía y es el de la producción de MOM6.

Se responde con experimentos en MOM6 y en pyOCEAN, más allá del lock exchange.

```bash
validation/mom6_ke_schemes/run_case.sh <caso> KE_ARAKAWA|KE_GUDONOV [DAYMAX] [ENERGYSAVEDAYS]   # MOM6
python validation/mom6_ke_schemes/compare_energy.py                                               # series de energía de MOM6
NUMBA_NUM_THREADS=5 python validation/mom6_ke_schemes/seiche_pyocean.py                           # seiche interno, pyOCEAN
NUMBA_NUM_THREADS=5 python validation/mom6_ke_schemes/lock_pyocean.py                             # lock exchange, pyOCEAN
```

Las salidas de MOM6 van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_ke/`. Cada caso conserva su propio `MOM_override` (que elige la coordenada) y le agrega el esquema.

**Qué esquema usa cada ejemplo de MOM6:** Godunov solo en `lock_exchange`, `adjustment2d` (sus tres modos) y `external_gwave`; Arakawa en todos los demás, incluidos OM4/OM5.

## MOM6 con los dos esquemas (energía de `ocean.stats.nc`)

| Caso | Energía final / inicial (Arakawa, Godunov) | Energía cinética final, Godunov/Arakawa |
|---|---|---|
| `lock_exchange` (capas) | 0.968, 0.911 | 0.88 |
| `sloshing/rho` (oscilación interna, 5 días) | 0.998, 0.971 | 0.97 |
| `adjustment2d/rho` (ajuste geostrófico) | 0.955, 0.939 | 0.75 |
| `flow_downslope` capas / ρ / z / σ | 0.970 / 0.942 / – / 1.050; 0.968 / 0.948 / – / 1.047 | 0.92 / 1.42 / 0.93 / 0.93 |
| `Phillips_2layer` (120 días) | iguales | 1.00 (la inestabilidad no se desarrolla: el flujo sigue uniforme en x; sin información) |

- **`sloshing`:** el período es el mismo con los dos esquemas. La energía cinética cae 0.2 % por período con Arakawa y 1.1 % con Godunov.
- **Frentes del `lock_exchange`** (salida cada 1000 s; el intervalo de medición real es de 15 000 s): ≈ 1.01 U con Arakawa y ≈ 1.08 U con Godunov. En MOM6 el esquema apenas cambia el frente, pero Godunov disipa más energía.

## pyOCEAN con los dos esquemas

**Seiche interno de modo 1:** L = 60 km, H = 200 m, N = 0.02 s⁻¹, 20 capas, período teórico 26.18 h.

| Coordenada | Período (Arakawa / Godunov) | Energía cinética pico a pico por medio período (Arakawa / Godunov) |
|---|---|---|
| σ euleriano, forma de flujo (referencia) | 26.21 h | 0.9975 |
| ALE-σ | 26.18 / 26.18 h | 0.9971 / 0.9951 |
| ALE z\* | 26.18 / 26.18 h | 0.9971 / 0.9951 |
| ALE híbrido | 26.22 / 26.21 h | 0.9950 / 0.9931 |

**Lock exchange, 6 h:**

| Caso | Frente 1 m (denso, ligero) | Cuerpo 5 m (denso, ligero) | Ganancia de RPE [J] | max\|u\| [m/s] |
|---|---|---|---|---|
| híbrido 10 capas, Arakawa | 1.12, 1.12 U | 1.12, 1.12 U | 4.0e6 | 1.46 |
| híbrido 10 capas, Godunov | 1.12, 1.05 U | 1.05, 1.05 U | 4.9e6 | 0.94 |
| híbrido 21 capas, Arakawa | 1.19, 1.05 U | 1.19, 1.05 U | 1.2e6 | 0.88 |
| híbrido 21 capas, Godunov | 1.19, 0.98 U | 1.12, 1.05 U | 8.3e5 | 0.73 |
| ALE-σ 10 capas, Arakawa | 0.84, 0.77 U | 0.77, 0.77 U | 5.8e8 | 0.88 |
| ALE-σ 10 capas, Godunov | 0.91, 0.91 U | 0.91, 0.84 U | 2.1e8 | 0.67 |
| ALE-σ 20 capas, Arakawa | 0.91, 0.84 U | 0.84, 0.84 U | 5.7e8 | 0.96 |
| ALE-σ 20 capas, Godunov | 0.98, 0.98 U | 0.91, 0.91 U | 1.9e8 | 0.72 |
| MOM6, 21 capas (Godunov) | 1.05, 1.05 U | 1.05, 1.05 U | ~0 (capas puras) | 0.56 |

## Lectura

- **Ondas internas:**
  - en los dos modelos, Godunov amortigua cerca del doble que Arakawa (pyOCEAN: 0.49 % contra 0.29 % de energía por medio período; MOM6: 1.1 % contra 0.2 % por período);
  - el período no cambia;
  - con Arakawa, ALE queda prácticamente igual que el σ euleriano.
- **Corrientes de densidad en pyOCEAN:** Godunov es mejor en todo:
  - frentes más cerca de Benjamin;
  - menos velocidad máxima;
  - en σ, tres veces menos mezcla espuria (2e8 contra 6e8 J). Con Arakawa la energía que llega a la escala de malla termina como mezcla.
- **Más capas (21, como MOM6):** max|u| baja de 0.94 a 0.73 m/s y la mezcla cae de 4.9e6 a 8.3e5 J. Pero la punta densa empeora (1.19 U). La punta rápida no es un problema de resolución: es numérico, y MOM6 con 21 capas no lo tiene.
- **Defecto encontrado:** `initial_state` del modo híbrido muestrea T en la rejilla z\* construida con `dz_top`, que en la híbrida es solo el piso z\* (por ejemplo 19 × 0.5 m + una capa de 190 m). Con estratificación continua pierde la estructura del fondo, y el seiche explotó, con una capa vacía. `seiche_pyocean.py` lo evita muestreando en capas uniformes. Hay que corregirlo en el modelo.

## Qué debe mejorar pyOCEAN

1. **`initial_state` híbrido:** muestrear la condición inicial en capas uniformes (o en σ) y regenerar desde ahí.
2. **Punta densa rápida** (1.12–1.19 U contra 1.05 en MOM6), con 10 y con 21 capas. Candidatos ahora que la forma de flujo ya no domina:
   - grosor PPM en el transporte (B2b, cuyo diff quedó guardado);
   - viscosidad con grosor aguas arriba en la capa de fondo (`hvel`).
3. **Energía cinética por defecto:** Godunov (decisión de Esteban, 2026-10-12): pyOCEAN es costero/regional, donde dominan frentes, plumas y corrientes de densidad. Arakawa (`ke_scheme='arakawa'`) queda como opción para estudios de ondas internas, donde amortigua la mitad.
