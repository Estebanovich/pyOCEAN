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

**Lock exchange, 6 h** (híbrido con estado inicial en capas σ, 2026-10-12; con 21 capas, objetivos de densidad con cada masa de agua en el centro de su intervalo, ver abajo; transporte de grosor `upwind` = celda donante, `ppm` = MOM6 continuity_PPM):

| Caso | Frente 1 m (denso, ligero) | Cuerpo 5 m (denso, ligero) | Ganancia de RPE [J] | max\|u\| [m/s] |
|---|---|---|---|---|
| híbrido 10, Arakawa, upwind | 1.12, 1.12 U | 1.12, 1.12 U | 4.0e6 | 1.55 |
| híbrido 10, Godunov, upwind | 1.12, 1.05 U | 1.05, 1.05 U | 3.0e6 | 0.94 |
| híbrido 10, Arakawa, ppm | 0.98, 0.98 U | 1.05, 0.98 U | 4.1e6 | 1.93 |
| híbrido 10, Godunov, ppm | 1.19, 1.05 U | 1.26, 1.05 U | 8.9e6 | 0.99 |
| híbrido 21, Arakawa, upwind | 1.19, 1.12 U | 1.40, 1.12 U | 7.8e6 | 1.17 |
| híbrido 21, Godunov, upwind | 1.19, 1.05 U | 1.26, 1.05 U | 4.2e6 | 0.92 |
| híbrido 21, Arakawa, ppm | 0.98, 0.98 U | 0.98, 0.98 U | 4.9e6 | 1.62 |
| **híbrido 21, Godunov, ppm (= MOM6)** | **1.05, 0.98 U** | **1.05, 0.98 U** | **8.8e5** | **0.74** |
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
- **Las primeras filas de 21 capas no valían.** Con `linspace(ρ_l − 1, ρ_d + 1, 22)` (paso 1/3) las densidades del agua, 990 y 995, caen exactamente sobre interfaces objetivo, así que cada masa de agua tiene dos capas válidas. El arranque z\* viejo la dejaba en una por desempate; el arranque σ la reparte en dos (2.9 + 17.0 m) y `keep_isopycnal` las conserva. Ahora el margen es 0.78125, con el agua en el centro de su intervalo.
- **21 capas con PPM y Godunov reproduce MOM6:** punta y cuerpo 1.05 / 0.98 U (MOM6 1.05), max|u| 0.74 m/s (MOM6 0.56; criterio ≤ 0.84), RPE 8.8e5 J, sin truncamientos. Es la configuración de MOM6 (21 capas, continuity_PPM, KE Godunov). PPM cuesta 28 % más por paso (5.1 contra 4.0 ms con 21 capas).
- **Con 10 capas PPM no ayuda con Godunov** (punta 1.19, cuerpo 1.26 U, RPE 8.9e6). Con Arakawa da los frentes exactos (0.98 U), pero max|u| 1.6–1.9 m/s, por la energía que no se disipa.
- **Upwind con 21 capas empeora** (cuerpo denso 1.26–1.40 U): la celda donante necesita más resolución horizontal que PPM.
- **Defecto corregido:** `initial_state` híbrido muestreaba T en el piso z\* de `dz_top` (por ejemplo 19 × 0.5 m + una capa de 190 m) y perdía la estratificación del fondo; el seiche explotó. Ahora arranca en capas σ y deja que `init_iterations` construya las isopicnas, como MOM6 (`THICKNESS_CONFIG = "uniform"`, `MOM_state_initialization.F90:271`; `REGRID_ACCELERATE_INIT`, `:488-507`). El seiche híbrido da lo mismo que con el parche anterior.

## Qué debe mejorar pyOCEAN

1. ~~`initial_state` híbrido~~ ✅ (capas σ + `init_iterations`).
2. **Transporte de grosor por defecto:** `thickness_flux='ppm'` con 21 capas cumple el criterio de la punta; con 10 capas no. Decisión pendiente de Esteban: PPM por defecto en ALE (como MOM6) o upwind.
3. **Energía cinética por defecto:** Godunov (decisión de Esteban, 2026-10-12): pyOCEAN es costero/regional, donde dominan frentes, plumas y corrientes de densidad. Arakawa (`ke_scheme='arakawa'`) queda como opción para estudios de ondas internas, donde amortigua la mitad.
4. **Regla híbrida con densidades sobre un objetivo:** agua de densidad exactamente igual a un objetivo puede quedar en dos capas. En el océano real es un caso de medida cero, pero vale un test o un desempate explícito.
