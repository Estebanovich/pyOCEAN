# Planicie de marea: mojado-secado de pyOCEAN contra CROCO `TIDAL_FLAT`

Caso `TIDAL_FLAT` de CROCO, corrido como problema 2DH (barotrópico) en los dos modelos y comparado celda por celda. No tiene solución exacta: prueba la línea de costa móvil con una marea fuerte, advección de momento y arrastre de fondo, contra la implementación de otro modelo.

**Configuración** (`OCEAN/ana_grid.F`, `analytical.F`, `TEST_CASES/croco.in.Tidal_flat`):
- Canal de 100 km con 200 celdas de 500 m, f = 0.
- Fondo: h₁ = 16 m y hᵢ = 16 − 20 (i − 1)/201 m para i ≥ 2. Llega a −3.9 m en el extremo este; los últimos ~19 km son una planicie que se inunda y se seca.
- Frontera oeste: η_ext = 2 sin(2πt/12 h). Paredes en el resto.
- Arrastre cuadrático C_d = 2.5e-3 sobre la profundidad total, sin viscosidad.
- Advección de momento upwind de tercer orden (CROCO `UV_HADV_UP3`, pyOCEAN `upwind3`).
- Umbral de secado 0.1 m (`D_wetdry` de CROCO), Δt = 15 s, 36 h (3 periodos).

## Cómo reproducirlo

```bash
python validation/tidal_flat/run_tidal_flat.py --croco ~/pyOCEAN_refs/croco_runs/tidal_flat/tidal_flat_his.nc
python validation/tidal_flat/make_reference.py ~/pyOCEAN_refs/croco_runs/tidal_flat/tidal_flat_his.nc   # → tests/data
```

**CROCO** (en `~/pyOCEAN_refs/croco_runs/tidal_flat`; `jobcomp` igual que en `validation/thacker`):
- El caso original es 3D, con GLS y sedimentos (MUSTANG). En el bloque `TIDAL_FLAT` de `cppdefs.h`:
  - `#undef` de `SOLVE3D`, `SALINITY`, `GLS_MIXING`, `TS_*`, `UV_VIS2`, `SPONGE`, `T_FRC_BRY`, `MUSTANG` y `NEW_S_COORD`;
  - `UV_HADV_WENO5` → `UV_HADV_UP3`;
  - `#define ZETA_DRY_IO`.
- En `TEST_CASES/croco.in.Tidal_flat`: `NDTFAST` = 1, `NWRT` = 60 (cada 15 min), `VISC2` = 0, `RDRG2` = 2.5e-3.
- Corre en ~1 s.

**Frontera.** CROCO usa `OBC_M2CHARACT` con `OBC_REDUCED_PHYSICS`. Con esa opción ignora ū_ext = 0 y toma como velocidad externa la interior extrapolada (`u2dbc.F`). Queda un Flather u_b = u_int − √(g/D)(η − η_ext), que en la práctica fija η ≈ η_ext. En pyOCEAN el equivalente es Flather con ū_ext = u de la primera cara interior en el paso anterior. Con ū_ext = 0, pyOCEAN da la mitad del rango de marea (2.5 contra 4.1 m a 10 km) y la comparación no tiene sentido.

## Resultados (último periodo, 24–36 h)

| x [km] | Rango de η py / CROCO [m] | Dif. rms de η [cm] | ū máx py / CROCO [m/s] | Dif. rms de ū [m/s] |
|---|---|---|---|---|
| 10 | 4.06 / 4.06 | 1.2 | 1.21 / 1.21 | 0.007 |
| 50 | 4.04 / 4.05 | 1.1 | 1.03 / 1.02 | 0.006 |
| 80 | 2.19 / 2.20 | 1.1 | 0.87 / 0.86 | 0.033 |
| 85 | 1.16 / 1.17 | 0.9 | 0.72 / 0.72 | 0.029 |
| 90 | 0.14 / 0.15 | 0.4 | 0.21 / 0.18 | 0.019 |

- **η donde ambos están mojados:** diferencia rms 1.1 cm, máxima 5.1 cm.
- **Línea de costa:** pyOCEAN 77.75–90.25 km, CROCO 77.25–90.25 km. La diferencia rms es 0.35 km y la máxima 0.5 km (una celda). En el reflujo pyOCEAN seca con una celda de retraso: el frente retiene un poco de agua.
- **Volumen:** no se recorta nada.

**Sensibilidad** (rms de η / máximo de la línea de costa):
- θ = 0.5, 0.55 o 0.6: 1.1 cm / 0.5 km, sin cambio.
- Advección centrada: igual que upwind3.
- Sin advección: 1.5 cm.
- Δt = 60 s: 1.1 cm (máximo 10.8 cm).
- d_min = 0.05 m: 1.6 cm y la línea de costa se separa 2 km. Es efecto de medirla con un umbral distinto del de CROCO (la película de 5–10 cm cuenta como mojada), no de la dinámica.

**Test de regresión:** `tests/test_wetting_drying.py::TestTidalFlatVsCroco` compara con `tests/data/tidal_flat_croco_reference.npz`:
- η en las estaciones: rms < 2 cm y máximo < 6 cm;
- ū: rms < 4 cm/s;
- línea de costa: dentro de una celda;
- ningún volumen recortado.

## Caso 3D original de CROCO (GLS k-ε, log-law z₀ = 1e-4 m, N = 10)

Hay dos corridas en `~/pyOCEAN_refs/croco_runs/`, ambas sin MUSTANG (los sedimentos no afectan la hidrodinámica) y con `ZETA_DRY_IO`:
- `tidal_flat3d`: la configuración original;
- `tidal_flat3d_matched`: sin `UV_VIS2` ni `SPONGE`, y con `UV_HADV_UP3`.

Las dos difieren en < 0.4 cm, así que la viscosidad, la esponja y el esquema de advección no importan.

**El 3D frena menos que el 2D con C_d = 2.5e-3.** La inundación llega a 92.75 km (contra 90.25) y el rango a 50 km es 4.7 m (contra 4.0). La C_d efectiva del 3D, bustr/(ρ₀ū²), depende de la profundidad:

| D [m] | 0.2–0.5 | 1–2 | 4–8 | 12–20 |
|---|---|---|---|---|
| CROCO 3D | 2.8e-3 | 2.1e-3 | 1.6e-3 | 1.3e-3 |
| log-law [κ/(ln(D/z₀) − 1)]² | 3.2e-3 | 2.3e-3 | 1.7e-3 | 1.4e-3 |

**Mejora a pyOCEAN:** `physics/bottom_friction.py::log_law_cd` (Soulsby 1997), en `run_tidal_flat.py --z0 1e-4`.

| pyOCEAN 2D contra CROCO 3D (último periodo) | η rms (todas las celdas mojadas) | η rms (estaciones) | Línea de costa máx |
|---|---|---|---|
| C_d = 2.5e-3 constante | 27.0 cm | ~30 cm | 3.5 km |
| **log-law, z₀ = 1e-4 m** | **4.5 cm** | 6.6 cm | 1.5 km |
| log-law, z₀ = 5e-5 m (solo diagnóstico) | 2.2 cm | — | 1.0 km |

El residuo (pyOCEAN ~0.1 m menos de rango y secado más lento en bajamar) equivale a ~12 % menos de fricción efectiva en CROCO. El perfil de GLS no es log en toda la columna: arriba la velocidad es mayor (a 50 km, u fondo/ū/superficie = 0.95/1.25/1.37 m/s). Un modelo 2D no puede representar eso sin ajustar z₀, y no lo ajustamos.

**Tests:** `TestTidalFlatVsCroco3D`, con la referencia `tests/data/tidal_flat_croco_3d_reference.npz`:
- con C_d constante el rms en las estaciones debe ser > 20 cm;
- con log-law, < 8 cm.

## Qué debe mejorar pyOCEAN (de este caso)

1. ✅ **Arrastre log-law dependiente de la profundidad** en los modelos 2D: hecho. Pendiente aplicarlo en `run_tides.py` y `run_tides_wd.py`, que siguen con C_d constante; es justo la física de las planicies del Alto Golfo.
2. **Modelo 3D homogéneo con mojado-secado:**
   - capas sigma sobre la profundidad total;
   - viscosidad vertical implícita;
   - esfuerzo de fondo log-law en la primera capa;
   - η con `WetDryFreeSurface` sobre el transporte integrado (Casulli & Cheng 1992).

   Hoy el 3D de pyOCEAN (z-levels, Fase 1) no tiene viscosidad vertical, arrastre de fondo, advección de momento ni mojado-secado.
3. **Cierre turbulento para la capa límite de fondo.** El KPP solo actúa desde la superficie. Para mareas hace falta GLS k-ε (Umlauf & Burchard 2003) o, como primer paso, una viscosidad parabólica ν = κu*z(1 − z/D).
4. **Frente de mojado-secado:** pyOCEAN seca una celda más tarde que CROCO en el reflujo; está relacionado con el `xfail` de Thacker.
