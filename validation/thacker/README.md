# Thacker (1981): mojado-secado contra la solución exacta y contra CROCO

Caso `THACKER` de CROCO (`THACKER_2DV`), reproducido con `core/dynamics/wetting_drying.py`.

**Configuración:**
- Canal de 200 km con 200 celdas de 1 km.
- Fondo h = 10 m · (1 − x²/80 km²); más allá de ±80 km hay tierra emergida.
- Superficie inicial plana con η₀ = 0.1, u = 0, sin fricción, f = 0.
- Δt = 60 s, 72 h (7.2 periodos de 9.97 h).

**Solución exacta:** η = A₀ x cos ωt + η₀² D sin² ωt, con u = −η₀ ω L sin ωt uniforme, A₀ = 2η₀D/L y ω² = 2gD/L².

## Cómo reproducirlo

```bash
python validation/thacker/run_thacker.py --theta 0.5 --dmin 0.05 \
    --croco ~/pyOCEAN_refs/croco_runs/thacker/thacker_his.nc
```

**CROCO** (en `~/pyOCEAN_refs/croco_runs/thacker`, hecho con `create_config.bash`):
- `cppdefs.h`: `#define THACKER`, `#undef REGIONAL` y `# define ZETA_DRY_IO` en el bloque de THACKER.
  - Sin `ZETA_DRY_IO`, `wrt_his.F` escribe ζ + h en algunas celdas secas, y el archivo mezcla η con la profundidad total.
- `jobcomp` en arm64 necesita tres cambios:
  - usar el cpp de GCC (`/opt/homebrew/bin/cpp-15`);
  - quitar `-mcmodel=medium`;
  - agregar `-L$(nc-config --libdir)`.
- Corre en 1.7 s, con Dcrit = 0.01 m.

## Resultados (Δx = 1 km)

| | RMS de η (72 h) | RMS del último periodo | Línea de costa (rms) | Volumen |
|---|---|---|---|---|
| **pyOCEAN** (θ = 0.5, D_min = 0.05 m) | 4.6 cm | 6.9 cm | 0.8 km | exacto (4e-16) |
| pyOCEAN (θ = 0.5, D_min = 0.01 m) | 9.7 cm | 15.4 cm | 1.4 km | exacto |
| **CROCO** (D_crit = 0.01 m) | 2.0 cm | 2.9 cm | 0.6 km | exacto |

**Convergencia de pyOCEAN** (RMS tras un periodo): 10.2, 3.8, 1.8 y 1.05 cm con Δx = 4, 2, 1 y 0.5 km.

## Limitación conocida: disipación en el frente

pyOCEAN pierde ~1 % de la amplitud del plano por periodo (A/A₀ = 0.936 tras 7 periodos); CROCO pierde ~0.2 % (0.984). La fase casi no deriva en ninguno de los dos. La causa es que, en pyOCEAN, una cara que se moja arranca con u = 0, lo que actúa como sumidero de momento en la línea de costa. CROCO no anula la velocidad en las caras secas, solo enmascara el flujo.

**Lo que probé y descarté:** extrapolar la velocidad a las caras recién mojadas, desde la cara vecina o desde la cara aguas arriba. Empeora el primer periodo y luego bombea energía (la amplitud sube por encima de 1), por las caras que parpadean entre mojado y seco.

**Pendiente:** un frente que conserve el momento (Stelling & Duinmeijer 2003), o la estrategia de CROCO. El test `test_front_dissipation_as_low_as_croco` queda como `xfail` estricto, para que la meta siga visible.
