# BayIW: ondas internas generadas por el seiche de una bahía (pyOCEAN σ vs MITgcm)

Réplica baroclínica del experimento de tesis de Esteban Cruz Isidro (MITgcm `BayIW_Rectan`, escenario realista de febrero). El seiche barotrópico ya está validado en `validation/bayiw_seiche/`.

**Verdad:** no hay solución exacta. La referencia es MITgcm, y la teoría lineal da c₁ y la dispersión del modo 1.

## Configuración

| | MITgcm | pyOCEAN |
|---|---|---|
| Malla horizontal | 560 × 352, 5 km en el núcleo y márgenes estirados hasta 13.3 km | igual (lee `delX`/`delY`) |
| Vertical | 50 niveles z (1 a 45.8 m, Σ = 599.9 m) | 50 capas σ con las fracciones de `delR`: iguales a los niveles z en el océano de 599.9 m y comprimidas en la bahía de 164 m (0.27 m arriba, ~10 m a 120 m de profundidad) |
| Estratificación | T de febrero, S uniforme 33.6, EOS lineal | igual (S uniforme, así que T basta) |
| Gradiente de presión | z (sin error en reposo) | jacobiano de densidad con el perfil inicial como ρ_ref (sin error en reposo) |
| Advección de trazadores | centrada de 2.º orden + K_h = 100 m²/s | WENO5-Z + K_h = 100 a lo largo de σ sobre T − T_ref(z) |
| Momento | A_h = 100, ν = 1e-5 | igual; momento en forma de flujo con WENO5 |
| Viento | pulso meridional ≤ 0.099 N/m² a t = 2.5 h, registros cada 1200 s | mismo archivo y misma interpolación temporal; ρ₀ = 999.8 |
| Fronteras S/O/E | Orlanski + esponja de 30 celdas (1200 s → 60 s) sobre la velocidad total | Flather + Orlanski + la misma esponja, solo sobre la parte baroclínica y la anomalía de T |
| Δt | 30 s | 300 s (superficie libre y mezcla vertical implícitas) |

## Ejecución

```bash
NUMBA_NUM_THREADS=5 python run_bayiw.py bay   --threads 5      # 120 h
NUMBA_NUM_THREADS=5 python run_bayiw.py nobay --threads 5
python compare_iw.py                                           # Δη, ΔW110, ΔT110 contra Output_IWBay
```

Para reanudar: `--restart out/restart_<caso>_realistic.nc`. Las series guardadas se conservan hasta el tiempo del restart.

## Salida

En `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_iw/` (si el SSD no está montado, en `out/`). Formato NetCDF4 CF-1.8, escrito por `iw_output.py`:

| Archivo | Contenido | Formato | Tamaño (120 h) |
|---|---|---|---|
| `pyocean_iw_<caso>_realistic_stations.nc` | 7 estaciones: ζ, T, u, v en centros de capa y w en interfaces, cada 900 s. CF `timeSeriesProfile`, dims (station, time, s_rho\|s_w) | float64, zlib (exacto) | ~4 MB |
| `pyocean_iw_<caso>_realistic_fields.nc` | sección Y–Z en el eje de la bahía (i = 280) y w, T a 110 y 250 m en una caja alrededor de la bahía | float32, zlib, un *chunk* por registro | ~110 MB |
| `restart_<caso>_realistic.nc` | estado completo para reanudar bit a bit | float64 | 1.2 GB |

- **Coordenada vertical:** CF `ocean_sigma_coordinate`, z = ζ + σ (h + ζ) con σ ∈ [−1, 0]. Cualquier lector CF (xarray + cf_xarray, Panoply, ncview) reconstruye z.
- **Tiempo:** segundos desde el inicio del modelo (unidades "seconds since 2000-01-01").
- **Robustez:** los archivos se escriben con nombre temporal y se renombran, así que un corte durante la escritura no borra la salida anterior.
- **Configuración de la corrida:** en el atributo global `run_config` (JSON).
- **Corridas viejas:** `convert_npz.py` pasa los `.npz` de las primeras corridas a este formato y verifica la conversión: estaciones exactas, campos con error relativo ≤ 6e-8.

## Historial de la corrida de referencia (2026-10-11)

La primera corrida `bay` se hizo por tramos, reanudando desde restarts bit a bit:

| Tramo | Código | Hilos | s/paso |
|---|---|---|---|
| 0–12 h | WENO5 en Numba (sin hilos), resto NumPy | 1 (con 2–3 corridas concurrentes) | ~8.5 |
| 12–36 h | igual | 5 (solo WENO paralelo) | 5.0 |
| 36–120 h | advección de trazadores fusionada en Numba, Thomas compilado, CG con LU desfasado | 5 | 3.3 |

**Corridas válidas (2026-10-11, tras corregir el viento):** bay 0–12 h (código original), 12–36 h (restart original), 36–120 h (código actual, ~2.9 s/paso, 49.6 min); nobay 0–12 h (original) y 12–120 h (código actual, 62.5 min).

**Errores del script de validación encontrados y corregidos** (clase *a*, no del modelo):
1. **Segundo pulso de viento.** MITgcm se corrió en dos etapas: `data.stage1` (0–60 h) con el archivo de viento y ciclo de 60 h, y `data.stage2` (60–120 h) sin viento. `wind_at` repetía el pulso a las 60 h y re-excitaba el seiche a 5 mm. Test: `tests/test_bayiw_wind.py`.
2. **Hueco de 84–96 h en la serie del bay.** La corrida se mató entre la escritura del restart y la del `.npz`. Ahora la salida se escribe con nombre temporal y se renombra.
3. **Nivel de W en MITgcm.** `W[k]` está en la cara RF[k], no en RC[k]: el "W a 110 m" de la tesis está a −106.6 m.
4. **Estaciones que son tierra en el nobay** (Seiche, Seiche2): MITgcm guarda 0 en tierra, así que el "Δ" de la tesis ahí es la serie del bay sola.

Las diferencias entre versiones del código son de redondeo (cada kernel se prueba contra la versión NumPy a 1e-12–1e-13).

**Rendimiento:** MITgcm usa ~1–1.4 h de reloj por corrida de 120 h con 8 procesos MPI (Δt = 30 s, ~11 h-núcleo). pyOCEAN usa ~80 min con 5 hilos (Δt = 300 s, ~2–3 h-núcleo con el código de 36–120 h).

## Resultados (`compare_iw.py`, 2026-10-11)

**Seiche barotrópico en la bahía** (amplitud de η_bay en la estación Seiche, bloques de 8 h):

| Bloque | MITgcm | pyOCEAN |
|---|---|---|
| 0–8 h | 6.30 mm | 6.25 mm |
| 8–16 h | 3.41 mm | 3.08 mm |
| 16–24 h | 1.50 mm | 1.19 mm |
| 24–32 h | 0.33 mm | 0.41 mm |
| 32–120 h | 0.004–0.05 mm | 0.06–0.14 mm |

El decaimiento coincide hasta ~32 h (e-fold 0–60 h: 7.5 h en MITgcm, 9.7 h en pyOCEAN). Después pyOCEAN se estanca en ~0.1 mm, con un pico espectral en **0.125 cph (~8 h)** que MITgcm no tiene.

**Ondas internas a 110 m, 27–120 h:**

| Estación | Resultado |
|---|---|
| Seiche (cabecera) | T: rms 4.3 vs 3.9 mK, corr **0.92**; W: corr **0.82**; misma frecuencia del seiche (0.257 cph) |
| A, C, Gamma (océano abierto) | correlaciones de W y T de 0.36 a 0.90; Δη (de solo 3–12 µm) con corr 0.45–0.73 |
| A, Omega, Gamma | banda casi inercial 4–6× más energética que MITgcm |
| A, B, C, Gamma | banda del seiche 0.15–0.5× la de MITgcm |
| **Seiche2 (j = 325, primera fila dentro de la bahía)** | **T: rms 59 vs 3 mK; varianza en la banda del seiche ~10⁴×** |

**Diagnóstico de Seiche2.** En la sección del eje, la varianza espuria de T está en j = 324–325, justo en el escalón de la boca (600 → 164 m). Crece con el tiempo: 11 mK a las 5–27 h, 21 mK a las 27–60 h, 96 mK a las 60–120 h. Oscila con ~8 h, igual que el residuo de η. Ese escalón tiene **rx0 = 0.57 y rx1 = 684**, cuando ROMS recomienda rx0 < 0.2 y rx1 < 3–7. Es el problema clásico de las coordenadas σ sobre topografía abrupta (Haney 1991; Mellor et al. 1994 [verificar]). MITgcm no lo tiene porque usa niveles z con celdas parciales.

## Qué debe mejorar pyOCEAN

1. **Escalones abruptos en σ (clase c, coordenada).** Con rx1 = 684, ninguna corrección del gradiente de presión basta. La solución de fondo es la etapa C: ALE con z* o híbrida z–σ, para que la boca sea un escalón en z como en MITgcm. Mientras tanto:
   - el verificador de dominio debe rechazar o advertir σ con rx1 > ~7;
   - falta un test idealizado (escalón + estratificación + perturbación) que mida el crecimiento espurio.
2. **Exceso casi inercial cerca de la boca y al oeste** (A, Omega: 4–6×). Hay que revisarlo cuando se quite la contaminación del escalón; puede ser la misma fuente.
3. **Menos energía en la banda del seiche en el océano abierto** (0.15–0.5×). Candidatos (clase c): WENO5 frente a centrado + K_h, Δt de 300 s frente a 30 s, esponja solo baroclínica. Se separa con la corrida de sensibilidad de baja difusión pendiente.

