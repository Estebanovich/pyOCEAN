# BayIW 3-D recortada: pyOCEAN contra MITgcm en el mismo corte

- **Spec:** `docs/superpowers/specs/2026-10-12-bayiw-cut-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-12-bayiw-cut.md`

## Qué es

Un corte de 200 × 160 celdas (~5 km) de la malla BayIW de Esteban, i = 180–379 y j = 192–351, con 50 niveles `delR`.

- **Contiene:** la bahía, el viento de la etapa 1 y las 7 estaciones (A, B, C, Gamma, Omega, Seiche2, Seiche).
- **Casos:** dos capas y lineal, con bahía y sin bahía; 60 h.
- **Fase 1 (este README):** los dos modelos en el mismo corte, radiando hacia un exterior en reposo.
- **Fase 2:** anidados en las corridas completas de Esteban.

**Geometría.** Las celdas OB de MITgcm están dentro de su dominio: impone T en la celda OB (`obcs_apply_ts.F:205`) y u/v en la cara junto al interior (`obcs_apply_uv.F:205`). Por eso el dominio de pyOCEAN es el interior de MITgcm, 198 × 159 (i = 181–378, j = 193–351), y la cara abierta de pyOCEAN es esa cara de MITgcm. El norte es la costa.

**MITgcm:**
- su código `BayIW_Rectan_2layer/code`, con `SIZE.h` para 5 procesos MPI (sNx = 40, nPx = 5);
- `data.stage1` con los archivos del corte y Δt = 30 s;
- **solo Orlanski** en oeste, este y sur, sin `useOBCSbalance` ni esponja. La sección 2-D mostró que la esponja y el balance de la configuración de Esteban reflejan el pulso barotrópico.

**pyOCEAN:**
- z\* (50 capas `delR`) e híbrida (25 capas, 10 objetivos ligeros), Δt = 120 s, energía cinética Arakawa;
- Flather + Orlanski 0.45 + esponja de 30 celdas sobre u′;
- inicio desde los niveles de MITgcm (`initial_state(..., h_src=bayiw.mitgcm_source(...))`), con su profundidad efectiva (`bayiw.mitgcm_depth`).

**Muestreo:**
- Las diagnósticas de MITgcm son promedios de 900 s; se fechan a la mitad del intervalo (t − 450 s).
- pyOCEAN guarda instantáneas cada 900 s, remapeadas de forma conservativa (PPM_H4) a los niveles fijos de MITgcm, con η en el nivel superior (`validation/bayiw_2d/section.to_mitgcm_levels`).

```bash
bash validation/bayiw_cut/run_mitgcm.sh build
bash validation/bayiw_cut/run_mitgcm.sh run 2layer bay radiate            # × {2layer, linear} × {bay, nobay}
NUMBA_NUM_THREADS=5 python validation/bayiw_cut/run_pyocean.py 2layer bay --coord zstar   # y --coord hybrid
python validation/bayiw_cut/compare.py 2layer zstar radiate
```

**Salidas:** en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/`, con las tablas completas en `compare_<caso>_<coord>_radiate.txt`.

**Costo por corrida de 60 h:**

| Modelo | Tiempo |
|---|---|
| MITgcm (5 MPI, Δt = 30 s) | 4.4 min |
| pyOCEAN z\* (Δt = 120 s) | 11.2–11.9 min (0.37–0.40 s/paso) |
| pyOCEAN híbrida (Δt = 120 s) | 5.4–5.6 min (0.18–0.19 s/paso) |

## Resultados de la fase 1

Todas las corridas terminan con **cero truncamientos**, max|Σh − H − η| ≤ 8e-13 m y un cambio de calor ≤ 3e-9 (el flujo por los bordes abiertos).

### Seiche (Δη bahía − sin bahía, 0–27 h)

| | Período MITgcm / py [h] | Rango Seiche [mm] | Rango Seiche2 [mm] | Correlación |
|---|---|---|---|---|
| dos capas z\* | 3.86 / 3.79 | 16.19 / 16.17 | 5.55 / 5.48 | 0.99–1.00 ✅ |
| dos capas híbrida | 3.86 / 3.79 | 16.19 / 16.40 | 5.55 / 5.64 | 1.00 ✅ |
| lineal z\* | 3.86 / 3.79 | 15.67 / 15.65 | 5.32 / 5.26 | 1.00 ✅ |
| lineal híbrida | 3.86 / 3.79 | 15.67 / 15.67 | 5.32 / 5.27 | 1.00 ✅ |

El período queda a −1.8 %, dentro del ±2 %. Con 27 h, la resolución espectral es gruesa; en el dominio completo, pyOCEAN dio 3.83 h y MITgcm 3.81–3.83 h (`validation/bayiw_seiche/`).

### Ondas internas: caso lineal (criterio T110: ±20 % y correlación ≥ 0.8)

Amplitud rms py/MITgcm y correlación, con bahía (sin bahía es igual a ±0.02, salvo V110 en Gamma, 0.52):

| Estación | z\* T110 | z\* V110 | híbrida T110 | híbrida V110 |
|---|---|---|---|---|
| Seiche | 1.07 · 0.99 ✅ | 1.04 · 0.99 | 2.53 · 0.53 ❌ | 0.95 · 0.98 |
| Seiche2 | 1.01 · 1.00 ✅ | 1.01 · 0.99 | 3.04 · 0.29 ❌ | 1.10 · 1.00 |
| Omega | 0.98 · 1.00 ✅ | 1.02 · 1.00 | 0.96 · 1.00 ✅ | 1.02 · 1.00 |
| C | 0.98 · 0.99 ✅ | 1.05 · 0.97 | 0.97 · 0.99 ✅ | 1.04 · 0.97 |
| A, B | 1.01–1.03 · 1.00 | 1.03 · 1.00 | 0.99–1.01 · 1.00 | 1.02 · 0.99 |
| Gamma | 0.60 · 0.90 | 0.63 · 0.71 | 0.59 · 0.90 | 0.63 · 0.71 |

- **Efecto de la bahía (ΔT110):** z\* 0.87–1.02, correlación 0.64–1.00; híbrida 0.83–1.00, correlación 0.66–1.00. Pasa en los dos.
- **Error del mapa de T a 110 m:** z\* 0.04 / 0.15 / 0.20 a las 6 / 24 / 60 h; híbrida 4.7 / 6.1 / 5.4.

### Ondas internas: dos capas (criterio: altura de la isoterma central, ±20 % y correlación ≥ 0.8)

En dos capas la interfaz apenas se mueve: ~4 cm rms. La T de los niveles fijos alrededor de ella mide sobre todo cuánto se difunde la interfaz. En Gamma, MITgcm y z\* cambian T en ±0.09 °C en los niveles 30 y 31 con desplazamiento cero.

Por eso se añadió, **sin criterio y como métrica complementaria**, el desplazamiento por calor:

  ζ = −Σ_k (T_k − T_k(0)) Δz_k / (T_sup − T_inf)

La difusión conserva el calor de la columna, así que ζ no ve el ensanchamiento de la interfaz, solo su desplazamiento (`compare.heat_displacement`). El criterio del spec (la isoterma) se mantiene.

| Estación | z\* isoterma | z\* ζ | z\* V110 | híbrida isoterma | híbrida ζ | híbrida V110 |
|---|---|---|---|---|---|---|
| Seiche | 1.01 · 0.99 ✅ | 1.05 · 0.99 | 0.91 · 0.96 | 4.29 · −0.27 ❌ | 3.67 · −0.34 | 0.96 · 0.93 |
| Seiche2 | 1.16 · 1.00 ✅ | 1.07 · 0.99 | 1.38 · 0.94 | 6.22 · −0.27 ❌ | 6.50 · −0.11 | 1.06 · 1.00 |
| Omega | 1.16 · 0.98 ✅ | 1.07 · 0.99 | 1.02 · 1.00 | 0.07 · 0.52 ❌ | 0.08 · 0.44 | 0.66 · 0.80 |
| Omega sin bahía | 1.21 · 0.96 ❌ | 1.10 · 0.99 | 1.07 · 0.94 | 0.04 · 0.31 ❌ | 0.05 · 0.22 | 0.52 · 0.73 |
| C | 1.12 · 0.98 ✅ | 1.03 · 0.99 | 1.04 · 0.98 | 0.89 · 0.84 ✅ | 0.95 · 0.92 | 1.07 · 0.93 |
| A / B | 1.13–1.14 · 0.99 | 1.05–1.07 · 1.00 | 1.01–1.17 · 0.91–0.95 | 0.39–0.56 · 0.45–0.55 | 0.41–0.59 · 0.52–0.55 | 0.52–0.53 · 0.71 |

Otras métricas de dos capas:
- **U110 en A y B:** z\* 1.04–1.19; híbrida 0.53–0.57, retrasada ~2.5 h.
- **T110:** z\* sale 1.4–3.5 veces la de MITgcm (la interfaz más difundida, ver abajo); en la híbrida, T110 casi no cambia (0.01–0.13).
- **Efecto de la bahía (ΔT110):** pasa en A, B, C y Gamma. Falla en Omega: z\* 3.42 con correlación 0.21; híbrida 0.01.
- **Error del mapa de T a 110 m:** z\* 0.30 / 0.54 / 0.51; híbrida ~1.0.

### Δt = 30 contra 120 s (z\*, dos capas con bahía)

En todas las estaciones, la isoterma y V110 cambian ≤ 0.02 en el cociente de amplitudes y ≤ 0.01 en la correlación (0.38 s/paso, 46 min). El paso de tiempo no influye.

## Diagnóstico de la híbrida

1. **En reposo (lineal, sin viento, 12 h):**
   - |u| < 3e-4 m/s, pero T110 muestreada cambia 0.2 °C dentro de la bahía.
   - Con K_h = 0 pasa lo mismo: no es la difusión a lo largo de capas.
   - En una columna de 126 m de la bahía, la híbrida tiene 10 capas z de 1–2 m, luego 4 capas gruesas (7.6, 38, 43 y 22 m) y 11 capas vacías.
   - La T de las capas vacías deriva (27.01 → 27.26 °C) sin masa y sin cambiar el calor de la columna. Pero entra en la reconstrucción PPM del muestreo y mueve los niveles fijos cerca del fondo.
2. **Dos capas, columnas nativas (sin muestreo):** el desplazamiento por calor confirma la tabla.
   - En Omega, la interfaz entre la capa cálida de 99 m y la fría de 484 m se mueve 0.003 m rms, frente a 0.043 m en MITgcm.
   - En Seiche y Seiche2 se mueve 0.11 y 0.30 m, frente a 0.03 y 0.05 m.
   - No es un artefacto del muestreo.
3. **Energía cinética Godunov en lugar de Arakawa:** no cambia nada. ζ nativo py/MITgcm con Arakawa → Godunov: Omega 0.08 → 0.08, Seiche 3.70 → 3.39, Seiche2 6.49 → 6.76, C 0.95 → 0.95, A 0.60 → 0.60, B 0.40 → 0.40. El esquema de momento queda descartado.

## Clasificación

| Diferencia | Clase | Razonamiento |
|---|---|---|
| Gamma lineal 0.6, V110 0.71 (z\* e híbrida) | (c) | Gamma está a 22 celdas del borde sur, dentro de la esponja de 30 celdas de pyOCEAN; MITgcm usa solo Orlanski (decisión del spec §1.3) |
| z\* dos capas: T110 1.4–3.5×; isoterma de Omega sin bahía 1.21; efecto de bahía en Omega | (c) | z\* difunde la interfaz unas 2 veces más que MITgcm (Gamma: ±0.17–0.22 frente a ±0.09 °C en Seiche y Omega). El desplazamiento por calor, que no ve la difusión, coincide: 1.03–1.10, correlación ≥ 0.99. Igual que en la sección 2-D |
| Híbrida lineal en la bahía: T110 2.5–3×, mapa 5× | (c) probable | Pocas capas activas en aguas someras (25 capas con objetivos para toda la columna de 600 m) más el muestreo sensible a la T de capas vacías. Falta separar las dos partes |
| Híbrida dos capas: la interfaz casi quieta en Omega y 3–6× en la bahía; U110 la mitad en A y B | **(a) candidato** | Mismo patrón que la sección 2-D (borde de la plataforma 0.27–0.61). z\* con el mismo momento coincide, así que está en algo propio de la coordenada híbrida. Se diagnostica aparte antes de usar la híbrida en la fase 2 |

## Resultados de la fase 2: anidado en la corrida completa (z\*)

**Configuración:**
- **pyOCEAN:** `open_boundary_state` en modo `'radiating'`, con los bordes de la corrida completa cada 900 s e interpolados en el tiempo; Flather hacia (η, ū) exterior, Orlanski con nudging y la misma esponja de 30 celdas, ahora hacia ese exterior.
- **MITgcm:** OBCS prescribe de u, v y T, sin balance ni esponja. **No prescribe η.** Con la superficie libre lineal de las corridas de Esteban, `obcs_check.F:561-569` rechaza los `OB*etaFile`. La η del borde solo se aplica con `NONLIN_FRSURF` (`update_etah.F:75`, `calc_r_star.F:170`, `calc_surf_dr.F:188`), y su `CPP_OPTIONS.h` lo tiene en `#undef` (línea 109). Se decidió con Esteban mantener la superficie lineal, la misma física que la "verdad".
- **Ventana:** 0–59.5 h. El calendario periódico de OBCS pone el estado inicial (registro 179) en t > 215 400 s.
- **Salidas:** las tablas completas están en `compare_<caso>_zstar_full.txt`. pyOCEAN cuesta 0.52–0.54 s/paso, frente a 0.37–0.40 radiando: +35 % por evaluar el exterior en cada paso.

**El criterio del spec degenera.** MITgcm anidado reproduce su propia corrida completa con error relativo ≤ 0.01 en todas las estaciones y en el mapa. La excepción es Gamma, a 22 celdas del borde sur: 0.07–0.17. Es el mismo código con la misma malla y el mismo Δt, alimentado con sus propios bordes. Su error ya no es "el piso del anidado" sino casi cero, así que la regla "pyOCEAN ≤ 1.5 × MITgcm" sale FAIL en casi todo. Las tablas guardan esos FAIL tal cual. La lectura útil es comparar el error de pyOCEAN en las dos fases.

**Error relativo rms de pyOCEAN z\*, fase 1 (contra MITgcm radiando) / fase 2 (contra la corrida completa):**

| Caso · métrica | Seiche | Seiche2 | A | B | C | Omega | Gamma |
|---|---|---|---|---|---|---|---|
| dos capas · V110 | 0.30/0.30 | 0.55/0.55 | 0.40/0.40 | 0.42/0.42 | 0.19/0.19 | 0.10/0.10 | 0.66/0.06 |
| dos capas · isoterma | 0.14/0.13 | 0.19/0.19 | 0.20/0.22 | 0.22/0.23 | 0.25/0.28 | 0.27/0.27 | 0.07/0.01 |
| dos capas · ζ | 0.16/0.15 | 0.16/0.15 | 0.11/0.09 | 0.07/0.07 | 0.16/0.17 | 0.14/0.14 | 0.96/0.40 |
| lineal · V110 | 0.17/0.17 | 0.11/0.11 | 0.10/0.10 | 0.10/0.10 | 0.26/0.26 | 0.08/0.08 | 0.71/0.74 |
| lineal · T110 | 0.19/0.19 | 0.10/0.10 | 0.06/0.06 | 0.04/0.04 | 0.11/0.11 | 0.08/0.06 | 0.52/0.54 |

(Con bahía; sin bahía es igual a ±0.03, salvo la isoterma y la V110 de Omega en dos capas: 0.37 y 0.36 en las dos fases.)

- **El anidado no añade error en el interior.** En 60 h, lo que entra por los bordes no llega a las estaciones, y el error de pyOCEAN es el mismo que su diferencia con MITgcm en la fase 1. Es decir, la diferencia de modelos ya clasificada: la interfaz más difundida en z\*, (c).
- **Seiche:** pasa en dos capas. En el caso lineal, el período en Seiche sale con error 0.018: es la resolución espectral de 27 h, 3.79 frente a 3.86 h, igual que en la fase 1.
- **Gamma:** está en la esponja de pyOCEAN.
  - En dos capas, el anidado la arregla: V110 0.66 → 0.06.
  - En el caso lineal no: V110 0.74 y T110 0.54, mientras que MITgcm anidado da 0.07 y 0.02.
  - La esponja relaja u′ hacia el perfil del borde copiado a lo largo de la normal (como `OBData` hoy), y Gamma está 22 celdas adentro. Con una onda interna que se propaga, ese perfil copiado no es el exterior local. Es la limitación ya anotada en `validation/internal_wave/README.md`: falta un exterior 3-D en toda la banda.
- **Efecto de la bahía (ΔT110):** 0.68–0.75 en dos capas (Omega 3.3) frente a 0.08–0.21 en el lineal. En dos capas, T110 mide sobre todo la difusión de la interfaz (fase 1).
- **Mapa de T a 110 m:** dos capas 0.30 / 0.58 / 0.56; lineal 0.02 / 0.16 / 0.20. Son las mismas cifras que en la fase 1.

### Clasificación de la fase 2

| Diferencia | Clase | Razonamiento |
|---|---|---|
| Criterio 1.5 × MITgcm anidado en FAIL | (d) del diseño del benchmark | MITgcm anidado en sí mismo es exacto (≤ 0.01), así que el piso es cero. El error de pyOCEAN anidado es igual al de la fase 1 |
| Gamma lineal 0.74 / 0.54 | (b) | Falta un exterior 3-D en la banda de esponja; la esponja hacia el perfil del borde copiado no representa una onda que se propaga |
| Interior en dos capas (isoterma 0.13–0.37, mapa 0.56) | (c) | Igual que en la fase 1: difusión de la interfaz en z\* |
| MITgcm no prescribe η | (c) | Restricción de MITgcm con superficie libre lineal; su η del borde sale del transporte |

## Qué debe mejorar pyOCEAN (las dos fases)

1. **La híbrida con una interfaz marcada (fase 1).** La interfaz responde mal al viento lejos de la costa y exagera la respuesta dentro de la bahía. Es la prioridad: es el caso de la tesis de Esteban. Se diagnostica aparte, en la sección 2-D.
2. **Un exterior 3-D en la banda de esponja.** Con anidado, la esponja de 30 celdas (150 km) relaja hacia el perfil del borde copiado y daña Gamma en el caso lineal. Hace falta `OBData` en toda la banda, o una esponja más angosta al anidar.
3. **La T de las capas vacías.** Deriva sin masa. No cambia el calor, pero contamina cualquier reconstrucción vertical que la use en su plantilla.
4. **La difusión de la interfaz en z\*** (unas 2 veces la de MITgcm). Ya conocida; afecta T en niveles fijos, no el desplazamiento.
5. **Costo.** z\* tarda 2.5–2.7 veces lo de MITgcm con 5 procesos por hora simulada, y el anidado añade 35 % (el exterior evaluado en cada paso). La híbrida, 1.2–1.3 veces. El perfil de rendimiento está pendiente.
6. **El diseño del criterio de anidado.** Un modelo anidado en sí mismo no da un piso de error. Para el próximo benchmark de anidado, el piso debe venir de otro modelo, o de MITgcm con otra malla o resolución.
