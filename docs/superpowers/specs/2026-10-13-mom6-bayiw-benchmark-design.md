# pyOCEAN contra MOM6 en BayIW: costo y resultados con un modelo que remapea

**Fecha:** 2026-10-13. **Estado:** diseño aprobado por Esteban en la conversación; falta revisar el texto escrito.
**Rama:** `mom6-benchmark`, desde `main`.

**Motivo:**
- Tras el plan de rendimiento 2, pyOCEAN z\* tarda 9.1 s por hora simulada en el corte BayIW, frente a 4.4 s de MITgcm. Pero MITgcm no remapea y el paso ALE sí.
- La comparación justa de costo es contra un modelo que también regridea y remapea en cada paso: MOM6 (ALE). HYCOM (hybgen) queda para después.
- MOM6 da además una segunda referencia de resultados *con* remapeo. Permite saber si los problemas abiertos de nuestra híbrida de dos capas son nuestros o propios de la coordenada isopícnica:
  - en 2-D, la V110 variable;
  - en 3-D, el retraso de 1.5–3 h en el borde de la plataforma.

**Regla de Esteban: simulaciones sí, pero claramente baratas.**
- Se reutiliza todo lo ya medido de pyOCEAN y MITgcm.
- Solo se corre MOM6, con corridas cortas.
- La única corrida 3-D larga es opcional.

## 1. Preguntas y dónde se responde cada una

1. **¿Cuánto cuesta MOM6 haciendo lo mismo?**
   - En el corte BayIW (198 × 159), 3 h simuladas por coordenada: z\* con 50 capas `delR` e HYCOM1 con 25 capas y el piso hasta el fondo (`bayiw.hybrid_dz_top(25)`). Dos capas con bahía, viento y fronteras radiantes.
   - Se mide el costo por hora simulada con 5 procesos MPI y el reparto por componente (los `clocks` de MOM6: dinámica, continuidad, regrid/remap, trazadores, PGF, barotrópico).
   - Se compara con `profile_final.txt` de pyOCEAN y con MITgcm, sin repetir esas corridas.
2. **¿La híbrida de MOM6 se comporta como la nuestra?**
   - Sección 2-D (1 × 352 × 50, 60 h), en los casos de dos capas y realista, con y sin bahía.
   - MOM6 z\* e HYCOM1, comparados con MITgcm `orl` y con pyOCEAN, con las métricas de `validation/bayiw_2d` y el desplazamiento de la interfaz por calor ζ.
3. **El retraso 3-D (opcional):** una corrida de MOM6 HYCOM1 en el corte, dos capas con bahía, **30 h**. Solo si la sección 2-D no lo aclara. Se compara ζ en A, B y Omega con MITgcm y con pyOCEAN.

## 2. Configuración de MOM6

- **Build:** MOM6 + FMS de `~/pyOCEAN_refs`, recompilado con `-O2` en un directorio aparte: `~/pyOCEAN_refs/MOM6/build_opt`. El build de depuración `.testing/build/symmetric` (`-O0`) no se toca. MPI con 5 procesos.
- **Entradas** (NetCDF), generadas desde `bayiw` y reutilizando `section.py` y `cut.py`:
  - malla cartesiana estirada (`dx`, `dy`) y batimetría con la profundidad efectiva de MITgcm (`bayiw.mitgcm_depth`);
  - T inicial en los niveles de MITgcm (en el caso realista, la T equivalente en densidad);
  - viento meridional cada 1200 s.
- **Física igual a la del benchmark:**
  - EOS lineal (ρ₀ = 999.8, α = 2e-4 con S constante);
  - viscosidad horizontal 100 m² s⁻¹, vertical 1e-5;
  - difusividades 100 y 1e-5;
  - f constante 6.97e-5;
  - energía cinética Arakawa (`KE_SCHEME = KE_ARAKAWA`);
  - sin fricción de fondo y con deslizamiento libre en el fondo y las paredes, como MITgcm en BayIW (`data.stage1`: `no_slip_bottom` y `no_slip_sides` = `.FALSE.`, sin `bottomDrag*`; en MOM6, `CDRAG = 0`, `BOTTOMDRAGLAW = False` y `NOSLIP = False`).
- **Coordenadas:**
  - `REGRIDDING_COORDINATE_MODE = "ZSTAR"` con los 50 `delR`;
  - `"HYCOM1"` con 25 capas, los mismos objetivos de densidad (`bayiw.centred_targets` / `profile_targets`) y el mismo piso;
  - remapeo `PPM_H4` como el nuestro.
- **Pasos de tiempo:**
  - **resultados:** Δt baroclínico de 120 s, como pyOCEAN;
  - **costo:** además, el Δt más grande que MOM6 admita sin inestabilidad en el corte (lo que haría un usuario), probado con 1 h simulada.
- **Fronteras:**
  - en el corte, oeste, este y sur abiertos con la radiación de MOM6 (`OBC_SEGMENT_00x = ... FLATHER,ORLANSKI`), sin esponja;
  - en la sección, sur abierto y paredes en x.
- **Salida:** η, T y v en las estaciones y la sección cada 900 s, como en los benchmarks; el estado final.

## 3. Métricas y criterios

- **Costo:** s por hora simulada y reparto por componente, frente a pyOCEAN y MITgcm. Sin umbral: se reporta y se señalan las piezas donde pyOCEAN pierde más de 2 veces contra la misma pieza de MOM6.
- **Resultados (sección 2-D):** las tablas de `validation/bayiw_2d/compare.py` con MITgcm como referencia, para MOM6 y pyOCEAN lado a lado, y ζ en las estaciones.
- **Lectura del caso de dos capas:**
  - si MOM6 HYCOM1 tiene los mismos errores que pyOCEAN, son propios de la coordenada (clase c);
  - si MOM6 se parece a MITgcm y pyOCEAN no, el error es nuestro (clase a): se diagnostica antes de seguir.
- **Toda diferencia** se clasifica de (a) a (d).

## 4. Piezas

**`validation/mom6_bayiw/`:**
- `build_mom6_opt.sh`: el build `-O2`.
- `inputs.py`: malla, batimetría, T inicial, viento y objetivos de densidad en NetCDF desde `bayiw`, para la sección y para el corte.
- `config.py`: `MOM_input`, `MOM_override`, `diag_table` e `input.nml` para cada caso.
- `run_mom6.sh`: `build` y `run <section|cut> <case> <bay> <zstar|hycom1> [hours]`. MPICH se cuelga en `MPI_Finalize`, igual que en `validation/mom6_ke_schemes/run_case.sh`.
- `read_mom6.py` y `compare.py`: lectores y tablas.
- `README.md`.

## 5. Pruebas (sin simulaciones)

- Las entradas generadas coinciden con `bayiw` en esquinas y estaciones.
- El `MOM_input` generado contiene los parámetros pedidos (coordenada, capas, Δt, EOS, viscosidades, fronteras).
- El lector reconstruye un campo conocido de un NetCDF sintético con el formato de MOM6.

## 6. Presupuesto de cómputo (estimado)

| Corrida | Estimado |
|---|---|
| Build `-O2` de MOM6 | una vez, ~15 min |
| Costo en el corte (z\* e HYCOM1, 3 h c/u, más la prueba de Δt) | ~10 min |
| Sección 2-D (dos capas y realista × con y sin bahía × z\* e HYCOM1: 8 corridas de 60 h) | ~1–2 min c/u |
| Corrida 3-D opcional (HYCOM1, 30 h) | ~10 min |

Si alguna corrida pasa de 3 veces su estimado, se reporta antes de seguir.

## 7. Fuera de alcance

- HYCOM (hybgen): otro benchmark.
- La fase 2 (anidado) con MOM6.
- La salinidad.
- Optimizar pyOCEAN en este plan: aquí solo se mide.
