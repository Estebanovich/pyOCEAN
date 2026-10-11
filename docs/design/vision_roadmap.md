# Visión y hoja de ruta de pyOCEAN: qué modelo necesita la comunidad

Documento de diseño. Fecha: 2026-10-11.

**Fuentes:**
- cuatro revisiones: modelos existentes, foros y trackers de usuarios, adjuntos/ML y fronteras numéricas;
- búsquedas de literatura en Consensus;
- lo aprendido en los benchmarks de `validation/`.

Las referencias marcadas **[verificar]** vienen de memoria y hay que comprobarlas antes de citarlas en un artículo.

## 1. Objetivo

Que pyOCEAN sea el mejor modelo regional y costero disponible, no un modelo hecho a la medida de un usuario. "Mejor" se mide en cuatro ejes:

1. **Precisión** contra la verdad: soluciones exactas, laboratorio y observaciones. Parecerse a otros modelos no basta.
2. **Costo:** tiempo por día simulado a igual error.
3. **Robustez:** no explota con batimetría real, mojado-secado ni fronteras abiertas.
4. **Usabilidad y transparencia:**
   - instalación sin compilar;
   - preparación del dominio integrada;
   - cada ecuación con su referencia y su test;
   - resultados reproducibles bit a bit.

Regla para cada benchmark: clasificar cada diferencia como (a) error nuestro, (b) física faltante, (c) elección numérica o (d) error del modelo de referencia. Solo (a) y (b) cambian el modelo. Una mejora cuenta si es física general y mejora más de un caso; nunca se calibra contra un solo caso.

## 2. El panorama: qué hace mejor cada modelo

| Capacidad | Mejor hoy | Por qué |
|---|---|---|
| Coordenada vertical | MOM6 (ALE general); HYCOM (concepto híbrido) | Cualquier coordenada vía paso lagrangiano + regrid/remapeo (Adcroft et al. 2019; Griffies et al. 2020) [verificar] |
| Poca mezcla espuria | MOM6 isopícnico/ALE; momento upwind de 3er orden (ROMS) y WENO (Oceananigans) | Ilicak et al. 2012: ROMS mezcla menos que MITgcm/MOM por su momento upwind; z̃ y ALE reducen la mezcla bajo mareas (Megann 2024; Gibson et al. 2017) |
| Paso de tiempo / superficie libre | ROMS/CROCO split-explicit (Shchepetkin & McWilliams 2005); SCHISM semi-implícito para costa | Exactitud y conservación; sin restricción barotrópica de CFL |
| Mojado-secado | SCHISM, D-Flow FM, ADCIRC; subgrid de Casulli (2009) | Implícito y robusto; batimetría subgrid |
| Anidamiento | AGRIF de dos vías (CROCO, NEMO) | Conservativo, refina en espacio y tiempo |
| Turbulencia | GOTM (biblioteca) | k-ε, k-ω, GLS, segundo momento |
| No hidrostático | MITgcm; CROCO-NBQ | Maduro; explícito y sin solver global |
| Adjunto / asimilación | MITgcm/ECCO (TAF, Tapenade); ROMS 4D-Var | Rutina operativa |
| GPU / rendimiento | Oceananigans (nativo); Veros y FESOM2-JAX (JAX) | Diseño nativo; los ports con directivas (MPAS-O, ICON-O, NEMO) dan ganancias modestas o frágiles |
| Usabilidad | Oceananigans (scripts y ejemplos) | Juicio cualitativo |

**Debilidades documentadas:**
- **HYCOM:** generador híbrido con efectos numéricos negativos; sin mojado-secado, sin no hidrostática, sin GPU; pasos de tiempo antiguos (Halliwell 2004).
- **Modelos σ (ROMS, CROCO, FVCOM, SCHISM):** error del gradiente de presión, que obliga a suavizar la batimetría (rx0/rx1).
- **Modelos z:** mezcla espuria dominada por la advección vertical, y peor con mareas (Griffies et al. 2000; Garinet et al. 2024).
- **MITgcm:** su adjunto clásico (TAF) es propietario; sin GPU.

## 3. Huecos que pyOCEAN puede resolver

### 3a. Huecos técnicos del campo

1. **Ningún código combina ALE + mojado-secado + mareas + escalas de costa a cuenca.**
   - MOM6 tiene ALE sin mojado-secado.
   - SCHISM y D-Flow FM tienen mojado-secado sin ALE.
   - HYCOM no tiene ninguno de los dos.

   pyOCEAN ya tiene mojado-secado de Casulli (2D y 3D sigma), mareas con SAL/JSL y la coordenada híbrida planeada.
2. **Mojado-secado 3D baroclínico:** abierto en todo el campo (capas que colapsan, mezcla del agua que se seca).
3. **Mezcla espuria casi nunca se diagnostica de forma rutinaria.** Medirla en cada corrida (RPE, decaimiento de varianza, transformación de masas de agua) sería un diferenciador.
4. **No existe un modelo regional y costero 3D diferenciable.**
   - Sí hay ya: FESOM2-JAX (global, 2026), Veros-autodiff, Oceananigans + Enzyme y AegirJAX (aguas someras no hidrostático costero, sin 3D).
   - Nadie tiene gradientes a través de mojado-secado, mareas, fronteras abiertas y anidamiento.
5. **Fronteras abiertas ajustadas a mano.** El problema está mal planteado para las ecuaciones primitivas (Blayo & Debreu 2005) [verificar]; no hay benchmarks públicos de reflexión ni de balance de masa.
6. **Sin suites de prueba compartidas y versionadas.** COMODO, DOME y OMIP se corren de forma esporádica.

### 3b. Lo que piden los usuarios (foros y trackers, 2023–2026)

En orden de frecuencia:

1. **Preparar el dominio regional** (malla, batimetría, condiciones iniciales y de frontera, forzamiento). Es el tema número 1 en MOM6, MITgcm, ROMS, NEMO y Veros; han nacido herramientas externas solo para esto (regional-mom6, ROMS-Tools, CrocoDash).
2. **Inestabilidades y confusión con las fronteras abiertas.** Por ejemplo: MOM6 #753 y #1033, Oceananigans #3628 (45 comentarios), MITgcm #145 abierto desde 2018.
3. **Combinar la marea con el modelo padre de baja frecuencia en la frontera,** sin que la esponja mate la marea (mitgcm-support, septiembre de 2026).
4. **Instalación y compilación.** MITgcm #846 en Mac M3, SCHISM #13 con conda, regional-mom6 #79.
5. **Explosiones del mojado-secado.** En el foro de ROMS el desarrollador llegó a recomendar no usar la opción; MOM6 #235.
6. **Restarts no reproducibles.** MOM6 #975, #208, #508; SCHISM #140.
7. **Salidas y diagnósticos.** Un archivo por procesador, presupuestos que no cierran.
8. **Error del gradiente de presión en topografía empinada.**
9. **Elección del Δt y localización del punto donde explotó la corrida.**
10. **Anidamiento.**
11. **Documentación y ejemplos.**
12. **Reproducibilidad y CI.**
13. **Acoplamiento** con biogeoquímica, oleaje y hielo.
14. **GPU.**
15. **Asimilación de datos y ML**, poca evidencia pero creciente.

## 4. Posicionamiento de pyOCEAN

> El modelo regional y costero que se instala con `pip`, prepara su dominio solo, no explota en la costa, mide su propia mezcla numérica, reproduce bit a bit y da gradientes de todo respecto a todo.

Diferenciadores, en orden de valor sobre esfuerzo:

1. **Preparación del dominio integrada, con diagnósticos legibles.**
   - Revisa batimetría (rx0/rx1, celdas aisladas, fosas como la de Guaymas en GEBCO), archivos de frontera (NaN, cobertura, balance de masa), forzamiento y Δt.
   - Cuando la corrida explota, guarda el estado y dice dónde y qué término. Ya existe en parte (`run_tides.py` guarda el estado al explotar).
2. **Frontera de marea + modelo padre:** Flather con marea (ya existe), esponja que solo relaja la baja frecuencia, y benchmarks públicos de reflexión y balance de masa.
3. **Mojado-secado robusto con benchmarks públicos.** Ya están Thacker, TIDAL_FLAT 2D/3D y el Alto Golfo. Siguen el subgrid de Casulli (2009) y el frente que conserva momento (Stelling & Duinmeijer 2003).
4. **Restarts bit a bit y CI de regresión de respuestas**, incluyendo la historia del AB y el estado de frontera.
5. **Salida CF/xarray/SGRID y Zarr compatible con anemoi-datasets, con presupuestos cerrados.** Habilita Pangeo y ML.
6. **Diferenciabilidad de diseño** (sección 6).
7. **Mezcla espuria medida en línea** (RPE y decaimiento de varianza de Burchard & Rennau 2008) en cada corrida.

No conviene asumir temprano: el port a GPU masivo, el anidamiento de dos vías y el acoplamiento completo con oleaje y hielo. Equipos grandes (Oceananigans, Veros, MOM6) ya cubren eso, y cuesta mucho.

## 5. Decisiones numéricas (frontera del estado del arte)

| Tema | Decisión propuesta | Motivo |
|---|---|---|
| Coordenada vertical (Fase 3) | **Marco ALE general:** paso lagrangiano, regrid, remapeo PPM/PQM (White & Adcroft 2008). z*, σ, z̃, híbrida HYCOM y adaptativa (Hofmeister et al. 2010) como *funciones objetivo* | Una sola ruta de código; z̃ reduce la mezcla con ondas internas y mareas (Megann et al. 2022; Megann 2024) |
| Gradiente de presión | Volumen finito de Adcroft et al. (2008) [verificar], con prueba de monte submarino y de reposo sobre PAC_MEX en CI | Sirve para z, σ e híbrida; con EOS no lineal |
| Pasos de tiempo | Reemplazar AB2 por AB3 o LF-AM3 (Lemarié et al. 2015) [verificar]. Modo barotrópico split-explicit para la Fase 3; conservar θ-implícito para mojado-secado y validación | AB2 es débilmente inestable para oscilaciones sin disipación; split-explicit elimina SuperLU (65 % del paso) |
| Advección | UP3 como base; WENO5 para trazadores; momento en forma vector-invariante con Coriolis que conserva energía; WENO de vorticidad (Silvestri et al. 2024) como opción | Mezcla espuria controlada por Re_Δ (Ilicak 2012) |
| Mojado-secado | Subgrid de Casulli (2009) + frente de Stelling & Duinmeijer (2003) | Resuelve el `xfail` de Thacker y la sensibilidad a la topografía intermareal |
| No hidrostático | Opcional en la Fase 4, solo si Δx ≲ H (fisión de ondas solitarias); corrección tipo SWASH/Klingbeil en capas σ | Vitousek & Fringer 2011 [verificar] |
| Solver elíptico | Ya: reutilizar la LU cuando la matriz es fija; PyAMG + CG con warm start cuando cambia la máscara. Después: split-explicit en Numba/JAX | — |
| Turbulencia | GLS k-ε ya portado de CROCO; agregar k-ω y Mellor-Yamada con la misma interfaz (estilo GOTM); KPP sin la difusividad de fondo que suprime el espectro de ondas internas (Thakur et al. 2022) | — |

## 6. Diferenciabilidad, adjunto y ML

**Ruta técnica: núcleos en JAX (un segundo backend) más unas pocas derivadas escritas a mano.** No un adjunto escrito a mano, que serían dos modelos que mantener, ni Enzyme (exige Julia o MLIR).

- **Diseño desde ya:** núcleos en estilo funcional, sin modificar arreglos in situ. Fue la lección de Veros. Los tests actuales (tolerancias de 1e-9 a 1e-10) pasan a ser tests de equivalencia entre el backend NumPy/Numba y el de JAX.
- **Superficie libre implícita:** la pasada inversa resuelve con la transpuesta; como la matriz es simétrica definida positiva, es la misma LU que ya está guardada. Alternativa: `jax.lax.custom_linear_solve` con CG o Lineax. Por el teorema de la función implícita no se guardan las iteraciones.
- **Solver tridiagonal (Thomas):** derivada propia cuya pasada inversa es otro Thomas con los coeficientes transpuestos, en O(N).
- **Mojado-secado y `max`/`min`:** el paso hacia adelante se queda con los umbrales duros (bit a bit igual a lo validado) y solo el gradiente usa versiones suavizadas (softplus, sigmoide).
- **Memoria:** checkpointing con `jax.checkpoint` o `RecursiveCheckpointAdjoint` de Diffrax (Revolve).
- **Hardware:** jax-metal es experimental y las GPUs de Apple no tienen FP64. Localmente se usa JAX en CPU con x64; a escala, GPUs NVIDIA. Esto afecta la Fase 5, que hoy dice MLX.

**Aplicaciones, en orden de valor sobre esfuerzo:**
1. Exportar a Zarr con metadatos xgcm/CF.
2. Núcleo 2DH de marea en JAX para invertir C_d espacial, `drgscl` de JSL, constantes de frontera y correcciones de batimetría contra los 31 mareógrafos de PAC_MEX. Es justo el error de Guaymas que los barridos de C_d uniforme no corrigieron.
3. Mojado-secado diferenciable, validado contra diferencias finitas en TIDAL_FLAT.
4. Sensibilidades tangente-lineal y adjunta: qué celdas de frontera o batimetría controlan la M2 en Guaymas (el equivalente de *observation impact* en ROMS).
5. Solver Thomas y GLS diferenciables, para calibrar el 3D.
6. Término de corrección neuronal entrenado a través del solver, al estilo NeuralGCM.
7. Emulador regional o frontera aprendida con anemoi-training o el código de ACE/Samudra, entrenado con HYCOM-IOA del Golfo de México (comparable a Lupin-Jimenez et al. 2025).

**Anemoi** (Apache-2.0, en incubación) todavía no tiene océano y supone convenciones atmosféricas: nube de puntos lat/lon, pasos de 6 h, sin fronteras laterales abiertas. Lo reutilizable es anemoi-datasets (Zarr), anemoi-graphs (mallas con refinamiento costero, como el modelo *stretched-grid* "Bris" de MET Norway) y anemoi-training. Usarlo en océano regional sería trabajo pionero.

## 7. Marcador (se actualiza en cada benchmark)

| Caso | Verdad | pyOCEAN | Referencias | Estado |
|---|---|---|---|---|
| Seiche barotrópico BayIW | MITgcm (sin solución exacta) | periodo 3.83 h, RMSE 0.18–0.48 mm | MITgcm 3.81–3.83 h | ✅ igual a MITgcm |
| Thacker (mojado-secado) | Exacta | 4.6 cm, costa 0.8 km | CROCO 2.0 cm, 0.6 km | ❌ atrás (frente; `xfail`) |
| TIDAL_FLAT 2DH | CROCO | 1.1 cm | — | ✅ igual |
| TIDAL_FLAT 3D | CROCO (GLS) | 1.2 cm con 3D σ + GLS | — | ✅ igual |
| Mareas del Pacífico mexicano | 31 mareógrafos TICON | RSS mediana 1.44 cm | TPXO10 1.79 cm | ✅ adelante |
| Acapulco M2 | TICON | 0.6 cm | TPXO10 5–6 cm | ✅ adelante |
| Alto Golfo de California M2 | TPXO10 y TICON | amplitud rms 3.8 cm (log-law, z₀ = 1e-3), fase 12.7° (−7 a −9° a lo largo del eje, heredado de la frontera) | TPXO10 = referencia | 🔶 cerca; falta la fase |
| Banda de marea, HYCOM-IOA | IOC | 5.1 cm | HYCOM-IOA 4.9 cm | 🔶 similar |
| Lock exchange: frente | Benjamin ½√(g′H) = 0.495 | 0.463 / 0.486 m/s | CROCO 0.463 / 0.486 | ✅ igual a CROCO |
| Lock exchange: mezcla espuria (ΔRPE/RPE₀, 17 h) | Ilicak 2012 | 3.2e-5 (WENO-Z), 2.9e-5 (JS) | CROCO 2.2e-5 | 🔶 1.3–1.5× CROCO (paso de tiempo de los trazadores) |
| Seiche interno lineal (periodo) | 2L/√(g′h₁h₂/H) | +5 a +7 % (converge) | — | 🔶 |
| Monte submarino (gradiente de presión) | Reposo exacto | — | ROMS, CROCO | ⬜ pendiente |
| Restart bit a bit (σ estratificado + GLS, incl. arranque AB3) | Corrida continua | idéntico (`np.array_equal`), en CI | MOM6 y SCHISM fallan | ✅ (pendiente: corridas 2DH de marea) |
| Verificador de dominio (Alto Golfo GEBCO crudo) | Fosa de Guaymas encontrada a mano | la detecta solo (cuerpo aislado de 6 celdas, 189 m) + 1 cuerpo y 9 fosas más candidatos a artefacto | — | ✅ |
| Solver de superficie libre con mojado-secado (Alto Golfo, 96 273 celdas) | — | CG + LU desfasado: 1–3 iteraciones, 0.053 s/paso (antes Jacobi ~110 it, 0.079 s/paso) | — | ✅ 1.5× |
| Rendimiento | — | 33 días PAC_MEX 650×500 en 11 min, 1 núcleo | — | ⬜ falta medir HYCOM con la misma configuración |

## 8. Hoja de ruta propuesta

Reorganiza las fases del `CLAUDE.md`. Cambiar el plan del `CLAUDE.md` requiere la aprobación de Esteban.

**Etapa A, cimientos de calidad (corto plazo): ✅ cerrada el 2026-10-11.** Quedan abiertos el restart y la salida CF en las corridas 2DH de marea, y la RPE del lock exchange a 1.3–1.5× de CROCO (lado del momento; ver `validation/lock_exchange/README.md`).
- Diagnósticos de mezcla espuria (RPE, decaimiento de varianza) y casos *lock exchange* y seiche interno.
- Restart bit a bit con test de CI.
- AB3 o LF-AM3.
- Reutilizar la LU; PyAMG para mojado-secado.
- Salida CF/Zarr.
- Verificador de dominio (rx0/rx1, fosas, celdas aisladas, Δt) y diagnóstico post-mortem cuando una corrida explota.
- Aplicar el arrastre log-law a las corridas regionales.

**Etapa B, física 3D:**
- Advección vertical y estratificación en el 3D σ.
- Gradiente de presión de volumen finito con prueba de monte submarino.
- Momento vector-invariante.
- Subgrid de Casulli y frente de Stelling & Duinmeijer.
- Esponja que separa baja frecuencia y marea.
- Seiche baroclínico BayIW contra MITgcm.

**Etapa C, ALE híbrido (la Fase 3 actual):**
- Marco ALE con remapeo PPM/PQM y funciones objetivo z*, σ, z̃, híbrida y adaptativa.
- Modo barotrópico split-explicit.
- *Overflow* (DOME) y monte submarino como referencia de entrada.

**Etapa D, diferenciable:**
- Backend JAX del 2DH de marea con derivadas propias en la superficie libre y en Thomas.
- Inversión contra mareógrafos y mapas de sensibilidad del adjunto.
- Después, el 3D y el GLS.

**Etapa E, ondas internas y marea baroclínica (la Fase 4 actual):**
- Marea interna en PAC_MEX en lugar del arrastre JSL.
- KPP sin difusividad de fondo.
- No hidrostático opcional.

**Etapa F, operativo y ML (la Fase 5 actual):**
- Ciclo de pronóstico contra HYCOM-IOA usando observaciones independientes.
- Emulador regional y frontera aprendida (Anemoi o ACE).
- GPU NVIDIA con JAX.

Criterio de avance entre etapas: cada etapa cierra con sus filas del marcador en ✅ o con la diferencia explicada.
