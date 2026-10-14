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
| Monte submarino (gradiente de presión, rx1 = 10.7) | Reposo exacto | max\|u\| a 10 d: σ jacobiano + ρ_ref 0.41 cm/s; σ volumen finito + ρ_ref 1.36; **z\* (ALE) + volumen finito + ρ_ref 0.10**; sin ρ_ref en σ 42–60 | Beckmann & Haidvogel 1993: ~1 cm/s | 🔶 z\* adelante; todas las configuraciones tienen un modo que se duplica cada ~2 días (σ jacobiano 9 cm/s a 20 d), por investigar (`validation/seamount/README.md`) |
| Escalón de la boca de BayIW 600 → 164 m (reposo, picnoclina a 80 m, 12 h) | Reposo exacto | σ + jacobiano + ρ_ref 7.3 cm/s; z\* + volumen finito 5e-12 m/s | — | ✅ z\* al nivel de redondeo (`tests/test_sigma_ale.py`) |
| Onda interna modo 1 analítica (canal plano 200 m, oeste impuesto `'clamped'`, este radiante + esponja) | u = u₀ cos(πz/H) sin(ωt − kx) con decaimiento viscoso | z\* e híbrida: amplitud ±5 %, fase < 3.4° (KE Arakawa); la KE Godunov por defecto amortigua ~9 % más en 5 km con dx = 100 m (primer orden en dx) | — | ✅ (`tests/test_sigma_ale.py`; encontró y corrigió esponja en borde impuesto y exterior de borde sin datos) |
| `internal_wave` de MITgcm (onda de modo 1 impuesta, pendiente subcrítica 0.03, 10 períodos) | MITgcm con la misma configuración | modo 1 en la parte plana: Arakawa z\*/híbrida u 0.91–1.01, T′ 0.98–1.08, fase ≤ 7°; Godunov u 0.83–0.99; flujo de energía a 2 km 0.92–1.04; 50 % del flujo a 4.53–4.85 km (MITgcm 4.61) | MITgcm = referencia | ✅ con Arakawa; 🔶 con Godunov (por defecto, c) (`validation/internal_wave/README.md`) |
| Sección 2-D BayIW (eje de la bahía, 1 × 352 × 50, 60 h; dos capas y lineal, con y sin bahía) | MITgcm con la misma sección, paredes en x y frontera sur solo Orlanski (la de Esteban refleja el pulso barotrópico: índice 0.57 contra 0.02) | seiche Δη: rango ±3 %, correlación 1.00 (z\*, híbrida 25 y 50); lineal: z\* T110/T250/V110 0.97–1.01 (corr. 1.00), híbrida 50 0.97–1.05 (corr. ≥ 0.99), híbrida 25 igual salvo T110 en la bahía somera (3–6×); dos capas: z\* T110 1.3–2.3× cerca de la interfaz (difusión, c), híbrida 25 V110 en la bahía 0.90–0.93 (corr. ≥ 0.96) pero 0.27–0.61 en la boca, híbrida 50 ruido de ~1 mm/s en capas casi desvanecidas | MITgcm = referencia | 🔶 lineal ✅; dos capas abierto (capas vacías, boca). Corrigió dos errores (a): estado inicial híbrido desde la malla de origen y profundidad efectiva de MITgcm (`validation/bayiw_2d/README.md`) |
| Corte 3-D BayIW, fase 1 (200 × 160 × 50, ~5 km, 60 h; dos capas y lineal, con y sin bahía; radiando a un exterior en reposo) | MITgcm en el mismo corte, 5 MPI, solo Orlanski | z\*: seiche período −1.8 %, rango ±1 %, corr. 1.00; lineal T110/V110 0.98–1.07 (corr. ≥ 0.97); dos capas isoterma 1.01–1.21, desplazamiento por calor 1.03–1.10 (corr. ≥ 0.99), T110 1.4–3.5× (difusión de la interfaz, c); Δt 30 = 120 s (≤ 2 %). Híbrida 25: seiche igual; lineal ✅ fuera de la bahía, T110 2.5–3× dentro; dos capas: interfaz casi quieta en Omega (0.08×), 3–6× en la bahía, U110 0.5× en A y B, igual con Godunov | MITgcm = referencia | 🔶 z\* ✅; híbrida en dos capas: candidato (a), en diagnóstico (`validation/bayiw_cut/README.md`) |
| Corte 3-D BayIW, fase 2 (anidado en las corridas completas de Esteban, z\*) | MITgcm anidado (OBCS prescribe u, v, T; η no se puede con superficie lineal) | η en las estaciones: pyOCEAN 0.03–0.11 frente a MITgcm anidado 0.05–0.17 (fase 1: 0.27–1.00), así que el anidado funciona; en V110/isoterma/ζ/T110/mapa MITgcm anidado es casi exacto (≤ 0.02 fuera de Gamma), así que el criterio 1.5× degenera ahí, y el error de pyOCEAN es igual al de la fase 1; efecto de la bahía de dos capas 0.68–0.75 frente a 0.13–0.19 (difusión de la interfaz en z\*, c); Gamma lineal 0.74 dentro de la esponja (esponja hacia el perfil del borde copiado) | Corrida completa de Esteban | 🔶 anidado ✅ (η mejor que MITgcm anidado); falta exterior 3-D en la banda de esponja (b) |
| Híbrida con interfaz marcada y caso realista (sección 2-D y corte 3-D, 2026-10-13) | MITgcm en la misma sección/corte | tres correcciones: piso z\* hasta el fondo (`bayiw.hybrid_dz_top`, aviso del modelo si no llega), inicialización regrideada sobre los niveles de origen como MOM6 + `snap_to_jumps` (a), `keep_isopycnal` conserva interfaces cuya meta queda entre sus dos capas (a); 2-D dos capas ζ 0.94–1.29 (antes 0.05–1.70), interfaz 8 m (MITgcm 8–9); realista (T equivalente en densidad) z\* e híbrida = MITgcm en 2-D y 3-D (T110 0.93–1.12, V110 1.01–1.04); 3-D dos capas: amplitud 0.87–1.36 pero retraso de 1.5–3 h en el borde de la plataforma; lineal: bahía somera T110 2.5–5.6 | MITgcm = referencia | 🔶 realista ✅; dos capas: amplitud ✅, fase en 3-D abierta; lineal en la bahía abierto (`validation/bayiw_2d/README.md`, `validation/bayiw_cut/README.md`) |
| MOM6 en BayIW (sección 2-D, 60 h; costo en el corte) | MOM6 z\*/HYCOM1 (`-O2`, misma física; HYBGEN no corre en este build) y MITgcm | z\*: pyOCEAN = MOM6 en todas las métricas (incluidos T110 inflada y V110 de Seiche2, propios de z\*); realista: híbrida de pyOCEAN mejor que HYCOM1 de MOM6 en la bahía (Seiche2 V110 0.99 frente a 2.60), aunque las dos híbridas tienen un error local cerca del fondo de la cabecera (error de sección 6.7 frente a 14.5; z\* 0.04); dos capas: pyOCEAN mantiene la interfaz (8 m; MOM6 HYCOM1 28 m) pero desplaza la interfaz de más (ζ 1.17–1.65; MOM6 1.02–1.15); costo igual a igual Δt, a ±10 % (z\* 9.1 frente a 9.0 s/h, híbrida 5.0 frente a 4.7; con Δt 600 s 1.8/1.9 y 1.0/1.0); ALE de MOM6 18.6–22.8 % del paso frente a ~13.5 % del remapeo de pyOCEAN; advección de T de pyOCEAN (WENO5, 3 etapas) ~11 % frente a 4.8 % de MOM6 | MITgcm = referencia; MOM6 = segundo modelo con remapeo | 🔶 z\* ✅ (= MOM6); costo ✅ (= MOM6); ζ de la híbrida en dos capas: candidato (a), pendiente (`validation/mom6_bayiw/README.md`) |
| Mezcla espuria frente a Re_Δ: lock exchange, overflow y onda interna de Ilicak (2012), 3 ν_h cada uno | Benjamin (frente); comparación entre modelos con la misma coordenada | z\*: = MOM6 z\* en el lock exchange (RPE ±1.6 %, frentes iguales, 0.86–0.96 U), ≈ MITgcm en el overflow (dRPE/dt 0.58 frente a 0.52 con ν = 1000) pero con extremos nuevos de T de WENO5-Z (≤ 4.7 °C en el 0.25 % del volumen), 1.7–2.6 veces menos mezcla que MOM6 z\* y MITgcm en la onda interna **pero amortigua la oscilación más que todos con ν bajo (0.39–0.45 frente a 0.55–0.72)**; dRPE/dt baja 3.6 veces de Re_Δ 2480 a 2.5 (MOM6 3.7, MITgcm 5.6); híbrida: ~1000 veces menos mezcla que z\* y frentes 0.96–1.04 U, pero T fuera de rango en capas < 1 m (3.2–33 °C) y no completa el overflow con ν ≤ 100 | MOM6 z\*/ρ, CROCO σ (esquemas REGIONAL: extremos de hasta 14 °C, RPE no comparable), MITgcm z\* | 🔶 z\* ✅ en mezcla; amortiguamiento de la onda interna (a)/(b) a diagnosticar; extremos de WENO5-Z (a); híbrida: capas delgadas y robustez (a) (`validation/ilicak_benchmark/README.md`) |
| Seamount de Beckmann & Haidvogel en cuatro modelos (10 días, sin viscosidad) | velocidad nula | σ con PGF de volumen finito + ρ_ref 0.22 cm/s; z\* 0.08 cm/s | CROCO σ 36 cm/s, MOM6 σ 154 cm/s (crecen), MITgcm z\* 0 exacto; MOM6 z\* trunca desde el día 1 (configuración sin aclarar) | ✅ (MOM6 pendiente) |
| Costo en los casos de Ilicak (Δt máximo estable, 1 proceso, s por hora simulada) | — | lock exchange 0.046, overflow 0.40 (Δt 80 s, el mayor), onda interna 0.0017, seamount 0.20 | MOM6 0.076 / 4.3 / 0.0025 / 0.19 (σ); CROCO 0.015 / 0.94 / 0.0008 / 0.12; MITgcm 0.106 / 4.7 / 0.0038 / 0.027; HYCOM compila pero no corre (sin entradas) | ✅ frente a MOM6 y MITgcm en 2-D; 🔶 CROCO 1.7–3 veces más barato salvo en el overflow, MITgcm 7.6 veces en el seamount |
| Escalón de la boca de BayIW, reposo en capas híbridas (25 capas, estratificación lineal, 6 h sin viento) | Reposo exacto | antes 1.6 cm/s (inicio en capas σ); con `initial_state(h_src=...)` y la profundidad de MITgcm 3.3e-4 m/s; canal con escalón < 1e-6 m/s en 1 h | — | ✅ (`tests/test_sigma_ale.py`) |
| Lock exchange híbrido isopicno (ALE, 21 capas como MOM6, 6 h; cada masa de agua en una capa desvanecida al otro lado) | Benjamin ½√(g′H); MOM6 1.05 U, max\|u\| 0.56 m/s | momento vectorial invariante + `keep_isopycnal` + viscosidad de MOM6; con 10 capas y upwind: cuerpo 1.05 / 1.05 U, ganancia de RPE 3.0e6 J (MOM6 RHO 4.5e7, ALE-σ 2.1e8); punta densa a 1 m 1.12 U, max\|u\| 0.94 m/s (antes, con forma de flujo: cuerpo 0.84 / 0.14 U). Con 21 capas y `thickness_flux='ppm'` (configuración de MOM6, por defecto desde 2026-10-12): punta y cuerpo 1.05 / 0.98 U, max\|u\| 0.74 m/s, RPE 8.8e5 J (`validation/mom6_ke_schemes/`) | MOM6 (`validation/mom6_lock_exchange/`) | ✅ con 21 capas + PPM (= MOM6); con 10 capas PPM da cuerpo denso 1.26 U |
| Transporte de Ekman (viento uniforme, plano f) | τ/(ρ₀f) inercial exacto | error máx 0.045 % en un periodo inercial | — | ✅ |
| Fronteras abiertas: onda interna modo 1 saliendo de un canal (error interior vs dominio 5× mayor, 16 h) | Dominio grande | cerrado 0.82; Flather + Orlanski 0.16; + esponja de 10 celdas 0.015. ALE (Plan C, 2026-10-12; error en niveles z, σ con esa métrica 0.80 / 0.149): z\* 0.79 / 0.16 / 0.016, híbrida 0.77 / 0.198 / 0.024 (la entrada desde el exterior en z mete agua de otra clase de densidad en las capas isopicnas desplazadas; por índice de capa daría 0.180) | Jensen 1998, Chapman 1985: radiación sola refleja | ✅ con esponja (pendiente: Marsaleix 2009 PRM para radiación sin esponja; benchmark público de reflexión) |
| Restart bit a bit (σ estratificado + GLS, incl. arranque AB3) | Corrida continua | idéntico (`np.array_equal`), en CI | MOM6 y SCHISM fallan | ✅ (pendiente: corridas 2DH de marea) |
| Verificador de dominio (Alto Golfo GEBCO crudo) | Fosa de Guaymas encontrada a mano | la detecta solo (cuerpo aislado de 6 celdas, 189 m) + 1 cuerpo y 9 fosas más candidatos a artefacto | — | ✅ |
| Solver de superficie libre con mojado-secado (Alto Golfo, 96 273 celdas) | — | CG + LU desfasado: 1–3 iteraciones, 0.053 s/paso (antes Jacobi ~110 it, 0.079 s/paso) | — | ✅ 1.5× |
| Rendimiento | — | 33 días PAC_MEX 650×500 en 11 min, 1 núcleo | — | ⬜ falta medir HYCOM con la misma configuración |
| Kit de columna ALE: remapeo PCM/PLM/PPM_H4 (nivel 1) | Fortran de MOM6 `remapping_core_h` | max \|Δ\| ≤ 5e-12 en 120 columnas, incluidas capas vacías, de 1e-10 m y de 1 mm en los extremos (fuera de la comparación: la reducción de orden cuenta capas activas, MOM6 cuenta todas) | MOM6 = referencia | ✅ igual a MOM6 |
| Kit de columna ALE: reglas z\*, σ e interpolación P1M_H2 (nivel 1) | Fortran de MOM6 `coord_zlike`, `coord_sigma`, `regrid_interp` | z\*/σ ≤ 3e-14 m; P1M bit a bit (pendiente: `build_hycom1_column` completo) | MOM6 = referencia | ✅ igual a MOM6 |
| Regla híbrida HYCOM1 frente a `hybgen` de HYCOM (puerto de MOM6), 10 ciclos | — (algoritmos distintos) | objetivos estilo OM4: Δz medio 4.0 m en BayIW y 0.8 m estratificada (≤ 0.4 capas en el interior); objetivos desde la superficie: 13–25 capas por las z fijas de hybgen y su modo nulo | hybgen | ✅ no se porta hybgen (2026-10-11); Plan B: transición somera por ventana y objetivos ligeros arriba (`target_densities(n_light)`); Plan B2: las capas sin masa toman la densidad de la capa con masa de arriba (relleno de hybgen), columna «164 m, fondo ligero» a 10 ciclos más cerca de hybgen (Δz medio 55.8 → 51.7 m, 6.2 → 3.3 m con objetivos OM4) (`validation/mom6_column/README.md`) |

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
