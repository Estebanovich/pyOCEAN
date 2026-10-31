# Visión y hoja de ruta de pyOCEAN: qué modelo necesita la comunidad

Documento de diseño. Fecha: 2026-10-11 (sección 9 y Etapa N agregadas el 2026-10-15).

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
| Mezcla espuria frente a Re_Δ: lock exchange, overflow y onda interna de Ilicak (2012), 3 ν_h cada uno | Benjamin (frente); comparación entre modelos con la misma coordenada | z\*: = MOM6 z\* en el lock exchange (RPE ±1.6 %, frentes iguales, 0.86–0.96 U), ≈ MITgcm en el overflow (dRPE/dt 0.61 frente a 0.52 con ν = 1000), sin extremos nuevos de T desde el FCT de Zalesak del 2026-10-14 (antes ≤ 4.7 °C con WENO5-Z; ahora ≤ 1.5e-8 °C), en la onda interna mezcla como MITgcm y amortigua como los demás (0.74 / 0.58 / 0.26 frente a 0.65–0.72 / 0.55–0.65 / 0.24–0.26) desde la corrección del 2026-10-14 (`theta_bc` = ½: antes el θ = 0.6 de la superficie libre amortiguaba el transporte baroclínico de las capas, 0.39–0.45); dRPE/dt baja 3.6 veces de Re_Δ 2480 a 2.5 (MOM6 3.7, MITgcm 5.6); híbrida (plan de robustez, 2026-10-14: limitador de flujo saliente, acoplamiento viscoso de capas delgadas sesgado aguas arriba como MOM6, FCT con cotas 3-D, θ_bc = ½): 250–1200 veces menos mezcla que z\*, T en el rango exacto (antes 3.2–33 °C), completa el overflow con ν = 100 y 1000 (MOM6 ρ se detiene con ν ≤ 100, HYCOM con 100; max\|u\| 17 m/s, antes 47); con ν = 1 solo 1 de 5 trayectorias (lámina de capas evanescentes junto a la pared, ya existía: 3 de 5 sin la corrección δ de la media vertical), onda interna 0.50 / 0.63 / 0.27 (antes 0.45 / 0.43 / 0.25), frentes 0.94–1.06 U (el acoplamiento frena 2–5 % con ν = 0.1) | MOM6 z\*/ρ, CROCO σ (esquemas REGIONAL: extremos de hasta 14 °C, RPE no comparable), MITgcm z\* | 🔶 z\* ✅; híbrida ✅ en rango de T y robustez; abiertos (a): onda interna con ν = 0.01 (0.50 frente a 0.65–0.72), caras delgadas del overflow hasta 1.9× las gruesas (2.0× con ν = 100; δ uniforme en la columna), lámina de capas evanescentes con ν = 1 (`validation/ilicak_benchmark/README.md`) |
| Seamount de Beckmann & Haidvogel en cuatro modelos (10 días, sin viscosidad) | velocidad nula | σ con PGF de volumen finito + ρ_ref 0.22 cm/s; z\* 0.08 cm/s | CROCO σ 36 cm/s, MOM6 σ 154 cm/s (crecen), MITgcm z\* 0 exacto; MOM6 z\* trunca desde el día 1 (configuración sin aclarar) | ✅ (MOM6 pendiente) |
| Costo en los casos de Ilicak (Δt máximo estable, 1 proceso, s por hora simulada) | — | lock exchange 0.049, overflow 0.45 (Δt 80 s, el mayor; híbrida 1.01 con Δt 40 s), onda interna 0.0019, seamount 0.22 (repetido el 2026-10-14 con la máquina libre) | MOM6 0.073 / 4.1 / 0.0024 / 0.18 (σ); CROCO 0.015 / 0.91 / 0.0008 / 0.12; MITgcm 0.101 / 4.5 / 0.0037 / 0.025; HYCOM, solo en el overflow, 5.41 (Δt 20 s, marginal, con ν = 1000; 1.43 con 5 procesos; con ν = 100 se detiene por `neg. dp`, sin diagnosticar) | ✅ frente a MOM6, MITgcm y HYCOM en 2-D; 🔶 CROCO 1.9–3.3 veces más barato salvo en el overflow, MITgcm 8.8 veces en el seamount |
| Overflow DOME (Legg et al. 2006, Caso 1, geometría de MOM6; 10 y 5 km, 50 días) | Legg 2006 (Figs. 15–17) y MITgcm z con la misma física | z\*: b_f a 1.6–3 % de MITgcm, transporte de la pluma ≤ 9–18 %, trayectoria 1–3.5 celdas más abajo, α_E +29–47 % (c, en estudio); σ entre z y la isopicna; híbrida b_f −16 a −17.5 contra −12 de Legg isopicno con ν constante (falta mezcla por Ri, b; con GLS en ALE, N3: −12.99, ver la fila «DOME híbrido + GLS»); entrada exacta en volumen; T y s en rango exacto, sin truncaciones con Δt 600 s, 4b/4d no aparecen; tendencia con la resolución como Legg; costo 14 s/día a 10 km (5 hilos) | MITgcm "legg" = Legg a 7 % en b_f y 5 % en la trayectoria lejana; MOM6 ρ no comparable (su tinte no entra, d); MITgcm 4.6 s/día, MOM6 12.0 s/día (Δt 1200) | 🔶 z\* ✅ en b_f y transporte; abiertos (a): inestabilidad con θ_bc = ½ a 10 km / 1200 s (Δt máximo la mitad que MOM6, plan aparte) y entrada por frontera abierta de la híbrida promediada en capas gruesas; (b) mezcla por Ri (Smagorinsky ya está, ver su fila) (`validation/dome/README.md`) |
| Escalón de la boca de BayIW, reposo en capas híbridas (25 capas, estratificación lineal, 6 h sin viento) | Reposo exacto | antes 1.6 cm/s (inicio en capas σ); con `initial_state(h_src=...)` y la profundidad de MITgcm 3.3e-4 m/s; canal con escalón < 1e-6 m/s en 1 h | — | ✅ (`tests/test_sigma_ale.py`) |
| Lock exchange híbrido isopicno (ALE, 21 capas como MOM6, 6 h; cada masa de agua en una capa desvanecida al otro lado) | Benjamin ½√(g′H); MOM6 1.05 U, max\|u\| 0.56 m/s | momento vectorial invariante + `keep_isopycnal` + viscosidad de MOM6; con 10 capas y upwind: cuerpo 1.05 / 1.05 U, ganancia de RPE 3.0e6 J (MOM6 RHO 4.5e7, ALE-σ 2.1e8); punta densa a 1 m 1.12 U, max\|u\| 0.94 m/s (antes, con forma de flujo: cuerpo 0.84 / 0.14 U). Con 21 capas y `thickness_flux='ppm'` (configuración de MOM6, por defecto desde 2026-10-12): punta y cuerpo 1.05 / 0.98 U, max\|u\| 0.74 m/s, RPE 8.8e5 J (`validation/mom6_ke_schemes/`) | MOM6 (`validation/mom6_lock_exchange/`) | ✅ con 21 capas + PPM (= MOM6); con 10 capas PPM da cuerpo denso 1.26 U |
| Transporte de Ekman (viento uniforme, plano f) | τ/(ρ₀f) inercial exacto | error máx 0.045 % en un periodo inercial | — | ✅ |
| Fronteras abiertas: onda interna modo 1 saliendo de un canal (error interior vs dominio 5× mayor, 16 h) | Dominio grande | cerrado 0.82; Flather + Orlanski 0.16; + esponja de 10 celdas 0.015. ALE (Plan C, 2026-10-12; error en niveles z, σ con esa métrica 0.80 / 0.149): z\* 0.79 / 0.16 / 0.016, híbrida 0.77 / 0.198 / 0.024 (la entrada desde el exterior en z mete agua de otra clase de densidad en las capas isopicnas desplazadas; por índice de capa daría 0.180) | Jensen 1998, Chapman 1985: radiación sola refleja | ✅ con esponja (pendiente: Marsaleix 2009 PRM para radiación sin esponja; benchmark público de reflexión) |
| Restart bit a bit (σ estratificado + GLS, incl. arranque AB3) | Corrida continua | idéntico (`np.array_equal`), en CI | MOM6 y SCHISM fallan | ✅ (pendiente: corridas 2DH de marea) |
| Estuario idealizado de Lorenz et al. (2021) (canal 500 × 1 × 60, E, P, río y M2; experimentos A–D, 600 ciclos) | GETM (Tabla 2); la relación de mezcla exacta (eq. 32) | relación exacta a 1e-10 en σ + GLS estratificado y ALE-σ (balance de varianza cerrado); las diferencias de la Tabla 2 son sobre todo de clase (c), cierre turbulento y advección: ALE-σ con cierre constante mezcla 3× GETM en A, sin tapón en B y C, intercambio 6× en D; σ + GLS con el tapón 3–4 g/kg más salado; B y C (ambas) y D σ + GLS no llegan a estado estacionario en 600 ciclos; dV de GETM sin explicar; con GLS en ALE (N3) ver la fila «Lorenz ALE-σ + GLS» | GETM: relación exacta a precisión de máquina | 🔶 balance exacto; magnitudes de mezcla no cerradas (`validation/lorenz_estuary/README.md`, «Qué debe mejorar pyOCEAN») |
| Columnas de CROCO con GLS k-ε (N3; Kato–Phillips, Willis–Deardorff, ciclo diurno, FORCED_EKBBL/DBLEEK/NONROTBBL/OSCNONROTBBL; 5×5 periódica, Δt 30 s) | CROCO (mismo caso, compilado localmente); Kato–Phillips contra Price (1979) | σ, z\* e híbrida: profundidad de mezcla de Kato–Phillips = CROCO (−0.0 %, híbrida +0.1 %) y +4.4 % sobre Price (CROCO también +4.4 %); Willis–Deardorff < 1e-3 del rango; ciclo diurno ≤ 0.4 %; DBLEEK ≤ 0.8 %; NONROTBBL ≤ 0.9 %; OSC amplitud −0.4 %, fase 0°; EKBBL giro en la capa del fondo **75.99°** contra 75.77° (62.8° antes de la corrección (a) del arrastre log con la velocidad completa, Task 9b), k máx 0.1297 contra 0.1312; z\* con capas fijas = σ a ≤ 3e-9 | CROCO = referencia | ✅ (`validation/turbulence_columns/README.md`; la regla del umbral de capas delgadas de la GLS en ALE bajó el RMS de D_m en el ciclo diurno híbrido de 0.72 a 0.48 m) |
| Columnas de MOM6/CVMix con KPP (N3; `CVmix_SCM_tests`: wind_only, cooling_only, mech_only, skin_warming_wind; 5 días) | MOM6 con CVMix (mismo caso, `MOM_override` sin cambios) | profundidad de la capa límite y MLD₀.₀₃ a ≤ 0.3 % en z\* y σ; T idéntica a 4.8e-7 °C (float32 de MOM6) en enfriamiento y mezcla mecánica; con viento la velocidad inercial difiere 0.6–1.3 % (c, paso de tiempo, sin aislar); contra la KPP propia de CROCO 13–27 % menos profunda (c, y el α de la EOS lineal de CROCO sin 1/ρ₀, d) | MOM6/CVMix = referencia | ✅ (`validation/turbulence_columns/README.md`, «Columnas MOM6/CVMix»); falta mezcla interior por Ri en `'kpp'` |
| Lorenz et al. (2021), ALE-σ + GLS (N3, Δt T_M2/746, `hbbl` 0.05 m) | GETM (Tabla 2); relación exacta (eq. 32) | se acerca a σ + GLS dentro de 0–4 % en S_max, Me, Q_in y s_in; B, C y D cierran la relación a 1.6e-10, 1.9e-10 y 1.1e-10, **A a 1.5e-9** (límite 5e-10): ganancia de redondeo de un solo signo del remapeo ALE (+1.4 ε Σ V S² por paso, `remap.py:369-372`, igual que MOM6 `MOM_remapping.F90:939-940`), clase (c), **corregida el 2026-10-30** (remapeo exacto en contenido: A 2.9e-11; Me de A 306.6 contra 281.0 por divergencia de trayectoria, variabilidad de la ventana); el sesgo por paso explica el error absoluto solo en A (en B y D 48–60 %) | GETM, σ + GLS, ALE-σ con ν constante | 🔶 acercó el ALE a σ + GLS; criterio 4 cumplido en A–D (A desde el remapeo exacto del 2026-10-30); magnitudes de mezcla frente a GETM siguen siendo (c) |
| DOME híbrido + GLS (N3; 10 km, 50 días, Δt 600 s) | Legg 2006 (iso, ν 0.5 = −12.0e-3); MITgcm z | b_f **−12.99e-3** (−16.18e-3 con ν constante; MITgcm legg −10.85; z\* −10.07), 76 % del camino hacia Legg; T en el rango exacto de las fuentes; **un** truncamiento CFL (día 31.8, cara de 1.3 cm, v = 8.96 m/s) y 1072 eventos del limitador (0 con ν constante); max\|u\| 5.7 contra 3.2 m/s; costo 20.9 contra 14.9 s/día; el control ν_v = 1e-5 constante revienta por CFL a los 38 s | Legg 2006; MITgcm | 🔶 b_f ✅; criterio «sin truncamientos nuevos» no cumplido por un evento (c/b); Smagorinsky: ver la fila siguiente; falta separar el efecto de la entrada (`validation/dome/README.md`) |
| Viscosidad horizontal de Smagorinsky laplaciana (forma de tensor de esfuerzos con grosor, MOM6; `smagorinsky=C`, apagada por defecto = bit a bit; cota de estabilidad de MOM6 reescalada a AB3, 0.055 Δx²/Δt) | Ilicak 2012 (Fig. 19b); MITgcm legg y Legg 2006 (DOME); MOM6 `MOM_hor_visc.F90` | Corridas ALE de después de la corrección I1 (47f2730: media de cara sin recortar de MOM6; antes ν, incluido ah, caía a ≈ 0 junto a cada escalón, a), con un control de operador C ≈ 0 que separa el cambio de operador del término: DOME híbrido + GLS, 10 km, 50 días, C = 0.15: truncamientos CFL 1 → 0 y eventos del limitador 1072 → 1 (el control conserva 1 y 641: es el término), max\|u\| en caras gruesas 5.52 → 3.54 m/s (MITgcm 2.04; 3.56 ya con el operador solo), b_f −12.99e-3 → −13.31e-3 (−13.05 el control; se aleja de Legg −12.0e-3), α_E −26 % por el término, trayectoria 3–9 km más abajo, costo +2.4 %; overflow z\* de Ilicak (ν_h 100 + C 0.5): ΔRPE −48 % (−47 % contra el control, que mueve −1.4 %), dRPE/dt −52 %, sin extremos nuevos (la tendencia del artículo); Lorenz D σ + GLS a T_M2/498 completa 150 ciclos con C = 0.15, pero el control sin Smagorinsky también (max\|u\| 1.46 contra 2.06 m/s), relación exacta 8.5e-12; costo por paso del término activo: BayIW cut z\* 1.041, DOME 0.992 | Ilicak 2012: menos mezcla espuria con Smagorinsky (MITgcm); MOM6 DOME propio usa Smagorinsky biarmónico | 🔶 criterio 5 ✅ (para los truncamientos de este caso) y 8 ✅; criterio 7: Smagorinsky no extiende el Δt estable, producción sigue en T_M2/746 (c); criterio 6 `kv_low` (ν_v 1e-4, ah 50, C 0.15): completa 50 días (MOM6 y MITgcm paran en el día 3–4 sin Smagorinsky) pero con 61 truncamientos, 239 337 eventos del limitador y capas interiores de la cuenca profunda acelerándose desde el día 37 hasta 8.29 m/s el día 50 (99.5 % del umbral), no robusto (b, falta mezcla por cizalla). Abiertos: contador `n_visc_limited` no separa el tope de Smagorinsky del de caras delgadas; biarmónica fuera de alcance; el smoke de `ilicak_overflow_zstar` cuesta 1.08× con C = 0 (causa sin aislar) (`validation/dome/README.md`, `validation/ilicak_benchmark/README.md`, `validation/lorenz_estuary/README.md`, `validation/bayiw_cut/README.md`) |
| Mezcla interior por cizalla de Jackson et al. (2008) (`shear_mixing='jackson'`, puerto de MOM6 `kappa_shear`; suma al cierre constante o KPP, rechazada con GLS; apagada por defecto = bit a bit) contra la verdad no hidrostática de Legg et al. (2006), DOME 10 km | Columna: MOM6 compilado; DOME: no hidrostático de Legg (0.5 km, digitalizado salvo E) y sus corridas de 10 km | Columna = MOM6 bit a bit en 18 columnas (tolerancia 1e-10). DOME híbrida ν_v 1e-4 + Smagorinsky 0.15, sin → con Jackson Pr = 1: b_f −17.91 → −12.84e-3 (verdad −8.75 dig.; isopicnal de Legg a 10 km −11.07), E de la esquina ≈ 0 → 4.88e-4 (isopicnal de Legg a 10 km 9.4e-4, banda de definición ×2), T(x′ = −200) 10.1 → 15.9 Sv (verdad 22.5 dig.), α_E 1.9e-5 → 4.77e-4 (1.03e-3 dig.), clase con más trazador en x′ = −300 −19…−18 → −17…−16 (verdad −10…−9 dig.), trayectoria a ≤ 2 celdas del isopicnal de Legg (se acerca en x′ = −300 y −500 km, se aleja en −200: 11 → 21 km); truncamientos 61 → 0, limitador 239 337 → 0, máx. diario de \|u\| 5.99 → 2.45 m/s; Pr = 0 (configuración del DOME de MOM6): b_f −8.29 pero la pluma no baja (Y′ −50 km en x′ = −200); con ν_v = 0.5 Jackson no actúa (Ri > ¼); z\* y σ: E y velocidades mejoran, b_f casi igual (el −8.81 de z\* sería mezcla numérica [hipótesis], por medir con el balance de varianza); Ilicak overflow z\* ν_h 100: ΔRPE −24 %, mezcla física ≤ 0.24 de 1.34 W m⁻² (fuera de línea); costo por paso 1.096× DOME, 1.275× BayIW cut | Legg 2006: sus corridas de 10 km quedan igual de lejos del no hidrostático (b_f −11.07 / −11.67); GLS (N3) −12.99 | 🔶 se acerca en todas las métricas contra la verdad no hidrostática (criterio 4 en dirección ✅, magnitud abierta: resolución c, entrada de la híbrida a, Pr/κ0 c); robustez ✅ (criterio 5); `validation/dome/README.md` |
| Mezcla diapicna de la capa de fondo por la fricción (`bbl_mixing='lotw'`, puerto de MOM6/OM4 `add_LOTW_BBL_diffusivity` con la energía física de la fricción de pyOCEAN; suma al cierre constante, KPP o Jackson; solo ALE; apagada por defecto = bit a bit) contra la isopicna de 10 km de Legg et al. (2006), DOME 10 km | Columna: MOM6 compilado; DOME: isopicna de Legg a 10 km (misma física, con su mezcla BBL; b_f −11.07e-3 dig.) y, en segundo lugar, su no hidrostático (0.5 km, dig.) | Columna = MOM6 a ≤ 1.5e-17 en 15 columnas; el factor de `set_BBL_TKE` de MOM6 (√2 u\*, 2 C_d\|U\|³) medido y documentado como (d): γ = 0.2 aquí = 0.1 en MOM6. DOME híbrida ν_v 1e-4 + Smagorinsky 0.15 + Jackson, U_bg 0.05, control → BBL (tras la corrección I1, N² con las densidades fusionadas, clase a; corridas repetidas 2026-10-31): b_f −12.70 → −11.86e-3 (52 % del camino hacia −11.07, 71 % antes de I1; no hidrostático −8.75 dig.), clase con más trazador en x′ = −300 −17…−16 → −14…−13 (dig. −10…−9) y en x′ = −100 deja de dominar la clase de entrada, pero T(x′ = −300) 19.0 → 18.0 Sv (dig. 20.6; 20.8 antes de I1) y α_E 9.95e-4 → 6.40e-4 (dig. 1.03e-3) se alejan (abierto); E de la esquina 3.51e-4 → 2.66e-4 (se aleja de 9.4e-4: parte de la dilución pasa al canal, T_in 9.96 → 11.38 Sv, c); trayectoria más cerca aguas abajo (x′ ≤ −500), más lejos cerca de la esquina; 0 truncamientos y **0 eventos del limitador** (32 antes de I1, con capas de fondo de 3–4 m/s en los días 46–50, que desaparecieron). U_bg sola (U_bg 0 → 0.05): b_f −12.84 → −12.70. z\*: −9.04 → −8.76 (poco efecto: niveles de 144 m; I1 no lo cambia); σ: −11.39 → −10.18 (idéntica bit a bit tras I1: en σ no hay fusión). Teoría (`theory.py`): Ellison–Turner y Xu dan E = 0 (Ri_B ≥ 0.8), pluma subcrítica (Fr 0.34–0.45) que sigue a Nof; la BBL baja las interfaces con Ri < 0.25 aguas abajo a 1.1 (< 2–4 de Jackson §6). Costo por paso DOME 1.027×, BayIW cut 1.071× (criterio 6 no se cumple en BayIW; causa [hipótesis, el perfil la apoya]: la fase 7 encendida con el cierre constante) | Legg 10 km: isopicna −11.07, z −11.67; Jackson U_bg 0 −12.84; GLS (N3) −12.99 | 🔶 criterio 4: dirección ✅ y 52 % de la magnitud en b_f, robusta tras I1 (0 eventos del limitador), el transporte aguas abajo se aleja; criterio 6 ❌ en BayIW; `validation/dome/README.md` |
| Núcleo N3 sin regresión (matriz dorada y smoke de 9 casos de N1) | `main` | 9/9 corridas de humo idénticas bit a bit a `main`; tras las correcciones de la revisión final (2026-10-30) 7/9 hashes idénticos: `bayiw_cut_zstar` y `bayiw_cut_hybrid` difieren solo en T de las columnas de tierra (inactivas), que la corrección M1 deja en su 11.4 °C inicial en vez de la deriva de 3e-5 °C del solver vertical de `main`; toda celda mojada y η, u, v, h bit a bit; matriz dorada N1 idéntica salvo `sigma_gls`, `sigma_loglaw`, `sigma_parabolic` y dos hashes GLS, que cambian por diseño con el arrastre log con velocidad completa y v ≠ 0 (Task 9b, clase a); costo (mediana por paso, dos mediciones): `bayiw_cut_zstar` 1.012 y 1.024× `main`, `bayiw_cut_hybrid` 0.992 y 1.017×, `ilicak_overflow_zstar` 1.037 y 1.000× (el límite del spec es 1.01×; superado por 1–2 % en un caso, ruido de la máquina o costo de N3, sin aislar) | `main` | 🔶 bit a bit ✅; costo ≈ +1–2 % en el corte BayIW z\* |
| Mezcla numérica de k/ε por el remapeo ALE (N3, anti-objetivo 3 del spec: no debe dominar la física) | balance de varianza de k (Σ w k² antes y después de `GLSClosure.remap`) o diagnóstico equivalente | medida solo de forma indirecta: híbrida frente a σ en Kato–Phillips, D_m ≤ 0.1 %; sin cuantificar en el ciclo diurno híbrido (RMS D_m 0.48 m frente a σ) ni en DOME híbrido + GLS (revisión final I2, 2026-10-30; `validation/turbulence_columns/`) | — | 🔶 abierto |
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
- Paralelismo MPI **opcional** (agregado el 2026-10-30, decisión de Esteban), después del split-explicit: con la superficie libre implícita actual el sistema global no se reparte bien entre procesos. Por defecto siguen los hilos de Numba en una sola máquina (`pip install pyocean`, sin MPI ni compiladores); `pip install pyocean[mpi]` agrega `mpi4py` y el usuario lo activa según su equipo (`parallel="mpi"` o lanzando con `mpirun`). Descomposición del dominio en bloques con halos, MPI entre bloques + hilos de Numba dentro de cada uno (el esquema híbrido de MOM6 y HYCOM). Criterios de cierre: 1 y N procesos idénticos bit a bit (como ya se exige entre 1 y 5 hilos); el pre-flight avisa si se pide MPI sin `mpi4py` o si los bloques quedan demasiado chicos para sus halos; escalamiento medido en Ometeotl en un caso PAC_MEX.
- *Overflow* (DOME) y monte submarino como referencia de entrada. DOME corrido el 2026-10-15 (🔶: z\* ✅ contra MITgcm; abiertos la inestabilidad con θ_bc = ½ a 10 km / 1200 s y la entrada de la híbrida por frontera abierta; la mezcla por Ri, con Jackson et al. 2008 desde el 2026-10-30, N3c; la mezcla de la capa de fondo por la fricción desde el 2026-10-31, N3d). Falta el modo barotrópico split-explicit.

**Etapa N, núcleo modular y experiencia de usuario (agregada el 2026-10-15; diseño en la sección 9):**
- Va **antes de la Etapa D** (JAX necesita núcleos funcionales separados) y antes de terminar el split-explicit de la Etapa C, que debe entrar ya como componente `FreeSurface`.
- N1. Contrato de componentes y driver con orden de pasos fijo, extraídos de `SigmaBaroclinic` sin cambiar la física. ✅ 2026-10-28: `core/components/` + driver de 10 fases (`core/model/driver.py`), bit a bit en 31 configuraciones y en las corridas de humo de los benchmarks; spec `docs/superpowers/specs/2026-10-28-n1-component-core-design.md`. Queda para N2: partir `SigmaWetDry.step` (viscosidad, arrastre, superficie libre, cierre) y que el cierre dé κ a los trazadores.
- N2. Salinidad, TEOS-10 y flujos de superficie completos (calor, onda corta, E − P) en el núcleo 3D, conectando `physics/eos` y `forcing/bulk.py`. N2a ✅ 2026-10-29: salinidad activa, EOS lineal y TEOS-10 en (T, S), agua dulce real E − P y ríos, balance exacto de varianza de S y T, TEF en línea, frontera de elevación prescrita, GLS k-ε estratificado y el benchmark del estuario de Lorenz et al. (2021) (`validation/lorenz_estuary/`). Abiertos para N2b: calor y onda corta de superficie en el 3D (hecho en N3, flujo prescrito), cierre turbulento y ajuste convectivo en ALE (hecho en N3: GLS y KPP en ALE), viscosidad horizontal de respaldo para AB3 + WENO5, error de interfaz del regrid híbrido en las celdas de los extremos (0.1–1.2 m, ya existía en el camino solo-T), acumuladores de TEF en el restart.
- N3. KPP como segundo cierre con la misma interfaz que GLS. ✅ 2026-10-30 (rama `n3-closures`; spec `docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md`): el cierre es un componente (`ConstantClosure`, `GLSClosure`, `KPPClosure`); GLS k-ε estratificada en ALE (z\*, σ, híbrida) con remapeo de k y ε en la malla dual y fusión de capas delgadas con umbral relativo `min(10 cm, max(1 mm, 1 % D/N))`; KPP de CVMix/MOM6 en σ y ALE sin difusividad de fondo por defecto; flujo de calor prescrito con onda corta de Paulson & Simpson (1977); fronteras periódicas en x e y; arrastre de ley de pared en ALE con la velocidad completa del fondo (corrige el giro de Ekman del fondo, 62.8° → 75.99° contra CROCO 75.77°); columnas contra CROCO (GLS) y MOM6/CVMix (KPP) en `validation/turbulence_columns/`; Lorenz ALE-σ + GLS y DOME híbrido + GLS. Cerrada con cuatro criterios abiertos, no corregidos: la relación exacta de Lorenz en A queda en 1.5e-9 (límite 5e-10; ganancia de redondeo del remapeo ALE, clase c; corregida el 2026-10-30 con el remapeo exacto en contenido: 2.9e-11, bits de ALE cambiados a nivel 1e-14), un truncamiento CFL en el DOME híbrido con GLS (día 31.8), el costo del smoke (`bayiw_cut_zstar` 1.012× y 1.024× frente al límite 1.01×, causa sin aislar; las 9 corridas son idénticas bit a bit) y la mezcla numérica de k/ε por el remapeo ALE (anti-objetivo 3 del spec), medida solo de forma indirecta (híbrida frente a σ, D_m ≤ 0.1 % en Kato; sin término de remapeo de la varianza de k; revisión final I2). Abiertos: mezcla interior por Ri en KPP, ALE sin remapeo de la historia de AB3, GLS contra GOTM en columna, bordes abiertos en el arrastre log (componente transversal cero).
- N3b. Viscosidad horizontal de Smagorinsky (laplaciana, tensor de esfuerzos con grosor, MOM6; `smagorinsky=C`, apagada por defecto y bit a bit; spec `docs/superpowers/specs/2026-10-30-smagorinsky-viscosity-design.md`, notas `docs/design/smag_refs/`). ✅ 2026-10-30 (rama `smagorinsky`; números de ALE recorridos después de la corrección I1 de la revisión final, 47f2730): DOME híbrido + GLS sin truncamientos con C = 0.15 (1 → 0; el control de operador C ≈ 0 conserva el truncamiento) y max\|u\| en caras gruesas 5.52 → 3.54 m/s (casi todo por el operador de tensor); overflow de Ilicak ΔRPE −48 % con C = 0.5 (−47 % contra el control); Lorenz D no gana Δt (el control sin Smagorinsky también completa a T_M2/498); `kv_low` completa pero no es robusto (61 truncamientos); costo +2.4 % en DOME. Abiertos: contador `n_visc_limited`, mezcla por cizalla (Jackson 2008) con ν_v bajo, biarmónica, Δt de producción de Lorenz.
- N3c. Mezcla interior por cizalla de Jackson et al. (2008) (puerto de MOM6 `kappa_shear`, `shear_mixing='jackson'`, `shear_prandtl`, `jackson_params`; spec `docs/superpowers/specs/2026-10-30-jackson-shear-mixing-design.md`, notas `docs/design/shear_refs/`). ✅ 2026-10-30 (rama `jackson-shear`): kernel de columna = MOM6 compilado bit a bit, Numba con oráculo NumPy, se suma al cierre constante o KPP; métricas de Legg (2006) contra su no hidrostático digitalizado (`validation/dome/legg2006_nonhydro.json`); DOME híbrido `kv_low` + Smagorinsky sin truncamientos (61 → 0) y más cerca de la verdad no hidrostática en todas sus métricas, sin llegar (b_f −12.84 contra −8.75). Abiertos: entrada de la híbrida (a), la configuración de Jackson del DOME de MOM6 (KD = 1e-4, Pr = 0) como variante, κ_shear en la salida y balance de varianza para separar mezcla física y numérica, Xu et al. (2006) como verdad de alta resolución, por qué Jackson no actúa con ν_v = 0.5.
- N3d. Mezcla diapicna de la capa de fondo por la fricción (puerto de MOM6/OM4 `add_LOTW_BBL_diffusivity`, `bbl_mixing='lotw'`, `bbl_params`; spec `docs/superpowers/specs/2026-10-30-bbl-drag-mixing-design.md`, notas `docs/design/bbl_refs/`) y diagnóstico teórico de la pluma (`validation/dome/theory.py`). 🔶 2026-10-31 (rama `bbl-drag-mixing`): kernel = MOM6 compilado, energía física de la fricción (γ = 0.2 aquí = 0.1 en MOM6, clase d); DOME híbrida con Jackson + BBL (tras la corrección I1 de la revisión final, N² con las densidades fusionadas, clase a): b_f −12.70 → −11.86e-3 contra −11.07 de la isopicna de Legg (52 % del camino; 71 % antes de I1), 0 eventos del limitador (32 antes de I1), pero el transporte aguas abajo baja (T(x′ = −300) 19.0 → 18.0 Sv). Abiertos: por qué baja ese transporte, costo en BayIW (1.071×; [hipótesis] la fase 7 encendida con el cierre constante), Kd_BBL en la salida, E de la esquina con la dilución en el canal, DOME 5 km robusto (con Jackson 469 truncamientos; sin Jackson aborta).
- N4. `Simulation`: config tipada (TOML/YAML) guardada en la salida y el restart, presets con procedencia, pre-flight (verificador de dominio, CFL, compatibilidad de componentes, balance de masa en fronteras), post-mortem, salida CF/Zarr y presupuestos término a término por defecto.
- N5. Constructor de dominio regional integrado (GEBCO, HYCOM/GLORYS, TPXO, ERA5/WRF) con reporte del verificador.
- N6. Paquete instalable (`pyproject.toml`, `pip`/conda-forge), API pública estable, galería de ejemplos con resultado esperado.
- Criterio de cierre: N1 reproduce bit a bit (o a ≤ 1e-12) los 1388 tests y los benchmarks de `validation/`; N2 pasa un caso regional con calor y sal contra MITgcm o HYCOM; N4–N6 permiten correr el quickstart y un caso PAC_MEX desde una instalación limpia sin tocar `sys.path` ni métodos privados.

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

## 9. Arquitectura del núcleo y experiencia de usuario (2026-10-15)

Origen: revisión del código y del manual el 2026-10-15, comparada con la organización de MITgcm, MOM6, ROMS/CROCO, HYCOM, Oceananigans y Veros y con las quejas de la sección 3b. Pendiente antes de fijar las interfaces: leer la composición en el código de MOM6 (`src/core/MOM.F90`, `step_MOM`) y de Oceananigans (`HydrostaticFreeSurfaceModel`), con referencias `file:line`.

### 9a. Diagnóstico

pyOCEAN creció como varios núcleos en paralelo, cada uno con su estado y su paso de tiempo:

| Núcleo | Archivo | Uso |
|---|---|---|
| Fase 1, niveles z | `core/state.py` (`ModelState`), `core/dynamics/momentum.py`, `pressure.py`, `core/timestepping/ab2.py` | ya no se usa en producción (ALE z\* lo cubre) |
| 2D superficie libre implícita | `core/dynamics/free_surface.py` (`ImplicitFreeSurface`) | mareas PAC_MEX |
| 2D mojado-secado | `core/dynamics/wetting_drying.py` (`WetDryFreeSurface`) | Alto Golfo, Thacker, TIDAL_FLAT |
| 3D homogéneo σ | `core/dynamics/sigma_wd.py` (`SigmaWetDry`) | base del 3D estratificado |
| 3D estratificado σ/ALE | `core/dynamics/sigma_baroclinic.py` (`SigmaBaroclinic`) | todo lo nuevo; una clase de ~1400 líneas con 56 argumentos |

Huecos que esto produce:
- **Física construida pero desconectada del 3D:** `SigmaBaroclinic` usa una EOS lineal solo en T (`sigma_baroclinic.py:1351`), no tiene salinidad (docstring, línea 57) y `step()` solo recibe `tau_x`, `tau_y` (`:2246`). TEOS-10 (`physics/eos`), los flujos bulk NCAR/Kara (`forcing/bulk.py`), KPP (`physics/mixing/kpp.py`) y la advección de la Fase 1 no se usan en el 3D. Hoy no se puede correr un caso regional realista con calor y sal en 3D.
- **Experiencia de usuario:** sin paquete instalable (`sys.path.insert` en cada script; `environment.yml` no coincide con el entorno probado); cada caso es un script propio; los runners llaman métodos privados (`m._edge_face_depths()`, `m._mean_face_thickness()` en `validation/bayiw_cut/run_pyocean.py:31-32`) e indexan caras del C-grid a mano; el verificador de dominio y el post-mortem solo los llaman `run_tides_wd.py` y `run_bayiw.py`; los runners guardan `.npz` propios en lugar de `OutputWriter`; `ke_scheme="gudonov"` en la API pública.

### 9b. Cómo se organizan los demás

- **MITgcm:** núcleo en `model/src`; paquetes en `pkg/` que entran por puntos fijos del paso, configurados al compilar (CPP, `SIZE.h`).
- **MOM6:** driver `step_MOM` en `src/core`; `src/ALE`, `src/parameterizations`, `src/equation_of_state`; drivers solo-océano o acoplado; parámetros en `MOM_input` (cientos).
- **ROMS/CROCO:** `main3d` llama en orden fijo a `rho_eos`, `omega`, `step2d`, `step3d_uv`, `step3d_t`, `gls_mixing`/`lmd_vmix`; opciones en `cppdefs.h`.
- **Oceananigans:** el modelo se compone de objetos (`grid`, `buoyancy`, `closure`, `free_surface`, `momentum_advection`, `boundary_conditions`, `forcing`).
- **Veros:** `VerosState` con registro de variables, rutinas del núcleo y un `VerosSetup` por caso; `pip install`, `veros copy-setup`.

Patrón común: un estado, un driver con orden de pasos documentado, piezas intercambiables con interfaces claras, y casos que solo configuran.

### 9c. Diseño propuesto: tres capas con contrato

```
Capa 3 — Usuario       Simulation, presets, constructor de dominio, config, CLI/Jupyter
Capa 2 — Componentes   EOS, Closure, Advection, PGF, Momentum, FreeSurface, OpenBoundary, WetDry, SurfaceFluxes
                        cada uno con contrato: variables que lee y escribe (registro único), parámetros con
                        unidades, valor por defecto y cita, requisitos y chequeos
Capa 1 — Núcleos       funciones puras sobre arreglos (NumPy/Numba hoy, JAX en la Etapa D), sin modificar la entrada
```

- El driver arma el paso en orden fijo y valida la composición **antes del primer paso** (por ejemplo: KPP sin flujo de flotabilidad en superficie, TEOS-10 sin S, forzamiento que no cubre el dominio).
- `SigmaBaroclinic` pasa a ser un modelo armado con componentes. Los 2D de marea siguen como modelo barotrópico que comparte grid, forzamiento y fronteras. La Fase 1 en z se retira.
- Mismo API con `backend="numpy" | "jax"`; los tests actuales pasan a ser tests de equivalencia entre backends (sección 6).
- Paralelismo como opción del usuario, independiente del backend: `parallel="threads"` (por defecto, Numba en una máquina) o `"mpi"` (opcional, `pyocean[mpi]`, para clústeres como Ometeotl). Mismo resultado bit a bit con 1 o N procesos (sección 8, Etapa C).

Uso objetivo:

```python
import pyocean as po

domain = po.regional.build(region=(-118, -92, 12, 32), dx_km=4,
                           bathy="gebco2024", ocean="hycom_espc", tides="tpxo10",
                           atmosphere="era5", start="2026-02-01", days=30)
print(domain.report())            # rx0/rx1, fosas, cobertura de fronteras, Δt sugerido

model = po.Model(domain,
    coordinate=po.vertical.Hybrid(n_layers=41),
    eos=po.eos.TEOS10(),
    closure=po.closures.GLS("k-epsilon"),
    tides=po.tides.Flather(sal="hycom"),
    preset="coastal_tidal")       # el preset cita el benchmark que lo validó

sim = po.Simulation(model, dt=120, stop="30 days",
                    output=po.Output("pacmex.zarr", every="1 h", fields="standard"),
                    checkpoint="1 day")
sim.run()                         # pre-flight, salud en línea, post-mortem si explota
sim.budgets().plot()              # cierre término a término
```

### 9d. En qué seríamos mejores

| Problema en los otros | Respuesta de pyOCEAN |
|---|---|
| MITgcm: configuración al compilar | todo al correr; Numba compila solo |
| MOM6: cientos de parámetros, errores horas después | validación al armar el modelo, errores en lenguaje claro con la corrección sugerida |
| ROMS/CROCO, MOM6: preparación del dominio con herramientas externas (ROMS-Tools, regional-mom6, CrocoDash) | constructor de dominio dentro del modelo, reproducible (queja nº 1 de la sección 3b) |
| HYCOM: `blkdat.input` críptico | config legible = serialización de los parámetros tipados, guardada en salida y restart; `pyocean config diff` |
| Oceananigans: latencia de compilación, errores de tipos opacos, OBC confusos | composición igual de limpia en Python; fronteras con benchmarks públicos de reflexión y balance de masa |
| Todos: explota sin decir dónde; presupuestos que no cierran | presupuestos de calor, sal y volumen término a término, RPE, truncaciones y post-mortem por defecto |
| Ninguno en Fortran | inspección en vivo: `model.tendencies(state)` devuelve cada término con nombre, en Jupyter, y se puede continuar |

Principios:
1. Un parámetro, un lugar: unidades, valor por defecto, cita y benchmark; la documentación de cada componente se genera de su contrato.
2. Nada falla en silencio: todo chequeo posible se hace antes de correr.
3. Lo bueno está en el camino por defecto (verificador, post-mortem, CF/Zarr, presupuestos).
4. Presets con procedencia; nunca ajustados a un solo caso.
5. API estable: versionado semántico y alias con aviso al renombrar (`gudonov` → `godunov`).
6. Instalación con `pip`/conda-forge en una Mac, sin compilar.

Orden de trabajo: Etapa N de la sección 8 (N1 → N6). N1–N3 cambian la calidad del modelo; N5 es lo que más atrae usuarios.
