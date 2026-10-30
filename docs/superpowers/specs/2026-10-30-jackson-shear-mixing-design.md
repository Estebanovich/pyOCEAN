# Mezcla vertical por cizalla de Jackson, Hallberg & Legg (2008)

**Fecha:** 2026-10-30. **Estado:** decisiones de diseño aprobadas por Esteban el 2026-10-30 (incluidas las dos de la revisión de constructor: corrida principal y geometría); el documento completo, pendiente de su lectura. Revisión de constructor hecha.
**Contexto:** después de N3 y de la viscosidad de Smagorinsky (`main` en `b5079a6`), antes de N2b. Lecturas de referencia (con `archivo:línea`) en `docs/design/shear_refs/`:
- `mom6_kappa_shear.md`: `MOM_kappa_shear.F90` (ecuaciones, iteración, integración en el tiempo, capas delgadas, parámetros, orden de llamada);
- `cvmix_shear.md`: Large et al. (1994) y Pacanowski & Philander (1981) en CVMix/MOM6;
- `croco_and_others.md`: CROCO `LMD_RIMIX`, MITgcm KPP/GGL90, configuraciones DOME y `flow_downslope` de MOM6;
- `pyocean_current.md`: qué hay hoy en pyOCEAN (GLS, KPP, interfaz de cierres) y dónde se conecta un esquema nuevo.

## Por qué

En los overflows la mezcla que fija la densidad final de la pluma es la de Kelvin–Helmholtz en la interfaz, donde el número de Richardson Ri = N²/S² baja de ~0.25. pyOCEAN solo la representa con el GLS k-ε estratificado. Con el cierre constante, o con KPP, no hay mezcla interior por cizalla (`core/components/closure.py:652`). Los benchmarks lo muestran:

- **DOME híbrido con ν constante** (`validation/dome/README.md`, punto 3): b_f −16.2e-3 m s⁻² contra −11.0 / −12.0 del isopicnal de Legg y −11.7 de su z. La capa densa casi no se diluye. Clase (b).
- **DOME `kv_low`** (ν_v = 1e-4): 61 truncamientos CFL aun con Smagorinsky; las capas interiores de la cuenca profunda llegan a 8.29 m/s porque nada las acopla. Clase (b).
- **DOME híbrido + GLS:** b_f −12.99e-3, pero con transporte de la pluma plano (18–22 Sv contra 35–37 de MITgcm). No hay un segundo esquema contra el cual separar lo que hace el cierre de lo que hace la coordenada.

Jackson et al. (2008) es el esquema que MOM6 y OM4 usan para overflows (Adcroft et al. 2019; Hallberg & Adcroft 2009 lo corren en DOME). Resuelve una TKE estacionaria y un κ implícito en la columna, con una escala de longitud ligada al ancho de la zona de Ri bajo. Está diseñado para pasos de tiempo largos. El isopicnal de Legg (2006) usa la mezcla por Ri de Hallberg (2000), su antecesora.

**Objetivo de Esteban (2026-10-30):** acercar el overflow de pyOCEAN a la **verdad no hidrostática de Legg et al. (2006)** (MITgcm no hidrostático a Δx·Δz = 500 m · 30 m, Caso 1). Si DOME no basta como verdad, se agregan otros casos con referencia de alta resolución.

## Criterios de éxito

1. **Columna contra MOM6 compilado.** En perfiles fijos (capa de cizalla estratificada, chorro de fondo tipo overflow, columna con capas de 1 mm–2 cm), κ y la viscosidad de pyOCEAN coinciden con `kappa_shear_column` de MOM6 (`~/pyOCEAN_refs/MOM6`, compilado como en `validation/mom6_column/`) con los mismos parámetros y Δt. La tolerancia la fija el constructor a partir de la tolerancia de iteración de MOM6 (`KAPPA_SHEAR_TOL_ERR` 0.1) y la documenta. Se usa la rama **corregida** de la derivada de Newton (`KAPPA_SHEAR_ITER_BUG` = False, el valor por defecto de MOM6). La comparación es sobre la **misma columna ya fusionada** (prueba del kernel `kappa_shear_column`; revisión de constructor, decisión de Claude): la fusión de capas delgadas de pyOCEAN usa otro umbral que MOM6 (abajo), así que de extremo a extremo diferirían a propósito. La propiedad "igual κ con o sin capas delgadas" se prueba dentro de pyOCEAN (criterio 2).
2. **Propiedades físicas en columna:**
   - Ri ≥ Ri_c en toda la columna → κ = 0.
   - κ = 0 en la superficie y en el fondo.
   - Una columna con capas casi vacías da el mismo κ que la columna sin ellas (fusión de capas delgadas).
   - El κ promediado sobre Δt no crece sin límite con Δt: integración de Hallberg (2000) con subpasos, como `MOM_kappa_shear.F90:1191-1368`.
3. **Nada se rompe:** con el esquema apagado (valor por defecto), los 33 goldens y las corridas de humo dan resultados idénticos bit a bit. Además:
   - restart bit a bit con el esquema activo;
   - el resultado es el mismo con 1 y con 5 hilos;
   - el pre-flight rechaza `gls` + cizalla y cualquier parámetro inválido, reportando todos los errores juntos.
4. **DOME 10 km contra la verdad no hidrostática de Legg.** Se agregan a `validation/dome/` las métricas de Legg para el Caso 1 (revisión de constructor, decisión de Claude):
   - **E medio en la región de la esquina** (Legg ec. 14; −50 < x < 50 km, −50 < y < 0 km, borde de la pluma s = 0.01): 8.6e-4 en el no hidrostático, y los valores de 10 km z e isopicnal de su Fig. 14. Es la métrica principal de entrainment, porque Legg la compara entre resoluciones y ahí ocurre la mayor parte de la mezcla.
   - α_E(x) al oeste de la entrada (ec. 12, Fig. 3b), como métrica secundaria.
   - Transporte de la pluma contra x (Fig. 3a).
   - Trazador por clase de flotabilidad contra la distancia a lo largo del talud (Fig. 4), promediado en la segunda mitad de la corrida como en Legg.

   Sus curvas no hidrostáticas se leen de las figuras del PDF y se marcan **[digitalizado]**. Siguen también b_f (Fig. 15), trayectoria (Figs. 16–17) y centroide a 50 días contra sus corridas de 10 km.

   Corridas (decisión de Esteban, 2026-10-30, tras la revisión de constructor):

   | Coordenada | Corridas |
   |---|---|
   | Híbrido | **`kv_low` (ν_v = 1e-4) + Jackson Pr = 1 + Smagorinsky C = 0.15: la principal.** Es el equivalente del isopicnal de Legg a 10 km (ν_v = 1e-4 de fondo, mezcla por Ri de Hallberg 2000, Smagorinsky biharmónico; Legg 2006 p. 77). Su línea base sin Jackson ya existe (`pyocean_hybrid_kv_low_smag0.15`: 61 truncamientos, b_f −17.9e-3). |
   | Híbrido | la misma con Jackson Pr = 0 (sensibilidad, como el DOME de MOM6) |
   | Híbrido | ν_v = 0.5 + Jackson Pr = 1 (secundaria; equivalente al isopicnal de Legg con ν_v = 0.5, b_f −12.0) |
   | Híbrido | GLS (segundo punto, ya existe con y sin Smagorinsky) |
   | z\* y σ | ν_v = 1e-4 + Jackson Pr = 1 + Smagorinsky C = 0.15, para la dependencia con la coordenada |

   **Geometría** (decisión de Esteban, 2026-10-30): se mantiene la de MOM6 (canal de 100 km, pared oeste con esponja, fondo con arrastre cuadrático). Las diferencias con el no hidrostático de Legg (canal de 50 km, Orlanski al oeste, fondo no-slip) se clasifican (c), como el 2026-10-14. x se alinea con el centro de la entrada (x′ = x − 1050 km, como `metrics.py`). Una variante con la geometría de Legg queda fuera.

   Cada diferencia contra la verdad se clasifica de (a) a (d).

   **Se espera, sin ajustar nada:** la híbrida principal con Jackson diluye la pluma y queda más cerca de la verdad que su línea base sin Jackson (b_f −17.9e-3), en E de la esquina, en la distribución por clase y en b_f. Si no se acerca, se documenta la causa y el trabajo cierra con el criterio abierto, como en N3.
5. **Robustez de la corrida principal** (`kv_low` + Jackson Pr = 1 + Smagorinsky): completa 50 días sin truncamientos. Si no, se documentan el día, el lugar y la causa (post-mortem).
6. **Ilicak overflow:** el benchmark no cambia, porque el esquema está apagado por defecto (KD = 0 en todos los modelos; es una prueba de mezcla espuria). Se corre **una** sensibilidad z\* ν_h = 100 con Jackson (Pr = 1) y se reporta cuánta mezcla física agrega frente a la espuria medida por RPE.
7. **Costo:** se reporta el costo por paso con el esquema activo en DOME 10 km híbrido y en BayIW cut z\*. No hay límite fijo; Jackson es caro en MOM6. Si pasa de +30 % del paso, se perfila y se documenta qué parte cuesta.

**Fuera** (pasos posteriores):
- Large et al. (1994) y Pacanowski & Philander (1981) como opción barata, dentro de KPP.
- Modo de vértices (`VERTEX_SHEAR`).
- Ajuste convectivo: Jackson recorta N² ≥ 0 y no hace convección. El cierre constante tampoco la tiene; es un pendiente de N2a.
- El caso de Xu et al. (2006), con Nek5000 como verdad: es el siguiente paso condicionado si DOME no basta.
- Recorrer el no hidrostático de Legg.
- Combinar con GLS.

## Esquema (de `MOM_kappa_shear.F90`, ver `shear_refs/mom6_kappa_shear.md`)

- **Fuente de κ** (`:1646-1652`): K_src = 2γ|S|·(Ri_c S² − N²)/(Ri_c S² + F_c N²) si N² < Ri_c S²; si no, 0. Parámetros:
  - γ = 0.089 (`SHEARMIX_RATE`);
  - Ri_c = 0.25 (`RINO_CRIT`);
  - F_c = −0.97 (`FRI_CURVATURE`).
- **Ecuación de κ** (`:1783-1803`): κ/L_d² − ∂²κ/∂z² = K_src, con κ = 0 arriba y abajo.
  - 1/L_d² = (N²/λ² + f²)/Q + 1/L_bdry², con λ = 0.82.
  - L_bdry es la distancia a la superficie y al fondo (`:1039-1049`).
- **TKE estacionaria** (`:1709-1773`): 0 = ∂_z[(κ+κ0)∂_z Q] + (κ+κ0)S² − κN² − (Q − q0)√(c_N²N² + c_S²S²).
  - c_N = 0.24, c_S = 0.14.
  - Q = TKE_min en el fondo.
- **Iteración acoplada** (`:1701-2041`): Picard que pasa a Newton, hasta `MAX_RINO_IT` (50), con tolerancia 0.1 (κ0 + κ).
- **Integración en Δt** (`:1191-1368`):
  - suavizado previo de u, v, T, S con κ0 (`:1052-1088`);
  - hasta 13 subpasos con búsqueda del paso y predictor–corrector;
  - se devuelve κ **promediado en Δt**;
  - semilla κ = 1 m²/s en cada llamada (`:325`).
- **Salida:** κ_shear a T, S y trazadores; ν_shear = Pr · κ_shear a la velocidad (`:390-394`). Ambos se **suman** al cierre base, como MOM6 (`MOM_set_diffusivity.F90:571-581`, `MOM_vert_friction.F90:2163-2205`).
- **N²** de la EOS del modelo (lineal o TEOS-10), recortada a ≥ 0 (`:1499`).

Parámetros con los valores por defecto de MOM6, conjunto 1 de JHL (`:2106-2112`). No se ajustan a un caso.

## Restricciones

- Solo Python (NumPy y Numba); sin dependencias nuevas.
- Toda corrida larga exporta `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, usa ≤ 5 hilos en los núcleos P, una corrida a la vez, y escribe en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`. Si una corrida pasa de ~24 h, se consulta a Esteban.
- Con el esquema apagado, el camino actual no cambia ni en el orden de las operaciones.
- Cada ecuación cita Jackson et al. (2008) y `MOM_kappa_shear.F90:línea` en el docstring, con las variables con los nombres del artículo y sus unidades.
- Sin ajustar parámetros a un caso; Pr = 0 se reporta como sensibilidad, no como calibración.
- **Anti-metas:**
  - contar la cizalla dos veces (GLS + Jackson);
  - meter una difusividad de fondo a escondidas: κ0 solo suaviza la entrada, como en MOM6, y no se devuelve en κ;
  - cambiar los resultados de Ilicak o de cualquier benchmark con el esquema apagado.
- A `docs/literature/reading_list.md` (con DOI, nombre de archivo y ★ si se usa en el código) van:
  - Jackson et al. 2008 ★;
  - Xu et al. 2006;
  - Papadakis et al. 2003;
  - Odier et al. 2014;
  - van Reeuwijk et al. 2019;
  - Tseng et al. 2006;
  - Wang et al. 2008;
  - Petersen et al. 2015 (MPAS DOME);
  - Hallberg (2000), marcado [verificar] si no está en la biblioteca.

  Los primeros ocho ya están en la biblioteca local (descargados por Esteban el 2026-10-30).

## Decisiones de diseño (aprobadas por Esteban el 2026-10-30)

1. **Jackson et al. (2008) como esquema principal** (MOM6 `kappa_shear`).
2. **Pr = 1 por defecto, configurable; Pr = 0 como sensibilidad.** Con Pr = 1, la mezcla de Kelvin–Helmholtz también acopla las capas en la velocidad, que es lo que le falta a `kv_low`.
3. **Se suma a un cierre base** (constante o KPP), como en MOM6. Con GLS se rechaza.
4. **Validación:**
   - DOME contra la verdad no hidrostática de Legg, con sus métricas del Caso 1 y las corridas del criterio 4; la principal es la híbrida `kv_low` + Jackson Pr = 1 + Smagorinsky C = 0.15, y la geometría de MOM6 se mantiene, con sus diferencias clasificadas (c);
   - Ilicak con el esquema apagado y una sola sensibilidad;
   - columna contra MOM6 compilado antes de los casos 3-D;
   - Xu et al. (2006) como paso siguiente condicionado.
5. **Implementación:**
   - kernel de columna en Numba con oráculo NumPy;
   - rama corregida de Newton;
   - modo de puntos de trazador;
   - reutiliza la fusión de capas delgadas existente;
   - apagado por defecto y bit a bit.

## Decisiones tomadas por Claude (Esteban las revisa al leer)

- **Parámetros:** `shear_mixing` (`None` por defecto, o `'jackson'`) y `shear_prandtl` (float, 1.0) en `SigmaBaroclinic`. Los demás coeficientes van en un `JacksonParams` con los nombres de MOM6 en los comentarios, como `KPPParams`. Deja lugar a `'lmd94'` más adelante.
- **Forma en el código:** una envoltura aditiva `ShearAddedClosure(base, shear)` (opción (b) de `pyocean_current.md` §5).
  - Suma `viscosity` y `tracer_diffusivity` del cierre base y de la cizalla.
  - Llama `update` de los dos, y mantiene el estado y el restart de cada parte con sus prefijos.
  - Si el cierre base no devuelve κ (constante), la suma usa el `kv` constante como base.
  - El kernel vive en `physics/mixing/kappa_shear.py`, en índices (k, j, i) con k = 0 en el fondo; MOM6 va de arriba abajo, así que se invierte como en `kpp_column`.
- **Fase y nivel de tiempo:** fase de trazadores (7), como KPP, que es el orden de MOM6.
  - κ_shear se calcula con el estado del paso y lo usan los trazadores de ese paso.
  - ν_shear se usa con un paso de retraso en la viscosidad implícita del paso siguiente, como `Kv_shear` en MOM6.
- **Δt de Jackson** = el Δt del modelo; pyOCEAN no tiene un `DT_THERM` separado.
- **κ0 (`KD_KAPPA_SHEAR_0`)** = max(kv de fondo, 1e-7) m²/s, como MOM6 = max(KD, 1e-7). Con KPP sin fondo queda 1e-7.
- **Capas delgadas:** se reutiliza el umbral relativo por columna de pyOCEAN, min(10 cm, max(1 mm, 1 % D/N)), con el patrón de fusión de MOM6 (`:254-303`). El umbral 0.1√(κ0Δt) de MOM6 daría ~6 cm en DOME y ~2 mm con κ0 = 1e-7; la diferencia se documenta, como en GLS (N3 tarea 9c).
- **u, v en el centro de la celda** con la media de las dos caras, como `_ale_closure_forcing` y como `find_uv_at_h` de MOM6. ν_shear pasa a las caras como la media de las dos columnas.
- **ALE:** ν_shear se conserva por índice de interfaz tras el regrid, igual que MOM6 por defecto (`REMAP_AUXILIARY_VARS` = False). κ no es estado, porque se re-siembra cada paso.
- **Restart:** solo ν_shear con retraso (`shear_kv`), como `kpp_akv`.
- **Salida y registro:** `kd_shear` y `kv_shear` en interfaces, en el registro único de variables; opcionales en la salida.
- **Mojado y secado, tierra, columnas secas:** κ = ν = 0, y la columna seca queda quieta bit a bit, como en la corrección M1 de N3.
- **Métricas de Legg en `validation/dome/`:**
  - trazador por clase de flotabilidad y transporte en la ventana de 13 días, con la definición de Legg (s = 0.01 como borde de la pluma);
  - E a partir del aumento de transporte con x.

  Los valores no hidrostáticos van a un archivo de datos digitalizados con su figura y página de origen.

## Lo que decide el constructor (y lo documenta)

- La tolerancia del criterio 1 y cómo se compila o llama `kappa_shear_column` de MOM6 para la prueba de columna. Reutilizar el arnés de `validation/mom6_column/`.
- La estructura de los kernels (paralelismo por columnas con `prange`, buffers) y el oráculo NumPy.
- Los perfiles concretos de las pruebas de columna y sus valores esperados.
- Cómo se digitalizan las curvas de Legg (herramienta, puntos, incertidumbre de lectura).
- Si la ventana de 13 días se mide en la misma corrida de 50 días (salidas intermedias) o en corridas aparte.
