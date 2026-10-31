# Diagnóstico teórico de la pluma y mezcla de capa de fondo por arrastre

**Fecha:** 2026-10-30. **Estado:** alcance, meta y energía física aprobados por Esteban el 2026-10-30. Revisión de constructor hecha. Pendiente: su lectura del documento completo.

**Contexto:** después de la mezcla por cizalla de Jackson (`main` en `77dbb44`). Lecturas de referencia en `docs/design/bbl_refs/`:
- `mom6_bbl_mixing.md`: MOM6 `add_drag_diffusivity` / `add_LOTW_BBL_diffusivity` / `set_BBL_TKE`, configuraciones DOME y OM4, Apéndice A de Legg (2006) y dónde se conecta en pyOCEAN, todo con `archivo:línea`;
- `entrainment_theory.md`: Xu et al. (2006), Ellison–Turner, Jackson, van Reeuwijk, Odier, Ezer y Price, más qué predicen para DOME y qué papers faltan.

## Por qué

Con Jackson, DOME 10 km híbrido (`kv_low` + Smagorinsky 0.15) llega a b_f −12.84e-3 m s⁻². La verdad no hidrostática de Legg es −8.75 (digitalizada). La corrida isopicna de 10 km de Legg, que tiene casi la misma física que pyOCEAN, da −11.07.

Esa corrida isopicna incluye una **mezcla diapicna de capa de fondo alimentada por la energía del arrastre** (Legg Ap. A, γ = 0.2). Sin ella, Legg ve una lámina densa, viscosa y casi sin diluir (Fig. A.1). MOM6 la trae activa en su DOME (`add_drag_diffusivity`) y en OM4 (`add_LOTW_BBL_diffusivity`). pyOCEAN no la tiene: es física faltante, clase (b).

Esteban quiere además saber si la teoría analítica revela algo que el modelo no contempla. La lectura de la biblioteca local apunta a tres cosas que habría que medir con un diagnóstico de la pluma:
- si la cizalla de la interfaz está resuelta (Jackson pide ≥ 2–4 puntos en la zona de Ri bajo);
- el papel del transporte de Ekman de fondo (Ezer 2005);
- que el α_E de nuestra tabla (1.03e-3, digitalizado de la Fig. 3b) no es el número que Legg imprime (8.6e-4, E de la esquina).

**Meta de Esteban (2026-10-30):** la referencia principal es la **corrida isopicna de 10 km de Legg** (b_f −11.07e-3), porque tiene la misma física. La no hidrostática queda como referencia secundaria.

## Metas

**A. Diagnóstico teórico de la pluma** (posproceso; no cambia el modelo). Se calcula a lo largo de la trayectoria Y(x) de la pluma para las corridas pyOCEAN, MITgcm (Legg y física común) y, donde aplique, para los números de Legg:
- espesor h (traza s > 0.01), velocidad ΔU respecto del ambiente, Δb, Ri_B = Δb h/ΔU² y Fr local;
- el Ri de gradiente en la interfaz (caras de la pluma), su histograma y el número de celdas con Ri < 0.25 a través de la interfaz;
- los números de Legg (Tabla 2): Ekman friccional ν_v/(h₀² f) y drag-Ekman C_d Δb₀ tanα/(f² h₀);
- el transporte de Ekman de fondo talud abajo, definido por el constructor y documentado;
- la velocidad a lo largo del talud contra g′s/f (Nof 1983);
- la predicción de Ellison–Turner y de Xu (E(Ri_B), E(Ri_g)) contra el E medido.

Además se aclara la definición de α_E (Legg ec. 12 contra ec. 14) y se corrige la tabla de `validation/dome/` si hace falta.

**B. Mezcla BBL por arrastre** en el modelo: `bbl_mixing=None|'lotw'` en `SigmaBaroclinic`, apagada por defecto. Es un port de MOM6 `add_LOTW_BBL_diffusivity` más la energía de `set_BBL_TKE`, en σ, z\* e híbrido.

**C. Corridas y comparación** contra la corrida isopicna de 10 km de Legg, con cada diferencia clasificada de (a) a (d).

## No metas

- `add_drag_diffusivity` (el esquema de capas de MOM6 DOME). Depende de las densidades objetivo de capa (`GV%Rlay`) y MOM6 desaconseja su modo `SIMPLE` con regridding. La forma de una sola capa de Legg (A.2) tampoco se implementa. Si el resultado lo pide, se agrega después como variante.
- Reproducir el caso de Xu (2006) con Nek5000. La ley de Xu entra en el diagnóstico.
- La mezcla BBL dentro de KPP o el GLS.
- Cambiar la viscosidad: MOM6 solo produce Kd y no hay número de Prandtl.

## Antimetas

- Ajustar `bbl_effic`, `cdrag` o cualquier parámetro para acercarse a Legg. Se usan los valores de MOM6/Legg (γ = 0.2) y no se tunea a un caso.
- Romper algo existente: con `bbl_mixing=None` todo queda bit a bit (goldens, smoke 9/9).
- Reportar el diagnóstico teórico como verdad: las leyes de laboratorio sin rotación sobreestiman ~10× en DOME (Legg §3). Sirven para explicar, no como meta.

## Criterios de éxito

1. **Columna contra MOM6 compilado.** El Kd de BBL de pyOCEAN coincide con `add_LOTW_BBL_diffusivity` de MOM6, compilado como en `validation/mom6_column/`. La comparación toma la energía (`BBL_meanKE_loss`, `ustar_BBL`) de `set_BBL_TKE` en columnas fijas:
   - estratificada;
   - N² ≤ 0;
   - f = 0;
   - capas de 1 mm.

   La tolerancia es de 1e-10 relativo, salvo que el constructor justifique otra. Esta comparación resuelve el posible factor 2 de `BBL_meanKE_loss` que la lectura marcó como [verificar].
2. **Propiedades en columna.**
   - Kd ≥ 0.
   - El trabajo contra la flotabilidad no excede BBL_EFFIC × la energía disponible.
   - Kd = 0 si no hay arrastre.
   - Una columna con capas casi vacías da el mismo Kd que sin ellas.
   - Una columna seca queda quieta bit a bit.
3. **Nada se rompe.**
   - Apagada, todo queda bit a bit.
   - Restart bit a bit con el esquema activo. No tiene estado propio, pero lo que lea de `ctx` debe reconstruirse igual.
   - 1 = 5 hilos.
   - El pre-flight rechaza `bbl_mixing` con `closure='gls'` (doble conteo con la condición de fondo de k) y cualquier parámetro inválido, reportando todos los errores juntos.
4. **DOME 10 km contra la corrida isopicna de 10 km de Legg.** La corrida principal es híbrido `kv_low` + Smagorinsky 0.15 + Jackson Pr 1 + BBL.
   - Se reportan b_f, E de la esquina, α_E, transporte, trazador por clase y trayectoria contra la isopicna de 10 km de Legg (−11.07) y, en segundo lugar, contra la no hidrostática.
   - Criterio: **la dirección** hacia −11.07 con su magnitud reportada, sin nuevos truncamientos ni eventos del limitador.
   - El diagnóstico (meta A) debe explicar la diferencia que quede.
5. **El diagnóstico teórico** produce, para cada corrida, una tabla y una figura con Ri_B, Fr, celdas con Ri < 0.25, los números de Ekman, el transporte de Ekman y Nof. El README de DOME lleva una sección "qué dice la teoría", con cada afirmación citada a `archivo §` y lo inferido marcado.
6. **Costo:** ≤ +5 % por paso en DOME y en BayIW cut, medido como en `scripts/shear_cost.py`.
7. **z\* y σ:** las mismas corridas con y sin BBL, clasificadas.

## Decisiones de diseño

- **Esquema LOTW de OM4**, no `add_drag`. Es independiente de la coordenada, que es lo que necesitan σ, z\* e híbrido. Legg usó la forma de una capa (A.2) y MOM6 DOME usa `add_drag`. La diferencia de esquema con la referencia es una elección numérica (c) y se documenta.
- **Aditivo:** Kd_total = Kd_base + Kd_shear + Kd_BBL, como OM4 (`SD:1782-1783`).
- **La energía** sale de lo que el arrastre de pyOCEAN realmente quita, no de u\*³/√C_d. Se guardan en `ctx`, durante la fase implícita, las tasas r y las fracciones f_k sobre hbbl. Así queda correcto con `drag_bg_vel` ≠ 0 y con la ley logarítmica.
- **Energía física, no la de MOM6** (decisión de Esteban, 2026-10-30, tras la revisión de constructor). La energía es el trabajo que el arrastre de pyOCEAN realmente quita, ≈ C_d|U|³ promediado de caras a centros. u\* = √(|τ_b|/ρ₀) del mismo arrastre (`ctx.ustar2_bot`), así que incluye U_bg y la ley logarítmica de forma consistente. Si la columna contra MOM6 compilado confirma que `set_BBL_TKE` da ≈ 2 C_d|U|³ y ≈ √2 u\* (suma sobre las 4 caras, `SD:2132-2137`), el factor se documenta como clase (d). Entonces γ = 0.2 en pyOCEAN equivale a 0.1 en MOM6/OM4, y eso se dice al comparar. El criterio 1 compara el kernel LOTW con la energía y la u\* de MOM6 como entradas, y una prueba aparte comprueba el factor de `set_BBL_TKE`.
- **Parámetros** en una dataclass `BBLParams`, con valores por defecto de MOM6/OM4:
  - `bbl_effic` 0.2;
  - `max_decay` 200 m;
  - `von_karman` 0.41;
  - `use_omega` True (piso Ω² de N²);
  - `kd_max` 0.1 m²/s, el de OM4. Debe ser > 0 porque LOTW lo usa donde N²·dz = 0.
- **Capas delgadas:** se fusionan antes, con el umbral y el patrón de Jackson/GLS (`thin_layer_threshold`).

## Decisiones de Claude (a revisar por Esteban)

1. **`drag_bg_vel` en DOME.** pyOCEAN corre DOME con 0, y MOM6 DOME y Legg con 0.05 m/s. Esto cambia la energía disponible en zonas lentas. Se agrega `--drag-bg-vel` al runner. Las corridas con BBL usan 0.05, como la referencia, y se agrega una corrida de control sin BBL con 0.05 para separar los dos efectos. La línea base existente no se toca.
2. **Velocidades para u\*:** las de la fase implícita (u nueva, como `ctx.ustar2_bot`), documentadas como diferencia con MOM6, que usa las de inicio de paso. Es clase (c).
3. **El diagnóstico** vive en `validation/dome/theory.py`, con pruebas en `tests/test_dome_benchmark.py`, y no en el núcleo.
4. **Línea base del criterio 4** (revisión de constructor): la dirección hacia −11.07 se juzga contra el control con el mismo `drag_bg_vel` 0.05 y sin BBL, así el efecto de la BBL queda aislado. También se reporta contra la corrida principal de Jackson (U_bg = 0, b_f −12.84), para separar el efecto de U_bg.
5. **Solo ALE** (σ, z\* e híbrido de ALE; revisión de constructor). En el σ heredado (no ALE) el arrastre vive dentro de `SigmaWetDry.column_step`, sin r ni f_k; el pre-flight rechaza `bbl_mixing` ahí, igual que `bottom_drag='quadratic'`. DOME corre siempre en ALE.
6. **Columna contra MOM6 sin fusión** (revisión de constructor, como en Jackson). El kernel sin fusionar se compara con MOM6 a 1e-10, incluida la columna de 1 mm. La ruta con fusión de capas delgadas se prueba solo por la igualdad del criterio 2 (MOM6 no fusiona en LOTW).

## Lo que queda para el constructor

- La estructura del port (oráculo NumPy y kernel Numba, como `kappa_shear`).
- La definición operativa del transporte de Ekman de fondo y de la "interfaz" de la pluma.
- La forma exacta del driver Fortran para la columna.
- Las figuras.
- El orden de las corridas: una a la vez y en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`.

## Restricciones

- Solo Python.
- ≤ 5 hilos en P-cores, con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`.
- Nunca `pip install`; numpy 1.26.4.
- Commits con rutas explícitas.
- Salidas fuera del repo.
- Literatura: primero la biblioteca local y luego Consensus, con 1–2 búsquedas como máximo. Los papers usados van a `docs/literature/reading_list.md`, incluidos los faltantes de `entrainment_theory.md` que se citen.
- Leer el código de MOM6 antes de implementar (ya hecho, `docs/design/bbl_refs/`).
- Consultar a Esteban si una corrida pasa de ~24 h.
