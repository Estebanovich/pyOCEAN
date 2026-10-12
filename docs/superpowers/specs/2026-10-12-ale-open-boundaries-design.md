# Fronteras abiertas en el modo ALE (Plan C)

**Fecha:** 2026-10-12. **Estado:** diseño aprobado por Esteban en la conversación; falta revisar el texto escrito.
**Spec padre:** `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` (§6.1 fronteras, §9 fase 5).
**Rama:** `ale-plan-b`.

## 1. Objetivo y alcance

El modo ALE de `SigmaBaroclinic` (`vertical_coordinate="ale"`) hoy solo corre en dominios cerrados; el constructor rechaza `open_boundaries` con un `ValueError` ("ALE Plan C"). Este plan habilita las fronteras abiertas con las mismas piezas que el modelo σ: Flather, Orlanski por capa, esponja y entrada de trazadores. Es el paso previo al benchmark de BayIW baroclínico contra MITgcm, que tendrá su propio spec.

**Alcance A (elegido por Esteban):** lo necesario para BayIW y el canal de la onda interna, con el exterior en reposo o con un perfil fijo:
- Flather (sin cambios);
- Orlanski por capa sobre la parte baroclínica ponderada con los grosores;
- transporte por capa en la cara abierta con el grosor de la celda interior;
- entrada de trazadores desde un perfil exterior guardado en z y remapeado a las capas actuales;
- esponja con objetivos en z.

**Fuera de alcance:**
- el reservorio de trazadores con L > 0 y el nudging τ_in/τ_out (spec padre §6.1, puntos 6 y 4), que entran cuando haya datos externos que varíen en el tiempo (anidamiento en HYCOM-IOA);
- columnas de frontera que se secan (H ≤ 0) en ALE;
- el restart con ALE (Plan D);
- la frontera por relaciones de polarización (Marsaleix et al. 2009b).

## 2. Enfoque: un solo camino con pesos

En σ, la separación en media y parte baroclínica usa Δσ fijo (`_baroclinic_part`). Se generaliza con pesos por cara w_k:

- **σ:** w = Δσ. El resultado es idéntico bit a bit al actual.
- **ALE:** w son las fracciones de grosor de la cara del paso, `face_fractions` del grosor medio de cara de hⁿ, las mismas que usa la superficie libre para ū.

Orlanski, la esponja de velocidad y el transporte en la cara abierta usan el mismo código en los dos modos. Lo único exclusivo de ALE es el estado exterior en z (§3.5). Es lo que hace MOM6, que tiene una sola rutina de fronteras para cualquier coordenada. Se descartó duplicar las rutas para ALE porque el código se desincroniza y cada arreglo futuro habría que hacerlo dos veces.

## 3. Componentes

### 3.1 Separación media/baroclínica con pesos

    ū = Σ_k w_k u_k,   u′_k = u_k − ū,   Σ_k w_k = 1   ⇒   Σ_k w_k u′_k = 0

Una cara sin agua (Σ grosor = 0) tiene w = 0 y conserva u.

### 3.2 Orlanski por capa

La fórmula implícita actual (ROMS `u3dbc`; Marchesiello et al. 2001, solo la parte normal) se aplica a u′ en el mismo índice de capa de la cara de frontera y de las dos caras interiores, como MOM6 (`MOM_open_boundary.F90:2950-2972`) y CROCO (`u3dbc.F:112-160`):

    r = clip((φ₁ⁿ − φ₁ⁿ⁺¹)(φ₁ⁿ − φ₂ⁿ) / (φ₁ⁿ − φ₂ⁿ)², 0, r_max),   φ_Bⁿ⁺¹ = (φ_Bⁿ + r φ₁ⁿ⁺¹)/(1 + r)

Después se re-centra con los pesos de la cara de frontera (Σ w u′ = 0), para que Orlanski no altere el transporte de Flather. u′ⁿ y u′ⁿ⁺¹ se separan cada uno con sus propios pesos: los de hⁿ, que son los únicos disponibles en el paso.

### 3.3 Esponja de velocidad

Es la relajación implícita actual (Martinsen & Engedahl 1987; perfil de MITgcm OBCS), aplicada solo a u′ con los pesos de 3.1. La media ū y h no se relajan.

### 3.4 Transporte por capa en la cara abierta

    Q_k = (h_k / Σ_j h_j) · Q_bt + L · h_k · (θ u′_kⁿ⁺¹ + (1 − θ) u′_kⁿ)

- h_k es el grosor de la celda interior (MOM6 `MOM_continuity_PPM.F90:1237-1248`).
- u′ se separa con las fracciones de esa celda, de modo que Σ_k Q_k = Q_bt exacto, con Q_bt el flujo de Flather de la superficie libre.
- Con σ se reduce a la fórmula actual (h_k/Σh = Δσ_k, Σh = D).
- Reemplaza en las caras abiertas al transporte interior (PPM o upwind); las caras interiores no cambian.

### 3.5 Estado exterior en z

- En `initial_state` (ALE) se guardan `h_ext` y `T_ext`: los grosores y la T del estado inicial ya regenerado (para la híbrida, después de `init_iterations`). Están en reposo, con η = 0.
- En cada paso, para las columnas de frontera y de esponja:
  1. el perfil se estira a la columna actual, h_ext·(H + η)/H, igual que z\* (spec padre §6.1, punto 1: "se ajustan al fondo"), lo que conserva la T media de la columna;
  2. se remapea de forma conservativa (`core.vertical.remap.remap_columns`, el esquema del modelo) a las capas destino.
- Destinos:
  - **entrada de trazadores** en las caras abiertas: capas hⁿ de la celda de frontera (`_open_faces`, valor aguas arriba de Marsaleix et al. 2006: la celda al salir, el exterior al entrar; también para la pila T²);
  - **esponja de T:** capas h† después de la física vertical.
- Los arreglos tienen el tamaño del dominio, pero solo se remapean las columnas de la máscara frontera ∪ esponja.
- En σ, `T_ext` sigue en capas σ como hoy.

### 3.6 Flather

Sin cambios: exterior en reposo (η = 0, ū = 0) en la superficie libre implícita. En ALE se le pasan los mismos argumentos `flather_*` que en σ.

### 3.7 Orden en el paso ALE con fronteras

1. Superficie libre con Flather.
2. Esponja de u′.
3. Orlanski.
4. Cero en las capas de fondo delgadas (`zero_thin_bed_velocity`, como hoy).
5. Transportes: interiores con PPM o upwind; abiertos según 3.4.
6. h† lagrangiano.
7. Trazadores, con la entrada desde `T_ext` remapeado a hⁿ.
8. Física vertical.
9. Esponja de T hacia `T_ext` remapeado a h†.
10. Regrid y remap.

## 4. Casos difíciles y errores

1. **Capas desvanecidas en la frontera** (z\* junto al fondo, capas sin masa de la híbrida):
   - Orlanski se aplica igual por índice;
   - la regla de capas de fondo delgadas pone u = 0 donde la cara mide menos que `h_thin_bed`;
   - una capa con h = 0 no transporta en la cara abierta (Q_k ∝ h_k), así que su T de entrada no influye.
2. **Columnas que se secan:** en ALE, si una cara abierta toca una celda activa con H ≤ 0, el constructor da un `ValueError` que lo explica. En σ sigue como hoy.
3. **Híbrida:** el remapeo conservativo da la T media de cada capa isopicna de la frontera; si no cae en su objetivo, el regrid del paso la corrige como en cualquier columna.
4. **API:**
   - `open_boundaries`, `sponge_cells`, `sponge_tau`, `orlanski_cmax` y `sponge_tracer` se usan igual que en σ;
   - se quita el `ValueError` de "Plan C";
   - `T_ext` se mantiene y en ALE se acompaña de `h_ext`.
5. **Restart con ALE:** sigue rechazado con su error actual (Plan D).

## 5. Pruebas (TDD)

**Unitarias:**
1. **Separación con pesos:**
   - con w = Δσ es idéntica a la actual;
   - con fracciones arbitrarias, incluidas capas vacías, Σ w u′ = 0 a redondeo;
   - una cara sin agua conserva u.
2. **Transporte en la cara abierta:**
   - Σ_k Q_k = Q_bt a redondeo con grosores arbitrarios y capas desvanecidas;
   - una capa con h = 0 tiene Q = 0;
   - con grosores proporcionales a Δσ reproduce la fórmula σ.
3. **Perfil exterior:**
   - remapear sobre la misma malla es la identidad;
   - el estiramiento por (H + η)/H conserva la T integrada;
   - sobre capas isopicnas da la media de cada capa.
4. **Error de configuración:** cara abierta junto a una celda con H ≤ 0 en ALE → `ValueError`.

**Integración** (z\* e híbrida, con PPM por defecto):
1. **Trazador uniforme** con fronteras abiertas y una perturbación de η que sale por Flather:
   - T se queda en 7.0 (1e-11);
   - Σh = H + η (1e-12);
   - el volumen sale del dominio y las caras de frontera llevan flujo.
2. **Estratificación en reposo** con fronteras abiertas y esponja: max|u| ≤ 1e-10 m/s en z\*.
3. **Canal de onda interna modo 1** (tests actuales de σ, 16 h, error interior contra el canal de 300 km). Mismos umbrales que σ:
   - cerrado > 0.3;
   - Orlanski < 0.2;
   - Orlanski + esponja < 0.05.

   Se reportan los valores medidos junto a los de σ (0.82 / 0.157 / 0.015).
4. **Regresión σ:**
   - todos los tests σ actuales pasan sin cambios;
   - durante la implementación se compara bit a bit una corrida σ con fronteras y esponja antes y después del cambio.

## 6. Criterio de éxito

- Las pruebas de §5 pasan con z\* y con la híbrida, y la suite completa pasa.
- Se actualizan:
  - la fila "Fronteras abiertas" del marcador (`docs/design/vision_roadmap.md` §7) con los números de ALE;
  - `CLAUDE.md`;
  - el spec padre (§9, fase 5 hecha con alcance A).

## 7. Después de este plan

**Benchmark BayIW baroclínico contra MITgcm** (spec aparte):
- configuración `BayIW_Rectan_2layer` / `_linear`;
- resolución reducida primero (σ a resolución completa tarda 6.5 s/paso);
- z\* y luego la híbrida;
- estaciones de `Output_IWBay`.
