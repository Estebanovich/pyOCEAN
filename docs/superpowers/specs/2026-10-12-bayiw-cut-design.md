# BayIW 3-D recortada: pyOCEAN contra MITgcm en el mismo corte, y anidada en la corrida completa

**Fecha:** 2026-10-12. **Estado:** diseño aprobado por Esteban en la conversación, sección por sección. Falta revisar el texto escrito.
**Rama:** `bayiw-cut`, desde `main` (faf4f04).

**Contexto:**
- Es el paso 3 del orden re-acordado (`2026-10-12-bayiw-2d-section-design.md` §1).
- Usa lo aprendido en la sección 2-D: la profundidad efectiva de MITgcm, el inicio desde los niveles de MITgcm, el muestreo en niveles fijos y la frontera "solo Orlanski".
- Usa también el estado exterior que varía en el tiempo (`2026-10-12-baroclinic-obc-internal-wave-design.md`).
- **Decisión de Esteban:** hay dos fases. Primero los dos modelos en el mismo corte con radiación; después anidados en su corrida completa.

## 1. Corte, casos y costo

### 1.1 Corte

- **Índices:** i = 180–379, j = 192–351 de la malla BayIW (200 × 160 × 50; celdas de ~5 km, unos 1000 × 800 km).
- **Contenido:**
  - la bahía (i = 274–285, j = 324–347);
  - el viento (más del 10 % del máximo en i = 220–340, j ≥ 250);
  - las 7 estaciones: A (230, 315), B (330, 315), C (280, 265), Gamma (280, 215), Omega (280, 315), Seiche2 (280, 325) y Seiche (280, 345).
- **Fronteras:** oeste, este y sur abiertas; norte, la costa.
- **Datos:** `dx`, `dy`, la batimetría con la profundidad efectiva de MITgcm (`bayiw.mitgcm_depth`), la T inicial y el viento, recortados de los binarios de Esteban (`bayiw.load_inputs`).

### 1.2 Casos y modelos

- **Casos:** dos capas y lineal, con bahía y sin bahía; 60 h, la ventana forzada de la etapa 1.
- **pyOCEAN:**
  - z\* (50 capas `delR`) e híbrida (25 capas, 10 objetivos ligeros), Δt = 120 s;
  - energía cinética Arakawa (valor por defecto desde 2026-10-12), la física de MITgcm (spec 3-D §3) y `mix_diagnostic=False`;
  - inicio desde los niveles de MITgcm (`initial_state(..., h_src=bayiw.mitgcm_source(...))`);
  - una corrida de control a Δt = 30 s (z\*, dos capas con bahía) mide el efecto del paso de tiempo.
- **MITgcm:**
  - el código de Esteban (`BayIW_Rectan_2layer/code`), con `SIZE.h` para 200 × 160 y 5 procesos MPI (por ejemplo sNx = 40, nPx = 5, sNy = 160);
  - su `data.stage1` con los archivos del corte, en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/`.
- **Costo estimado:** pyOCEAN ~18 min por corrida; MITgcm ~40 min con 5 procesos. Si una corrida de MITgcm pasa de 2 h, se reporta antes de seguir.

### 1.3 Fase 1: el mismo corte con radiación

Los dos modelos radian hacia un exterior en reposo:
- **MITgcm:** solo Orlanski en los tres bordes, sin `useOBCSbalance` ni esponja. La sección 2-D mostró que la esponja sobre la velocidad total y `balanceFacS = -1` reflejan el pulso barotrópico; índice 0.57 frente a 0.02 con solo Orlanski.
- **pyOCEAN:** Flather + Orlanski 0.45 + esponja de 30 celdas sobre u′ (60 → 1200 s), como el benchmark 3-D.

### 1.4 Fase 2: anidado en la corrida completa de Esteban

**La "verdad":** las corridas completas de Esteban:
- `BayIW_Rectan_2layer/run_expand` y `run_expand_nobay`;
- `Esteban_SSD/…/BayIW_Rectan_linear/run_expand` y `run_expand_nobay`.

Todas guardan el estado MNC cada 900 s en 8 teselas de 140 × 176.

**Bordes:** u, v, T y η en los bordes del corte, extraídos cada 900 s:
- **pyOCEAN:** con `open_boundary_state` en modo `'radiating'`, a partir de un `OBData` por borde. `zw` son los niveles de MITgcm, `u_normal` la velocidad normal en las caras del borde, y `eta` y `ubar` vienen de esos mismos datos. Se interpolan linealmente en el tiempo con `interpolated_boundary`.
- **MITgcm:** con OBCS prescribe (`useOBCSprescribe`, archivos OB por borde a 900 s) y los mismos campos, sin balance ni esponja.

**Comparación:** los dos anidados contra la corrida completa, en el interior del corte (sin las 10 celdas junto a los bordes).

## 2. Métricas y criterios

**Muestreo:** igual que en la sección 2-D. pyOCEAN se remapea de forma conservativa (PPM_H4) a los niveles *fijos* de MITgcm, con η en el nivel superior, y se muestrea igual que la salida de MITgcm.

1. **Seiche:** Δη = η_bahía − η_sinbahía en Seiche y Seiche2, de 0 a 27 h. Hay unos 7 ciclos de 3.8 h, así que el período espectral sí se resuelve. Se reportan período, rango y correlación.
2. **Ondas internas:**
   - T110, T250 y V110 (y U110 en A y B) en las 7 estaciones, con y sin bahía, de 0 a 60 h: amplitud rms, correlación y desfase;
   - en dos capas, además, la altura de la isoterma central (20.645 °C).
3. **Efecto de la bahía:** ΔT110 (bahía − sin bahía) en las estaciones.
4. **Mapa de T a 110 m** a las 6, 24 y 60 h: error rms relativo de la anomalía.
5. **Salud:** truncamientos, max|Σh − H − η|, cambio de calor y costo por paso.

**Criterios de la fase 1** (pyOCEAN contra MITgcm en el mismo corte):
- seiche: período ±2 %, rango ±10 %, correlación ≥ 0.95;
- ondas internas en Seiche, Seiche2, Omega y C: amplitud ±20 % y correlación ≥ 0.8 (T110, o la isoterma en dos capas);
- efecto de la bahía: mismo signo y orden de magnitud;
- z\* frente a híbrida y Δt = 120 frente a 30 s: se reportan sin umbral.

**Criterio de la fase 2:** en las métricas 1–4, el error de pyOCEAN anidado contra la corrida completa no pasa de 1.5 veces el de MITgcm anidado. Este último es el piso de error del anidado mismo.

**Toda diferencia** se clasifica de (a) a (d). Si es (a), me detengo y se lo reporto a Esteban.

## 3. Piezas y pruebas

**`validation/bayiw_cut/`:**
- **`cut.py`:**
  - el recorte de las entradas;
  - las entradas y los namelists de MITgcm (desde `data.stage1`, con `data.obcs` en las variantes `radiate` y `nest`);
  - el modelo pyOCEAN;
  - la lectura MNC del corte y de la corrida completa (unión de teselas por X/Y);
  - para la fase 2, la extracción de los bordes a `OBData` interpolados y a los archivos OBCS de MITgcm.
- **`run_mitgcm.sh`:** `build`, y `run <case> <bay> radiate|nest`.
- **`run_pyocean.py`:** series en las estaciones, el mapa de T a 110 m y la isoterma.
- **`compare.py`:** las métricas del §2, reutilizando `validation/bayiw_2d/compare.py` y `section.py` donde se pueda.
- **`README.md`:** tablas, clasificación y "qué debe mejorar pyOCEAN".

**Tests baratos** (`tests/test_bayiw_cut.py`):
- el corte coincide con los binarios en las esquinas y en las estaciones;
- los archivos de MITgcm (entradas y OBCS) tienen el orden de bytes big-endian y las formas correctos;
- la unión de teselas reproduce un campo conocido;
- los bordes extraídos coinciden con la salida completa en un instante;
- el modelo del corte se construye y da 3 pasos, con y sin anidado.

**Documentación:** una fila en el marcador, `CLAUDE.md` y la memoria.

## 4. Fuera de alcance

- La BayIW completa (paso 4, opcional).
- La salinidad.
- Un exterior 3-D en toda la banda de esponja (la siguiente pieza de `OBData`).
- La reflexión de la frontera sur en las corridas completas de Esteban: es parte de su "verdad" y no se corrige.
