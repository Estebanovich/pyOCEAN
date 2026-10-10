# Validación de mareas contra mareógrafos

Fecha: 2026-10-10. Script: `compare_ticon_tpxo.py`. Salidas: `out/ticon_vs_tpxo_*.csv`.

## Bases de datos evaluadas

| Fuente | Qué ofrece | Control de calidad | Estado (2026-10-10) |
|---|---|---|---|
| **TICON-3** (Hart-Davis et al. 2022; PANGAEA doi:10.1594/PANGAEA.951610, CC-BY-4.0) | 40 constantes armónicas en 3471 mareógrafos, derivadas de GESLA-3, con σ de amplitud y fase, % de faltantes y huecos | Exige ≥ 70 % de datos válidos y ≥ 1 año | ✅ Descargado en `~/pyOCEAN_refs/sealevel/ticon3`. Usado aquí |
| **UHSLC / JASL** (Univ. de Hawái + NOAA NCEI) | Series horarias "research quality" (QC de tiempo, unidades y datum) y "fast delivery" | Dos etapas: variabilidad y datum | ⚠️ Servidor inaccesible al consultar (timeout) |
| **IOC Sea Level Station Monitoring Facility** | Datos crudos en tiempo casi real, alta frecuencia | **Ninguno**: datos tal cual del sensor | ✅ Accesible; útil para detectar fallas actuales |
| **Servicio Mareográfico Nacional (UNAM)** | 27 estaciones (12 en el Pacífico, 15 en el Golfo y el Caribe) | Entrega datos **crudos**; el procesamiento requiere convenio | ✅ Sitio accesible (mareografico.unam.mx); descarga por solicitud |
| **NOAA CO-OPS** | Datos horarios verificados de estaciones de EE. UU. en el Golfo | Verificados | ✅ API accesible |
| **GESLA-3** (Haigh et al. 2023) | Series horarias o mejores, de 5119 mareógrafos | Banderas por dato; reporta saltos de datum en ~21 % de los registros y desfases de tiempo frecuentes | Pendiente de descarga |

Notas sobre la calidad de las fuentes:

- **Metadatos de TICON-3:** la columna "fuente" tiene etiquetas mal asignadas en México. Por ejemplo, Cabo San Lucas aparece como `gesla.sfwmd` (Florida) y Mazatlán como `gesla.rws` (Holanda). No se usa para nada.
- **Registros duplicados:** varios sitios aparecen dos veces, con el mismo registro bajo dos coordenadas o dos fuentes.

## Método: separar defectos del mareógrafo de errores del modelo

Para cada registro costero se comparan las 8 constantes de HYCOM contra TPXO10-atlas-v2, interpolado con la bilineal que excluye tierra (`forcing/tides.py`). Luego se ajustan conjuntamente un desfase de reloj τ y una ganancia κ (`diagnostics/tide_gauges.py`):

    h_obs ≈ κ · h_TPXO · e^{−iω_c τ}

| Firma | Interpretación |
|---|---|
| τ ≈ un número entero de horas, κ ≈ 1 | **Huso horario**: hora local de México (UTC−6/−7) registrada como UTC, o el horario de verano |
| κ ≈ 10ⁿ | **Unidades** (mm frente a cm) |
| κ < 0 | **Sensor invertido** |
| κ < 1 y τ de decenas de minutos | **Atenuación y retraso** físicos: laguna, estuario, puerto, o falta de resolución del atlas |
| Residuo grande sin firma simple | **Misfit**: modelo, efectos locales u otra falla |

**Registros del mismo sitio.** Además se compara cada registro contra los otros registros del mismo sitio, sin pasar por TPXO. Un registro que difiere de sus hermanos con una firma instrumental, y que además es el peor contra TPXO, es un **defecto confirmado**. Si varios registros coinciden entre sí pero no con TPXO, el problema es del **modelo o del entorno local**.

**Nodos someros.** Se descartan los mareógrafos cuyos nodos de mar en TPXO tienen menos de 5 m. TPXO10 tiene **amplitudes espurias de 1.5 a 2.4 m** en nodos de 2 m de profundidad, por ejemplo en la Florida Bay.

## Resultados (estaciones mexicanas)

**Pacífico (34 registros):** 20 sin bandera, con **RSS mediano de 1.5 cm**, |ΔG_M2| de 1.7° y |ΔA_M2| de 0.6 cm.
- **Acapulco**: 4 registros, de 1952 a 2017, coinciden entre sí. TPXO **subestima M2 en ~30 %** (12.8 frente a 18.5 cm); la fase está bien. Es un problema del atlas o una amplificación en la bahía, no del mareógrafo.
- **Topolobampo**: 2 registros coinciden entre sí, con 60 min de retraso respecto a TPXO. Es la laguna o la falta de resolución del atlas (profundidad de TPXO de 9 m).
- **La Paz** (33 min) y **San Quintín** (100 min): retrasos de laguna.
- **Ixtapa–Zihuatanejo (17.64°N)**: misfit pequeño (1.8 cm).

**Golfo y Caribe (18 registros):** 5 sin bandera, con **RSS mediano de 1.3 cm** y |ΔG_M2| de 0.4°. Veracruz da 0.45 cm.
- **Defecto confirmado — Progreso, 1980–1984:** fase de M2 desplazada 175° y desfase de **6.0 h** respecto a los otros dos registros de Progreso (1952–1994 y 2012–2013). Es un error de huso horario o de signo en ese registro.
- **Defecto confirmado — Puerto Morelos, registro 2007–2014:** amplitud **10 veces menor** que el otro registro del mismo periodo y que TPXO. Es un error de unidades.
- **Tampico**: 3 registros coinciden entre sí; TPXO sobrestima en ~14 %. Es la desembocadura del Pánuco o el atlas.
- **Dos Bocas**: κ = 0.4 y 73 min de retraso. Es la dársena del puerto.
- **Laguna de Términos (Cd. del Carmen) y Coatzacoalcos**: TPXO no es utilizable ahí (menos de 5 m o sin nodos de mar).

## Advertencias

- TICON-3 resume registros **históricos**. Una estación sin bandera aquí puede estar fallando hoy. Para el estado actual (sensores sin mantenimiento, picos, valores planos, saltos de datum) hay que aplicar control de calidad a los datos crudos recientes de IOC o del SMN.
- Un retraso de 1 h puede ser horario de verano o físico. Sin registros hermanos no se puede distinguir.
- El RSS contra TPXO en sitios abrigados mide sobre todo la resolución del atlas, no el mareógrafo.
