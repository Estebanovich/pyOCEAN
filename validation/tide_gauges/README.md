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

- TICON-3 resume registros **históricos**. Una estación sin bandera aquí puede estar fallando hoy. El estado actual se evalúa abajo, con los datos crudos recientes de IOC/SMN.
- Un retraso de 1 h puede ser horario de verano o físico. Sin registros hermanos no se puede distinguir.
- El RSS contra TPXO en sitios abrigados mide sobre todo la resolución del atlas, no el mareógrafo.

---

# Estado actual de la red: QC de datos crudos IOC / SMN-UNAM

Fecha: 2026-10-10. Periodo: 2026-07-12 → 2026-10-10 (90 días).
Scripts: `fetch_ioc.py` (descarga, en caché en `~/pyOCEAN_refs/sealevel/ioc/`) y `qc_ioc_mexico.py`.
Módulo: `diagnostics/sealevel_qc.py` (37 tests). Salidas: `out/ioc_qc_series.csv`, `out/ioc_qc_twins.csv`, `out/ioc_qc_clock.csv` y figuras `out/ioc_qc_*.png`.

La IOC SLSMF retransmite casi en tiempo real la red del SMN, **sin ningún control de calidad**. La API entrega como máximo ~31 días por petición. Casi todas las estaciones del SMN tienen dos sensores (radar `rad` y flotador `flt`; Manzanillo, además, burbujeador `bub`). Se excluye el canal `bat`, que es el voltaje de la batería.

## Método

| Revisión | Cómo | Qué no debe confundir |
|---|---|---|
| Reporte | Último dato, cobertura, duplicados, marcas de tiempo desordenadas o en conflicto, huecos | — |
| Errores gruesos | \|h − marea TPXO − mediana\| > 2.5 m | Marejada de tormenta |
| Picos | Desviación respecto a una mediana móvil de 15 muestras > max(6σ_local, 5 cm), en rachas de ≤ 3 muestras | Tsunami o seiche: son coherentes y elevan la σ local |
| Sensor pegado | Valores idénticos mientras la marea esperada cambia > max(3 × resolución, 2 cm) | Estoa cuantizada a 1 cm |
| Saltos de datum | En el residuo del análisis armónico propio: el mismo cambio a escala de 10 min, 1 h y un día de marea (24.84 h), > 5σ del ruido corto, y nivel estable a ambos lados (> 3σ_local) | Marejada (cambia en horas), seiches de plataforma con frentes abruptos (Progreso) |
| Reloj dentro del periodo | τ y κ por ventanas de 3 días contra la predicción con las **constantes propias** del periodo (κ > 0, \|τ\| ≤ 7.5 h); el salto o la deriva se deciden con significancia (`clock_changes`, z > 5) | El ciclo vivas–muertas: contra una referencia de otra época, τ oscila ±45 min con el batimiento K1–O1 |
| Contra su historia | Constantes de los 90 días (P1 y K2 inferidas de K1 y S2 con las razones de TPXO) contra el registro TICON-3 del mismo sitio, excluyendo los registros con defecto confirmado | Retraso físico de laguna (está en ambas épocas) |
| Sensores gemelos | h₂ ≈ a + κ h₁(t − τ) con incrementos de 1 h de series suavizadas 1 h; diferencia alineada → escalones y deriva | La marea, la marejada y las seiches se cancelan |

**Lecciones del primer pase (corregidas, cada una con un test):**
- En macOS, BLAS usaba 9 hilos. Ahora el script fija 1 hilo.
- La ventana larga de 6 h convertía la marea residual en "escalones". Ahora es un día de marea.
- La referencia TICON de otra época generaba "saltos de reloj" falsos. Ahora la referencia es el análisis armónico propio.
- En Zihuatanejo, κ = 0.73 entre gemelos era **dilución por regresión**: el radar ve seiches y el pozo del flotador no. Ahora se suaviza 1 h antes del ajuste.
- La malla de tiempo no estaba anclada a múltiplos del paso, así que los gemelos que empezaban a otra hora no compartían instantes.

**Fiabilidad de los saltos de datum con un solo sensor.** Las 8 candidatas que se pudieron contrastar con un gemelo independiente fueron **rechazadas** por él. En costas energéticas, un salto de 5–10 cm con un solo sensor es solo un candidato. La confirmación necesita un gemelo independiente o una nivelación.

## Resultados

**Red:** 58 códigos IOC en México.
- **17 sin datos en el periodo:** Acapulco (`acap` 2017, `acap2` abril 2025), Cabo San Lucas 2017, Manzanillo_MX 2017, Mazatlán (ambos, agosto 2022), Ensenada, Isla Cedros, Loreto, La Paz (`lpaz`), Alvarado2 (julio 2025), Sánchez Magallanes2 (noviembre 2025), Huatulco (mayo 2026), Isla Mujeres (mayo 2026), Salina Cruz2 (abril 2026), Puerto Morelos2 (mayo 2024), Cd. del Carmen2.
- **Intermitentes:** Acapulco Club de Yates, Lerma (Campeche), Sisal, Frontera y Telchac reportan solo el 2–10 % del tiempo, en ~50 ráfagas. Telchac está **mudo desde el 5 de octubre**, Isla Clarión **desde el 21 de septiembre** y Tuxpan2 solo desde el 24 de septiembre.
- **34 series activas en 28 estaciones.** Contra su historia TICON, la mediana es RSS = 2.3 cm.

**Fallas de sensor:**

| Estación | Hallazgo | Evidencia |
|---|---|---|
| **Veracruz2 (`vera2`, flotador)** | **No sigue el mar**: plano en ~0.07 m durante los 90 días mientras la marea oscila ±0.4 m. Probablemente el flotador está atorado | M2 = 0.0 cm; κ = −0.01 contra TPXO y contra su gemelo `vera/rad` |
| **Isla Clarión (`iclr`, radar)** | **Escala ×0.1 y signo invertido**: la marea registrada es exactamente −0.10 × TPXO (τ = −2 min, ajuste limpio). TPXO es coherente en ese punto: mar abierto, 3000 m, M2 ≈ 40 cm alrededor de la isla. Después se calló el 21 de septiembre | Error de configuración del sensor, no del océano |
| **La Paz2 (flotador vs radar)** | **Deriva relativa** de ~9 cm desde el **9 de agosto**, en escalera (+3.7 cm/mes); antes estaban estables con −4.5 cm. Uno de los dos sensores deriva y hace falta nivelar en sitio | `out/ioc_qc_twin_lpaz2_flt_lpaz2_rad.png` |
| **Celestún (`clst`, flotador)** | Caídas repetidas a −4 m (encoder/flotador); marea **amortiguada al 64 %** y 15 min de retraso respecto a su registro de 2010–11; ganancia por ventana inestable (0.4–1.6) | Sospecha de pozo tapado o falla mecánica |
| **Sánchez Magallanes (`smag`, flotador)** | Marea **amortiguada al 74 %** respecto a 2016–18 (rango ~25 cm frente a ~60 cm de TPXO). Puede ser azolve de la boca o pozo tapado | Verificar en sitio |
| **Puerto Chiapas (`chia/flt`)** | Solo 10 % de cobertura, datum 7.6 m distinto, 20 cm rms contra el radar: sensor fuera de servicio. El radar `chia/rad` y `made/flt` están bien (2 cm rms) | — |
| **Lázaro Cárdenas2 radar** | 13.7 picos/día y 10 errores gruesos; los flotadores no tienen ese problema | Radar ruidoso (oleaje, aves o montaje) |
| **Puerto Morelos (`pumo`)** | 30 errores gruesos, 51 % de cobertura | — |

**Metadatos y retransmisión:**
- **El mismo sensor aparece con dos códigos:**

  | Par | Desfase |
  |---|---|
  | `laza/flt` ≡ `laza2/flt` | −1 min |
  | `zihu/rad` ≡ `zihu2/flt` | −1 min |
  | `ptan` ≡ `ptan2` | +1 min |
  | `puert/rad` ≡ `puer2/rad` | +1 min |
  | `prog` ≡ `prog2` | 0 |
  | `sian` ≡ `sian2` | 0 |
  | `tuxp/flt` ≡ `tuxp2/rad` | 0 |

  En dos pares, además, **la etiqueta del tipo de sensor está mal**: `zihu/rad` es en realidad el flotador de `zihu2`, y `tuxp2/rad` es el flotador de `tuxp`. El desfase de 1 min entre retransmisiones del mismo sensor sugiere convenciones distintas de marca de tiempo (inicio o fin del promedio). No afecta a la marea, pero sí a la llegada de un tsunami.
- **Marcas de tiempo duplicadas en conflicto:** Lázaro Cárdenas (865), Puerto Vallarta (692), Puerto Madero (188) y Cd. del Carmen (106).

**Sin anomalías relevantes:** Salina Cruz, Puerto Ángel, Puerto Vallarta2 (flotador vs radar: κ = 1.00, 1.5 cm rms), Zihuatanejo2 (flotador vs radar: κ = 0.99), Isla Mujeres2 (flotador vs radar: 0.4 cm rms), Veracruz (radar), Sian Ka'an2 y Manzanillo (burbujeador vs radar: κ = 0.95 con 3 min de retraso del burbujeador, solo 15 días de traslape). Huatulco2 tiene 6.8 h con el sensor pegado y el radar de Manzanillo 4.6 picos/día: son menores.

**Reloj:** ninguna estación activa muestra un cambio de huso horario, un salto de reloj ni una deriva significativos en el periodo. El defecto de huso horario de Progreso (1980–84) es solo histórico.

## Limitaciones

- 90 días no separan K1 de P1 ni S2 de K2: se infieren con las razones de TPXO. En marea diurna de baja amplitud (Golfo), la ganancia contra la historia tiene una incertidumbre de ~±15 %. Progreso y Progreso2, que son el mismo sensor, dan κ = 0.85 y 0.97 según su cobertura. Por eso solo se reporta κ < 0.8 o κ > 1.25.
- Los registros TICON de referencia son de otra época y a veces de otro punto del puerto (hasta 5 km).
- Los datos del SMN por solicitud (procesados) permitirían confirmar las fallas; aquí solo se usa la retransmisión cruda de IOC.
