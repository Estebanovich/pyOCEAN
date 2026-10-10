# Arquitectura de forzamientos multi-fuente

Documento de diseño para la Fase 2.
Fecha: 2026-10-10.

## 1. Objetivo

pyOCEAN debe aceptar los forzamientos oficiales que se usan en la operación y la investigación sin que el núcleo del modelo sepa de dónde vienen.

Principio, tomado de MOM6 y del flujo de preproceso de HYCOM:
**el núcleo del modelo solo ve variables canónicas en la malla del modelo.**
Cada fuente externa es un *adaptador* que traduce nombres, unidades, convenciones de tiempo, signos y mallas a ese formato canónico.

```
 Fuente externa ─► Adaptador ─► Dataset canónico (CF, unidades SI) ─► Regrid a malla C ─► Caché NetCDF ─► Modelo
 (ERA5, GFS,       (sources/)   (nombres y unidades fijos)            (interp.py)         (preproc/)      (surface_tendency,
  ESPC, TPXO…)                                                                                             OBData, IC, nudging)
```

## 2. Categorías de forzamiento y fuentes

### 2.1 Atmósfera (superficie)
Variables canónicas: `u10`, `v10` [m/s], `t2m` [K], `q2m` [kg/kg], `msl` [Pa], `sw_down` y `lw_down` [W/m²], `precip` [m/s].
Opcionalmente `taux`/`tauy` [N/m²] y `qnet` [W/m²] cuando la fuente ya da los flujos.

| Fuente | Resolución | Uso típico | Notas de conversión |
|---|---|---|---|
| ERA5 (ECMWF/C3S) | 0.25°, horaria | Reanálisis, hindcast | Radiación y precipitación acumuladas (÷3600, asignar a t − 30 min); q se obtiene de d2m + msl |
| GFS (NCEP) | 0.25°, 1–3 h | Pronóstico operacional | GRIB2; flujos promediados con ventana que se reinicia cada 6 h (hay que desacumular) |
| NAM / HRRR (NCEP) | 12 km / 3 km | Pronóstico regional (EUA y Golfo) | Malla Lambert, requiere regrid curvilíneo |
| ECMWF IFS HRES / open data | 0.1–0.25° | Pronóstico | Mismas convenciones que ERA5 |
| CFSR / CFSv2 | ~0.2–0.5° | Reanálisis acoplado | GRIB2 |
| JRA55-do / JRA3Q | 0.5° / 0.375° | Forzamiento de modelos oceánicos (OMIP) | Ya corregido para océano; q a 10 m |
| MERRA-2 (NASA) | 0.5°×0.625° | Reanálisis | Flujos promediados por hora |
| WRF propio | Variable | Acoplado o regional (p. ej. el WRF del ICAyCC-UNAM) | Rotar vientos de la malla del modelo a coordenadas geográficas |

### 2.2 Océano (condiciones iniciales, fronteras abiertas, nudging)
Variables canónicas: `temp` (θ, °C), `salt` (PSU), `u`, `v` [m/s], `ssh` [m], en (z, lat, lon).

| Fuente | Resolución | Notas |
|---|---|---|
| **ESPC-D-V02 (HYCOM/NRL)** | 1/12°, 41 niveles z (salida interpolada) | Sustituyó a GOFS 3.1 en 2024. THREDDS/OPeNDAP en hycom.org |
| GOFS 3.1 (HYCOM) | 1/12°, 41 z | Histórico (lo usa HYCOM-IOA) |
| HYCOM nativo (`.a/.b`, capas híbridas) | 1/12° o 1/25° | Para el anidamiento de la Fase 3; requiere un lector `.a/.b` y remapeo de capas |
| Copernicus GLORYS12 / Global Analysis-Forecast (CMEMS) | 1/12°, 50 niveles | Variables `thetao`, `so`, `uo`, `vo`, `zos`; requiere `copernicusmarine` |
| RTOFS (NOAA) | 1/12° | Pronóstico operacional (HYCOM) |
| SODA, ORAS5 | 0.25–0.5° | Reanálisis climáticos |
| WOA23 | 1/4° | Climatología, para condición inicial y relajación |

Puntos críticos de conversión:
- **Temperatura:** CMEMS da θ potencial; algunos productos dan temperatura in-situ o conservativa. Convertir a θ con `gsw`.
- **Profundidad:** positiva hacia abajo en las fuentes, z negativa en pyOCEAN.
- **SSH:** cada producto tiene su propio nivel de referencia (geoide o media). Restar la media del dominio y ajustar el transporte barotrópico neto.
- **Fallas:** rellenar tierra con extrapolación horizontal (sea-over-land) antes de interpolar.

### 2.3 Mareas
Variables canónicas: amplitud y fase por constituyente para `eta`, `ubar` y `vbar` (transporte o velocidad barotrópica), además del potencial de marea.

| Fuente | Notas |
|---|---|
| **TPXO10-atlas v2 / TPXO9 (OSU, Egbert & Erofeeva 2002)** | Constituyentes M2, S2, N2, K2, K1, O1, P1, Q1, M4, MS4, MN4 y de largo periodo. NetCDF, malla C propia (h, u, v). Licencia académica: no redistribuir |
| FES2022 (AVISO/LEGOS) | 34 constituyentes, 1/30°; alternativa con licencia abierta |
| Constituyentes de estaciones (mareógrafos SMN-UNAM, NOAA CO-OPS) | Para validación, no como forzamiento |

Implementación de la marea:
- Reconstruir η(t), ū(t), v̄(t) con correcciones nodales (f, u) y argumento astronómico V0, siguiendo Foreman (1977) o Pawlowicz et al. (2002, T_TIDE). Estas correcciones son el detalle que más se equivoca en la práctica.
- Aplicar la condición de **Flather (1976)** en la frontera abierta para la parte barotrópica:
  ū_n = ū_ext + √(g/H)·(η − η_ext).
- Para el potencial de marea dentro del dominio (equilibrio + Love numbers, factor 1 + k − h ≈ 0.69), la referencia es Arbic et al. (2004). Es opcional en dominios regionales pequeños y relevante para el Golfo completo.
- **Dependencia del núcleo:** las mareas requieren superficie libre con paso barotrópico adecuado (implícita o split-explicit, paso 3 de la Fase 2). Sin eso no conviene activar la marea.

### 2.4 Ríos
Variables canónicas: `Q` [m³/s], `T_river`, `S_river` (≈0) por punto de descarga.

| Fuente | Notas |
|---|---|
| CONAGUA (BANDAS) | Grijalva, Usumacinta, Papaloapan, Coatzacoalcos, Pánuco… |
| USGS NWIS | Mississippi (Tarbert Landing), Atchafalaya, Mobile… |
| GloFAS (Copernicus) | Global, útil donde no hay estaciones |
| Dai & Trenberth (2002) | Climatología |

### 2.5 Otras (para fases posteriores)
- **Oleaje:** WW3 (NOAA GFS-Wave), ERA5-wave; para Stokes drift y el acople de la Fase 5.
- **Clorofila / Kd490:** Copernicus/NASA, para la penetración de onda corta (esquema KPAR de HYCOM).
- **SST/SSS satelital** (OSTIA, GHRSST, SMAP): para nudging y validación.

## 3. Estructura de código propuesta

```
forcing/
  surface.py          # (existente) aplica flujos → tendencias
  bulk.py             # algoritmos bulk (NCAR, COARE, Kara)
  canonical.py        # nombres CF, unidades, validador de Dataset canónico
  interp.py           # regrid horizontal/vertical/temporal a malla C (puntos T, U, V)
  tides.py            # reconstrucción armónica + correcciones nodales
  sources/
    __init__.py       # registro: SOURCES = {"era5": ERA5Source, ...}
    base.py           # clase abstracta ForcingSource
    era5.py
    gfs.py
    espc_hycom.py     # ESPC-D-V02 / GOFS 3.1 vía OPeNDAP
    cmems.py
    tpxo.py
    fes.py
    rivers.py
boundaries/obcs/      # (existente) OBData; añadir eta/ubar/vbar barotrópicos y Flather
preproc/              # scripts CLI que generan archivos canónicos en caché
```

Interfaz mínima de cada adaptador:

```python
class ForcingSource(ABC):
    kind: Literal["atmosphere", "ocean", "tide", "river"]

    @abstractmethod
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """Return the raw source subset (lazy, xarray/dask)."""

    @abstractmethod
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables, convert units/signs/time stamps to the canonical CF form."""
```

Toda la lógica de malla y tiempo vive en `interp.py`, compartida por todas las fuentes. Cada adaptador solo traduce.

## 4. Decisiones de diseño

1. **Archivos canónicos en caché (NetCDF4) como contrato.** El modelo nunca lee directamente de OPeNDAP mientras corre; un paso `preproc` genera los archivos. Así es reproducible (como los `.a/.b` de forzamiento de HYCOM) y además funciona sin conexión en Ometeotl.
2. **Procedencia en atributos:** fuente, versión, URL, fecha de descarga y conversiones aplicadas.
3. **Mezclar fuentes por categoría** desde la configuración, por ejemplo atmósfera de GFS, fronteras de ESPC y marea de TPXO. Es exactamente el caso operacional de HYCOM-IOA.
4. **Superponer marea a fronteras sin marea:** η_obc = η_ESPC (filtrado o promedio diario) + η_TPXO(t). ESPC ya contiene parte de la señal de marea en sus salidas horarias, así que conviene filtrarla o usar promedios diarios para no contarla dos veces.
5. **Dependencias nuevas** (opcionales, por categoría): `cfgrib` o `eccodes` (GRIB2 de GFS), `copernicusmarine` (CMEMS), `cdsapi` (descarga de ERA5) y `xesmf` (regrid conservativo, opcional; por defecto bilineal con scipy/xarray).

## 5. Orden de implementación (dentro de la Fase 2)

1. `canonical.py` + `interp.py` + `sources/base.py`: la infraestructura común, con tests.
2. `bulk.py` (NCAR) + `sources/era5.py`: atmósfera, validando contra MITgcm.
3. `sources/espc_hycom.py`: condición inicial y fronteras desde ESPC/GOFS, que alimentan `OBData`.
4. Superficie libre implícita o split-explicit: prerrequisito para la marea.
5. `tides.py` + `sources/tpxo.py` + Flather en OBCS: validar contra mareógrafos (Veracruz, Progreso, Ciudad del Carmen).
6. `sources/gfs.py`, `sources/cmems.py` y `sources/rivers.py`.
7. Notebook de validación: subdominio del Golfo.

## 6. Referencias

- Arbic, B.K., Garner, S.T., Hallberg, R.W. & Simmons, H.L. (2004). *Deep-Sea Res. II* 51, 3069–3101.
- Egbert, G.D. & Erofeeva, S.Y. (2002). *J. Atmos. Oceanic Technol.* 19, 183–204.
- Flather, R.A. (1976). *Mém. Soc. R. Sci. Liège* 6(10), 141–164.
- Foreman, M.G.G. (1977). *Manual for tidal heights analysis and prediction*. Pacific Marine Science Report 77-10.
- Pawlowicz, R., Beardsley, B. & Lentz, S. (2002). *Comput. Geosci.* 28, 929–937.
- Lyard, F.H. et al. (2021). FES2014 global ocean tide atlas. *Ocean Sci.* 17, 615–649.
- Marchesiello, P., McWilliams, J.C. & Shchepetkin, A. (2001). *Ocean Modell.* 3, 1–20 (OBC + nudging).
