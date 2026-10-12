# Cómo configuran los usuarios de MOM6 el momento y las capas delgadas (2026-10-12)

Encuesta de 16 configuraciones reales. Las fuentes son `~/pyOCEAN_refs/MOM6-examples` (c1730b8; valores efectivos de `MOM_parameter_doc.all`) y GitHub (`gh api`). Los valores marcados con * son los de omisión del código actual de MOM6 (0e9a561), no los verificados en la versión que usó cada corrida.

En todas las configuraciones, PV_ADV_SCHEME = PV_ADV_CENTERED y CFL_TRUNCATE = 0.5.

**Abreviaturas:** S75E = SADOURNY75_ENERGY, S75Z = SADOURNY75_ENSTRO, BC = BOUND_CORIOLIS, ED = CORIOLIS_EN_DIS.

| Configuración | Coordenada | Coriolis / KE / BC / ED | KV | HMIX_FIXED / HBBL | MAXVEL | Δt |
|---|---|---|---|---|---|---|
| OM4_025, OM4_05 | HYCOM1 | S75Z / ARAKAWA / T / F | 1e-4 | 0.5 / 10 | 6 | 900, 1800 s |
| OM5_025 | HYCOM1 | S75Z / ARAKAWA / T / F | 0 | 0.5 / 10 | 6 | 900 s |
| lock_exchange | capas puras | S75E / **GUDONOV** / T / F | 1e-4 | 1e-10 / 1e-3 | 10 | 250 s |
| DOME | capas puras | S75E / ARAKAWA / T / F | 1e-4 | 20 / 10 | 10 | 1200 s |
| flow_downslope, seamount, ISOMIP | capas, ρ, σ, z\* | S75E / ARAKAWA / (T) / **T** | 1e-4 | 20 / 10 | 10 | 300–900 s |
| rotating_gravity_current | capas, z\* | S75E / ARAKAWA / (T) / T | 1e-5 | 20 / 10 | 50 | 20–30 s |
| CESM3 (`ESCOMP/MOM_interface`) | HYCOM1 | S75E\* (1/4°: S75Z) / ARAKAWA\* / T / F\* | 1e-6 | 0.5 / 10 | 6 | 1800 s |
| NOAA-EMC RTOFS_GLO | HYCOM1, 41 capas | S75E / ARAKAWA / (T) / T | 3e-5 | 0.5 / 10 | 6 | 300 s |
| CEFI NWA12 / NEP10 | z\*, 75 niveles | S75E / ARAKAWA / F / T | 0 | 0.9 / 10 | 6 | 600 / 400 s |
| ESMG NGOA, CCS1, Arctic6 | z\* | S75E (Arctic6: ARAKAWA_HSU90) / ARAKAWA\* / (T) / T | 0–1e-6 | 0.5–0.9 / 10 | 6 | 300–1200 s |
| COSIMA `regional-mom6` (= ACCESS-OM3 regional) | z\* | S75Z / ARAKAWA\* / T / F\* | 1e-5 | 0.5 / 10 | 6 | 50–600 s |
| ACCESS-OM3 25 km | z\* | S75Z / ARAKAWA / T / F | 0 | 0.5 / 10 | 6 | 900 s |
| NCAR regional Caribe | z\* | S75Z / – / T / – | 1e-4 | 0.5 / 10 | 6 | 450 s |

(T) = BOUND_CORIOLIS aparece en el registro, pero el código lo apaga si ED = T con S75E (`MOM_CoriolisAdv.F90` ~2165).

**Práctica común:**
- **Forma vectorial invariante en todas.**
  - KE_ARAKAWA en todas, salvo `lock_exchange`, que usa GUDONOV.
  - Coriolis en dos grupos: S75Z con BC en la producción GFDL/ACCESS/regional; S75E con ED en los idealizados con capas desvanecidas, en RTOFS y en CEFI/ESMG.
- **Otras opciones comunes:** `MASS_WEIGHT_IN_PRESSURE_GRADIENT = T` en OM4, NWA12, OM3 y RTOFS. El truncamiento es una red de seguridad registrada (U_TRUNC_FILE), no algo que deba ocurrir.

**Issues relevantes:**
- **NOAA-GFDL/MOM6#971:** NaN en un seamount con capas desvanecidas. La causa fue un error de índices en `vertvisc_coef`. Las velocidades en capas desvanecidas las controla el acoplamiento viscoso.
- **#810** (PR): `MASS_WEIGHT_IN_PGF_VANISHED_ONLY`, recomendado cuando el grosor mínimo es menor que 0.1 m.
- **#639** (PR): corrige `visc_rem` en el modo dividido.
- **#946** (PR, 2025): WENOVI{3,5,7}_PV_ENSTRO y KE_UP3. Todavía no lo usa ninguna configuración de producción.
- **ACCESS-NRI/access-om3-configs#697 y #465:** cambiaron a S75Z solo para coincidir con OM4. Los truncamientos se resolvieron bajando DT_THERM.
