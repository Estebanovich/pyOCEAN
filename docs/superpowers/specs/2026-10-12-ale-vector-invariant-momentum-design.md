# Momento en forma vectorial invariante para el modo ALE: diseño

**Fecha:** 2026-10-12. **Estado:** aprobado por secciones en la conversación (enfoque A, secciones 1–3); pendiente de revisión escrita.

**Spec padre:** `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` (§2 paso 1, §7 punto 6).

## 1. Motivación

En el lock exchange híbrido, la corriente de gravedad de pyOCEAN no forma la cabeza de Benjamin. MOM6, con la misma configuración, sí la forma (`validation/mom6_lock_exchange/`):

| Caso | Frente 1 m | Cuerpo 5 m | max\|u\| |
|---|---|---|---|
| MOM6, capas puras y RHO | 1.05 / 0.98–1.05 U | 0.98–1.05 U | 0.56 m/s |
| pyOCEAN, `keep_isopycnal`, momento en forma de flujo | 1.96 / 1.61 U | 0.84 / 0.14 U | 2.06 m/s |
| pyOCEAN, igual pero con advección en forma advectiva (`upwind3`) | 1.05 / 1.12 U | 1.12 / 1.12 U | 3.15 m/s (en una capa intermedia de 8.8 cm) |

- **Causa principal** (diagnóstico de 2026-10-12): la advección de momento en forma de flujo divide por el volumen de la celda de momento. En una capa que se adelgaza, empuja la punta con +3 a +14 × 10⁻⁴ m/s² y la capa densa se vuelve una película supercrítica (1.1–1.8 m/s).
- **Literatura:** es el problema de la advección y del Coriolis con grosor de capa muy variable que describen Bleck & Smith (1990, apéndice B) y Higdon (2005).
- **Modelos de referencia:** MOM6 y HYCOM usan la forma vectorial invariante. Las 16 configuraciones de MOM6 revisadas en GitHub la usan: OM4/OM5, CESM3, RTOFS, CEFI NWA12/NEP10, ESMG, ACCESS-OM3, `regional-mom6` y los casos idealizados.
- **Lo que falta además:** una capa intermedia delgada sigue acelerándose (3.15 m/s). En MOM6 eso lo controlan la viscosidad vertical implícita y el truncamiento CFL, no el esquema de advección.

## 2. Alcance

- **Solo el modo ALE** (`vertical_coordinate="ale"`). La advección horizontal de momento y Coriolis pasan a la forma vectorial invariante.
- **El modo σ euleriano no cambia:** queda bit a bit igual, con la forma de flujo WENO5 y su Coriolis de energía, que reproducen el lock exchange de CROCO.
- **En ALE no hay advección vertical de momento:** en el paso lagrangiano ω = 0, como ahora.
- **Fuera de alcance:**
  - WENO vectorial invariante (Silvestri et al. 2024; MOM6 PR #946), como mejora posterior medible contra este diseño;
  - `CORIOLIS_EN_DIS`, el grosor `hvel` de la capa límite de fondo y `visc_rem`. Son el plan B del §6.

## 3. Discretización

Fuente: MOM6 `src/core/MOM_CoriolisAdv.F90` (commit 0e9a561). Se usa la configuración del `lock_exchange` de MOM6: SADOURNY75_ENERGY con `BOUND_CORIOLIS`, energía cinética KE_ARAKAWA o KE_GUDONOV, y PV_ADV_CENTERED.

**Ecuación** (paso lagrangiano, por capa k):

    ∂u/∂t = q·(vh)‾ − ∂K/∂x + (gradiente de presión, viscosidad, viento)
    ∂v/∂t = −q·(uh)‾ − ∂K/∂y + …

con q = (f + ζ)/h_q y K = ½|u|².

1. **Vorticidad relativa en las esquinas**, en forma de circulación (`:595-610`):
   ζ_q = (δx(v·Δy) − δy(u·Δx))/A_q.
   Deslizamiento libre: ζ = 0 en esquinas con tierra (máscara de esquina).
2. **Grosor en la esquina, promedio por área de las 4 celdas mojadas** (`:337, 364-367, 429-437, 625-629`):
   h_q = Σ(A h)/Σ A.
   q = (f_q + ζ_q)·Σ A/(Σ A h + V_neglect), con V_neglect = 1e-20 m × (1e-4 m)².
3. **Término q·transporte, Sadourny (1975) de energía** (`:818-823, 1071-1076`):

       CAu[j,I] =  ¼/Δx_u · ( q[j+1,I]·(Qy[j+1,I−1] + Qy[j+1,I]) + q[j,I]·(Qy[j,I−1] + Qy[j,I]) )
       CAv[J,i] = −¼/Δy_v · ( q[J,i]·(Qx[J−1,i] + Qx[J,i]) + q[J,i+1]·(Qx[J−1,i+1] + Qx[J,i+1]) )

   Qx y Qy son los transportes de capa [m³/s] que la continuidad aguas arriba usó en el paso anterior (`st.Qx`, `st.Qy`), como MOM6 usa los `uh` de su última continuidad (`MOM_dynamics_split_RK2.F90:566, 903`). En el primer paso, sin transportes previos, se usa L·u·h̄ con el grosor medio de cara.
4. **`BOUND_CORIOLIS`** (`:1008-1023, 1260-1275`): CAu se recorta al intervalo [min, max] de {(f + ζ)_N·v_NE, (f + ζ)_N·v_NW, (f + ζ)_S·v_SE, (f + ζ)_S·v_SW}; CAv igual, con −(f + ζ)·u. Se aplica antes de restar ∇K. Es la protección de capas delgadas: q·vh ≈ (f + ζ)·v·h_cara/h_q puede crecer si h_q ≪ h_cara.
5. **Energía cinética** en el centro de la celda (`gradKE`, `:1377-1535`):
   - Arakawa (por defecto): K = ¼(A_u u_E² + A_u u_W² + A_v v_N² + A_v v_S²)/A_T;
   - Godunov (`ke_scheme="gudonov"`, el del `lock_exchange`): K = ½(max(u_m² A_uE, u_p² A_uW) + max(v_m² A_vN, v_p² A_vS))/A_T, con u_p = max(u_W, 0), u_m = min(u_E, 0) y lo mismo para v; solo cuentan las velocidades que entran;
   - el término es −δx K/Δx_u y −δy K/Δy_v.
6. **Caras secas o de tierra:** tendencia cero, igual que la máscara actual (`wet_u`, `wet_v`).

## 4. Capas delgadas

- **Viscosidad vertical:** el código no cambia. `vertical_implicit` (`core/dynamics/sigma_wd.py:88`) ya resuelve la forma ponderada por grosor de MOM6 (`MOM_vert_friction.F90:711-783`): dz_k·u_k más el acoplamiento Δt·ν/(½(dz_k + dz_k−1)), con dz igual al grosor medio de cara y un piso de 1e-6 m. Con ν = 1e-4 y Δt = 20 s, una capa de menos de ~1 cm toma la velocidad de sus vecinas.
- **Truncamiento CFL** (`vertvisc_limit_vel`, `MOM_vert_friction.F90:2555-2715`):
  - al final del paso ALE, después del remapeo de las velocidades y de la regla de capas de fondo delgadas, toda velocidad de capa con |u|·Δt/Δx_donante > 0.5 se fija en ±0.9 × 0.5 × Δx_donante/Δt;
  - se aplica al final para no romper la consistencia entre η y Σh de ese paso; solo afecta al paso siguiente;
  - se cuentan los recortes en caras de grosor medio > 1 cm (`h_thin_bed`), como MOM6 cuenta solo h_u > 3 Å (`:2584, 2619`);
  - el estado guarda el total (`SigmaState.n_trunc`);
  - si se supera `max_truncations` (1000 por defecto; el `MAXTRUNC` de MOM6), el modelo se detiene con un `RuntimeError` que dice celda, capa, velocidad y grosor.
- **Se conservan:**
  - la ponderación por masa del gradiente de presión solo junto a capas desvanecidas (MOM6 `MASS_WEIGHT_IN_PGF_VANISHED_ONLY`, PR #810);
  - el transporte aguas arriba con recorte al fondo (B2a);
  - el relleno de densidad de las capas sin masa (B2a);
  - `keep_isopycnal` (B2c).
- **Se retira el limitador de la forma de flujo en celdas desvanecidas** solo en ALE con `momentum="vector_invariant"`, porque esa forma ya no se usa ahí. Con `momentum="flux"` en ALE se queda como está.

## 5. Interfaz

- `SigmaBaroclinic(..., momentum: str | None = None, ke_scheme: str = "arakawa", max_truncations: int = 1000)`:
  - `momentum=None` toma `"vector_invariant"` en ALE y `"flux"` en σ euleriano;
  - `momentum="vector_invariant"` en σ euleriano da un `ValueError` claro;
  - `ke_scheme` acepta `"arakawa"` o `"gudonov"`.
- Módulo nuevo `core/dynamics/vector_invariant.py`, con funciones puras sobre arreglos (u (N, ny, nx+1), v (N, ny+1, nx), h (N, ny, nx), esquinas (ny+1, nx+1)):
  - `vorticity_q(u, v, dx_u, dy_v, area_q, mask_q)`;
  - `pv_q(zeta, f_q, h, area_t, mask_t)`;
  - `coriolis_pv(q, Qx, Qy, dx_u, dy_v, zeta_abs, u, v, bound=True)`;
  - `kinetic_energy(u, v, area_u, area_v, area_t, scheme)` y su gradiente.

  Cada función cita su `file:line` de MOM6.
- `SigmaState` gana `n_trunc: int = 0`. El restart ya rechaza los modelos ALE, así que no cambia.

## 6. Pruebas y criterios de aceptación

**Unitarias** (`tests/test_vector_invariant.py`):
1. **Plano f con flujo uniforme** (h uniforme, ζ = 0): CAu = 0, CAv = −f·u exacto, ∇K = 0.
2. **Equivalencia con el Coriolis actual:** con h uniforme, ζ = 0 y transportes L·u·h, q·(uh, vh) es idéntico al `_coriolis` de energía actual.
3. **Sadourny de energía no hace trabajo:** con campos aleatorios y sin recorte, Σ(Qx·CAu·Δx_u + Qy·CAv·Δy_v) = 0 al redondeo.
4. **Vorticidad:** una rotación sólida da ζ = 2Ω exacto en el interior, y ζ = 0 en esquinas con tierra.
5. **Energía cinética:** con u uniforme, K = u²/2; Godunov usa solo las velocidades que entran.
6. **`BOUND_CORIOLIS`:** en una capa desvanecida junto a una gruesa, |CAu| ≤ max|(f + ζ)·v| de sus vecinas; sin NaN cuando h_q = 0.
7. **Truncamiento:** con Courant 0.7 la velocidad queda en 0.45·Δx/Δt; solo cuentan las caras de más de 1 cm; al pasar `max_truncations` se detiene con la ubicación.
8. **Acoplamiento viscoso:** una capa de 1 mm entre dos gruesas toma la velocidad de sus vecinas en un paso (ν = 1e-4), aunque su propia tendencia sea grande.

**Modelo, sin regresión:**
- **σ euleriano:** idéntico bit a bit (toda la suite σ).
- **ALE-σ:** lock exchange a Benjamin ±10 %; onda interna dentro de 10 % del σ euleriano.
- **Escalón z\* en reposo:** ≤ 1e-10 m/s.
- **Monte submarino z\*** con volumen finito, perfil de referencia y Coriolis: ≤ 0.15 cm/s el día 10 (hoy 0.11).

**Modelo, objetivo:** lock exchange híbrido con la viscosidad de MOM6 (KV = 1e-4, KH = 0.01; el helper del test la usa):
- frentes a 1 m y a 5 m dentro de ±10 % de Benjamin (salen los dos xfail de `tests/test_sigma_ale.py`);
- max|u| ≤ 1.5 × el de MOM6 (≤ 0.84 m/s);
- ganancia de RPE < 0.2 × la de ALE-σ;
- cero truncamientos;
- Σh = H + η y calor conservados.

**Validación:**
- fila nueva en `validation/mom6_lock_exchange/compare.py`;
- marcador del roadmap (§7) actualizado;
- costo por paso medido contra la forma de flujo WENO5 en la malla de 160×100×30 y reportado, sin optimizar todavía.

**Plan B si el objetivo no se cumple** (en este orden; hay que parar y reportar antes de pasar al siguiente):
1. `CORIOLIS_EN_DIS`, la variante de los casos idealizados de MOM6 con capas desvanecidas (`:439-449, 733-815`);
2. grosor aguas arriba/armónico en la capa límite de fondo para la viscosidad (`hvel`, `MOM_vert_friction.F90:1456-1536`).

## Referencias

- Sadourny, R. (1975). J. Atmos. Sci. 32, 680–689.
- Arakawa, A. & Lamb, V. R. (1981). Mon. Wea. Rev. 109, 18–36.
- Bleck, R. & Smith, L. T. (1990). J. Geophys. Res. 95, 3273–3285 (apéndice B).
- Higdon, R. L. (2005). J. Comput. Phys., doi:10.1016/j.jcp.2004.12.011 (volumen y páginas [verificar]).
- Adcroft, A. et al. (2019). JAMES 11, 3167–3211 (OM4: forma vectorial invariante, Sadourny).
- Silvestri, S. et al. (2024). JAMES, doi:10.1029/2023MS004130 (WENO vectorial invariante; fuera de alcance).
- Ilıcak, M. et al. (2012). Ocean Modell. 45–46, 37–58 (lock exchange).
- MOM6 commit 0e9a561: `MOM_CoriolisAdv.F90`, `MOM_vert_friction.F90`.
- Encuesta de 16 configuraciones de MOM6 (2026-10-12): `validation/mom6_lock_exchange/MOM6_configs.md`.
