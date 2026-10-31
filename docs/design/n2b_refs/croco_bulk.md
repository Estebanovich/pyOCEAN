# CROCO: flujos bulk aire-mar (ruta por defecto COARE3.0 + gustiness + BULK_LW)

Fuente: `~/pyOCEAN_refs/croco` (git ab63b58). Rutas relativas a `OCEAN/` salvo que se indique. Lectura de referencia, no verdad: cuando se dice "como CROCO" es lo que hace el codigo, no necesariamente la fisica de Fairall et al. (2003).

## 0. Opciones de compilacion (valores por defecto)

- `cppdefs.h:178-185`: `BULK_FLUX`, `BULK_GUSTINESS`, `BULK_LW` definidos; `BULK_ECUMEV0/V6`, `BULK_WASP`, `SST_SKIN`, `ANA_DIURNAL_SW`, `ONLINE`, `READ_PATM`, `ABL1D` indefinidos (`cppdefs.h:179-197`). `SFLUX_CFB` indefinido (`cppdefs.h:213`), luego `CFB_WIND_TRA`/`CFB_STRESS` no existen en el camino por defecto (mb = 1, `bulk_flux.F:52`; el modo con corrientes en `cppdefs_dev.h:686-692`).
- `QCORRECTION`, `SFLX_CORR` solo se definen en la rama `# else` de `BULK_FLUX` (`cppdefs.h:206-212`), es decir NO con bulk. `get_vbc.F:313-331` dice "si BULK_FLUX, la correccion se hace en bulk_flux", pero `bulk_flux.F` no contiene `dqdt` (grep vacio): no hay correccion de flujo (comentario obsoleto).
- `RAIN_FLUX` no se define en `cppdefs.h`/`cppdefs_dev.h` (solo se usa en `bulk_flux.F:637` y `forces.h:287`): sin flujo de calor por lluvia ni `EmP`; el agua dulce entra como flujo virtual de sal (ver 2.9).
- `SALINITY`, `MASKING`, `SOLVE3D` definidos en el bloque REGIONAL (`cppdefs.h:134,138,142`) [verificar cual bloque aplica a la configuracion concreta de pyOCEAN].
- `IterFl = 3` iteraciones (`bulk_flux.F:48`).
- Precision: el codigo usa `REAL` por defecto; `jobcomp:242,249` compila con `-r8` / `-fdefault-real-8 -fdefault-double-8` (reales de 8 bytes, `-O2`, sin `-ffast-math`; ifort con `-fp-model precise`). Ver seccion 4.

## 1. Entradas (unidades, alturas, SST, corriente)

Campos atmosfericos (`forces.h:295-302`, lectura/escala `get_bulk.F`):

| variable | unidad en el codigo | notas |
|---|---|---|
| `tair` | degC, a 2 m | `blk_ZT = 2.0` (`bulk_flux.F:96`) |
| `rhum` | fraccion (RH < 2) o g/kg (RH >= 2) | `spec_hum` `bulk_flux.F:769-774`; humedad a 2 m |
| `uwnd`, `vwnd` | m/s, a 10 m (`blk_ZW = 10.0`, `:95`), en puntos u y v | NO se rotan ni usan corriente relativa |
| `wspd` | m/s, en rho: `sqrt(0.25(uwnd(i)+uwnd(i+1))^2 + 0.25(vwnd(j)+vwnd(j+1))^2)` (`get_bulk.F:768-769`, `989-990`); si el campo es un escalar: `sqrt(u^2+v^2)` (`:888`) | |
| `radlw` | W/m2 en el fichero, positivo hacia abajo, **solo LW descendente** con `BULK_LW`; en memoria cinematico: `radlw = radlwg/(rho0*Cp)` (`get_bulk.F:384`, `:960`) | |
| `radsw` -> `srflx` | W/m2 **neto** (albedo ya aplicado fuera: CROCO no aplica albedo; `srflx = radsw`, `get_bulk.F:742,962`), cinematico `/(rho0*Cp)` | |
| `prate` | cm/dia en el fichero; `stf_scale(isalt) = 0.01/86400` (`get_bulk.F:386`) -> m/s, positivo = lluvia | |
| presion | constante `psurf = 100000.0 Pa` (`params_bulk.h:30-33`, `READ_PATM` indefinido) | |

- Tiempo: interpolacion lineal entre dos registros al instante `time+0.5*dt` (`get_bulk.F:723,941-951`; mismo esquema en `online_set_bulk.F:97-120`).
- ERA5: CROCO no convierte acumulados (J/m2 -> W/m2, signos, dewpoint -> RH, etc.). Los `online_*bulk*.F` solo cambian que se lee (con `ERA_ECMWF` usa SW neto y no el descendente, `online_get_bulk.F:92-96`, `online_set_bulk.F:79-83`); la conversion ocurre en las herramientas de preproceso, no en el nucleo. pyOCEAN debe hacer esa conversion antes de llamar al esquema [verificar contra CROCO_TOOLS si se quiere reproducir el preproceso, no leido].
- SST: `TseaC = t(i,j,N,nrhs,itemp)` = capa superior del modelo, **sin** cool skin ni warm layer (`SST_SKIN` indefinido, `bulk_flux.F:224-229`). SSS para evaporacion/sal: `t(i,j,N,nrhs,isalt)`. Corriente superficial: no se usa (sin `CFB`).
- Constantes del modelo: `g = 9.81`, `Cp = 3985.0`, `vonKar = 0.41` (`scalars.h:633,640,643`); `rho0` se lee de `croco.in` (1025.d0 en `croco.in:61`; `read_inp.F:2386-2388`).

## 2. Algoritmo por defecto

Constantes (`params_bulk.h`): `CtoK = 273.16` (OJO, no 273.15), `blk_Rgas = 287.0596736665907`, `blk_Rvap = 461.5249933083879`, `blk_Cpa = 1004.708857833067`, `ip00 = 1e-5`, `rdocpd = Rgas/Cpa`, `r_gas = 8.314510`, `mm_dryair = 28.9644e-3`, `mm_water = 18.0153e-3`, `cpvir = Rvap/Rgas - 1`, `MvoMa = mm_water/mm_dryair`, `eps = 1e-8`, `r3 = 1/3`, `pis2 = 2 atan(1)`, `sqr3 = sqrt(3)`, `pis2osqr3 = pis2/sqr3`, `blk_beta = 1.2`, `blk_Zabl = 600`, `emiss_lw = 0.985`, `SigmaSB = 5.6697e-8` (`:1-28`). `LMOmin/LMOmax` (`:19-20`), `dWstar0` etc. (`:23-25`) existen pero NO se usan en la ruta COARE (solo en ECUME).

2.1 Humedad especifica del aire (`spec_hum`, `bulk_flux.F:758-775`): `e_s[mb] = (1.0007 + 3.46e-6*0.01*psfc)*6.1121*exp(17.502*TairC/(240.97+TairC))`; si RH < 2: `e = e_s*RH`, `Q = MvoMa*e/(psfc*0.01 - 0.378*e)`; si no, `Q = 0.001*RH`. (Usa `psurf`, no la presion a 2 m.)

2.2 Presion y densidad del aire (`exner_patm_from_tairabs`, `:779-804`, llamada `:231`): `pair = psfc`; 3 iteraciones: `q_sat = qsat(TairK, pair, 1.0)`; `xm = mm_dryair + (Q/q_sat)(mm_water - mm_dryair)`; `pair = psfc*exp(-g*xm*z/(r_gas*TairK))` con **g local = 9.80665** (`:787`, sombrea al 9.81) y z = `blk_ZT` = 2 m; `iexna = (pair*ip00)^(-rdocpd)`. `iexns = (psurf*ip00)^(-rdocpd)` (`:177`). `rhoAir = patm*(1+Q)/(Rgas*TairK*(1+MvoMa*Q))` (`:233`). `TairK = TairC + CtoK`.

2.3 Humedad de saturacion en el mar (`qsat`, `:808-835`): `psat` [Pa] `= exp(alpw - betaw/T - gamw*ln T)` (T > 273.16; `alpw=60.2227554, betaw=6822.40088, gamw=5.13926744`), rama de hielo (`T <= CtoK`) con `alpi=32.62117980819471, betai=6295.421338904806, gami=0.5631331575423155`; `psat *= coeff`; `qsat = MvoMa*psat/(patm + (MvoMa-1)*psat)`. En el mar: `Qsea = qsat(TseaK, psurf, 0.98)` (`:235`): **factor 0.98 de salinidad, con psurf (1e5)**, no con patm a 2 m.

2.4 Gradientes (`:246-263`): viento `delW = sqrt(wspd0^2 + 0.25)` (gustiness inicial 0.5 m/s), con `wspd0 = max(wspd, 0.1*min(10, ZW)) = max(wspd, 1.0)` (`:222`); `delQ = Q - Qsea`; `delT = TairC*iexna - TseaC*iexns + CtoK*(iexna - iexns)` (diferencia de temperatura potencial, referida a psurf).

2.5 Primera estimacion (`:429-472`): `Wstar = 0.035*delW*Log10oLogZw` (`Log10oLogZw = ln(10e4)/ln(ZW*1e4) = 1`, `:110`); `VisAir = air_visc(TairC)` = `1.326e-5(1+6.542e-3 T + 8.301e-6 T^2 - 4.84e-9 T^3)` (`:898-908`); `charn = 0.011`; `Ch10 = 0.00115`; `Ribcu = -ZW/(Zabl*0.004*beta^3)`;
`iZo10 = g*Wstar/(charn*Wstar^3 + 0.11*g*VisAir)`; `iZoT10 = 0.1*exp(vonKar^2/(Ch10*ln(10*iZo10)))`; `CC = ln(ZW*iZo10)^2/ln(ZT*iZoT10)`; `Ri = g*ZW*(delT + cpvir*TairK*delQ)/(TairK*delW^2)`; `ZoLu = CC*Ri/(1+Ri/Ribcu)` si Ri < 0 (inestable) o `CC*Ri/(1+3Ri/CC)` si Ri >= 0; `psi_u = psiu(ZoLu)`; `Wstar = delW*vonKar/(ln(ZW*iZo10) - psi_u)`; `ZoLt = ZoLu*ZT/ZW`; `psi_t = psit(ZoLt)`; `logts10 = ln(ZT*iZoT10)`; `cff = vonKar/(logts10 - psi_t)`; `Tstar = delT*cff`; `Qstar = delQ*cff`. Sin limitar ZoL.

2.6 Charnock (`:497-503`, no WASP): `charn = 0.011`; si `delW > 18`: 0.018; si `delW > 10`: `0.011 + 0.125*(0.018-0.011)*(delW-10)`. Se calcula **una vez** con el delW de gustiness inicial (sqrt(w^2+0.25)) y NO cambia dentro del bucle.

2.7 Bucle (3 iteraciones, `:511-579`), en este orden:
 a. `iZoW = g*Wstar/(charn*Wstar^3 + 0.11*g*VisAir)` (`:516`)
 b. `Rr = Wstar/(iZoW*VisAir)`; `iZoT = max(8695.65, 18181.8*Rr^0.6)` (`:523-524`; z0t = 5.5e-5 Rr^-0.6 acotado a 1.15e-4, COARE3.0)
 c. `ZoLu = vonKar*g*ZW*(Tstar*(1+cpvir*Q) + cpvir*TairK*Qstar)/(TairK*Wstar^2*(1+cpvir*Q) + eps)` (`:527-529`) con Tstar/Qstar/Wstar de la iteracion previa; **sin clamp**.
 d. `psi_u = psiu_coare(ZoLu)`; `Wstar = delW*vonKar/(ln(ZW*iZoW) - psi_u)` (`:531-536`)
 e. `ZoLt = ZoLu*ZT/ZW`; `psi_t = psit_coare(ZoLt)`; `cff = vonKar/(ln(ZT*iZoT) - psi_t)`; `Tstar = delT*cff`; `Qstar = delQ*cff` (`:539-547`)
 f. Gustiness (`:549-556`): `Bf = -g/TairK*Wstar*(Tstar + cpvir*TairK*Qstar)`; `cff = blk_beta*(Bf*Zabl)^(1/3)` si Bf > 0, si no `cff = 0.2`; `delW = sqrt(wspd0^2 + cff^2)`. (No usa el factor (1+cpvir Q) del ECUME; `r3` real.)
 g. `ddelW = delW` (`:577`).
 Nota: tras la 3a iteracion Wstar NO se recalcula con el delW nuevo; luego `Cd = (Wstar/ddelW)^2` mezcla Wstar de la iteracion 3 (calculado con delW de la iteracion 2) con ddelW de la 3 (`:658`). Hay que reproducirlo literalmente.

2.8 Funciones de estabilidad (`:912-972`):
- `psiu(z)`: z <= 0: `chik=(1-15z)^0.25`; `psik = 2 ln(0.5(1+chik)) + ln(0.5(1+chik^2)) - 2 atan(chik) + pis2`; `chic=(1-10.15z)^(1/3)`; `psic = 1.5 ln(r3(chic^2+chic+1)) - sqr3 atan((2chic+1)/sqr3) + 2 pis2osqr3`; `psi = psic + (psik-psic)/(1+z^2)`. z > 0: `chic = -min(50, 0.35z)`; `psi = -((1+z) + 0.6667(z-14.28) exp(chic) + 8.525)`.
- `psit(z)`: z < 0: `chik=(1-15z)^0.25`; `psik = 2 ln(0.5(1+chik^2))`; `chic=(1-34.15z)^(1/3)`; `psic = 1.5 ln((chic^2+chic+1) r3) - sqr3 atan((2chic+1)/sqr3) + 2 pis2osqr3`; `psi = psic + (psik-psic)/(1+z^2)`. z >= 0 (incluye z = 0): `-((1+2z/3)^1.5 + 0.6667(z-14.28) exp(-min(50,0.35z)) + 8.525)`. Cuidado: psit(0) = -0.004524 (!= 0) por las constantes redondeadas 0.6667/8.525; psiu usa `<=` y devuelve 0 en 0.

2.9 Flujos (`:605-654`):
- `hfsen = -Cpa*rhoAir*Wstar*Tstar` [W/m2, positivo hacia arriba]; `Hlv = (2.5008 - 0.0023719*TseaC)*1e6`; `hflat = -Hlv*rhoAir*Wstar*Qstar`.
- LW: `hflw = radlw_cinematico - emiss_lw*rho0i*cpi*SigmaSB*TseaK^4` (positivo hacia abajo; emisividad solo en la emision, **no** hay termino (1-eps)*LW_down).
- Webb: `upvel = -1.61*Wstar*Qstar - (1+1.61*Q)*Wstar*Tstar/TairK`; `hflat = hflat + rhoAir*Hlv*upvel*Q`.
- Cinematicos: `hflat = -hflat*rho0i*cpi`; `hfsen = -hfsen*rho0i*cpi` (K m/s, negativo = enfria). `rho0i = 1/rho0`, `cpi = 1/Cp` (Cp = 3985).
- `stflx(itemp) = srflx + hflw + hflat + hfsen` (K m/s, positivo calienta el oceano; incluye todo el SW en la superficie).
- Agua dulce: `evap = -cp*hflat/Hlv` (m/s; `hflat` ya cinematico, incluye Webb) y `stflx(isalt) = (evap - prate)*S_top` (psu m/s, positivo concentra) (`:636-644`). Es **flujo virtual de sal**, no fuente de volumen ni de calor por lluvia (sin `RAIN_FLUX`).
- Mascaras: `stflx *= rmask` (`:648-651`), `rmask_wet` con WET_DRY.
- No hay cool skin, warm layer, calor de lluvia ni albedo.

2.10 Tension del viento (`:657-659`, `:696-744`): `aer = rhoAir*delW(1)` (con la delW final de gustiness), `cer = Cd = (Wstar/ddelW)^2`, ambos en puntos rho. A puntos u: `cff = 0.5(cer(i-1,j)+cer(i,j))`, `cff2 = 0.5(aer(i-1,j)+aer(i,j))`, `sustr(i,j) = cff*cff2*uwnd(i,j)*rho0i` (i = imin+1..imax); analogo para v: `svstr(i,j) = 0.5(cer(i,j-1)+cer(i,j)) * 0.5(aer(i,j-1)+aer(i,j)) * vwnd(i,j) * rho0i`. Unidades m2/s2; `* umask`/`vmask`. Es decir tau = rhoAir * Cd * delW * u (con delW incluyendo gustiness, y u la componente sin gustiness) y no usa |U| = wspd. Sin `CFB_STRESS` (sin termino de corriente) ni `WAVE_SMFLUX`.

## 3. Aplicacion en el modelo

- `step.F:366` llama `bulk_flux(tile)` en cada paso 3D; luego `sustr/svstr/stflx/srflx` pasan a las ecuaciones (`step3d_t.F:566`, `FC(N) = dt*stflx`).
- Calor: `stflx(itemp)` es el flujo superficial total. Con `LMD_SKPP`/`LMD_BKPP`/`GLS_MIXING` la penetracion se representa con `FC(k) = dt*srflx*swdk(k)` en las interfaces interiores (`step3d_t.F:582-586`); sin esos esquemas todo el SW se absorbe en la capa superior. `swdk = lmd_swfrac(z_w(k) - z_w(N))` (`step3d_t.F:445-458`), Paulson & Simpson (1977) Jerlov tipo I: `r1 exp(z/mu1) + (1-r1) exp(z/mu2)` con `mu1 = 0.35`, `mu2 = 23`, `r1 = 0.58` (`lmd_swfrac.F:~60-63`, tipo fijo `lmd_Jwt = 1`); el SW que llega al fondo no se redistribuye (pyOCEAN N3 ya difiere: reparte en h).
- `ANA_DIURNAL_SW` indefinido: no hay modulacion diurna del SW.
- Convenio de signos: todos los flujos superficiales son "hacia el oceano" positivos (cinematicos: K m/s, psu m/s, m2/s2).

## 4. Dependencias de precision y de compilacion

- `real(kind=4)` / `OPENACCNUMERIC`: solo `bulk_flux.F` (lineas 444-446, 457-459, 467-469, 533-535, 557-559, 797-799, 831-833, 861-863, 934-936, 960-962, 967-969) y `set_nudgcof.F:113,119`; la macro **no se define en ningun sitio del arbol** (grep), luego la ruta por defecto es **toda en doble precision** (con `-r8`). Si pyOCEAN quiere reproducir la variante GPU: truncar a float32 esos puntos (`CC`, `logus10`, `logts10`, `delW`, `pair`, `qsat`, `psiu`, `psit`).
- `REAL` = 8 bytes depende de `-fdefault-real-8` (gfortran) / `-r8` (ifort) en `jobcomp:242,249`; con otros flags las constantes de `params_bulk.h` serian float32. La precision de los literales sin sufijo (0.035, 0.6667, 8.525, 8695.65, 18181.8, 2.5008...) sigue al flag; `eps = 1.d-8` es siempre doble. [verificar] los flags con los que se compilen los casos comparados.
- Dependencias de opciones: `BULK_GUSTINESS` (cambia delW y Cd), `BULK_LW` (radlw descendente), `READ_PATM` (cambia psurf), `SST_SKIN`, `SFLUX_CFB`, `RAIN_FLUX`, `BULK_WASP`/`ECUME`, `MASKING`, `SALINITY`, `ABL1D`, `ANA_DIURNAL_SW`.
- Constantes con valores distintos en el mismo archivo: g = 9.81 (modelo) vs 9.80665 (barometrica, `:787`); CtoK = 273.16 vs 273.15; el rango extendido `imin..imax` (`:126-155`) solo afecta a los bordes, no a los valores.
- Un `real^(1/3)` con `**r3` y `**0.25`, `**0.6`, `**1.5` usan `pow` del compilador: dependen de la libm (diferencias ~1 ulp posibles).

## 5. Pseudocodigo en orden de ejecucion (por punto rho)

```
# constantes: params_bulk.h; g=9.81, Cp=3985, vonKar=0.41 (scalars.h); rho0 de croco.in
rho0i = 1/rho0 ; cpi = 1/Cp                                     # bulk_flux.F:171-173
iexns = (psurf*1e-5)**(-rdocpd), psurf = 1e5                    # :177
wspd0 = max(wspd, 1.0)                                          # :199,222
TairC = tair ; TairK = TairC + 273.16                           # :204-205
Q     = spec_hum(rhum, psurf, TairC)                            # :206, 758-775
TseaC = T_top ; TseaK = TseaC + 273.16                          # :227-229
(iexna, patm) = exner_patm(Q, TairK, ZT=2, psurf)               # :231, 779-804
rhoAir = patm*(1+Q)/(Rgas*TairK*(1+MvoMa*Q))                    # :233
Qsea   = qsat(TseaK, psurf, 0.98)                               # :235
delW   = sqrt(wspd0**2 + 0.25)                                  # :247
delQ   = Q - Qsea                                               # :260
delT   = TairC*iexna - TseaC*iexns + 273.16*(iexna - iexns)     # :262-263
# primera estimacion                                            # :429-472
Wstar = 0.035*delW ; VisAir = air_visc(TairC) ; charn=0.011 ; Ch10=0.00115
Ribcu = -ZW/(Zabl*0.004*beta**3)
iZo10 = g*Wstar/(charn*Wstar**3 + 0.11*g*VisAir)
iZoT10 = 0.1*exp(vonKar**2/(Ch10*log(10*iZo10)))
CC = log(ZW*iZo10)**2/log(ZT*iZoT10)
Ri = g*ZW*(delT + cpvir*TairK*delQ)/(TairK*delW**2)
ZoLu = CC*Ri/(1+Ri/Ribcu) if Ri<0 else CC*Ri/(1+3*Ri/CC)
Wstar = delW*vonKar/(log(ZW*iZo10) - psiu(ZoLu))
ZoLt = ZoLu*ZT/ZW ; psi_t = psit(ZoLt)
cff = vonKar/(log(ZT*iZoT10) - psi_t) ; Tstar = delT*cff ; Qstar = delQ*cff
charn = 0.011 ; if delW>18: 0.018 ; elif delW>10: 0.011+0.125*0.007*(delW-10)   # :498-503
for it in 1..3:                                                  # :511
    iZoW = g*Wstar/(charn*Wstar**3 + 0.11*g*VisAir)
    Rr = Wstar/(iZoW*VisAir) ; iZoT = max(8695.65, 18181.8*Rr**0.6)
    ZoLu = vonKar*g*ZW*(Tstar*(1+cpvir*Q)+cpvir*TairK*Qstar)/(TairK*Wstar**2*(1+cpvir*Q)+1e-8)
    Wstar = delW*vonKar/(log(ZW*iZoW) - psiu(ZoLu))
    ZoLt = ZoLu*ZT/ZW
    cff = vonKar/(log(ZT*iZoT) - psit(ZoLt)) ; Tstar = delT*cff ; Qstar = delQ*cff
    Bf = -g/TairK*Wstar*(Tstar + cpvir*TairK*Qstar)
    cff = beta*(Bf*Zabl)**(1/3) if Bf>0 else 0.2
    delW = sqrt(wspd0**2 + cff**2)
ddelW = delW                                                     # :577
hfsen = -Cpa*rhoAir*Wstar*Tstar                                  # :606
Hlv = (2.5008 - 0.0023719*TseaC)*1e6                             # :608
hflat = -Hlv*rhoAir*Wstar*Qstar                                  # :610
hflw  = radlw_kin - 0.985*rho0i*cpi*5.6697e-8*TseaK**4           # :617-618
upvel = -1.61*Wstar*Qstar - (1+1.61*Q)*Wstar*Tstar/TairK         # :621
hflat = hflat + rhoAir*Hlv*upvel*Q                               # :624
hflat = -hflat*rho0i*cpi ; hfsen = -hfsen*rho0i*cpi              # :630-631
stflx_T = srflx + hflw + hflat + hfsen                           # :633  [K m/s]
evap = -Cp*hflat/Hlv                                             # :636  [m/s]
stflx_S = (evap - prate)*S_top                                   # :644  [psu m/s]
aer = rhoAir*delW ; cer = (Wstar/ddelW)**2                       # :657-659
# luego, en u/v:                                                 # :706-717, 728-739
sustr(i,j) = 0.5*(cer(i-1)+cer(i))*0.5*(aer(i-1)+aer(i))*uwnd(i,j)*rho0i
svstr(i,j) = 0.5*(cer(j-1)+cer(j))*0.5*(aer(j-1)+aer(j))*vwnd(i,j)*rho0i
```
psiu/psit: seccion 2.8. Funciones auxiliares: `air_visc` `:898-908`, `qsat` `:808-835`.

## 6. Driver minimo en Fortran

- Las funciones `spec_hum`, `exner_patm_from_tairabs`, `qsat`, `air_visc`, `bulk_psiu_coare`, `bulk_psit_coare` (`bulk_flux.F:758-972`) solo incluyen `params_bulk.h`: se pueden compilar tal cual (necesitan `-DBULK_FLUX -DBULK_GUSTINESS -DBULK_LW`, `-fdefault-real-8 -fdefault-double-8`, `-I OCEAN/`, y cpp; el `#include "cppdefs.h"` de la linea 11 arrastra mucho: mas simple copiar el bloque a un fichero propio sin ese include).
- La rutina `bulk_flux_tile` depende de `param.h`, `grid.h`, `ocean3d.h`, `forces.h`, `scalars.h` (common blocks, `omp_get_thread_num`, `WESTERN_EDGE` y macros de teselas, arrays `A2d` privados): no llamarla. Extraer las lineas 199-659 a una subrutina `bulk_point(tair, rhum, wspd, radlw_kin, srflx, TseaC, S, prate, rho0, ...)` que devuelva `Wstar, Tstar, Qstar, delW, Cd, hfsen, hflat, hflw, evap`, mas un `program` que lea un CSV de entradas y escriba las salidas en `ES25.17`. `g`, `Cp`, `vonKar` se declaran como `parameter` locales (9.81, 3985.0, 0.41), `rho0` como argumento. Sin dependencias MPI/OpenMP en el bloque escalar (solo en el reparto por teselas y en `!$acc`, que son comentarios).
- Para la tension, basta con `Cd` y `aer` por punto; el promedio a u/v es aritmetico y se puede probar en un par de puntos.

## Lista [verificar]

1. Valor de `rho0` en los casos de comparacion (1025.d0 en `croco.in:61`; cada caso lo lee de su `croco.in`).
2. Flags de compilacion reales de los casos de `croco_runs` (si se usa `-r8`; no pude localizarlos).
3. Que bloque `#ifdef` de `cppdefs.h` aplica a la configuracion que pyOCEAN reproduzca (`SALINITY`, `MASKING`, `GLS_MIXING`/`LMD_SKPP` para la penetracion de SW), y si el caso a comparar redefine `RAIN_FLUX`, `SST_SKIN` o `BULK_WASP`.
4. Conversion de ERA5 (acumulados, humedad) en CROCO_TOOLS: no esta en este arbol.
5. Que `QCORRECTION`/`SFLX_CORR` quedan inactivos con bulk (inferido de `cppdefs.h:206-212`, no ejecutado).
6. Diferencias de 1 ulp por `pow`/`exp`/`log` entre libm de gfortran y NumPy: no medidas.
7. Tipo de agua de Jerlov: `lmd_Jwt = 1` fijo en `lmd_swfrac.F`; numeros `mu1/mu2/r1` leidos pero no re-verificados por linea exacta (aprox. `lmd_swfrac.F:34-40`).
