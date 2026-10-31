# Teoría de entrainment en corrientes de gravedad — lectura de la biblioteca local (2026-10-30)

Rutas relativas a `/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura/`. Lo inferido y no leído va
[inferencia]; lo citado de memoria, [verificar]. El archivo `Ezer_2004*.md` contiene en realidad Ezer (2005), Ocean
Modelling 9:151–168.

## Xu et al. (2006)

- **Pregunta (Xu §1, ecs. 5–7).** Una ley algebraica E ≡ w_E/ΔU = f(Ri) para modelos isopicnos, con el Ri de capa de Hallberg. Turner 1986 aplicado capa por capa "carece de base experimental".
- **Verdad (Nek5000, §3, Tabla 1).**
  - Canal de 10 × 2 km con pendiente de 1–4°, 400–1000 m de profundidad.
  - **Sin rotación.** No-slip en el fondo, ambiente homogéneo.
  - Ra = 5e6, Pr = 1, entrada con h₀ = 200 m.
  - 4000 elementos de grado 6, ~30 m [inferencia].
- **Entrainment (§4.1, ecs. 10–15).** E = dh/dX y E(t) = (h̄ − h̄₀)/ℓ, con el techo en S′ = 0.2 psu. En Nek5000, E decae con la distancia y crece con θ.
- **HYCOM a Δx = 100 m (§4.2).** Ley E = E₀(1 − Ri/Ri_c) (ec. 16), óptimo E₀ = 0.20 y Ri_c = 0.25.
  - Con Ri_c = 0.25, el Ri de la interfaz queda clavado en Ri_c y E es insensible a E₀: la mezcla se autorregula (§4.2.4–4.2.5).
  - Turner original sobremezcla; Turner × 0.15 da un E plano (§4.3.1).
- **Arrastre (§4.3.3).** Cambiar C_D dos órdenes de magnitud mueve el avance de la pluma solo ~10 %.
- **Validación (§5).** Mediterráneo con HYCOM a 1/12°: P2 da la profundidad y salinidad observadas. KPP la deja demasiado profunda y Turner demasiado somera.
- **Límites (§4.3, §6).** No reproduce la cabeza (rasgo no hidrostático). La dependencia con la resolución horizontal queda abierta.

## Leyes

1. **Ellison–Turner / Turner 1986.**
   - E = (0.08 − 0.1 Ri_B)/(1 + 5 Ri_B) para Ri_B < 0.8, con Ri_B = Δb h/ΔU² (Xu ecs. 3–4; Legg ec. 2).
   - Laboratorio sin rotación.
2. **Xu 2006:** E = 0.20(1 − Ri/0.25), con Ri de gradiente o de capa.
3. **Jackson 2008.**
   - F = F₀(1 − Ri/Ri_c)/(1 + α Ri/Ri_c) (ecs. 10–11).
   - Para una capa de mezcla da Turner con un Ri de gradiente < Ri_B (ec. 9).
   - Necesita ≥ 2–4 puntos dentro de la región de Ri bajo (§6); una resolución vertical insuficiente da un Ri espuriamente alto (§4b).
4. **van Reeuwijk 2019 (DNS):** E = 0.31(Ri_max − Ri) con Ri_max ≈ 0.15 (ecs. 3.9, 4.25).
5. **Price 1979:** E ≈ 5e-4 R_v⁻⁴; la conservación de momento controla la tasa.
6. **Odier 2014 (laboratorio).**
   - Ri bulk ≈ 0.3 frente a Ri_g de 0.04 a 0.25; E ≈ 0.03–0.04.
   - E depende también de Re; compatible con Cenedese & Adduce 2010 (§6.2).
7. **Özgökmen 2009:** la mezcla crece ~67 % de Re = 1e3 a 1e6.
8. **Mezcla BBL por energía de arrastre:** Legg Ap. A (ecs. A.1–A.2), usada en las corridas isopicnas de Legg y en OM4 (Adcroft 2019 §2.2.4). Ver `mom6_bbl_mixing.md`.
9. **Detrainment de Baines:** C_d + α_E − 0.2 B^0.4 sin θ = 0 (Legg ec. 16).
10. **Nof 1983:** u_t = g′s/f (Ezer ec. 2).
11. **Price & Baringer 1994 (tubo de corriente):** solo citado [verificar].

## Qué dicen para DOME

- **Ellison–Turner.** Legg calcula Ri_B ≈ 0.4–0.5 cerca de la cima del talud, lo que predice E = 8e-3–4e-2. El valor medido es mucho menor (E de la esquina 8.6e-4, ec. 14) porque la rotación modifica las velocidades (§3). Las leyes de laboratorio sin rotación sobreestiman ~10×.
- **Números adimensionales de Legg (Tabla 2).**
  - Fr de entrada = 1.
  - Ekman friccional ν_v/(h₀² f): > 0.1 da flujo laminar (Cenedese 2004).
  - Drag-Ekman C_d Δb₀ tanα/(f² h₀) = 0.127.
- **Ezer.** Aguas abajo Fr ≈ 0.4–0.6, subcrítico. Con K_M = 0 la pluma no baja (análogo a nuestro Pr = 0) y el transporte de Ekman de fondo es el 22 % del transporte talud abajo.
- **Legg (§4).** Con ν_v grande el camino baja 40 km más.
- **El α_E de Legg.** El texto imprime 8.6e-4 como E de la esquina (ec. 14). Nuestro 1.03e-3 es un α_E digitalizado de la Fig. 3b. Hay que aclarar la definición.

## Ingredientes que pyOCEAN podría estar omitiendo

1. **Mezcla BBL por arrastre** (Legg Ap. A, OM4). Sin ella, la pluma se arrastra casi sin diluir.
2. **Umbral y cizalla subresuelta.** Con Ri_c = 0.25 el total lo fija la cizalla resuelta (Xu §4.2.5) y Jackson pide ≥ 2–4 puntos en la zona de Ri bajo.
3. **Bulk frente a gradiente.** Ri_g ≪ Ri_B (Odier §6.1).
4. **Momento y Prandtl.** El transporte de Ekman de fondo (Ezer) podría explicar parte de los 15.9 Sv frente a 22.5 [inferencia].
5. **Re.** El Re ≈ 1400 de la verdad de Legg la vuelve una cota inferior del entrainment [inferencia].
6. **Fr subcrítico y rotación.** Ninguna ley local incluye f [inferencia].
7. **Resolución de la interfaz.** Legg ec. 1: Δz < h y Δx < h/tanα.

## Citados que faltan en la biblioteca

Todos [verificar] salvo lo que imprimen Xu y Benjamin:

- **Leyes base:** Turner 1986 (JFM 173:431–471); Ellison & Turner 1959 (JFM 6:423–448).
- **Tubo de corriente:** Price & Baringer 1994 (Prog. Oceanogr. 33:161–200); Baringer & Price 1997a, b.
- **Esquemas para modelos:** Hallberg 2000 (MWR 128:1402–1419); Chang et al. 2005 (Ocean Model. 10:342–368); Özgökmen et al. 2004a y b.
- **Laboratorio:** Cenedese et al. 2004 (JPO 34:188–203); Cenedese & Adduce 2008 y 2010; Baines 2001 y 2002; Britter & Linden 1980.
- **Observaciones:** Girton & Sanford 2003; Peters & Johns 2005.
- **Teoría:** Miles 1961; Nof 1983.
- **Capa de fondo y talud en modelos:** Killworth & Edwards 1999; Winton et al. 1998.
- **Sensibilidad en DOME:** Legg et al. 2008 y 2009.
- **Mezcla turbulenta:** Shih et al. 2005.
