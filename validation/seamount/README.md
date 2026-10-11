# Monte submarino: error del gradiente de presión en σ

Es la prueba estándar de Beckmann & Haidvogel (1993, §2c):
- **Geometría:** monte gaussiano h = H₀ − H·exp(−(x² + y²)/L²), con H₀ = 4500 m, H = 4050 m (δ = 0.9) y L = 25 km (pendiente máxima 13.7 %), en plano f con f₀ = 1e-4. Caja cerrada de 384 × 288 km, Δx = 6 km, 20 capas σ.
- **Estado inicial:** océano en reposo con isopicnas horizontales, ρ(z) = ρ₀ + 3·(1 − e^{z/1000}) kg/m³.
- **El gradiente de presión exacto es cero:** toda velocidad que aparezca es error numérico.
- **Pendientes:** rx0 = 0.27 y rx1 = 10.7, más que los límites usuales (0.2 y 3–7).

```bash
python validation/seamount/run_seamount.py --pgf twoterm
python validation/seamount/run_seamount.py --pgf jacobian
python validation/seamount/run_seamount.py --pgf jacobian --ref
```

## Resultados (10 días)

| Gradiente de presión (con Coriolis, f₀ = 1e-4) | max\|PGF\| en t = 0 | max\|u\| día 1 / 2 / 10 |
|---|---|---|
| Dos términos, ρ′ = ρ − ρ₀ | 4.0e-5 m/s² (≈ 40 cm/s geostrófico) | 49.9 / 60.2 / 42.2 cm/s |
| Jacobiano de densidad (Song 1998), ρ′ = ρ − ρ₀ | 3.7e-5 | 49.9 / 60.2 / 42.2 cm/s |
| **Jacobiano + perfil de referencia del estado inicial** (`rho_ref="auto"`) | **1.1e-6** | **0.17 / 0.21 / 0.41 cm/s** |
| Beckmann & Haidvogel 1993 (su modelo espectral) | — | ~1 cm/s a 10 días |

**El jacobiano de segundo orden no basta.** Comparado con los dos términos, la velocidad cambia muy poco (diferencia rms de u de 2e-6 contra u rms de 1.7e-2) y solo cambia η (0.018 contra 0.003 m):
- su ventaja está casi toda en la parte uniforme en la vertical, cerca de la superficie, que absorbe la superficie libre;
- la cizalla, que es lo que genera la corriente espuria, es la misma en ambos.

Marsaleix et al. (2009) lo advierten: los jacobianos de bajo orden no son automáticamente mejores. El jacobiano queda como esquema por defecto porque es exacto para estratificación lineal sobre cualquier topografía, y nunca peor.

**Lo que funciona es restar el perfil de referencia** (Haney 1991; método "balanceado" de Berntsen et al. 2015):
- El perfil ρ_ref(z) se estima del promedio horizontal de la densidad del propio modelo; no hace falta conocerlo analíticamente.
- La presión de un ρ_ref(z) horizontalmente uniforme no tiene gradiente a z fija, así que desaparecen los dos términos grandes que se cancelaban.
- Reduce la corriente espuria ~100 veces y deja a pyOCEAN por debajo del error de Beckmann & Haidvogel con rx1 = 10.7.

**Nota:** la primera versión de esta prueba corrió sin Coriolis, porque el modelo estratificado aún no lo tenía; daba 45 cm/s sin perfil de referencia y 0.14–0.44 con él. Con rotación (Coriolis de malla C que conserva energía) el error con perfil de referencia crece despacio, de 0.2 a 0.41 cm/s en el día 10. Por eso falta la prueba larga.

## Pendiente

- **Jacobiano de alto orden** (Shchepetkin & McWilliams 2003) o **volumen finito** (Adcroft et al. 2008; Engwirda et al. 2017). Ambos ya están en la biblioteca local. Sirven para las anomalías respecto al perfil de referencia, que importan en casos realistas con frentes sobre el talud.
- **Perfil de referencia que evoluciona:** recalcularlo cada cierto tiempo en corridas largas con forzamiento.
- **Prueba de 180 días** (Marsaleix et al. 2009) y la energía cinética espuria integrada: el error con perfil de referencia todavía crece al día 10.
