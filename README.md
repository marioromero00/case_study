# Pricing Intelligence para Planta Salmonera

**Case Study — Vinson Digital**
Forecast de precios a 18 meses + solución digital recurrente
Mario Romero · Junio 2026

---

## 1. Resumen

Este repositorio contiene el análisis, el forecast y la propuesta de solución digital para el caso: "una compañía salmonera que busca profesionalizar sus decisiones de pricing. A partir de una base histórica de precios semanales por producto y calibre". Para esot se construye un forecast a 18 meses con incertidumbre explícita y se diseña una capacidad analítica recurrente que el equipo comercial puede usar semana a semana.

El principio rector del trabajo: **el desafío no es solo predecir precios, sino decidir mejor.**

---

## 2. Datos

**Fuente:** un único archivo CSV (`Dataset_case_study.csv`) con encabezados en tres niveles (producto / calibre / lectura), provisto por el cliente.

**Cobertura:**

| Concepto | Valor |
|---|---|
| Series | 6 (2 productos × 3 calibres) |
| Frecuencia | Semanal |
| Horizonte histórico | ene 2022 – mar 2026 (~4.3 años) |
| Semanas por serie | 221 |
| Completitud | ~98% (212–217 datos válidos por serie) |

**Productos:**

| Producto | Unidad | Calibres | Rango observado |
|---|---|---|---|
| Brazil HON | USD/kg FCA | 10-12, 12-14, 14-16 lb | 5.35 – 10.07 |
| USA Fillet | USD/lb FOB Miami | 2/3, 3/4, 4/5 | 5.08 – 7.78 |

**Variable objetivo:** precio real semanal.
- Brazil HON: promedio de las dos lecturas reales (`P Real X` / `P Real XII`).
- USA Fillet: columna `P Real`.

---

## 3. Preprocesamiento

Decisiones de limpieza conservadoras, pensadas para no introducir sesgo:

1. **Normalización de estructura:** la tabla jerárquica (producto/calibre/lectura) se convierte a formato largo (tidy) y luego a una matriz ancha por serie.
2. **Indexación temporal:** cada serie se re-indexa a frecuencia semanal regular anclada a lunes (`asfreq('W-MON')`), garantizando continuidad temporal.
3. **Tratamiento de nulos (~2-3%):** interpolación lineal acotada.
   - `limit=3`: solo se rellenan huecos de hasta 3 semanas consecutivas.
   - `limit_area="inside"`: solo se interpola **entre** observaciones reales; nunca se extrapola en los extremos de la serie.
   - Huecos mayores a 3 semanas se conservan como nulos: no se fuerza el relleno.
4. **Carga incremental:** la función `append_incremental` integra nuevas semanas al histórico y, ante fechas duplicadas, conserva la última versión. Esta función es la base de la capacidad recurrente de la solución digital.

---

## 4. Análisis exploratorio (hallazgos)

- **Volatilidad:** Brazil HON es ~1.6× más volátil que USA Fillet (≈31% vs ≈19% anualizado). El precio USD/kg FCA reacciona más a la oferta de cosecha; el filete FOB Miami es más estable por su mercado de destino.
- **Tendencia:** Brazil HON con tendencia bajista sostenida (~-13% YoY) tras los máximos de 2022-2024; USA Fillet casi plano (-0.3% a -2.9%).
- **Estacionalidad (recurrente en ambos mercados):** pico en semanas 10–16 (marzo–abril, Cuaresma/Semana Santa), +15% a +20% sobre la media anual; valle en semanas 35–42 (sept–oct), hasta -15% bajo la media.
- **Anomalías:** episodios puntuales detectados como residuos > 3σ en la descomposición STL (ejemplificada sobre Brazil HON 12-14 lb). Eventos de oferta/demanda que rompen el patrón estacional.

---

## 5. Modelamiento

### Modelos en competencia

| Modelo | Rol | Descripción |
|---|---|---|
| Naive estacional | Línea base (rival a vencer) | Proyecta el precio de hace 52 semanas — benchmark difícil de superar para commodities mean-reverting. |
| Prophet (logístico) | Candidato | Logístico acotado, prior rígido (0.02) — estable, captura estacionalidad anual y tendencia no lineal. |
| Ensemble Naive + Prophet | **Ganador** | Promedio simple de los dos modelos anteriores: cancela errores idiosincráticos y reduce el MAPE en todos los horizontes. |
| SARIMA | Referencia estadística | Modelo clásico estacional (1,1,1)(1,0,1,26) — alternativa para comparación. |

### Validación: backtesting con origen móvil

No se usa un split aleatorio, porque rompería la causalidad temporal de la serie. En su lugar:

- **4 folds** con ventana de entrenamiento creciente (expansiva).
- **Paso de 13 semanas** entre cortes.
- Cada fold proyecta **78 semanas** (18 meses) hacia adelante.
- El error se mide segmentado en tres tramos de horizonte:
  - Corto: semanas 1–26
  - Medio: semanas 27–52
  - Largo: semanas 53–78
- **Métrica: MAPE** (robusto, excluye divisiones por valores cercanos a cero).

### Resultados del backtesting (MAPE medio — 4 folds · 6 series)

| Modelo | Corto 1-26 | Medio 27-52 | Largo 53-78 | **Global** |
|--------|-----------|------------|------------|----------|
| **Ensemble Naive+Prophet** | **6.56 %** | **7.56 %** | **9.78 %** | **7.96 %** |
| Naive estacional | 6.84 % | 8.17 % | 10.45 % | 8.49 % |
| Prophet (orig) | 7.46 % | 7.79 % | 10.65 % | 8.63 % |
| SARIMA | 12.09 % | 9.04 % | 15.28 % | 12.14 % |

El Ensemble gana en **todos los horizontes** (−6.3 % vs Naive solo): el Naive ancla el patrón estacional anual sin asumir tendencia; Prophet aporta estructura paramétrica que corrige el drift intra-año. El promedio simple reduce la varianza sin añadir parámetros.

---

## 6. Forecast a 18 meses

- Modelo seleccionado: **Ensemble Naive + Prophet** (MAPE global 7.96 %).
- El **P50** es el promedio de la predicción Naive estacional y la de Prophet logístico.
- Las **bandas P10–P90** provienen de Prophet (modelo probabilístico), usado como estimador de la incertidumbre.
- La incertidumbre se ensancha con el horizonte: la confianza a corto plazo no es la misma que a 18 meses.

### Supuestos del forecast

1. **Tendencia logística acotada (Prophet):** los precios tienen techo y piso de mercado; no crecen ni caen indefinidamente.
2. **Patrón estacional anual (Naive):** el precio de hace 52 semanas es el ancla estacional del ensemble.
3. **Continuidad del régimen histórico:** no ocurren shocks estructurales nuevos (sanitarios, regulatorios, de mercado) durante el período proyectado.
4. **Solo precios propios:** el forecast se construye con el histórico de cada serie, sin variables externas.

---

## 7. Uso comercial

El forecast se traduce en decisiones según el horizonte donde el modelo es más confiable:

- **Negociar (corto plazo, alta confianza):** definir piso/techo de precio por calibre; decidir vender ahora o esperar.
- **Planificar (mediano plazo, confianza media):** planificación comercial y presupuestaria; priorizar mercados/productos; anticipar ventanas estacionales.
- **Gestionar escenarios (transversal):** construir escenarios optimista/base/pesimista desde las bandas; alertas cuando el precio real se aparta de lo esperado.

---

## 8. Solución digital (capacidad recurrente)

Ciclo semanal automatizado:

```
Histórico base
  → Carga semanal incremental (subir CSV)
  → Reentrenamiento automático
  → Selección del mejor modelo por horizonte
  → Forecast actualizado
  → Dashboard + alertas
  ↺ (realimentación al histórico)
```

- **Recurrente y automática:** cada semana se incorpora el dato nuevo y el sistema se re-entrena.
- **Auto-seleccionable:** se re-elige el mejor modelo por horizonte en cada actualización.
- **Accionable:** el resultado llega como dashboard + alertas, no como archivo técnico.

La función `append_incremental` ya implementa la lógica de carga incremental, y la carpeta `app/` contiene una primera versión funcional de la interfaz: el flujo "subir CSV de la semana → forecast actualizado" es demostrable, no solo conceptual.

MVP ppta digial funcional: https://salmonforecasting.streamlit.app/

---

## 9. Riesgos, limitaciones y próximos pasos

### Limitaciones

- **Una sola variable:** el modelo solo usa el histórico del propio precio.
- **Sin variables externas:** quedan fuera FX, oferta de Noruega/Chile, demanda, costos e inventarios.
- **Sensible a shocks:** eventos sanitarios o logísticos imprevistos se detectan tarde, no se anticipan.
- **Precisión decreciente:** a mayor horizonte, mayor incertidumbre.

### Próximos pasos

1. **Incorporar variables externas** (FX, oferta, demanda): palanca de mayor impacto en precisión.
2. **Automatizar el pipeline:** carga, reentrenamiento y publicación sin intervención manual.
3. **Dashboard ejecutivo:** interfaz lista para el equipo comercial.
4. **Medición continua:** monitorear la precisión semana a semana y recalibrar.

---

## 10. Reproducibilidad

### Estructura del repositorio

```
case_study/
├── app/                 # Solución digital: dashboard + carga incremental
├── data/                # Datos crudos y procesados
├── notebooks/
│   ├── 00_eda.ipynb     # Análisis exploratorio (EDA)
│   ├── 01_data_prep.ipynb   # Preprocesamiento y limpieza
│   └── 02_modeling.ipynb    # Backtesting y forecast (banco de pruebas de modelos)
├── src/                 # Módulos reutilizables (carga incremental, helpers)
├── requirements.txt     # Dependencias
└── README.md
```

### Cómo ejecutar

```bash
# 1. Clonar el repositorio
git clone https://github.com/marioromero00/case_study.git
cd case_study

# 2. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar los notebooks en orden
jupyter notebook notebooks/00_eda.ipynb
# jupyter notebook notebooks/01_data_prep.ipynb
# jupyter notebook notebooks/02_modeling.ipynb

# 4. Levantar la app / dashboard
# (ver instrucciones específicas en la carpeta app/)
```

**Stack:** Python · pandas · numpy · statsmodels (STL) · Prophet · matplotlib.

