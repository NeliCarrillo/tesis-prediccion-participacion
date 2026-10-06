# Evaluación y comparación de los modelos (Sprint 6)

Análisis de la fase de validación y evaluación: comparan la red bayesiana, la LSTM y la
extrapolación proporcional sobre las mismas predicciones de validación cruzada. Ninguno
reentrena los modelos oficiales ni modifica sus resultados: leen las predicciones de
`3_lstm/resultados/` y `4_red_bayesiana/resultados/`, y guardan sus tablas en
`resultados/` y sus figuras en `figuras/resultados/`.

Cada script comprueba sus propios resultados con `assert` y, cuando reproduce un valor
publicado (Tablas 16, 19 y 20), exige que coincida con el informe.

## Qué respalda cada script

| Script | Qué calcula | Dónde aparece en el informe |
|---|---|---|
| `comparacion_modelos.py` | RMSE, MAE y R² de los tres métodos por asignatura e hito | Tablas 16, 17 y 21 |
| `figura_lstm_vs_extrapolacion_tabla16.py` | Figura de RMSE y R² de la LSTM y la extrapolación | Figura 14 |
| `figura_comparacion_modelos.py` | Figura de RMSE de los tres métodos | Figura 19 |
| `casos_atipicos.py` | Predicciones de los tres casos atípicos | Figura 18 |
| `figura_brier.py` | Puntaje de Brier por asignatura e hito y referencias uniforme y marginal histórica, con los colores de las tablas | Tabla 18 y Figura 15 |
| `respaldo_evidencia.py` | Respaldo de cada combinación de evidencia, brecha en los casos con respaldo y alternativas para los casos sin respaldo | Tabla 19, Comparación LSTM y Bayes, Tabla F3 |
| `sensibilidad_evidencia.py` | Rango de sensibilidad por variable, global y por asignatura; predicción de cada caso con el valor observado, con cada estado y con la variable omitida; cambio de RMSE al omitir cada variable (aporte predictivo observado) y dirección del cambio entre estados con respaldo | Tabla de sensibilidad (Tabla 20) y Figura 17 |
| `margen_equivalencia.py` | Otras reglas de suavizado y cota con todos los trimestres | Comparación LSTM y Bayes, Tabla F2 |
| `rmse_con_techo.py` | RMSE de la red bayesiana tomando el techo como valor correcto | Comparación LSTM y Bayes, Apéndice F |
| `anomalias_pendiente.py` | Cambio de pendiente del RMSE con intervalo bootstrap, sesgos e hipótesis de la meta de participaciones | Comparación LSTM y Bayes, Tabla J2 |
| `anomalias_calendario.py` | Calendario, acontecimientos nacionales, casos sin participación y prueba de la indicación de evaluación en la LSTM | Figuras 20 y 21, Tabla J1, Apéndice J |
| `semillas_lstm.py` | Repite con las cinco semillas los análisis por estudiante que usan la semilla 42 (cambio de pendiente, casos sin participación, brecha con respaldo) la posición de cada semilla, la diferencia promedio frente a los otros métodos y la variación de la predicción individual entre semillas | Apéndice C (Tablas C2 y C3) y Resultados |
| `figura_posterior_atipicos.py` | Distribución posterior de los casos atípicos | No incluida en el informe |
| `rutas.py` | Agrega `4_red_bayesiana/codigo/` a `sys.path` para reutilizar el código de la red | Uso interno |

El intento de explicabilidad de la LSTM (pesos y combinaciones) está en
`3_lstm/adaptacion/explicabilidad_lstm.py`, junto al modelo que analiza.

## Ejecución

Desde la raíz del repositorio, con las dependencias de `5_prototipo/requirements.txt`:

```bash
python3 6_evaluacion/codigo/comparacion_modelos.py
python3 6_evaluacion/codigo/respaldo_evidencia.py
```

`anomalias_calendario.py` necesita además TensorFlow, porque carga los modelos finales
del prototipo para la prueba de la indicación de evaluación.
