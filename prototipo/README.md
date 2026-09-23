# Prototipo (Sprint 5)

Interfaz mínima para demostrar ambos modelos (LSTM y red bayesiana) sobre
un mismo caso académico. No es una aplicación de producción: una sola
vista, sin autenticación, sin base de datos, sin historial.

## Estado actual

### Tarjeta 1 (selección del caso) — completada

- **`caso.py`** — `CasoPrediccion` (dataclass inmutable: materia,
  trimestre, sección, estudiante_id, hito) y `OpcionesCaso`, que reutiliza
  `RedBayesiana/codigo_red/ensamblado.ensamblar_conjunto()` para enumerar
  las combinaciones realmente existentes (524 registros). No reimplementa
  esa lógica ni lee los CSV crudos por su cuenta.
- **`app.py`** — NiceGUI. Selectores encadenados asignatura → trimestre →
  sección → estudiante → hito; cada nivel solo se habilita y se llena
  cuando el anterior tiene un valor válido, así que no se puede construir
  una combinación inexistente. "Generar predicción" solo se habilita con
  los cinco valores completos. **Todavía no** llama a `lstm_service` ni a
  ningún servicio de Bayes — eso es tarjetas 4/5 (visualización), fuera de
  alcance de esta tarjeta.

### Tarjeta 2 (conexión con la LSTM) — modelos y servicio listos, sin conectar a la interfaz

Decisión metodológica aprobada: **Estrategia 2** — un modelo final por
asignatura × hito (12 en total), entrenado con **todos** los trimestres
disponibles (sin dejar ninguno fuera), semilla 42 (la misma que usa por
defecto la celda 34 del notebook, no elegida por desempeño). Estos
modelos son exclusivamente para el prototipo: una predicción del
prototipo sobre un registro histórico **no** es una nueva medición de
desempeño, porque ese registro pudo haber participado en el ajuste del
modelo que lo predice. Las métricas oficiales de la tesis siguen siendo
únicamente las de la validación cruzada (Sprint 2/4).

- **`services/lstm_service.py`** — `predict_lstm(caso)`. Las funciones de
  preparación de datos están copiadas verbatim (extraídas
  programáticamente, no retranscritas) de las celdas 5/8/11/14/17/20 del
  notebook validado.
- **`services/test_paridad_lstm.py`** — nivel A: compara la preparación
  de datos del servicio contra una ejecución real del notebook (celdas
  0-18); nivel B: verifica la aritmética de inferencia contra una
  reimplementación independiente. Ambos niveles superados sobre 4 casos
  reales (incluido uno con año académico faltante). Ejecutar con
  `python3 services/test_paridad_lstm.py`.
- **`artefactos_lstm/`** — 12 carpetas (`<materia>_h<hito>/`), cada una
  con `modelo.keras`, `escalador_x.json` (mean\_/scale\_ del
  `StandardScaler`, no el objeto serializado), `objetivo.json`
  (media/desviación) e `imputacion.json` (mediana de año académico, o
  `null` si esa asignatura no tuvo faltantes). `config_compartida.json`
  documenta las columnas, hitos y semilla usadas.
- **`LSTM/nuevo/predicciones_lstm_validacion_cruzada.csv`** — exportación
  aditiva (2 celdas nuevas al final del notebook, ninguna de las 56
  originales se modificó) de las 1.572 predicciones individuales
  (524 registros × 3 hitos) de la validación cruzada, semilla única 42.
  Sirve para comparaciones futuras de paridad (Tarjeta 7), no reemplaza
  ninguna métrica oficial.

**Pendiente:** conectar `predict_lstm` a `app.py` (tarjeta 4) y construir
el equivalente para Bayes (tarjeta 3 — no necesita artefactos persistidos,
ver más abajo).

### Red bayesiana (tarjeta 3, todavía no implementada)

- estructura (ya es determinista y barata de reconstruir:
  `red_bayesiana.construir_modelo_manual()`, no hace falta persistirla)
- CPD ajustadas (`ajuste_cpd.ajustar_cpd`, con el ESS de referencia) —
  reconstruibles en ~30 ms por asignatura con
  `ajuste_cpd.preparar_datos_asignatura` (ya existe, Sprint 4 carta 4)
- mapa de tema (`discretizacion.ajustar_mapa_temas`)
- medias de entrenamiento por estado
  (`valor_esperado.medias_entrenamiento_por_estado`, con un
  `trimestre_prueba` inexistente para obtener las medias sobre todo el
  histórico sin modificar la función)

Por lo barato que es reconstruir todo esto, no se persistirá nada en
disco para Bayes — se recalculará en memoria al arrancar el prototipo.

## Cómo correr el prototipo

Requiere el entorno `tesis-py311` (el mismo que usan los notebooks:
TensorFlow, pgmpy, pandas ya instalados ahí). NiceGUI se instaló en ese
mismo entorno.

```bash
cd prototipo
python3 app.py
```

Abre en `http://localhost:8080`.
