# Prototipo (Sprint 5)

Interfaz mínima para demostrar ambos modelos (LSTM y red bayesiana) sobre
un mismo caso académico. No es una aplicación de producción: una sola
vista, sin autenticación, sin base de datos, sin historial.

## Estado actual: tarjeta 1 (selección del caso)

Implementado en esta carpeta:

- **`caso.py`** — `CasoPrediccion` (dataclass inmutable: materia,
  trimestre, sección, estudiante_id, hito) y `OpcionesCaso`, que reutiliza
  `RedBayesiana/codigo_red/ensamblado.ensamblar_conjunto()` para enumerar
  las combinaciones realmente existentes (524 registros). No reimplementa
  esa lógica ni lee los CSV crudos por su cuenta.
- **`app.py`** — NiceGUI. Selectores encadenados asignatura → trimestre →
  sección → estudiante → hito; cada nivel solo se habilita y se llena
  cuando el anterior tiene un valor válido, así que no se puede construir
  una combinación inexistente. "Generar predicción" solo se habilita con
  los cinco valores completos.

**Lo que todavía NO hace `app.py`:** no ejecuta ni la LSTM ni la red
bayesiana. Al presionar "Generar predicción" solo construye y muestra el
`CasoPrediccion`, con un aviso de que la ejecución real depende de las
tarjetas 2 y 3.

## Por qué falta ejecutar los modelos (tarjetas 2 y 3)

Ninguno de los dos modelos tiene hoy un artefacto final persistido — ver
la auditoría de esta misma conversación. Antes de implementar
`lstm_servicio.py`/`bayes_servicio.py`, hay que decidir y ejecutar
(fuera de esta carpeta, en `LSTM/nuevo/` y `RedBayesiana/codigo_red/`)
cómo se entrena/ajusta el modelo "de producción" de cada asignatura, y
guardar los artefactos que la inferencia va a necesitar:

### LSTM (por asignatura y por hito → 4 × 3 = 12 modelos)

- modelo entrenado (`.keras`)
- `StandardScaler` de las características (ajustado sobre el mismo
  conjunto con que se entrenó ese modelo)
- media/desviación usadas para estandarizar el objetivo
- mediana de año académico usada para imputar (si aplica)
- orden exacto de las columnas (`ESTATICAS + DINAMICAS`) y longitud de
  secuencia (`hito`)

### Red bayesiana (por asignatura → 4 modelos, mismas CPD sirven para S4/S6/S8)

- estructura (ya es determinista y barata de reconstruir:
  `red_bayesiana.construir_modelo_manual()`, no hace falta persistirla)
- CPD ajustadas (`ajuste_cpd.ajustar_cpd`, con el ESS de referencia)
- mapa de tema (`discretizacion.ajustar_mapa_temas`)
- medias de entrenamiento por estado (`valor_esperado.medias_entrenamiento_por_estado`)

Ese ejercicio de "qué datos usa el modelo final" es una decisión
metodológica (¿todo el histórico? ¿todo salvo el trimestre más
reciente?) que debe explicitarse antes de generar los artefactos — no se
tomó todavía, a propósito.

## Cómo correr el prototipo

Requiere el entorno `tesis-py311` (el mismo que usan los notebooks:
TensorFlow, pgmpy, pandas ya instalados ahí). NiceGUI se instaló en ese
mismo entorno.

```bash
cd prototipo
python3 app.py
```

Abre en `http://localhost:8080`.
