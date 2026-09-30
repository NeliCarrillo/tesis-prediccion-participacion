# Red bayesiana

Carpeta para el desarrollo del modelo explicable (Sprint 3 en adelante), en
paralelo a `3_lstm/` para el modelo de contraste.

## Estructura

```
4_red_bayesiana/
├── exploracion/   scripts de las figuras descriptivas del conjunto de datos
│                  (sus imágenes están en figuras/metodo_datos_y_exploracion/)
├── codigo/        el modelo en sí: discretización, ensamblado, estructura del
│                  grafo, ajuste de CPD, inferencia, validación cruzada,
│                  comparación de datos faltantes y auditorías (Sprints 3 y 4)
├── notebooks/     notebooks de ejecución y auditoría, guardados con outputs
└── resultados/    CSV oficiales de la red: predicciones de validación cruzada,
                   comparación NaN frente a dropna y tablas de auditoría
```

Los análisis del Sprint 6 que comparan la red con la LSTM y la extrapolación
(respaldo, sensibilidad, techo, anomalías, margen) están en `6_evaluacion/` y
reutilizan este código. Las figuras de todo el proyecto están en `figuras/`,
con su índice en `figuras/INDICE.md`.

> Nota: las secciones siguientes describen los archivos tal como se
> documentaron en los Sprints 3 y 4; las rutas ya corresponden a la nueva
> organización del repositorio.

### `exploracion/`

- **`figura_tamano_grupo_exploracion.py`** → tamaño de las 14 secciones
  (Figura 3 del informe, sección «Exploración inicial de los datos
  cuantitativos»).
- **`figura_posicion_lista.py`** → distribución de la posición relativa en
  la lista, con la prueba chi-cuadrado que explica por qué no es
  perfectamente uniforme (Figura 7, sección «Codificación numérica de los
  campos cualitativos»).
- **`figura_discretizacion.py`** / **`figura_tema.py`** → versiones previas
  de las figuras de discretización (4 histogramas y exploración de tema),
  de una etapa anterior a mover tamaño/posición a las figuras de arriba.
  Puede que ya no estén referenciadas en la versión actual del informe —
  revisar antes de descartarlas.

### `codigo/`

- **`discretizacion.py`** — los cortes de la Tabla 12 para tamaño del grupo,
  posición en la lista y participaciones (semana y trimestre). Verificado
  contra los 437 registros y las 5.244 filas estudiante-semana.
- **`valor_esperado.py`** — convierte la distribución de salida de la red
  (categórica) en un valor continuo comparable con la LSTM (RMSE, R²), usando
  la media observada de cada estado, no el punto medio del intervalo.
- **`figura_grafo_bayesiano.py`** — el diagrama del grafo (10 nodos, 9 arcos),
  con orientación vertical, sin autobucles (nodo propio para «participaciones
  de la semana anterior»), medido con el renderer real de matplotlib para
  evitar cajas encimadas. `figura_grafo.py` es una versión anterior (más
  angosta, con autobucle) — mantenida por si hace falta comparar.
- **`estructura_aprendida.py`** — compara el grafo manual contra la
  estructura que aprende `pgmpy` (hill-climbing + BIC), sin restricciones y
  con lista negra temporal (Tabla 13), por asignatura. No genera figura, solo
  texto (arcos aprendidos, comparación y puntajes BIC).

```bash
python3 codigo/discretizacion.py          # verificaciones incluidas
python3 codigo/estructura_aprendida.py    # tarda por el hill-climbing
```

Las demás variables de la Tabla 11 (año que cursa, sección, sesiones,
evaluaciones) no requieren función propia: ya llegan discretas o se
discretizaron en una fase anterior (`2_preprocesamiento/generar_csv.py` y
`1_datos/plantillas/CATALOGO_Temas.xlsx`).

## Estado (Sprint 3)

Cartas 1 a 6 del Trello completas: mapeo (Tabla 11), discretización
(Tabla 12, Figura 7 en exploración inicial), temporalidad (Tabla 13),
estructura del grafo (Figura 9) y selección de la herramienta (pgmpy).
Pendiente de decidir: si incorporar hallazgos de `estructura_aprendida.py`
al diseño del grafo antes de Sprint 4 (construcción del modelo).

## Auditoría de ponderación y asistencia (Product Backlog del tutor)

Responde a dos observaciones de Fernando Torre Mora: por qué se pondera o
no se pondera durante el entrenamiento, y si la asistencia puede
reconstruirse a partir de la participación. Sustenta los Apéndices F y G
del informe.

- **`codigo/auditoria_ponderacion_asistencia.py`** — script de solo
  lectura (no modifica ningún pipeline). Reproduce, desde los datos y el
  código reales: tamaños por materia/trimestre/sección; sesiones por
  registro estudiante-sección (24 en todo el estudio salvo los 73 registros
  de Computación Emergente 2425-3, con 12); confirmación de que la LSTM
  adaptada no usa `sample_weight` ni `class_weight`; inventario de hojas de
  asistencia reales en `1_datos/originales/`; confirmación de que
  «asistencia» nunca entró a `generar_csv.py` ni a `1_datos/estandarizados/`; búsqueda en todo el historial de git de una corrida previa
  con asistencia; y la evaluación cuantitativa de la regla «participación >
  0 ⇒ presencia» (14,1 % de las filas con participación conocida quedan
  determinadas, 85,9 % quedan ambiguas).
- **`notebooks/auditoria_ponderacion_asistencia.ipynb`** — importa y
  ejecuta ese script (sin duplicar su lógica), guardado con outputs. Genera
  los CSV de apoyo en `resultados/`: `auditoria_sesiones_por_registro.csv`,
  `auditoria_inventario_asistencia.csv`,
  `auditoria_heuristica_participacion.csv`.

```bash
python3 codigo/auditoria_ponderacion_asistencia.py
```

No se introdujo ninguna ponderación correctora ni ninguna variable de
asistencia reconstruida en los modelos — ambos puntos quedan documentados,
no resueltos con un cambio de metodología.
