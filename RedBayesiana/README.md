# Red bayesiana

Carpeta para el desarrollo del modelo explicable (Sprint 3 en adelante), en
paralelo a `LSTM/` para el modelo de contraste.

## Estructura

```
RedBayesiana/
├── exploracion_inicial/   figuras descriptivas del conjunto de datos, sin
│                          conceptos de la red bayesiana (van en el informe
│                          antes del Sprint 3: exploración inicial de los
│                          datos cuantitativos y codificación numérica)
└── codigo_red/            el modelo en sí: discretización, comparabilidad,
                            estructura del grafo y su validación
```

### `exploracion_inicial/`

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

### `codigo_red/`

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
python3 codigo_red/discretizacion.py          # verificaciones incluidas
python3 codigo_red/estructura_aprendida.py    # tarda por el hill-climbing
```

Las demás variables de la Tabla 11 (año que cursa, sección, sesiones,
evaluaciones) no requieren función propia: ya llegan discretas o se
discretizaron en una fase anterior (`preprocesamiento/generar_csv.py` y
`CATALOGO_Temas.xlsx`).

## Estado (Sprint 3)

Cartas 1 a 6 del Trello completas: mapeo (Tabla 11), discretización
(Tabla 12, Figura 7 en exploración inicial), temporalidad (Tabla 13),
estructura del grafo (Figura 9) y selección de la herramienta (pgmpy).
Pendiente de decidir: si incorporar hallazgos de `estructura_aprendida.py`
al diseño del grafo antes de Sprint 4 (construcción del modelo).
