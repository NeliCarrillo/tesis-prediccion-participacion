# Preprocesamiento — Fase 2 del anteproyecto

Código que convierte los registros crudos de participación (`Datos Tesis/`, formatos heterogéneos
según el docente y el trimestre) en los CSV estandarizados y anonimizados que alimentan el
modelado (redes bayesianas / LSTM). Implementa la Fase 2 ("levantamiento, limpieza y
estandarización de datos") descrita en el anteproyecto.

## Estructura

| Archivo | Qué hace |
|---|---|
| `cronogramas.py` | Parsea los cronogramas (`.docx` y el `.xlsx` especial de Algoritmos 2425-2) a `{(materia, trimestre): {semana: tema}}`. |
| `limpieza.py` | Funciones puras de normalización: cédulas, artefacto de iniciales pegadas en nombres, cálculo de semana a partir de una fecha, decodificación de asistencia P/1/T/F/J. |
| `extractores.py` | Un extractor por cada formato de archivo fuente (por sesión, agregado por semana, híbrido con código de asistencia) + el caso especial de Estructuras 2526-1 (cruce por nombre). |
| `fuentes.py` | Catálogo declarativo de las 11 fuentes (qué archivo/hoja, qué materia/trimestre/sección, qué extractor usar) y la función que las despacha todas. |
| `anonimizacion.py` | Construye el mapa cédula → `estudiante_id` y lo aplica. |
| `validacion.py` | Chequeos post-procesamiento (cruce contra la columna `TOTAL` del Excel original, huecos de tema). Solo reporta cifras, nunca nombres ni cédulas. |
| `pipeline.py` | Orquesta todo lo anterior. Se puede correr como script (`python3 pipeline.py`) o importar (`from pipeline import ejecutar`). |
| `01_estandarizacion_y_anonimizacion.ipynb` | Notebook delgado: importa `pipeline`, lo corre, muestra el resumen. Toda la lógica vive en los `.py`, no en el notebook. |

Separar la lógica en módulos permite leer/probar cada pieza de forma aislada (p. ej. `extractores.py`
sin tener que abrir el notebook), y que el notebook en sí sea solo la capa de orquestación y
narrativa — no un único archivo con 300+ líneas de código mezcladas con explicaciones.

## Cómo correrlo

```bash
cd preprocesamiento
python3 pipeline.py
```

o abriendo `01_estandarizacion_y_anonimizacion.ipynb` en Jupyter/VS Code y ejecutando todas las celdas.
Ambos caminos llaman exactamente al mismo código y producen el mismo resultado.

**Dependencias:** `pandas`, `numpy`, `openpyxl`, `python-docx`.

## Salida

Todo se escribe en `Datos Tesis/_procesado/` (no versionado salvo por su ausencia — no hay `.csv`
comiteados en este repo; cada quien los regenera corriendo el pipeline):

- `<materia>_<trimestre>[_seccion]_participaciones.csv` — 13 archivos, uno por fuente. Esquema:

  | Columna | Descripción |
  |---|---|
  | `estudiante_id` | Anónimo, consistente entre archivos (`anon_001`, `anon_002`, ...) |
  | `numero_lista` | Solo si el archivo fuente lo trae directamente |
  | `materia`, `trimestre`, `seccion` | — |
  | `fecha` | `YYYY-MM-DD`; vacía si la fuente ya venía agregada por semana |
  | `semana` | Número de semana del trimestre |
  | `tema` | Del cronograma correspondiente a esa semana |
  | `participaciones` | Cantidad de intervenciones |
  | `tipo_participacion` | Reservado, no disponible aún en los datos fuente |
  | `asistencia` | Solo disponible en 2 de las 13 fuentes (las que traen código P/1/T/F/J) |

- `log_limpieza.txt` — bitácora pública de las transformaciones. No contiene cédulas ni nombres.
- `_confidencial/mapeo_estudiantes.csv` — el mapa cédula → `estudiante_id`.
- `_confidencial/log_limpieza_detalle.txt` — el puñado de notas que sí requieren nombrar a alguien
  (hoy, solo el detalle del cruce por nombre en Estructuras 2526-1).

⚠️ `Datos Tesis/_procesado/_confidencial/` está en `.gitignore` — son datos personales de
estudiantes y nunca deben subirse al repositorio.

## Decisiones de diseño que vale la pena conocer antes de tocar el código

- **Cómo se calcula `semana` cuando el archivo fuente no la da directamente**: se toma el lunes
  de la semana de la primera sesión registrada como inicio de la "semana 1" del trimestre, y se
  cuenta en bloques de 7 días desde ahí (`limpieza.semana_desde_fecha`). Se validó contra el único
  cronograma que trae fechas explícitas por semana (Algoritmos 2425-2) y reproduce exactamente la
  numeración real del profesor.
- **`participaciones` vs. el `TOTAL` de algunos Excel originales**: en Algoritmos 2425-2 sec1, 2 de
  31 estudiantes tienen un `TOTAL` en el Excel que no coincide con la suma de las columnas de
  fecha, porque la fórmula del archivo original no se extendió a la última columna. Este pipeline
  suma cada celda de fecha directamente, así que es más confiable que ese `TOTAL` para esos casos
  (ver `validacion.validar_total_algoritmos_2425_2_sec1`, que lo reporta en el log).
- **Duplicados**: Computación Emergente 2526-2 existe en dos libros (uno standalone y otro dentro
  del libro de Estructuras de Datos 2526-2); se usa el standalone.
- **Algoritmos 2526-2** no tenía archivo de participaciones propio (solo cronograma). Se completó
  con la hoja `ALGORITMOS` del libro de Estructuras de Datos 2526-2, que trae ese mismo trimestre.
- **Código `J` ("jubilado")** en los archivos híbridos: confirmado con el autor de la tesis que
  significa "asistió pero se retiró antes de terminar la clase" → se cuenta como presente, sin
  participaciones ese día.
- Quedan sin interpretar el emoji `⚕️` (8 celdas, Estructuras 2425-3), el emoji `⚖️` (4 celdas,
  Estructuras 2526-3) y una `p` minúscula suelta (1 celda, típico de un typo de `P`) — se dejan
  como `NaN` en vez de adivinar su significado.
