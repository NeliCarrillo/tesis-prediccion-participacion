# Preprocesamiento

Convierte los registros estandarizados en un conjunto de datos tabular, uno por sección.

## Entrada

Cada archivo de `Datos Tesis Upstream` lleva dos hojas añadidas al libro original:

- **`Estandar`** — la rejilla de participaciones: una fila por estudiante y dos
  subcolumnas por semana, una por sesión. Se considera sesión real la subcolumna
  que lleva el nombre del día, de modo que las secciones con una sola clase
  semanal quedan con una sesión por semana.
- **`Cronograma`** — una fila por sesión, con `semana`, `dia_sesion`, `tema` y
  `tipo_sesion`. El tema es el código del catálogo de la asignatura
  (`CATALOGO_Temas.xlsx`); vale `0` cuando la sesión no cubre contenido nuevo, y
  entonces `tipo_sesion` indica la causa.

## Salida

Un `.csv` por sección en `Datos Tesis Downstream`, con la misma estructura de
carpetas. Cada fila es un estudiante en una sesión.

## Criterios aplicados

- **Participaciones.** El número de la celda, redondeado hacia arriba (una media
  participación cuenta como una).
- **Asistencia.** Participar implica haber asistido. Los códigos `P`, `p`, `T`,
  `F` y `J` indican presencia sin participación; `⚕️` y `⚖️`, ausencia
  justificada. En las secciones que llevan registro de asistencia, la celda vacía
  significa ausencia; en las demás queda como dato faltante, que la red bayesiana
  puede manejar.
- **Sesiones no dictadas.** Cuando `tipo_sesion` es `sin_clase`, la participación
  y la asistencia quedan vacías: no hubo oportunidad de participar.
- **Año académico.** Diferencia entre el año calendario del trimestre y el año de
  ingreso que indican los cuatro primeros dígitos del carnet, más uno.
- **Anonimización.** La cédula se sustituye por un identificador consistente entre
  archivos. El mapeo se guarda en `Datos Tesis Upstream/_procesado/_confidencial/`,
  fuera del control de versiones.

## Ejecución

Requiere Python 3 con `pandas` y `openpyxl`:

```bash
pip install pandas openpyxl
```

```bash
python3 preprocesamiento/generar_csv.py
```

Se puede correr cuantas veces haga falta: reescribe los catorce CSV desde cero en cada
ejecución y elimina los que ya no correspondan a ningún archivo de origen, de modo que
la carpeta refleje siempre el estado actual de los registros. El mapeo de anonimización
sí se conserva entre ejecuciones, para que un mismo estudiante mantenga su identificador.

El resultado es reproducible: dos ejecuciones sobre una copia limpia del repositorio
producen los catorce archivos idénticos byte a byte, identificadores anónimos incluidos,
porque estos se asignan recorriendo los archivos en un orden fijo. El mapeo no se versiona
por contener cédulas, de modo que quien clone el repositorio lo regenera al ejecutar el
script y obtiene los mismos identificadores.
