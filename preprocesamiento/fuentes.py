"""Catalogo declarativo de las 11 fuentes de datos (13 archivos de salida) y su despacho
al extractor que corresponde segun su formato.

Mantener esto separado de extractores.py permite ver de un vistazo *que* se procesa y de
*donde* sale, sin mezclarlo con *como* se procesa cada formato.
"""
from datetime import datetime

from extractores import (
    process_simple_partic,
    process_hybrid_asistencia,
    process_weekly_aggregated,
    process_estructura_2526_1,
)

# --- Formato 1: una fila por estudiante, una columna por fecha de sesion -------------------

FUENTES_SIMPLES = [
    dict(
        nombre="algoritmos_2425-2_sec1",
        ruta="Algoritmos y Programacion/2425-2/Participacion sec 1 Alg 2425-2.xlsx", hoja="Partic",
        materia="Algoritmos y Programación", trimestre="2425-2", seccion="1",
        kwargs=dict(text_date_map={
            'Jan-06': datetime(2025, 1, 6), '08-Jan': datetime(2025, 1, 8), '13-Jan': datetime(2025, 1, 13),
        }),
    ),
    dict(
        nombre="algoritmos_2425-2_sec2",
        ruta="Algoritmos y Programacion/2425-2/Participacion sec 2 Alg 2425-2.xlsx", hoja="Partic",
        materia="Algoritmos y Programación", trimestre="2425-2", seccion="2",
    ),
    dict(
        nombre="algoritmos_2526-1",
        ruta="Algoritmos y Programacion/2526-1/Participación Algoritmos 2526-1.xlsx", hoja="Partic",
        materia="Algoritmos y Programación", trimestre="2526-1", seccion="",
        nota_extra="El archivo fuente no distingue seccion por estudiante (el libro 'Total' "
                    "menciona 'sec 1 y 5' de forma ambigua) -> columna 'seccion' se dejo vacia.",
    ),
    dict(
        nombre="computacion_emergente_2526-1",
        ruta="Computacion Emergente/2526-1/Particip emergente 2526-1.xlsx", hoja="Intervenciones",
        materia="Computación Emergente", trimestre="2526-1", seccion="",
        kwargs=dict(header_row=1),
        nota_extra="Seccion no especificada en el archivo fuente -> columna 'seccion' vacia.",
    ),
    dict(
        nombre="estructuras_2425-3",
        ruta="Estructura de Datos/2425-3/Participaciones Estructura de Datos 2425-3.xlsx", hoja="Partic",
        materia="Estructuras de Datos", trimestre="2425-3", seccion="",
        nota_extra="Seccion no especificada en el archivo fuente -> columna 'seccion' vacia.",
    ),
]

# --- Formato 2: columnas agrupadas bajo 'Semana N' con codigo de asistencia P/1/T/F/J -----

FUENTES_HIBRIDAS = [
    dict(
        nombre=f"algoritmos_2526-3_sec{sec}",
        ruta="Algoritmos y Programacion/2526-3/Asistencia y participacion Algoritmos 2526-3.xlsx",
        hoja=f"sec {sec}",
        materia="Algoritmos y Programación", trimestre="2526-3", seccion=sec,
    )
    for sec in ("1", "3", "6")
] + [
    dict(
        nombre="estructuras_2526-3",
        ruta="Estructura de Datos/2526-3/Asistencia y Participaciones Estructura de Datos 2526-3.xlsx",
        hoja="Asistencia",
        materia="Estructuras de Datos", trimestre="2526-3", seccion="2",
    ),
]

# --- Formato 3: ya agregado por semana (Sem 1 ... Sem 12), sin fecha de sesion ------------

FUENTES_SEMANALES = [
    dict(
        nombre="computacion_emergente_2526-2",
        ruta="Computacion Emergente/2526-2/Participaciones Emergente 2526-2.xlsx", hoja="COMPURACION EMERGENTE",
        materia="Computación Emergente", trimestre="2526-2", seccion="",
        kwargs=dict(header_row=1),
        nota_extra="Se uso el archivo standalone de 'Computacion Emergente/2526-2/' (no la hoja "
                    "duplicada dentro del libro de Estructuras de Datos 2526-2).",
    ),
    dict(
        nombre="estructuras_2526-2",
        ruta="Estructura de Datos/2526-2/Participaciones Estructura de Datos 2526-2.xlsx", hoja="ESTRUCTURAS-DE DATOS",
        materia="Estructuras de Datos", trimestre="2526-2", seccion="",
        kwargs=dict(header_row=1),
        nota_extra="El libro fuente tambien trae una hoja COMPURACION EMERGENTE para el mismo "
                    "trimestre, duplicada del archivo standalone (ver nota de computacion_emergente_2526-2), "
                    "por lo que no se proceso de nuevo aqui. La hoja ALGORITMOS del mismo libro si se "
                    "proceso por separado, ver fuente 'algoritmos_2526-2'.",
    ),
    dict(
        nombre="algoritmos_2526-2",
        ruta="Estructura de Datos/2526-2/Participaciones Estructura de Datos 2526-2.xlsx", hoja="ALGORITMOS",
        materia="Algoritmos y Programación", trimestre="2526-2", seccion="",
        kwargs=dict(header_row=1),
        nota_extra="Algoritmos 2526-2 no tenia archivo de participaciones propio (solo cronograma). "
                    "Por decision expresa del autor de la tesis, se tomo esta hoja del libro de "
                    "Estructuras de Datos 2526-2, que registra participaciones de Algoritmos para el "
                    "mismo trimestre con el mismo formato Sem 1 .. Sem 12.",
    ),
]

# --- Caso especial: cruce por nombre contra otra hoja del mismo libro ---------------------

FUENTE_ESTRUCTURAS_2526_1 = dict(
    nombre="estructuras_2526-1",
    ruta="Estructura de Datos/2526-1/Participaciones Estructura de Datos 2526-1.xlsx",
    materia="Estructuras de Datos", trimestre="2526-1", seccion="",
)


def procesar_todas_las_fuentes(base_dir, cronogramas):
    """Corre el extractor correspondiente sobre cada fuente declarada arriba.

    Devuelve (resultados, log_publico, log_confidencial):
      - resultados: {nombre_fuente: [filas...]}
      - log_publico: lista de strings, cada una ya prefijada con '[nombre_fuente] ...'
      - log_confidencial: idem, pero solo con las notas que pueden incluir nombres
    """
    resultados = {}
    log_publico = []
    log_confidencial = []

    def _registrar(nombre, notas_publicas, notas_confidenciales, nota_extra=None):
        for nota in notas_publicas:
            log_publico.append(f"[{nombre}] {nota}")
        if nota_extra:
            log_publico.append(f"[{nombre}] NOTA: {nota_extra}")
        for nota in notas_confidenciales:
            log_confidencial.append(f"[{nombre}] {nota}")

    for spec in FUENTES_SIMPLES:
        filas, notas_pub, notas_conf = process_simple_partic(
            base_dir / spec["ruta"], spec["hoja"], spec["materia"], spec["trimestre"], spec["seccion"],
            cronogramas, **spec.get("kwargs", {}))
        resultados[spec["nombre"]] = filas
        _registrar(spec["nombre"], notas_pub, notas_conf, spec.get("nota_extra"))

    for spec in FUENTES_HIBRIDAS:
        filas, notas_pub, notas_conf = process_hybrid_asistencia(
            base_dir / spec["ruta"], spec["hoja"], spec["materia"], spec["trimestre"], spec["seccion"],
            cronogramas)
        resultados[spec["nombre"]] = filas
        _registrar(spec["nombre"], notas_pub, notas_conf, spec.get("nota_extra"))

    for spec in FUENTES_SEMANALES:
        filas, notas_pub, notas_conf = process_weekly_aggregated(
            base_dir / spec["ruta"], spec["hoja"], spec["materia"], spec["trimestre"], spec["seccion"],
            cronogramas, **spec.get("kwargs", {}))
        resultados[spec["nombre"]] = filas
        _registrar(spec["nombre"], notas_pub, notas_conf, spec.get("nota_extra"))

    spec = FUENTE_ESTRUCTURAS_2526_1
    filas, notas_pub, notas_conf = process_estructura_2526_1(
        base_dir / spec["ruta"], spec["materia"], spec["trimestre"], spec["seccion"], cronogramas)
    resultados[spec["nombre"]] = filas
    _registrar(spec["nombre"], notas_pub, notas_conf)

    return resultados, log_publico, log_confidencial
