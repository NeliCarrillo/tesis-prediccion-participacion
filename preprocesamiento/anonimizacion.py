"""Anonimizacion global y consistente entre todos los archivos de salida.

Un mismo estudiante (misma cedula) que aparezca en varias materias o trimestres debe recibir
siempre el mismo estudiante_id, y la asignacion debe ser deterministica: si se vuelve a correr
el pipeline sobre los mismos datos crudos, el resultado tiene que ser identico byte a byte.
"""


def construir_mapa(cedulas_unicas):
    """cedulas_unicas: coleccion de cedulas (str). Se ordenan de forma ascendente por su valor
    numerico antes de asignar el id, para que el mapeo no dependa del orden de procesamiento
    de los archivos (que si puede variar entre corridas)."""
    cedulas_ordenadas = sorted(cedulas_unicas, key=int)
    return {cedula: f"anon_{i + 1:03d}" for i, cedula in enumerate(cedulas_ordenadas)}


def aplicar_mapa(filas, mapa):
    """Sustituye 'cedula_raw' por 'estudiante_id' en cada fila. La cedula nunca llega al CSV final."""
    resultado = []
    for fila in filas:
        fila_anon = dict(fila)
        cedula = fila_anon.pop('cedula_raw')
        fila_anon['estudiante_id'] = mapa[cedula]
        resultado.append(fila_anon)
    return resultado
