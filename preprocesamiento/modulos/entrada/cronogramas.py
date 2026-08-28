"""Extraccion de la relacion semana -> tema a partir de los cronogramas de cada materia/trimestre.

Los cronogramas viven en dos formatos dentro de "Datos Tesis Upstream":
  - La mayoria son .docx con una tabla de columnas Tema | Semana | Actividad Evaluada | Fecha | Porcentaje.
  - El cronograma de Algoritmos y Programacion 2425-2 es un .xlsx con un formato propio: dos filas
    por semana (la fecha de cada dia de clase, y debajo el tema visto ese dia).
"""
import re
import docx
import pandas as pd


def parse_docx_cronograma(path):
    """Lee un cronograma .docx y devuelve {semana:int -> tema:str}.

    Cuando varias filas de la tabla comparten el mismo numero de semana (p. ej. dos
    temas dictados la misma semana), los textos se concatenan con ' | '.
    """
    documento = docx.Document(path)
    tabla = documento.tables[0]
    resultado = {}
    for fila in tabla.rows[1:]:
        celdas = [c.text.strip() for c in fila.cells]
        if len(celdas) < 2:
            continue
        tema, semana = celdas[0], celdas[1]
        m = re.search(r'\d+', semana)
        if not m:
            continue
        numero_semana = int(m.group())
        tema = re.sub(r'\s+', ' ', tema).strip()
        if not tema:
            continue
        if numero_semana in resultado:
            resultado[numero_semana] += " | " + tema
        else:
            resultado[numero_semana] = tema
    return resultado


def parse_xlsx_cronograma_alg2425_2(path, hoja):
    """Caso especial: cronograma de Algoritmos 2425-2, con dos filas por semana
    (fila de fechas, fila de temas por dia de clase) en vez de la tabla estandar.
    """
    df = pd.read_excel(path, sheet_name=hoja, header=None)
    resultado = {}
    i = 7  # primera fila de datos 'Semana N'; la fila 6 es el encabezado
    while i < len(df) - 1:
        semana_val = df.iat[i, 1]
        if pd.isna(semana_val):
            i += 1
            continue
        try:
            numero_semana = int(semana_val)
        except (ValueError, TypeError):
            i += 1
            continue
        fila_temas = df.iloc[i + 1]
        temas = []
        for col in (2, 3):
            v = fila_temas[col]
            if isinstance(v, str) and v.strip():
                temas.append(re.sub(r'\s+', ' ', v.replace('\n', ' ')).strip())
        resultado[numero_semana] = " | ".join(temas)
        i += 2
    return resultado


# (materia, trimestre) -> (formato, ruta relativa a "Datos Tesis Upstream")
_RUTAS_CRONOGRAMA = {
    ("Algoritmos y Programación", "2425-2"): ("xlsx_alg2425_2", "Algoritmos y Programacion/2425-2/cronograma alg 2425-2.xlsx"),
    ("Algoritmos y Programación", "2526-1"): ("docx", "Algoritmos y Programacion/2526-1/Cronograma 2526-1.docx"),
    ("Algoritmos y Programación", "2526-2"): ("docx", "Algoritmos y Programacion/2526-2/Cronograma 2526-2.docx"),
    ("Algoritmos y Programación", "2526-3"): ("docx", "Algoritmos y Programacion/2526-3/Cronograma 2526-3.docx"),
    ("Computación Emergente", "2526-1"): ("docx", "Computacion Emergente/2526-1/Cronograma FPTSP25 2526-1.docx"),
    ("Computación Emergente", "2526-2"): ("docx", "Computacion Emergente/2526-2/Cronograma FPTSP25 2526-2 (3).docx"),
    ("Estructuras de Datos", "2425-3"): ("docx", "Estructura de Datos/2425-3/Cronograma Estructuras de Datos 2425-3.docx"),
    ("Estructuras de Datos", "2526-1"): ("docx", "Estructura de Datos/2526-1/Cronograma Estructuras de Datos 2526-1.docx"),
    ("Estructuras de Datos", "2526-2"): ("docx", "Estructura de Datos/2526-2/Cronograma Estructuras de Datos 2526-2.docx"),
    ("Estructuras de Datos", "2526-3"): ("docx", "Estructura de Datos/2526-3/Cronograma Estructuras de Datos 2526-3.docx"),
}


def cargar_cronogramas(base_dir):
    """Parsea todos los cronogramas conocidos. Devuelve {(materia, trimestre): {semana: tema}}."""
    cronogramas = {}
    for clave, (formato, ruta) in _RUTAS_CRONOGRAMA.items():
        ruta_completa = base_dir / ruta
        if formato == "docx":
            cronogramas[clave] = parse_docx_cronograma(ruta_completa)
        else:
            cronogramas[clave] = parse_xlsx_cronograma_alg2425_2(ruta_completa, "Cronograma LyM")
    return cronogramas


def tema_for(cronogramas, materia, trimestre, semana):
    """Devuelve el tema de esa semana, o '' si no hay cronograma cargado o esa semana
    no tiene tema registrado (p. ej. semana de examen/feriado)."""
    tabla = cronogramas.get((materia, trimestre), {})
    if semana is None:
        return ""
    try:
        return tabla.get(int(semana), "")
    except (ValueError, TypeError):
        return ""
