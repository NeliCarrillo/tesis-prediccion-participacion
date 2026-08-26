"""Un extractor por cada formato de archivo fuente detectado en la Fase 1.

Los 11 archivos/hojas de participacion procesados caen en 3 formatos:

  1. process_simple_partic     - una fila por estudiante, una columna por fecha de sesion.
  2. process_hybrid_asistencia - columnas de sesion agrupadas bajo "Semana N", con codigo de
                                  asistencia (P/1/T/F/J) en vez de solo un numero, y columnas
                                  "prepa N" intercaladas que no son participacion.
  3. process_weekly_aggregated - ya viene agregado como Sem 1 ... Sem 12, sin fecha de sesion.

Un cuarto caso, process_estructura_2526_1, es una variante de (1) donde primero hay que
resolver la cedula cruzando por nombre contra otra hoja del mismo libro.

Todas las funciones devuelven la misma forma: (filas, notas_publicas, notas_confidenciales).
  - filas: lista de dicts con 'cedula_raw' (se anonimiza despues, ver anonimizacion.py) y el
    resto de las columnas de salida.
  - notas_publicas: mensajes de log sin ningun dato personal (cifras, nombres de columnas).
  - notas_confidenciales: mensajes que si pueden incluir nombres (solo se usa en el cruce
    por nombre de Estructuras 2526-1); se guardan aparte y nunca se suben al repositorio.
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..entrada.cronogramas import tema_for
from .limpieza import (
    norm_cedula,
    limpiar_artefacto_nombre,
    normalizar_para_cruce,
    lunes_de_la_semana,
    semana_desde_fecha,
    interpretar_codigo_asistencia,
)


def _fila_base(cedula, numero_lista, materia, trimestre, seccion, fecha, semana, cronogramas,
                participaciones, asistencia):
    return {
        'cedula_raw': cedula,
        'numero_lista': numero_lista,
        'materia': materia,
        'trimestre': trimestre,
        'seccion': seccion,
        'fecha': fecha,
        'semana': semana,
        'tema': tema_for(cronogramas, materia, trimestre, semana),
        'participaciones': participaciones,
        'tipo_participacion': '',
        'asistencia': asistencia,
    }


def process_simple_partic(path, hoja, materia, trimestre, seccion, cronogramas,
                           header_row=0, text_date_map=None):
    """Formato 1: encabezado con una columna de fecha por sesion de clase y, debajo,
    un numero (participaciones) o vacio por estudiante."""
    df = pd.read_excel(path, sheet_name=hoja, header=None)
    header = df.iloc[header_row]
    data = df.iloc[header_row + 1:].reset_index(drop=True)

    columnas_fecha = []
    for i, v in enumerate(header):
        if isinstance(v, (pd.Timestamp, datetime)):
            columnas_fecha.append((i, pd.Timestamp(v).to_pydatetime()))
        elif text_date_map and isinstance(v, str) and v.strip() in text_date_map:
            columnas_fecha.append((i, text_date_map[v.strip()]))

    idx_cedula = idx_id = None
    for i, v in enumerate(header):
        if isinstance(v, str) and v.strip().upper() in ('CEDULA', 'CÉDULA', 'CI'):
            idx_cedula = i
        if isinstance(v, str) and v.strip().upper() == 'ID':
            idx_id = i
    if idx_cedula is None:
        raise ValueError(f"No se encontro columna de cedula en {path} / {hoja}")
    if not columnas_fecha:
        raise ValueError(f"No se encontraron columnas de fecha en {path} / {hoja}")

    fechas_ordenadas = sorted(set(d for _, d in columnas_fecha))
    inicio_trimestre = lunes_de_la_semana(fechas_ordenadas[0])

    filas = []
    n_estudiantes = 0
    codigos_especiales = {}
    for _, fila in data.iterrows():
        cedula_cruda = fila[idx_cedula]
        if pd.isna(cedula_cruda):
            continue
        cedula = norm_cedula(cedula_cruda)
        if cedula is None:
            continue
        n_estudiantes += 1
        numero_lista = fila[idx_id] if idx_id is not None and not pd.isna(fila[idx_id]) else None

        for i, fecha in columnas_fecha:
            valor = fila[i]
            if pd.isna(valor):
                participaciones = 0.0
            elif isinstance(valor, (int, float)):
                participaciones = float(valor)
            else:
                participaciones = np.nan
                codigos_especiales[str(valor)] = codigos_especiales.get(str(valor), 0) + 1
            semana = semana_desde_fecha(fecha, inicio_trimestre)
            filas.append(_fila_base(cedula, numero_lista, materia, trimestre, seccion,
                                     fecha.date().isoformat(), semana, cronogramas,
                                     participaciones, np.nan))

    notas = [
        f"Sesiones detectadas: {len(columnas_fecha)} (rango {fechas_ordenadas[0].date()} a "
        f"{fechas_ordenadas[-1].date()}). Inicio de termino asumido (lunes semana 1): {inicio_trimestre.date()}.",
        f"Estudiantes procesados: {n_estudiantes}. Filas estudiante-fecha generadas: {len(filas)}.",
    ]
    if codigos_especiales:
        notas.append(f"ADVERTENCIA: celdas con valores no numericos {codigos_especiales} tratadas como NaN en 'participaciones'.")
    return filas, notas, []


def process_hybrid_asistencia(path, hoja, materia, trimestre, seccion, cronogramas):
    """Formato 2: columnas de sesion agrupadas bajo 'Semana N' (fila 0, con celdas combinadas
    -> hay que propagar el valor hacia adelante) con codigo P/1/T/F/J en vez de un numero,
    y columnas 'prepa N' intercaladas que no son participacion y se ignoran."""
    df = pd.read_excel(path, sheet_name=hoja, header=None)
    fila_grupo = df.iloc[0].tolist()
    fila_encabezado = df.iloc[1].tolist()
    data = df.iloc[2:].reset_index(drop=True)

    etiquetas_semana = []
    ultima = None
    for v in fila_grupo:
        if isinstance(v, str) and v.strip() and not v.strip().upper().startswith('P=PRESENTE'):
            ultima = v.strip()
        etiquetas_semana.append(ultima)

    idx_cedula = idx_id = None
    columnas_fecha = []
    for i, v in enumerate(fila_encabezado):
        if isinstance(v, str) and v.strip().upper() in ('CEDULA', 'CÉDULA'):
            idx_cedula = i
        if isinstance(v, str) and v.strip().upper() == 'ID':
            idx_id = i
        if isinstance(v, (datetime, pd.Timestamp)):
            etiqueta = etiquetas_semana[i]
            m = re.search(r'\d+', etiqueta) if etiqueta else None
            columnas_fecha.append((i, pd.Timestamp(v).to_pydatetime(), int(m.group()) if m else None))
    if idx_cedula is None:
        raise ValueError(f"No se encontro columna de cedula en {path} / {hoja}")

    filas = []
    n_estudiantes = 0
    notas_codigos = {}
    for _, fila in data.iterrows():
        cedula_cruda = fila[idx_cedula]
        if pd.isna(cedula_cruda):
            continue
        cedula = norm_cedula(cedula_cruda)
        if cedula is None:
            continue
        n_estudiantes += 1
        numero_lista = fila[idx_id] if idx_id is not None and not pd.isna(fila[idx_id]) else None

        for i, fecha, semana in columnas_fecha:
            participaciones, asistencia, nota = interpretar_codigo_asistencia(fila[i])
            if nota:
                notas_codigos[nota] = notas_codigos.get(nota, 0) + 1
            filas.append(_fila_base(cedula, numero_lista, materia, trimestre, seccion,
                                     fecha.date().isoformat(), semana, cronogramas,
                                     participaciones, asistencia))

    notas = [
        f"Estudiantes procesados: {n_estudiantes}. Columnas de sesion usadas: {len(columnas_fecha)} "
        f"(se ignoraron columnas 'prepa N', que son notas de preparaduria, no participacion).",
    ]
    notas.extend(f"{nota} (x{cnt} celdas)" for nota, cnt in notas_codigos.items())
    notas.append(f"Filas estudiante-fecha generadas: {len(filas)}.")
    return filas, notas, []


def process_weekly_aggregated(path, hoja, materia, trimestre, seccion, cronogramas, header_row=1):
    """Formato 3: ya viene agregado por semana (Sem 1 ... Sem 12), sin fecha de sesion
    individual. No se inventa una fecha: la columna 'fecha' queda vacia."""
    df = pd.read_excel(path, sheet_name=hoja, header=None)
    header = df.iloc[header_row].tolist()
    data = df.iloc[header_row + 1:].reset_index(drop=True)

    idx_cedula = None
    columnas_semana = []
    for i, v in enumerate(header):
        if isinstance(v, str) and v.strip().upper() in ('CEDULA', 'CÉDULA', 'CÉDULA DE IDENTIDAD'):
            idx_cedula = i
        if isinstance(v, str):
            m = re.match(r'^Sem\s*(\d+)$', v.strip(), re.IGNORECASE)
            if m:
                columnas_semana.append((i, int(m.group(1))))
    if idx_cedula is None:
        raise ValueError(f"No se encontro columna de cedula en {path} / {hoja}")

    filas = []
    n_estudiantes = 0
    codigos_especiales = {}
    for _, fila in data.iterrows():
        cedula_cruda = fila[idx_cedula]
        if pd.isna(cedula_cruda):
            continue
        cedula = norm_cedula(cedula_cruda)
        if cedula is None:
            continue
        n_estudiantes += 1

        for i, semana in columnas_semana:
            valor = fila[i]
            if pd.isna(valor):
                participaciones = 0.0
            elif isinstance(valor, (int, float)):
                participaciones = float(valor)
            else:
                participaciones = np.nan
                codigos_especiales[repr(valor)] = codigos_especiales.get(repr(valor), 0) + 1
            filas.append(_fila_base(cedula, None, materia, trimestre, seccion,
                                     '', semana, cronogramas, participaciones, np.nan))

    semanas = [w for _, w in columnas_semana]
    notas = [
        f"Estudiantes procesados: {n_estudiantes}. Semanas ya agregadas en el origen: "
        f"Sem {min(semanas)} a Sem {max(semanas)} ({len(columnas_semana)} columnas). Sin fecha de sesion "
        f"individual en el archivo fuente -> columna 'fecha' queda vacia, se mantiene la granularidad "
        f"semanal original.",
        f"Filas estudiante-semana generadas: {len(filas)}.",
    ]
    if codigos_especiales:
        notas.append(f"ADVERTENCIA: celdas con valores no numericos {codigos_especiales} tratadas como NaN en 'participaciones'.")
    return filas, notas, []


def process_estructura_2526_1(path, materia, trimestre, seccion, cronogramas):
    """Caso especial: la hoja 'Hoja 1' (participaciones por fecha) no trae cedula, asi que
    se cruza por nombre normalizado contra la hoja 'Totales' (que si la trae) del mismo libro."""
    h1 = pd.read_excel(path, sheet_name="Hoja 1", header=None)
    totales = pd.read_excel(path, sheet_name="Totales", header=None)

    header = h1.iloc[0].tolist()
    data = h1.iloc[1:].reset_index(drop=True)
    columnas_fecha = [(i, pd.Timestamp(v).to_pydatetime()) for i, v in enumerate(header)
                       if isinstance(v, (datetime, pd.Timestamp))]
    fechas_ordenadas = sorted(set(d for _, d in columnas_fecha))
    inicio_trimestre = lunes_de_la_semana(fechas_ordenadas[0])

    nombre_a_cedula = {}
    for _, fila in totales.iloc[1:].iterrows():
        nombre, apellido, cedula = fila[0], fila[1], fila[2]
        if pd.isna(nombre) or pd.isna(apellido) or pd.isna(cedula):
            continue
        nombre_a_cedula[normalizar_para_cruce(f"{nombre} {apellido}")] = norm_cedula(cedula)

    filas = []
    n_estudiantes = 0
    sin_match = []
    for _, fila in data.iterrows():
        nombre_crudo = fila[0]
        if pd.isna(nombre_crudo):
            continue
        nombre_limpio = limpiar_artefacto_nombre(str(nombre_crudo).strip())
        cedula = nombre_a_cedula.get(normalizar_para_cruce(nombre_limpio))
        if cedula is None:
            sin_match.append(nombre_limpio)
            continue
        n_estudiantes += 1

        for i, fecha in columnas_fecha:
            valor = fila[i]
            if pd.isna(valor):
                participaciones = 0.0
            elif isinstance(valor, (int, float)):
                participaciones = float(valor)
            else:
                participaciones = np.nan
            semana = semana_desde_fecha(fecha, inicio_trimestre)
            filas.append(_fila_base(cedula, None, materia, trimestre, seccion,
                                     fecha.date().isoformat(), semana, cronogramas,
                                     participaciones, np.nan))

    notas = [
        f"Sesiones detectadas: {len(columnas_fecha)} (rango {fechas_ordenadas[0].date()} a "
        f"{fechas_ordenadas[-1].date()}). Inicio de termino asumido: {inicio_trimestre.date()}.",
        "Cedula obtenida cruzando 'Hoja 1' (sin cedula) con 'Totales' (con cedula) por nombre normalizado.",
        f"Estudiantes cruzados exitosamente: {n_estudiantes}. Sin match de cedula (excluidos del CSV): {len(sin_match)}.",
        f"Filas estudiante-fecha generadas: {len(filas)}.",
    ]
    notas_confidenciales = []
    if sin_match:
        notas_confidenciales.append(f"Nombres de 'Hoja 1' sin match en 'Totales' (excluidos): {sin_match}")
    return filas, notas, notas_confidenciales
