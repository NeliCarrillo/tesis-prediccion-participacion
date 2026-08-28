"""Utilidades de limpieza reutilizadas por los distintos extractores (ver extractores.py).

Cubren los problemas de calidad de datos detectados en la Fase 1 sobre los archivos crudos:
cedulas guardadas con formatos distintos, nombres con iniciales pegadas por error, fechas de
sesion en texto libre, numero de semana no siempre disponible, y codigos de asistencia
(P/1/T/F/J) que no son simplemente un numero de participaciones.
"""
import re
import unicodedata
from datetime import timedelta

import numpy as np
import pandas as pd


def norm_cedula(valor):
    """Normaliza una cedula (que puede venir como float, int o string) a un string de digitos."""
    if pd.isna(valor):
        return None
    try:
        return str(int(float(valor)))
    except (ValueError, TypeError):
        digitos = re.sub(r'\D', '', str(valor))
        return digitos or None


# Varios archivos traen el nombre precedido por sus propias iniciales pegadas sin espacio,
# p. ej. "FBFidel Eduardo Barreat Lemoine" (F de Fidel, B de Barreat). Se detecta como
# 2 mayusculas seguidas de una mayuscula+minuscula (el arranque real del nombre).
_INICIALES_PEGADAS_RE = re.compile(r'^[A-ZÁÉÍÓÚÑ]{2}([A-ZÁÉÍÓÚÑ][a-zà-ÿ].*)$')


def limpiar_artefacto_nombre(nombre):
    """Quita el prefijo de iniciales pegadas cuando esta presente; si no, devuelve el nombre tal cual."""
    if not isinstance(nombre, str):
        return nombre
    nombre = nombre.strip()
    m = _INICIALES_PEGADAS_RE.match(nombre)
    return m.group(1) if m else nombre


def _quitar_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


def normalizar_para_cruce(nombre):
    """Normaliza un nombre completo (sin acentos, minusculas, sin dobles espacios) para
    poder cruzarlo por nombre entre dos hojas que no comparten un identificador comun
    (caso de Estructuras de Datos 2526-1, ver extractores.process_estructura_2526_1)."""
    nombre = limpiar_artefacto_nombre(str(nombre))
    nombre = _quitar_acentos(nombre).lower()
    nombre = re.sub(r'[^a-z ]', ' ', nombre)
    return re.sub(r'\s+', ' ', nombre).strip()


def lunes_de_la_semana(fecha):
    return fecha - timedelta(days=fecha.weekday())


def semana_desde_fecha(fecha, inicio_trimestre):
    """Numero de semana del trimestre (1-indexado), asumiendo que 'inicio_trimestre' es el
    lunes de la semana 1. Ver README.md para la justificacion de este metodo."""
    return (fecha.date() - inicio_trimestre.date()).days // 7 + 1


# Emojis usados en algunas hojas de 'participaciones' en vez de un numero. Confirmados
# por el autor de la tesis; ambos implican una ausencia justificada (no se sabe la fecha
# exacta de reincorporacion, asi que se cuentan en 0 participaciones ese dia).
CODIGOS_AUSENCIA_JUSTIFICADA = {
    '⚕️': 'reposo medico',
    '⚖️': 'caso legal/gubernamental',
}


def interpretar_valor_participacion(valor):
    """Traduce una celda de una hoja 'simple' (sin codigo de asistencia, solo numero o
    codigo de ausencia justificada) a (participaciones, nota)."""
    if pd.isna(valor):
        return 0.0, None
    if isinstance(valor, (int, float)):
        return float(valor), None
    codigo = str(valor).strip()
    if codigo in CODIGOS_AUSENCIA_JUSTIFICADA:
        return 0.0, f"codigo {codigo!r} ({CODIGOS_AUSENCIA_JUSTIFICADA[codigo]}) tratado como ausencia justificada, participaciones=0"
    return np.nan, f"codigo especial no reconocido {codigo!r} tratado como NaN"


def interpretar_codigo_asistencia(valor):
    """Traduce el codigo de asistencia usado en los archivos 'hibridos' (P/1/T/F/J/numero)
    a (participaciones, asistencia, nota). 'nota' es None salvo que el codigo merezca
    documentarse explicitamente en el log (incluye codigos no reconocidos, nunca se adivina
    su significado en silencio)."""
    if pd.isna(valor):
        return 0.0, 0, None
    if isinstance(valor, (int, float)):
        return float(valor), 1, None

    codigo = str(valor).strip()
    if codigo in ('P', 'p'):
        # 'p' minusculo confirmado como typo de 'P', mismo significado.
        return 0.0, 1, None
    if codigo == 'T':
        return 0.0, 1, None
    if codigo == 'F':
        return 0.0, 1, "codigo 'F' (Fraude) tratado como asistencia=1, participaciones=0"
    if codigo == 'J':
        # Confirmado por el autor de la tesis: el estudiante asistio pero se retiro antes
        # de terminar la clase -> se cuenta como presente, sin participaciones ese dia.
        return 0.0, 1, ("codigo 'J' (jubilado): estudiante asistio pero se retiro antes de "
                         "terminar la clase -> asistencia=1, participaciones=0")
    if codigo in CODIGOS_AUSENCIA_JUSTIFICADA:
        return 0.0, 0, (f"codigo {codigo!r} ({CODIGOS_AUSENCIA_JUSTIFICADA[codigo]}) tratado como "
                         f"ausencia justificada: asistencia=0, participaciones=0")
    if re.match(r'^\d+(\.\d+)?$', codigo):
        return float(codigo), 1, None

    return np.nan, np.nan, f"codigo especial no reconocido {codigo!r} tratado como NaN"
