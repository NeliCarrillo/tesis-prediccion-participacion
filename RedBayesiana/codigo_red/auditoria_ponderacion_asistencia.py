"""Auditoría de ponderación/balanceo y de asistencia — tareas del Product
Backlog pedidas por el tutor.

Es un script de AUDITORÍA, no de pipeline: no modifica ni la LSTM ni la red
bayesiana, no genera datos nuevos para entrenamiento. Reproduce, con
evidencia del código y de los datos, los hallazgos usados para el apéndice
de ponderación y para el apartado de asistencia del informe.

Fuentes:
- Datos Tesis Downstream/  (11.700 filas estudiante-sesión, las que usan
  ambos modelos)
- Datos Tesis Upstream/    (planillas originales de los profesores, con las
  hojas de asistencia que nunca llegaron al pipeline estandarizado)

No requiere pgmpy ni TensorFlow — solo pandas/openpyxl.
"""
from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
CLAVE = ["materia", "trimestre", "seccion", "estudiante_id"]
NOTEBOOK_LSTM_ADAPTADO = RAIZ / "LSTM" / "nuevo" / "adaptacion_lstm_participaciones.ipynb"
NOTEBOOK_LSTM_ORIGINAL = RAIZ / "LSTM" / "viejo" / "lstm_computacion_emergente.ipynb"
GENERAR_CSV = RAIZ / "preprocesamiento" / "generar_csv.py"


def cargar_datos_crudos() -> pd.DataFrame:
    archivos = sorted(glob.glob(str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True))
    datos = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    datos["participaciones"] = pd.to_numeric(datos["participaciones"], errors="coerce")
    return datos


# ======================================================================
# 1. PONDERACIÓN Y BALANCEO
# ======================================================================

def tabla_tamanos(datos: pd.DataFrame) -> pd.DataFrame:
    """Registros por materia x trimestre x sección — la unidad que, sin
    ponderación, determina cuánto pesa cada grupo en el entrenamiento."""
    return (
        datos.groupby(["materia", "trimestre", "seccion"])["estudiante_id"]
        .nunique()
        .rename("n_estudiantes")
        .reset_index()
    )


def verificar_lstm_sin_ponderacion() -> dict:
    """Verifica, sobre el código real del notebook adaptado (no sobre lo que
    se recuerde de él), que el ajuste no usa `sample_weight` y que la
    versión de regresión no usa `class_weight`. Cuenta también las
    ocurrencias de `class_weight` en el notebook original, para dejar
    registrada su procedencia (el modelo de clasificación del que se
    adaptó, no el actual)."""
    def contar_en_celdas_de_codigo(ruta: Path, patron: str) -> int:
        nb = json.loads(ruta.read_text())
        return sum(
            "".join(c["source"]).count(patron)
            for c in nb["cells"]
            if c["cell_type"] == "code"
        )

    apariciones_sample_weight_adaptado = contar_en_celdas_de_codigo(
        NOTEBOOK_LSTM_ADAPTADO, "sample_weight"
    )
    apariciones_class_weight_adaptado = contar_en_celdas_de_codigo(
        NOTEBOOK_LSTM_ADAPTADO, "class_weight"
    )
    apariciones_class_weight_original = contar_en_celdas_de_codigo(
        NOTEBOOK_LSTM_ORIGINAL, "class_weight"
    )

    return {
        "sample_weight_en_notebook_adaptado_celdas_codigo": apariciones_sample_weight_adaptado,
        "class_weight_en_notebook_adaptado_celdas_codigo": apariciones_class_weight_adaptado,
        "class_weight_en_notebook_original_celdas_codigo": apariciones_class_weight_original,
    }


def asimetria_ce_2425_3(sesiones_registro: pd.DataFrame) -> dict:
    """Aísla y cuantifica la única excepción a las 24 sesiones por registro:
    Computación Emergente, trimestre 2425-3."""
    fila = sesiones_registro.loc[("Computación Emergente", "2425-3")]
    return {
        "materia": "Computación Emergente",
        "trimestre": "2425-3",
        "registros_afectados": int(fila["count"]),
        "sesiones_por_registro": int(fila["max"]),
        "sesiones_por_registro_resto_del_estudio": 24,
        "peso_relativo_frente_al_resto": round(fila["max"] / 24, 2),
    }


def sesiones_por_registro(datos: pd.DataFrame) -> pd.DataFrame:
    """Sesiones por registro estudiante-sección, por materia x trimestre —
    esto es lo que determina el peso IMPLÍCITO de cada registro en el ajuste
    de CPD de la red bayesiana (que cuenta a nivel de sesión, no de
    registro): un registro con más sesiones aporta más conteos a los nodos
    que le son propios (Año que cursa, Sección, Tamaño del grupo, objetivo),
    sin que nadie lo haya decidido como una ponderación."""
    n = datos.groupby(CLAVE).size().rename("n_sesiones").reset_index()
    return n.groupby(["materia", "trimestre"])["n_sesiones"].agg(["min", "max", "mean", "count"])


# ======================================================================
# 2. ASISTENCIA
# ======================================================================

def inventario_hojas_asistencia() -> list[dict]:
    """Recorre Datos Tesis Upstream/ y reporta, para cada archivo con una
    hoja de asistencia real, cuántas celdas (estudiante x sesión) tienen
    algún valor registrado. No toca ningún archivo confidencial de
    estudiantes (no abre mapeo_estudiantes.csv)."""
    import datetime as dt

    resultados = []
    archivos = sorted(glob.glob(str(RAIZ / "Datos Tesis Upstream" / "**" / "*.xlsx"), recursive=True))
    archivos = [a for a in archivos if "_procesado" not in a]

    for ruta in archivos:
        xl = pd.ExcelFile(ruta)
        for hoja in xl.sheet_names:
            if "asistencia" not in hoja.lower() and "sec" not in hoja.lower():
                continue
            df = xl.parse(hoja, header=None)
            if df.empty or df.shape[0] < 3:
                continue
            fila_header = 1
            header = df.iloc[fila_header]
            cols_fecha = [c for c in df.columns if isinstance(header[c], (pd.Timestamp, dt.datetime))]
            if not cols_fecha:
                continue
            col_cedula = 0 if "asistencia" in hoja.lower() else 1
            filas = df.iloc[2:]
            filas = filas[pd.to_numeric(filas[col_cedula], errors="coerce").notna()]
            n_estudiantes = len(filas)
            n_celdas = n_estudiantes * len(cols_fecha)
            llenas = int(filas[cols_fecha].notna().sum().sum()) if n_celdas else 0
            resultados.append({
                "archivo": str(Path(ruta).relative_to(RAIZ)),
                "hoja": hoja,
                "n_estudiantes": n_estudiantes,
                "n_sesiones_con_fecha": len(cols_fecha),
                "n_celdas": n_celdas,
                "celdas_llenas": llenas,
                "pct_lleno": round(100 * llenas / n_celdas, 1) if n_celdas else 0.0,
            })
    return resultados


def verificar_asistencia_fuera_del_pipeline(datos: pd.DataFrame) -> dict:
    """Confirma, sobre el código y los datos reales, que la asistencia
    nunca entró al pipeline estandarizado: ni como lógica en
    `generar_csv.py`, ni como columna en `Datos Tesis Downstream/`."""
    codigo = GENERAR_CSV.read_text(encoding="utf-8", errors="ignore")
    return {
        "menciones_asistencia_en_generar_csv": codigo.lower().count("asistencia"),
        "columna_asistencia_en_datos_downstream": "asistencia" in [c.lower() for c in datos.columns],
        "columnas_reales_datos_downstream": sorted(datos.columns.tolist()),
    }


def verificar_sin_corrida_historica_asistencia() -> dict:
    """Recorre TODO el historial de git buscando 'asistencia'. No asume que
    no existió una corrida histórica: lista cada commit que la menciona
    para que quede documentado qué es, en vez de solo afirmarlo."""
    salida = subprocess.run(
        ["git", "log", "--all", "--oneline", "-i", "--grep=asistencia"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout.strip()
    commits_por_mensaje = [
        {"commit": linea.split(" ", 1)[0], "mensaje": linea.split(" ", 1)[1]}
        for linea in salida.splitlines() if linea
    ]

    salida2 = subprocess.run(
        ["git", "log", "--all", "--oneline", "-S", "asistencia"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout.strip()
    commits_que_tocan_codigo_con_asistencia = [
        {"commit": linea.split(" ", 1)[0], "mensaje": linea.split(" ", 1)[1]}
        for linea in salida2.splitlines() if linea
    ]

    return {
        "commits_con_asistencia_en_el_mensaje": commits_por_mensaje,
        "commits_que_agregan_o_quitan_la_palabra_asistencia_en_algun_archivo": commits_que_tocan_codigo_con_asistencia,
    }


def asimetria_participo_implica_asistio(datos: pd.DataFrame) -> dict:
    """Cuantifica por qué 'si participó, asistió' no basta para reconstruir
    una variable de asistencia utilizable: solo etiqueta el caso
    participaciones>0 (asistencia=1); el caso, dominante, de
    participaciones==0 queda sin poder distinguirse entre ausente y
    presente-sin-participar."""
    con_participacion = datos["tipo_sesion"].notna() & datos["participaciones"].notna()
    base = datos[con_participacion]
    total = len(base)
    mayor_cero = int((base["participaciones"] > 0).sum())
    igual_cero = int((base["participaciones"] == 0).sum())
    return {
        "total_filas_con_dato_de_participacion": total,
        "participaciones_mayor_a_cero_asistencia_inferible": mayor_cero,
        "pct_asistencia_inferible": round(100 * mayor_cero / total, 1),
        "participaciones_igual_a_cero_asistencia_ambigua": igual_cero,
        "pct_asistencia_ambigua": round(100 * igual_cero / total, 1),
    }


if __name__ == "__main__":
    datos = cargar_datos_crudos()

    print("=" * 70)
    print("1. PONDERACIÓN Y BALANCEO")
    print("=" * 70)
    tam = tabla_tamanos(datos)
    print(tam.to_string(index=False))
    print(f"\nmin={tam.n_estudiantes.min()} max={tam.n_estudiantes.max()} "
          f"media={tam.n_estudiantes.mean():.1f} std={tam.n_estudiantes.std():.1f}")

    sxr = sesiones_por_registro(datos)
    print("\nSesiones por registro estudiante-sección, por materia x trimestre:")
    print(sxr.to_string())

    print("\nAsimetría CE 2425-3 (la única excepción a 24 sesiones por registro):")
    for k, v in asimetria_ce_2425_3(sxr).items():
        print(f"  {k}: {v}")

    print("\nVerificación de ponderación LSTM (sobre el código real del notebook):")
    for k, v in verificar_lstm_sin_ponderacion().items():
        print(f"  {k}: {v}")

    print()
    print("=" * 70)
    print("2. ASISTENCIA — inventario de hojas reales en Datos Tesis Upstream/")
    print("=" * 70)
    inventario = inventario_hojas_asistencia()
    if inventario:
        print(pd.DataFrame(inventario).to_string(index=False))
    else:
        print("(sin hojas de asistencia detectadas)")

    print("\nAsistencia fuera del pipeline estandarizado:")
    for k, v in verificar_asistencia_fuera_del_pipeline(datos).items():
        print(f"  {k}: {v}")

    print("\nAsimetría de 'participó => asistió' (sobre las filas con dato de participación):")
    for k, v in asimetria_participo_implica_asistio(datos).items():
        print(f"  {k}: {v}")

    print("\nBúsqueda de corridas históricas con asistencia (todo el historial de git):")
    historial = verificar_sin_corrida_historica_asistencia()
    for k, v in historial.items():
        print(f"  {k}: {v}")
