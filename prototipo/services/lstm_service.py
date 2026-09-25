"""Servicio de inferencia LSTM del prototipo — Sprint 5, tarjeta 2.

Las funciones `cargar_datos`, `codificar_campos_cualitativos`,
`agregar_por_semana`, `construir_tabla_estatica`, `construir_tensor_3d` y
`truncar_a_hito` (junto con `CLAVES`, `ESTATICAS`, `DINAMICAS`,
`N_SEMANAS`, `N_TEMAS`, `HITOS`) están copiadas **verbatim** — extraídas
programáticamente, no retranscritas a mano — de las celdas 5, 8, 11, 14,
17 y 20 de `LSTM/nuevo/adaptacion_lstm_participaciones.ipynb`. No se
modificó ni una línea: es la misma preparación de datos que usa el
pipeline validado de Sprint 2, para garantizar que este servicio construye
exactamente la misma representación que el notebook, no una aproximación.
La prueba de paridad que lo confirma vive en
`prototipo/services/test_paridad_lstm.py`.

Los modelos que este servicio carga NO son los de la validación cruzada
de Sprint 2: son los modelos finales del prototipo (Estrategia 2, Sprint
5), entrenados una vez con todos los trimestres disponibles de cada
asignatura. Una predicción de este servicio sobre un registro histórico
no debe interpretarse como una nueva medición de desempeño — ver
`prototipo/README.md`.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_DATOS = RAIZ / "Datos Tesis Downstream"
RUTA_ARTEFACTOS = RAIZ / "prototipo" / "artefactos_lstm"
RUTA_PREDICCIONES_VALIDACION = RAIZ / "LSTM" / "nuevo" / "predicciones_lstm_validacion_cruzada.csv"

# Debe coincidir con el diccionario usado al entrenar los modelos finales
# (ver el historial de esta sesión: script de entrenamiento de Sprint 5).
SLUGS = {
    "Algoritmos y Programación": "algoritmos_y_programacion",
    "Computación Emergente": "computacion_emergente",
    "Estructura de Datos": "estructura_de_datos",
    "Matemáticas Discretas": "matematicas_discretas",
}

# ============================================================
# A partir de aquí: extracción verbatim de las celdas 5/8/11/14/17/20 del
# notebook. No editar estas funciones para "mejorarlas" — cualquier cambio
# de comportamiento debe hacerse primero en el notebook y volver a
# extraerse, para no divergir del pipeline validado.
# ============================================================


def cargar_datos(carpeta):
    """Cargar y concatenar los archivos de todas las secciones.

    Parameters
    ----------
    carpeta : str or pathlib.Path
        Carpeta que contiene los archivos .csv, organizados por asignatura y trimestre.

    Returns
    -------
    pandas.DataFrame
        Tabla única con una fila por estudiante y sesión.
    """
    archivos = sorted(glob.glob(f"{carpeta}/**/*.csv", recursive=True))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .csv en {carpeta}")
    tablas = [pd.read_csv(a) for a in archivos]
    return pd.concat(tablas, ignore_index=True)


def codificar_campos_cualitativos(datos):
    """Convertir a valores numéricos los campos que la estructura estándar guarda como texto.

    El trimestre se convierte en un ordinal relativo al primero observado; el día de la
    sesión, en su posición dentro de la semana; la asignatura y el estudiante, en un
    identificador entero; y el tipo de sesión, en una variable indicadora por categoría.

    Parameters
    ----------
    datos : pandas.DataFrame
        Tabla con una fila por estudiante y sesión.

    Returns
    -------
    pandas.DataFrame
        La misma tabla con las columnas numéricas añadidas.
    """
    d = datos.copy()

    # Ordinales: conservan el orden del período y de la sesión dentro de la semana
    trimestres = sorted(d.trimestre.unique())
    d["trimestre_num"] = d.trimestre.map({t: i + 1 for i, t in enumerate(trimestres)})
    d["dia_num"] = d.dia_sesion.map({"lunes": 1, "martes": 1, "miercoles": 2, "jueves": 2})

    # Nominales de segmentación: distinguen sin ordenar y no entran como variable
    materias = sorted(d.materia.unique())
    d["materia_num"] = d.materia.map({m: i + 1 for i, m in enumerate(materias)})
    d["seccion_num"] = pd.to_numeric(d.seccion, errors="coerce")

    # Nominal sin orden: una variable indicadora por categoría
    for categoria in sorted(d.tipo_sesion.dropna().unique()):
        d[f"sesion_{categoria}"] = (d.tipo_sesion == categoria).astype(int)

    return d


CLAVES = ["materia", "trimestre", "seccion", "estudiante_id"]


def agregar_por_semana(datos):
    """Resumir las sesiones de cada semana en una única fila por estudiante y semana.

    Parameters
    ----------
    datos : pandas.DataFrame
        Tabla con una fila por estudiante y sesión, ya codificada.

    Returns
    -------
    pandas.DataFrame
        Tabla con una fila por estudiante y semana del trimestre, con los rasgos
        numéricos y el código de tema de la primera y de la segunda sesión.
    """
    # El tema solo cuenta en las sesiones de contenido; en las demás vale 0
    d = datos.copy()
    d["tema_contenido"] = np.where(d.tipo_sesion == "contenido", d.tema.fillna(0), 0).astype(int)

    semanal = d.groupby(CLAVES + ["semana"], as_index=False).agg(
        participaciones=("participaciones", "sum"),
        sesiones=("dia_num", "size"),
        evaluaciones=("sesion_evaluacion", "sum"),
    )

    # Tema de cada sesión, ordenadas por su posición en la semana
    temas = (d.sort_values("dia_num")
             .groupby(CLAVES + ["semana"]).tema_contenido
             .apply(lambda s: list(s)[:2] + [0] * (2 - len(list(s)[:2])))
             .apply(pd.Series))
    temas.columns = ["tema_1", "tema_2"]
    return semanal.merge(temas.reset_index(), on=CLAVES + ["semana"], how="left")


def construir_tabla_estatica(datos):
    """Reunir en una fila por estudiante-sección las variables que no varían en el trimestre.

    Parameters
    ----------
    datos : pandas.DataFrame
        Tabla con una fila por estudiante y sesión, ya codificada.

    Returns
    -------
    pandas.DataFrame
        Tabla con una fila por registro estudiante-sección.
    """
    estatica = datos.groupby(CLAVES, as_index=False).agg(
        anio_academico=("anio_academico", "first"),
        seccion_num=("seccion_num", "first"),
        numero_lista=("numero_lista", "first"),
        trimestre_num=("trimestre_num", "first"),
        materia_num=("materia_num", "first"),
    )

    # El tamaño del grupo es una propiedad de la sección, no del estudiante
    tamano = (datos.groupby(["materia", "trimestre", "seccion"]).estudiante_id
              .nunique().rename("tamano_grupo").reset_index())
    estatica = estatica.merge(tamano, on=["materia", "trimestre", "seccion"], how="left")
    # Posición relativa en la lista: comparable entre secciones de distinto tamaño
    estatica["posicion_lista"] = estatica.numero_lista / estatica.tamano_grupo

    return estatica.sort_values(CLAVES).reset_index(drop=True)


ESTATICAS = ["anio_academico", "seccion_num", "tamano_grupo", "posicion_lista"]
DINAMICAS = ["participaciones", "sesiones", "evaluaciones"]
N_SEMANAS = 12
N_TEMAS = 14          # códigos 0 a 13 del catálogo (Estructura de Datos fija el máximo); el 0 señala ausencia de contenido


def construir_tensor_3d(semanal, estatica):
    """Construir los tensores de entrada y el vector objetivo.

    Cada registro estudiante-sección ocupa una fila y se despliega en doce pasos
    temporales. El tensor numérico replica las variables estáticas y coloca los rasgos
    de la semana; el tensor de temas coloca los códigos de sus dos sesiones. El objetivo
    es la suma de participaciones del trimestre.

    Parameters
    ----------
    semanal : pandas.DataFrame
        Tabla con una fila por estudiante y semana.
    estatica : pandas.DataFrame
        Tabla con una fila por registro estudiante-sección.

    Returns
    -------
    tuple
        (tensor numérico (registros, semanas, características),
         tensor de temas (registros, semanas, 2), vector objetivo).
    """
    n_estaticas = len(ESTATICAS)
    numerico = np.zeros((len(estatica), N_SEMANAS, n_estaticas + len(DINAMICAS)), dtype=np.float32)
    temas = np.zeros((len(estatica), N_SEMANAS, 2), dtype=np.int32)

    # Replicación de las variables estáticas en cada paso temporal
    numerico[:, :, :n_estaticas] = estatica[ESTATICAS].to_numpy(dtype=np.float32)[:, None, :]

    # Colocación de los rasgos y los temas de cada semana en su paso
    indice = {clave: i for i, clave in
              enumerate(estatica[CLAVES].itertuples(index=False, name=None))}
    for fila in semanal.itertuples(index=False):
        i = indice[(fila.materia, fila.trimestre, fila.seccion, fila.estudiante_id)]
        paso = int(fila.semana) - 1
        numerico[i, paso, n_estaticas:] = (fila.participaciones, fila.sesiones,
                                           fila.evaluaciones)
        temas[i, paso] = (fila.tema_1, fila.tema_2)

    # Objetivo: participaciones acumuladas en todo el trimestre
    objetivo = (semanal.groupby(CLAVES).participaciones.sum()
                .reindex(list(indice)).to_numpy(dtype=np.float32))

    return numerico, temas, objetivo


HITOS = [4, 6, 8]


def truncar_a_hito(tensor, hito):
    """Conservar únicamente las semanas transcurridas hasta el hito.

    Parameters
    ----------
    tensor : numpy.ndarray
        Tensor 3D completo, de doce pasos temporales (numérico o de temas).
    hito : int
        Número de la semana hasta la que se observa, inclusive.

    Returns
    -------
    numpy.ndarray
        Tensor truncado, de forma (registros, hito, ...).
    """
    return tensor[:, :hito]


# ============================================================
# A partir de aquí: código propio del prototipo (Sprint 5). No proviene
# del notebook.
# ============================================================

_CACHE_DATOS: dict | None = None
_CACHE_ARTEFACTOS: dict[tuple[str, int], dict] = {}


def _preparar_datos() -> dict:
    """Ejecuta la preparación completa (celdas 5-18 del notebook) una sola
    vez y guarda el resultado en memoria para las siguientes llamadas."""
    global _CACHE_DATOS
    if _CACHE_DATOS is not None:
        return _CACHE_DATOS

    datos = cargar_datos(RUTA_DATOS)
    datos = codificar_campos_cualitativos(datos)
    semanal = agregar_por_semana(datos)
    estatica = construir_tabla_estatica(datos)
    X, X_temas, y = construir_tensor_3d(semanal, estatica)

    _CACHE_DATOS = {"estatica": estatica, "X": X, "X_temas": X_temas, "y": y}
    return _CACHE_DATOS


def construir_entrada_lstm(caso) -> tuple[np.ndarray, np.ndarray, float]:
    """A partir de un `CasoPrediccion`, localiza su fila en `estatica` y
    devuelve `(X_hito, X_temas_hito, acumulado)` — el tensor numérico y de
    temas truncados al hito del caso (con la forma (1, hito, …) que espera
    el modelo), y el acumulado de participaciones observado hasta ese
    hito (necesario para reconstruir la predicción final). No aplica
    imputación ni escalado — eso lo hace `predict_lstm`, que ya conoce los
    parámetros congelados del modelo de esa asignatura/hito."""
    datos = _preparar_datos()
    estatica = datos["estatica"]

    filtro = (
        (estatica.materia == caso.materia)
        & (estatica.trimestre == caso.trimestre)
        & (estatica.seccion.astype(str) == str(caso.seccion))
        & (estatica.estudiante_id == caso.estudiante_id)
    )
    indices = np.where(filtro.to_numpy())[0]
    if len(indices) != 1:
        raise ValueError(
            f"se esperaba exactamente 1 registro para {caso}, hay {len(indices)}"
        )
    i = indices[0]

    X_hito = truncar_a_hito(datos["X"][i : i + 1], caso.hito)
    X_temas_hito = truncar_a_hito(datos["X_temas"][i : i + 1], caso.hito)
    acumulado = float(X_hito[:, :, len(ESTATICAS)].sum(axis=1)[0])
    return X_hito, X_temas_hito, acumulado


def _cargar_artefactos(materia: str, hito: int) -> dict:
    clave = (materia, hito)
    if clave in _CACHE_ARTEFACTOS:
        return _CACHE_ARTEFACTOS[clave]

    slug = SLUGS[materia]
    carpeta = RUTA_ARTEFACTOS / f"{slug}_h{hito}"
    if not carpeta.exists():
        raise FileNotFoundError(
            f"no hay modelo final para {materia!r} / hito {hito} en {carpeta}"
        )

    modelo = tf.keras.models.load_model(carpeta / "modelo.keras")
    escalador = json.load(open(carpeta / "escalador_x.json"))
    objetivo = json.load(open(carpeta / "objetivo.json"))
    imputacion = json.load(open(carpeta / "imputacion.json"))

    artefactos = {
        "modelo": modelo,
        "mean": np.array(escalador["mean"], dtype=np.float32),
        "scale": np.array(escalador["scale"], dtype=np.float32),
        "media_objetivo": objetivo["media_objetivo"],
        "desviacion_objetivo": objetivo["desviacion_objetivo"],
        "mediana_anio_academico": imputacion["mediana_anio_academico"],
    }
    _CACHE_ARTEFACTOS[clave] = artefactos
    return artefactos


@dataclass(frozen=True)
class ResultadoLSTM:
    materia: str
    trimestre: str
    seccion: str
    estudiante_id: str
    hito: int
    prediccion_total: float


def _predecir_desde_tensor(
    materia: str, hito: int, X_hito: np.ndarray, X_temas_hito: np.ndarray, acumulado: float
) -> float:
    """Imputación de año académico, escalado y reconstrucción del total —
    el único cuerpo de esta transformación, para no duplicarlo entre
    `predict_lstm` (estudiante real) y `predict_lstm_manual` (estudiante
    hipotético): ambos terminan en un tensor con esta misma forma."""
    indice_anio = ESTATICAS.index("anio_academico")
    artefactos = _cargar_artefactos(materia, hito)

    X_hito = X_hito.copy()
    if np.isnan(X_hito[:, :, indice_anio]).any():
        mediana = artefactos["mediana_anio_academico"]
        if mediana is None:
            raise ValueError(
                f"año académico faltante pero el modelo de "
                f"{materia}/hito {hito} no registró una mediana de imputación"
            )
        X_hito[:, :, indice_anio] = np.where(
            np.isnan(X_hito[:, :, indice_anio]), mediana, X_hito[:, :, indice_anio]
        )

    pasos, caracteristicas = X_hito.shape[1], X_hito.shape[2]
    X_escalado = ((X_hito.reshape(-1, caracteristicas) - artefactos["mean"]) / artefactos["scale"])
    X_escalado = X_escalado.reshape(-1, pasos, caracteristicas)

    prediccion_estandarizada = artefactos["modelo"].predict(
        [X_escalado, X_temas_hito], verbose=0
    ).ravel()[0]
    return (
        acumulado
        + float(prediccion_estandarizada) * artefactos["desviacion_objetivo"]
        + artefactos["media_objetivo"]
    )


def predict_lstm(caso) -> ResultadoLSTM:
    """Predicción puntual del total trimestral para `caso`, usando el
    modelo final de su asignatura/hito (Estrategia 2 — entrenado con
    todos los trimestres disponibles, no con la validación cruzada de
    Sprint 2). No debe usarse para calcular métricas de desempeño."""
    X_hito, X_temas_hito, acumulado = construir_entrada_lstm(caso)
    total = _predecir_desde_tensor(caso.materia, caso.hito, X_hito, X_temas_hito, acumulado)

    return ResultadoLSTM(
        materia=caso.materia,
        trimestre=caso.trimestre,
        seccion=caso.seccion,
        estudiante_id=caso.estudiante_id,
        hito=caso.hito,
        prediccion_total=total,
    )


def _fila_referencia_seccion(materia: str, trimestre: str, seccion: str) -> int:
    """Índice de cualquier estudiante ya existente en una sección real
    (materia/trimestre/sección). Se usa solo para leer hechos de horario
    de esa sección (sesiones, evaluaciones, temas de cada semana), nunca
    para leer nada propio de un estudiante en particular."""
    datos = _preparar_datos()
    estatica = datos["estatica"]
    filtro = (
        (estatica.materia == materia)
        & (estatica.trimestre == trimestre)
        & (estatica.seccion.astype(str) == str(seccion))
    )
    indices = np.where(filtro.to_numpy())[0]
    if len(indices) == 0:
        raise ValueError(f"no existe la sección {materia}/{trimestre}/sección {seccion}")
    return int(indices[0])


def horario_real_seccion(materia: str, trimestre: str, seccion: str, hito: int) -> dict:
    """Hechos reales de la sección (tamaño del grupo, y calendario de
    sesiones/evaluaciones/temas por semana hasta el hito) -- los mismos
    que ya usa `construir_entrada_lstm_manual` para el estudiante nuevo,
    aquí solo expuestos para mostrárselos al usuario (qué queda "fijado"
    por la sección real, en vez de inventado)."""
    datos = _preparar_datos()
    i = _fila_referencia_seccion(materia, trimestre, seccion)
    fila = datos["estatica"].iloc[i]

    n_estaticas = len(ESTATICAS)
    indice_sesiones = n_estaticas + DINAMICAS.index("sesiones")
    indice_evaluaciones = n_estaticas + DINAMICAS.index("evaluaciones")

    semanas = []
    for semana in range(hito):
        semanas.append({
            "semana": semana + 1,
            "sesiones": float(datos["X"][i, semana, indice_sesiones]),
            "evaluaciones": float(datos["X"][i, semana, indice_evaluaciones]),
            "tema_1": int(datos["X_temas"][i, semana, 0]),
            "tema_2": int(datos["X_temas"][i, semana, 1]),
        })

    return {
        "tamano_grupo": float(fila["tamano_grupo"]),
        "semanas": semanas,
    }


def valores_reales_estudiante(caso) -> dict:
    """Valores reales (año académico, posición en la lista, participaciones
    semana a semana hasta el hito) de un estudiante histórico -- para
    precargar el formulario de «estudiante nuevo» como plantilla editable
    (nunca para mezclarlos con una predicción hipotética sin que el
    usuario los vea y pueda modificarlos)."""
    datos = _preparar_datos()
    estatica = datos["estatica"]
    filtro = (
        (estatica.materia == caso.materia)
        & (estatica.trimestre == caso.trimestre)
        & (estatica.seccion.astype(str) == str(caso.seccion))
        & (estatica.estudiante_id == caso.estudiante_id)
    )
    indices = np.where(filtro.to_numpy())[0]
    if len(indices) != 1:
        raise ValueError(f"se esperaba exactamente 1 registro para {caso}, hay {len(indices)}")
    i = indices[0]
    fila = estatica.iloc[i]

    indice_anio = ESTATICAS.index("anio_academico")
    indice_participaciones = len(ESTATICAS) + DINAMICAS.index("participaciones")

    anio = datos["X"][i, 0, indice_anio]  # estática: igual en cualquier semana
    participaciones = datos["X"][i, :caso.hito, indice_participaciones].tolist()

    return {
        "anio_academico": None if np.isnan(anio) else float(anio),
        "posicion_lista": float(fila["posicion_lista"]),
        "participaciones_semanales": participaciones,
    }


def construir_entrada_lstm_manual(
    materia: str,
    trimestre: str,
    seccion: str,
    hito: int,
    anio_academico: float | None,
    posicion_lista: float,
    participaciones_semanales: list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Para un estudiante hipotético (no presente en los datos históricos)
    que se uniría a una sección real ya existente: `tamano_grupo`,
    `seccion_num` y el horario de cada semana (sesiones, evaluaciones,
    temas) se toman de esa sección real -- son hechos idénticos para
    todos sus estudiantes (verificado: 0 de 204 combinaciones
    materia/trimestre/sección/semana del conjunto tienen alguna variación
    entre estudiantes), nunca se inventan. Lo único propio del estudiante
    nuevo que se pide es lo que de verdad varía por estudiante: año que
    cursa, posición en la lista y su propia participación en cada
    semana."""
    if len(participaciones_semanales) != hito:
        raise ValueError(
            f"se esperaban {hito} valores de participación semanal, "
            f"llegaron {len(participaciones_semanales)}"
        )

    datos = _preparar_datos()
    i = _fila_referencia_seccion(materia, trimestre, seccion)
    fila_estatica = datos["estatica"].iloc[i]

    n_estaticas = len(ESTATICAS)
    indice_anio = ESTATICAS.index("anio_academico")
    indice_seccion_num = ESTATICAS.index("seccion_num")
    indice_tamano_grupo = ESTATICAS.index("tamano_grupo")
    indice_posicion = ESTATICAS.index("posicion_lista")
    indice_participaciones = n_estaticas + DINAMICAS.index("participaciones")
    indice_sesiones = n_estaticas + DINAMICAS.index("sesiones")
    indice_evaluaciones = n_estaticas + DINAMICAS.index("evaluaciones")

    X_hito = np.zeros((1, hito, n_estaticas + len(DINAMICAS)), dtype=np.float32)
    X_hito[0, :, indice_anio] = np.nan if anio_academico is None else float(anio_academico)
    X_hito[0, :, indice_seccion_num] = float(fila_estatica["seccion_num"])
    X_hito[0, :, indice_tamano_grupo] = float(fila_estatica["tamano_grupo"])
    X_hito[0, :, indice_posicion] = float(posicion_lista)

    # Horario real de la sección (sesiones/evaluaciones ya vividas, iguales
    # para todos sus estudiantes) -- no se pide ni se inventa.
    X_hito[0, :, indice_sesiones] = datos["X"][i, :hito, indice_sesiones]
    X_hito[0, :, indice_evaluaciones] = datos["X"][i, :hito, indice_evaluaciones]
    # Único dato propio del estudiante nuevo por semana.
    X_hito[0, :, indice_participaciones] = np.array(participaciones_semanales, dtype=np.float32)

    X_temas_hito = datos["X_temas"][i : i + 1, :hito].copy()

    acumulado = float(np.sum(participaciones_semanales))
    return X_hito, X_temas_hito, acumulado


@dataclass(frozen=True)
class ResultadoLSTMManual:
    materia: str
    trimestre: str
    seccion: str
    hito: int
    prediccion_total: float


def predict_lstm_manual(
    materia: str,
    trimestre: str,
    seccion: str,
    hito: int,
    anio_academico: float | None,
    posicion_lista: float,
    participaciones_semanales: list[float],
) -> ResultadoLSTMManual:
    """Misma predicción que `predict_lstm`, para un estudiante hipotético
    que se uniría a una sección real ya existente (`materia`/`trimestre`/
    `seccion`) en vez de para un registro histórico."""
    X_hito, X_temas_hito, acumulado = construir_entrada_lstm_manual(
        materia, trimestre, seccion, hito, anio_academico, posicion_lista, participaciones_semanales
    )
    total = _predecir_desde_tensor(materia, hito, X_hito, X_temas_hito, acumulado)

    return ResultadoLSTMManual(
        materia=materia,
        trimestre=trimestre,
        seccion=seccion,
        hito=hito,
        prediccion_total=total,
    )


_CACHE_TABLA_ERROR: np.ndarray | None = None

_VECINDAD_MINIMA_ERROR_LOCAL = 2.0
_N_MINIMO_ERROR_LOCAL = 5


def _tabla_error_validacion() -> np.ndarray:
    """(real, predicho) de los 1572 casos de la validación cruzada oficial
    de la LSTM (Sprint 2, `LSTM/nuevo/adaptacion_lstm_participaciones.ipynb`,
    celda 32 `validacion_cruzada`) -- NO de los modelos Estrategia 2 de
    este prototipo. Lee el CSV ya exportado en Sprint 5 para paridad
    (`predicciones_lstm_validacion_cruzada.csv`); no reentrena nada."""
    global _CACHE_TABLA_ERROR
    if _CACHE_TABLA_ERROR is None:
        tabla = pd.read_csv(RUTA_PREDICCIONES_VALIDACION)
        _CACHE_TABLA_ERROR = tabla[["total_trimestre_real", "prediccion_lstm"]].to_numpy(dtype=float)
    return _CACHE_TABLA_ERROR


def error_local(prediccion: float) -> dict:
    """Margen de error empírico para una predicción de esta magnitud,
    estimado sobre los casos de la validación cruzada oficial cuya
    predicción histórica cae cerca de `prediccion` (vecindad de
    `_VECINDAD_MINIMA_ERROR_LOCAL` participaciones, ampliada si hay menos
    de `_N_MINIMO_ERROR_LOCAL` casos). No sustituye ni se mezcla con el
    RMSE/R² global ya reportado en el informe -- es una estimación local,
    adicional, propia del prototipo."""
    tabla = _tabla_error_validacion()
    reales, predichos = tabla[:, 0], tabla[:, 1]

    vecindad = _VECINDAD_MINIMA_ERROR_LOCAL
    mascara = np.abs(predichos - prediccion) <= vecindad
    while mascara.sum() < _N_MINIMO_ERROR_LOCAL and vecindad < 20:
        vecindad *= 2
        mascara = np.abs(predichos - prediccion) <= vecindad

    n = int(mascara.sum())
    if n == 0:
        return {"rmse_local": None, "mae_local": None, "n": 0, "vecindad": vecindad}

    diferencias = reales[mascara] - predichos[mascara]
    return {
        "rmse_local": float(np.sqrt(np.mean(diferencias ** 2))),
        "mae_local": float(np.mean(np.abs(diferencias))),
        "n": n,
        "vecindad": vecindad,
    }


def valor_real(caso) -> float:
    """Valor real observado (participaciones totales del trimestre) para
    un `CasoPrediccion` histórico -- nunca disponible para un estudiante
    hipotético. Se lee de la tabla estática, ya cacheada por
    `_preparar_datos()`."""
    datos = _preparar_datos()
    estatica = datos["estatica"]
    filtro = (
        (estatica.materia == caso.materia)
        & (estatica.trimestre == caso.trimestre)
        & (estatica.seccion.astype(str) == str(caso.seccion))
        & (estatica.estudiante_id == caso.estudiante_id)
    )
    indices = np.where(filtro.to_numpy())[0]
    if len(indices) != 1:
        raise ValueError(f"se esperaba exactamente 1 registro para {caso}, hay {len(indices)}")
    return float(datos["y"][indices[0]])
