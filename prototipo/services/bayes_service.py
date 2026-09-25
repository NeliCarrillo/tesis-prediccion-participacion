"""Servicio de inferencia de la red bayesiana — Sprint 5, tarjeta 3.

Reutiliza directamente, sin duplicar, los módulos ya validados de
Sprint 4 (`RedBayesiana/codigo_red/ensamblado.py`, `ajuste_cpd.py`,
`red_bayesiana.py`, `inferencia.py`, `valor_esperado.py`). No reimplementa
ninguna transformación: importa las mismas funciones que produjeron
`RedBayesiana/resultados/predicciones_bayesiana_s4_s6_s8.csv`.

No persiste artefactos en disco. La estructura del grafo es determinista
y el ajuste de CPD de una asignatura completa toma ~30 ms (medido en la
auditoría de Sprint 5) — se reconstruye en memoria la primera vez que se
pide cada asignatura y se cachea para las peticiones siguientes.

Estrategia 2 (la misma decisión aprobada para la LSTM): las CPD y las
medias de estado se ajustan con TODOS los trimestres disponibles de la
asignatura, no con la validación cruzada de Sprint 4
(`ajuste_cpd.preparar_fold`, que deja un trimestre fuera). Para esto se
reutiliza `ajuste_cpd.preparar_datos_asignatura` — ya existente desde la
carta 4 de Sprint 4, sin ningún fold — y `valor_esperado.
medias_entrenamiento_por_estado` con un `trimestre_prueba` centinela que
no existe en los datos, de modo que su propio filtro (`trimestre !=
trimestre_prueba`) deja pasar todo el histórico sin modificar esa
función.

Una predicción de este servicio sobre un registro histórico no debe
interpretarse como una nueva medición de desempeño — ver
`prototipo/README.md`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_CODIGO_RED = RAIZ / "RedBayesiana" / "codigo_red"
if str(RUTA_CODIGO_RED) not in sys.path:
    sys.path.insert(0, str(RUTA_CODIGO_RED))

from pgmpy.inference import VariableElimination  # noqa: E402

from ensamblado import ensamblar_conjunto, ESTADOS_BN  # noqa: E402
from ajuste_cpd import preparar_datos_asignatura, ajustar_cpd, ESS_SELECCIONADO  # noqa: E402
from red_bayesiana import construir_modelo_manual  # noqa: E402
from inferencia import _evidencia_c1, NODO_OBJETIVO, COLUMNAS_EVIDENCIA, ejecutar_inferencia  # noqa: E402
from valor_esperado import ESTADOS, medias_entrenamiento_por_estado, valor_esperado  # noqa: E402
from discretizacion import bin_anio, participaciones_semana as _discretizar_participaciones  # noqa: E402

# Centinela: ningún trimestre real se llama así. Ver docstring del módulo.
_SIN_HOLDOUT = "__sin_holdout_estrategia_2__"

_CACHE_DATOS: dict | None = None
_CACHE_MODELOS: dict[str, dict] = {}


def _preparar_datos() -> dict:
    """Ensambla (sesiones, reg, sem) una sola vez — misma función que usa
    Sprint 4, sin cambios."""
    global _CACHE_DATOS
    if _CACHE_DATOS is None:
        sesiones, reg, sem = ensamblar_conjunto()
        _CACHE_DATOS = {"sesiones": sesiones, "reg": reg, "sem": sem}
    return _CACHE_DATOS


def _cargar_modelo(materia: str) -> dict:
    """Ajusta (o recupera de caché) la CPD de `materia` con todo su
    histórico (Estrategia 2) y las medias de estado correspondientes."""
    if materia in _CACHE_MODELOS:
        return _CACHE_MODELOS[materia]

    datos = _preparar_datos()
    datos_bn, _reporte = preparar_datos_asignatura(datos["sesiones"], materia)
    modelo = construir_modelo_manual()
    modelo = ajustar_cpd(modelo, datos_bn, ESS_SELECCIONADO)
    medias = medias_entrenamiento_por_estado(datos["reg"], materia, _SIN_HOLDOUT)

    entrada = {
        "inferencia": VariableElimination(modelo),
        "medias": medias,
        "ess": ESS_SELECCIONADO,
    }
    _CACHE_MODELOS[materia] = entrada
    return entrada


def construir_evidencia_bayes(caso) -> tuple[dict, pd.Series, pd.Series]:
    """A partir de un `CasoPrediccion`, localiza su fila en `reg` y en
    `sem` (semana == hito) y construye la evidencia C1 con la misma
    función que usa Sprint 4 (`inferencia._evidencia_c1`), sin
    reimplementarla. Devuelve `(evidencia, fila_reg, fila_semana)`."""
    datos = _preparar_datos()
    reg, sem = datos["reg"], datos["sem"]

    filtro_caso = (
        (reg["materia"] == caso.materia)
        & (reg["trimestre"] == caso.trimestre)
        & (reg["seccion"].astype(str) == str(caso.seccion))
        & (reg["estudiante_id"] == caso.estudiante_id)
    )
    filas_reg = reg[filtro_caso]
    if len(filas_reg) != 1:
        raise ValueError(f"se esperaba exactamente 1 registro para {caso}, hay {len(filas_reg)}")
    fila_reg = filas_reg.iloc[0]

    filtro_semana = (
        (sem["materia"] == caso.materia)
        & (sem["trimestre"] == caso.trimestre)
        & (sem["seccion"].astype(str) == str(caso.seccion))
        & (sem["estudiante_id"] == caso.estudiante_id)
        & (sem["semana"] == caso.hito)
    )
    filas_semana = sem[filtro_semana]
    if len(filas_semana) != 1:
        raise ValueError(
            f"se esperaba exactamente 1 fila semana={caso.hito} para {caso}, hay {len(filas_semana)}"
        )
    fila_semana = filas_semana.iloc[0]

    evidencia = _evidencia_c1(fila_reg, fila_semana)
    return evidencia, fila_reg, fila_semana


def _tamano_grupo_seccion(materia: str, trimestre: str, seccion: str) -> str:
    """Estado discretizado de «Tamaño del grupo» de una sección real ya
    existente (mismo valor para todos sus estudiantes). Se usa para un
    estudiante hipotético que se uniría a esa sección: el tamaño del grupo
    es un hecho de la sección, no algo que deba inventarse ni pedirse a
    mano."""
    datos = _preparar_datos()
    reg = datos["reg"]
    filtro = (
        (reg["materia"] == materia)
        & (reg["trimestre"] == trimestre)
        & (reg["seccion"].astype(str) == str(seccion))
    )
    filas = reg[filtro]
    if filas.empty:
        raise ValueError(f"no existe la sección {materia}/{trimestre}/sección {seccion}")
    return filas.iloc[0]["Tamaño del grupo"]


@dataclass(frozen=True)
class ResultadoBayes:
    """Salida reutilizable por las tarjetas de visualización (4/5).
    `evidencia_omitida` lista las variables de `COLUMNAS_EVIDENCIA` que no
    entraron a la consulta (hoy, a lo sumo, «Año que cursa»)."""

    materia: str
    trimestre: str
    seccion: str
    estudiante_id: str
    hito: int
    evidencia_utilizada: dict
    evidencia_omitida: tuple[str, ...]
    posterior: dict[str, float]
    prediccion_continua: float
    ess: int


def _consultar(entrada: dict, evidencia: dict) -> tuple[dict[str, float], float]:
    """Ejecuta una consulta de inferencia con la evidencia dada sobre un
    modelo ya ajustado (`entrada`, de `_cargar_modelo`) y devuelve
    `(posterior, prediccion_continua)`. Es el único punto que llama a
    `VariableElimination.query` — lo reutilizan tanto `predict_bayes` como
    `explorar_sensibilidad`, para no duplicar la consulta ni el cálculo
    del valor esperado."""
    resultado = entrada["inferencia"].query(
        variables=[NODO_OBJETIVO], evidence=evidencia, show_progress=False
    )
    estados_orden = resultado.state_names[NODO_OBJETIVO]
    posterior = {estado: float(p) for estado, p in zip(estados_orden, resultado.values)}

    suma = sum(posterior.values())
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(f"posterior no normalizada (suma={suma}) con evidencia {evidencia}")

    return posterior, valor_esperado(posterior, entrada["medias"])


def predict_bayes(caso) -> ResultadoBayes:
    """Distribución posterior y valor esperado continuo para `caso`,
    usando el modelo final de su asignatura (Estrategia 2 — ajustado con
    todos los trimestres disponibles, no con la validación cruzada de
    Sprint 4). No debe usarse para calcular métricas de desempeño."""
    evidencia, _fila_reg, _fila_semana = construir_evidencia_bayes(caso)
    entrada = _cargar_modelo(caso.materia)

    posterior, prediccion = _consultar(entrada, evidencia)

    evidencia_omitida = tuple(c for c in COLUMNAS_EVIDENCIA if c not in evidencia)

    return ResultadoBayes(
        materia=caso.materia,
        trimestre=caso.trimestre,
        seccion=caso.seccion,
        estudiante_id=caso.estudiante_id,
        hito=caso.hito,
        evidencia_utilizada=dict(evidencia),
        evidencia_omitida=evidencia_omitida,
        posterior=posterior,
        prediccion_continua=prediccion,
        ess=entrada["ess"],
    )


@dataclass(frozen=True)
class ResultadoBayesManual:
    """Igual que `ResultadoBayes`, para un estudiante hipotético que no
    está en los datos históricos: se une a una sección real (`materia` /
    `trimestre` / `seccion` sí deben existir), así que no lleva
    `estudiante_id` ni `hito` propios de un registro real."""

    materia: str
    trimestre: str
    seccion: str
    evidencia_utilizada: dict
    evidencia_omitida: tuple[str, ...]
    posterior: dict[str, float]
    prediccion_continua: float
    ess: int


def predict_bayes_manual(
    materia: str,
    trimestre: str,
    seccion: str,
    anio_academico: float | None,
    participaciones_semana_actual: float | None,
    participaciones_semana_anterior: float | None,
) -> ResultadoBayesManual:
    """Misma consulta que `predict_bayes`, para un estudiante hipotético
    que se uniría a una sección real ya existente (`materia`/`trimestre`/
    `seccion`). «Tamaño del grupo» se toma de esa sección real, nunca se
    inventa. Los otros tres valores son datos crudos del estudiante nuevo
    (mismas unidades que en los datos históricos); se discretizan aquí con
    las mismas funciones ya validadas de `discretizacion.py` (`bin_anio`,
    `participaciones_semana`), no con una regla nueva. Cualquiera de los
    tres puede dejarse en `None` -- igual que con «Año que cursa» faltante
    en `predict_bayes`, la red marginaliza la variable no observada en vez
    de inventar un valor."""
    evidencia: dict[str, str] = {"Tamaño del grupo": _tamano_grupo_seccion(materia, trimestre, seccion)}

    estado_anio = bin_anio(anio_academico)
    if estado_anio is not None:
        evidencia["Año que cursa"] = estado_anio
    if participaciones_semana_actual is not None:
        evidencia["Participaciones de la semana"] = _discretizar_participaciones(
            participaciones_semana_actual
        )
    if participaciones_semana_anterior is not None:
        evidencia["Participaciones de la semana anterior"] = _discretizar_participaciones(
            participaciones_semana_anterior
        )

    entrada = _cargar_modelo(materia)
    posterior, prediccion = _consultar(entrada, evidencia)
    evidencia_omitida = tuple(c for c in COLUMNAS_EVIDENCIA if c not in evidencia)

    return ResultadoBayesManual(
        materia=materia,
        trimestre=trimestre,
        seccion=seccion,
        evidencia_utilizada=dict(evidencia),
        evidencia_omitida=evidencia_omitida,
        posterior=posterior,
        prediccion_continua=prediccion,
        ess=entrada["ess"],
    )


@dataclass(frozen=True)
class Escenario:
    """Un escenario hipotético de evidencia: qué pasaría con la
    predicción si `variable` tomara `valor_alternativo` (o se omitiera
    por completo), en vez del valor realmente observado. No es una
    medida de importancia ni de causalidad — es, literalmente, el
    resultado de volver a preguntarle al mismo modelo ya ajustado con
    una evidencia distinta."""

    valor_alternativo: str  # o "(sin esta evidencia)" para el caso marginalizado
    prediccion_continua: float
    es_el_valor_observado: bool


def _escenarios_variable(entrada: dict, evidencia: dict, variable: str) -> list[Escenario]:
    """Recalcula la predicción bajo cada uno de los demás estados posibles
    de `variable`, y bajo la marginalización completa (como si no se
    conociera). No reentrena ni reajusta nada: reutiliza el mismo modelo
    ya cacheado (`entrada`), cambiando únicamente el diccionario de
    evidencia en cada consulta. Es el único cuerpo de este cálculo —
    `explorar_sensibilidad` y `explorar_sensibilidad_manual` solo difieren
    en de dónde sale `evidencia`."""
    if variable not in evidencia:
        raise ValueError(
            f"'{variable}' no forma parte de la evidencia utilizada "
            f"(evidencia disponible: {sorted(evidencia)})"
        )

    valor_observado = evidencia[variable]

    escenarios = []
    for estado in ESTADOS_BN[variable]:
        evidencia_alternativa = dict(evidencia)
        evidencia_alternativa[variable] = estado
        _posterior, prediccion = _consultar(entrada, evidencia_alternativa)
        escenarios.append(Escenario(
            valor_alternativo=estado,
            prediccion_continua=prediccion,
            es_el_valor_observado=(estado == valor_observado),
        ))

    evidencia_sin_variable = {k: v for k, v in evidencia.items() if k != variable}
    _posterior, prediccion_sin = _consultar(entrada, evidencia_sin_variable)
    escenarios.append(Escenario(
        valor_alternativo="(sin esta evidencia)",
        prediccion_continua=prediccion_sin,
        es_el_valor_observado=False,
    ))

    return escenarios


def explorar_sensibilidad(caso, variable: str) -> list[Escenario]:
    """Para una `variable` que sí forma parte de la evidencia usada por
    `caso` (lanza `ValueError` si no).

    Pensado para responder, caso por caso, "¿qué predeciría la red si
    esta evidencia concreta fuera distinta?" — no para producir un
    ranking ni un porcentaje de influencia entre variables."""
    evidencia, _fila_reg, _fila_semana = construir_evidencia_bayes(caso)
    entrada = _cargar_modelo(caso.materia)
    return _escenarios_variable(entrada, evidencia, variable)


def explorar_sensibilidad_manual(materia: str, evidencia: dict, variable: str) -> list[Escenario]:
    """Igual que `explorar_sensibilidad`, para la evidencia de un
    estudiante hipotético (`ResultadoBayesManual.evidencia_utilizada`) en
    vez de un `CasoPrediccion` real."""
    entrada = _cargar_modelo(materia)
    return _escenarios_variable(entrada, evidencia, variable)


_CACHE_TABLA_ERROR: np.ndarray | None = None

_VECINDAD_MINIMA_ERROR_LOCAL = 2.0
_N_MINIMO_ERROR_LOCAL = 5


def _tabla_error_validacion() -> np.ndarray:
    """(real, predicho) de los 1572 casos de la validación cruzada oficial
    de Sprint 4 (`inferencia.ejecutar_inferencia`, 14 pliegues
    leave-one-trimestre-out, ESS=5) -- NO de los modelos Estrategia 2 de
    este prototipo. Se ejecuta una sola vez y se cachea en memoria; no
    escribe ningún archivo ni reentrena nada."""
    global _CACHE_TABLA_ERROR
    if _CACHE_TABLA_ERROR is None:
        resultados, _resumenes = ejecutar_inferencia()
        _CACHE_TABLA_ERROR = resultados[["total_trimestre_real", "prediccion_continua"]].to_numpy(
            dtype=float
        )
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
    hipotético. Se lee de `reg`, ya cacheado por `_preparar_datos()`."""
    datos = _preparar_datos()
    reg = datos["reg"]
    filtro = (
        (reg["materia"] == caso.materia)
        & (reg["trimestre"] == caso.trimestre)
        & (reg["seccion"].astype(str) == str(caso.seccion))
        & (reg["estudiante_id"] == caso.estudiante_id)
    )
    filas = reg[filtro]
    if len(filas) != 1:
        raise ValueError(f"se esperaba exactamente 1 registro para {caso}, hay {len(filas)}")
    return float(filas.iloc[0]["total_trimestre"])
