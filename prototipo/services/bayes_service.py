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

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_CODIGO_RED = RAIZ / "RedBayesiana" / "codigo_red"
if str(RUTA_CODIGO_RED) not in sys.path:
    sys.path.insert(0, str(RUTA_CODIGO_RED))

from pgmpy.inference import VariableElimination  # noqa: E402

from ensamblado import ensamblar_conjunto  # noqa: E402
from ajuste_cpd import preparar_datos_asignatura, ajustar_cpd, ESS_SELECCIONADO  # noqa: E402
from red_bayesiana import construir_modelo_manual  # noqa: E402
from inferencia import _evidencia_c1, NODO_OBJETIVO, COLUMNAS_EVIDENCIA  # noqa: E402
from valor_esperado import ESTADOS, medias_entrenamiento_por_estado, valor_esperado  # noqa: E402

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


def predict_bayes(caso) -> ResultadoBayes:
    """Distribución posterior y valor esperado continuo para `caso`,
    usando el modelo final de su asignatura (Estrategia 2 — ajustado con
    todos los trimestres disponibles, no con la validación cruzada de
    Sprint 4). No debe usarse para calcular métricas de desempeño."""
    evidencia, _fila_reg, _fila_semana = construir_evidencia_bayes(caso)
    entrada = _cargar_modelo(caso.materia)

    resultado = entrada["inferencia"].query(
        variables=[NODO_OBJETIVO], evidence=evidencia, show_progress=False
    )
    estados_orden = resultado.state_names[NODO_OBJETIVO]
    posterior = {estado: float(p) for estado, p in zip(estados_orden, resultado.values)}

    suma = sum(posterior.values())
    if abs(suma - 1.0) > 1e-6:
        raise ValueError(f"{caso}: posterior no normalizada (suma={suma})")

    prediccion = valor_esperado(posterior, entrada["medias"])

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
