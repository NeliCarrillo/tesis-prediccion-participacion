"""Convierte la salida de la red bayesiana (una distribución de probabilidad sobre
los estados de «Cantidad de participaciones del trimestre», Tabla 13) en un valor
continuo, para poder evaluarla con las mismas métricas que la LSTM (RMSE, R²)
sobre el total de participaciones del trimestre — Sprint 4, carta 6.

Cada estado se representa por la media observada del total continuo dentro de
ese estado (no por el punto medio del intervalo): para los estados acotados
ambas cosas casi coinciden, pero «12 o más» no tiene punto medio propio, y la
media empírica es el valor que minimiza el error cuadrático dentro del
estado, dado que no hay más información para distinguir dentro de él.

Corrección respecto de la versión anterior de este módulo: ya NO existe una
tabla `MEDIA_POR_ESTADO` global y fija. Esa tabla se calculó una sola vez
sobre los 437 registros previos a incorporar Matemáticas Discretas y sin
segmentar por asignatura ni por pliegue — mezclaba entrenamiento y prueba de
los 14 pliegues de la validación cruzada (Sprint 4, carta 5), lo cual filtra
información de cada trimestre de prueba hacia su propia evaluación. Ahora
`medias_entrenamiento_por_estado` calcula esa media específicamente por
asignatura y por pliegue, usando exclusivamente los registros
estudiante-sección de entrenamiento del pliegue correspondiente — nunca los
del trimestre de prueba.
"""
from __future__ import annotations

import pandas as pd

ESTADOS: tuple[str, ...] = ("0", "1-2", "3-5", "6-11", "12 o más")


def medias_entrenamiento_por_estado(
    reg: pd.DataFrame, materia: str, trimestre_prueba: str
) -> dict[str, float]:
    """Media continua observada de `total_trimestre` (el valor real antes de
    discretizar, ya calculado por `ensamblado._construir_registros`) por
    estado de «Cantidad de participaciones del trimestre», usando
    EXCLUSIVAMENTE los registros estudiante-sección de entrenamiento de este
    pliegue: misma `materia`, cualquier trimestre distinto de
    `trimestre_prueba`. Nunca usa registros del trimestre de prueba.

    No imputa ni usa una media global como respaldo: si algún estado de
    `ESTADOS` no tiene ningún registro de entrenamiento en este pliegue,
    lanza `ValueError` explícito en vez de aproximar.
    """
    train = reg[(reg["materia"] == materia) & (reg["trimestre"] != trimestre_prueba)]

    medias: dict[str, float] = {}
    sin_representante = []
    for estado in ESTADOS:
        valores = train.loc[
            train["Cantidad de participaciones del trimestre"] == estado, "total_trimestre"
        ]
        if len(valores) == 0:
            sin_representante.append(estado)
            continue
        medias[estado] = float(valores.mean())

    if sin_representante:
        raise ValueError(
            f"{materia!r}, pliegue con prueba={trimestre_prueba!r}: sin representante de "
            f"entrenamiento para el/los estado(s) {sin_representante} de "
            "«Cantidad de participaciones del trimestre». No se imputa ni se usa una "
            "media global como respaldo."
        )

    return medias


def valor_esperado(posterior: dict[str, float], medias: dict[str, float]) -> float:
    """Convierte una posterior discreta `{estado: probabilidad}` en un valor
    continuo: Σ P(objetivo=estado | evidencia) × media_train(estado).

    `medias` debe venir de `medias_entrenamiento_por_estado` (o tener la
    misma forma) — siempre calculadas solo con entrenamiento del pliegue
    correspondiente, nunca con datos globales ni de prueba."""
    faltan_posterior = set(ESTADOS) - posterior.keys()
    if faltan_posterior:
        raise ValueError(f"faltan estados en la posterior: {faltan_posterior}")
    faltan_medias = set(ESTADOS) - medias.keys()
    if faltan_medias:
        raise ValueError(f"faltan medias de entrenamiento para los estados: {faltan_medias}")
    return sum(medias[estado] * posterior[estado] for estado in ESTADOS)


if __name__ == "__main__":
    medias_ejemplo = {"0": 0.0, "1-2": 1.5, "3-5": 4.0, "6-11": 8.0, "12 o más": 18.0}
    posterior_ejemplo = {"0": 0.9, "1-2": 0.1, "3-5": 0.0, "6-11": 0.0, "12 o más": 0.0}
    v = valor_esperado(posterior_ejemplo, medias_ejemplo)
    assert abs(v - 0.15) < 1e-9

    try:
        valor_esperado({"0": 1.0}, medias_ejemplo)
    except ValueError:
        pass
    else:
        raise AssertionError("debía fallar por posterior incompleta")

    print("ok")
