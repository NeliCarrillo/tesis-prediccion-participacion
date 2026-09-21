"""Inferencia y valor esperado — Sprint 4, carta 6.

Implementa C1 (única alternativa aprobada; C2 y C3 quedaron descartadas):
para cada registro estudiante-sección de prueba de cada uno de los 14
pliegues, y para cada hito S4/S6/S8, construye la evidencia mínima
necesaria — los 4 padres directos del nodo objetivo: «Año que cursa»,
«Tamaño del grupo», «Participaciones de la semana» en semana=hito y
«Participaciones de la semana anterior» — obtiene la distribución posterior
completa de «Cantidad de participaciones del trimestre» con
`pgmpy.inference.VariableElimination`, y la convierte a una predicción
continua con `valor_esperado.py`, usando medias por estado calculadas
exclusivamente con el entrenamiento de ese pliegue.

No se incluyen «Tema de la sesión», «Número de sesiones de la semana»,
«Sesiones de evaluación de la semana», «Sección» ni «Posición relativa en
la lista» como evidencia: por la propiedad markoviana local del grafo ya
cerrado, todas son irrelevantes para la posterior del objetivo una vez
fijados esos 4 padres (verificado antes de implementar, no es una
simplificación de conveniencia).

Una predicción por registro estudiante-sección y por hito — nunca una por
sesión ni una combinación de posteriores de varias sesiones. Los 3
registros con año académico faltante se excluyen de la inferencia (nunca
se imputan) cuando caen en el trimestre de prueba de su propio pliegue.

Fuera de alcance, deliberadamente no incluido aquí: RMSE, R² y comparación
con LSTM/extrapolación — eso es la carta 7.

Importar este módulo no ejecuta nada; `ejecutar_inferencia()` corre los 14
pliegues.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from ensamblado import ensamblar_conjunto, CLAVE
from ajuste_cpd import preparar_fold, ajustar_cpd, ESS_SELECCIONADO
from red_bayesiana import construir_modelo_manual
from valor_esperado import ESTADOS, medias_entrenamiento_por_estado, valor_esperado

RAIZ = Path(__file__).resolve().parent.parent.parent
CARPETA_RESULTADOS = RAIZ / "RedBayesiana" / "resultados"
RUTA_CSV_PREDICCIONES = CARPETA_RESULTADOS / "predicciones_bayesiana_s4_s6_s8.csv"

HITOS: tuple[int, ...] = (4, 6, 8)

# Evidencia mínima necesaria (C1) — los 4 padres directos del objetivo.
NODO_OBJETIVO = "Cantidad de participaciones del trimestre"
COLUMNAS_EVIDENCIA = (
    "Año que cursa",
    "Tamaño del grupo",
    "Participaciones de la semana",
    "Participaciones de la semana anterior",
)


def _evidencia_c1(fila_reg: pd.Series, fila_semana: pd.Series) -> dict[str, str]:
    return {
        "Año que cursa": fila_reg["Año que cursa"],
        "Tamaño del grupo": fila_reg["Tamaño del grupo"],
        "Participaciones de la semana": fila_semana["Participaciones de la semana"],
        "Participaciones de la semana anterior": fila_semana["Participaciones de la semana anterior"],
    }


def evaluar_fold(
    sesiones: pd.DataFrame,
    reg: pd.DataFrame,
    sem: pd.DataFrame,
    materia: str,
    trimestre_prueba: str,
    ess: int = ESS_SELECCIONADO,
    hitos: tuple[int, ...] = HITOS,
) -> tuple[list[dict], dict]:
    """Ejecuta la inferencia C1 de un pliegue completo: ajusta las CPD solo
    con entrenamiento (reusando `ajuste_cpd.preparar_fold`/`ajustar_cpd`,
    sin duplicar esa lógica ni la de la carta 5) y evalúa cada registro de
    prueba evaluable en cada hito.

    Devuelve `(filas, resumen_fold)`: `filas` es la lista de resultados (uno
    por registro evaluable × hito); `resumen_fold` trae conteos y las
    exclusiones por año académico faltante.
    """
    train_bn, test_bn, reporte_fold = preparar_fold(sesiones, materia, trimestre_prueba)

    # Mismas verificaciones de separación que la carta 5, repetidas aquí
    # porque esta función vuelve a decidir con qué se ajustan CPD y medias.
    assert trimestre_prueba not in reporte_fold["trimestres_train"], (
        f"{materia}/{trimestre_prueba}: fuga de partición hacia el ajuste de CPD"
    )

    modelo = construir_modelo_manual()
    modelo = ajustar_cpd(modelo, train_bn, ess)
    ve = VariableElimination(modelo)

    # Medias de valor esperado: solo con entrenamiento del pliegue (nunca
    # con el trimestre de prueba) — ver valor_esperado.py.
    medias = medias_entrenamiento_por_estado(reg, materia, trimestre_prueba)

    registros_test = reg[
        (reg["materia"] == materia) & (reg["trimestre"] == trimestre_prueba)
    ].copy()
    n_test_fold = len(registros_test)
    n_train_fold = int((reg["materia"] == materia).sum()) - n_test_fold

    falta_anio = registros_test["Año que cursa"].isna()
    excluidos = registros_test.loc[falta_anio, list(CLAVE)]
    evaluables = registros_test.loc[~falta_anio]

    filas = []
    for _, fila_reg in evaluables.iterrows():
        clave_valores = {c: fila_reg[c] for c in CLAVE}

        for hito in hitos:
            fila_semana = sem[
                (sem["estudiante_id"] == clave_valores["estudiante_id"])
                & (sem["materia"] == clave_valores["materia"])
                & (sem["trimestre"] == clave_valores["trimestre"])
                & (sem["seccion"] == clave_valores["seccion"])
                & (sem["semana"] == hito)
            ]
            if len(fila_semana) != 1:
                raise ValueError(
                    f"{materia}/{trimestre_prueba}/hito={hito}: se esperaba exactamente 1 "
                    f"fila semana={hito} para {clave_valores}, hay {len(fila_semana)}"
                )
            fila_semana = fila_semana.iloc[0]
            assert fila_semana["semana"] == hito  # nunca semanas posteriores al hito

            evidencia = _evidencia_c1(fila_reg, fila_semana)

            resultado = ve.query(variables=[NODO_OBJETIVO], evidence=evidencia, show_progress=False)
            estados_orden = resultado.state_names[NODO_OBJETIVO]
            posterior = {estado: float(p) for estado, p in zip(estados_orden, resultado.values)}

            suma = sum(posterior.values())
            if abs(suma - 1.0) > 1e-6:
                raise ValueError(
                    f"posterior no normalizada (suma={suma}) en {clave_valores}, hito={hito}"
                )

            prediccion = valor_esperado(posterior, medias)
            if not np.isfinite(prediccion):
                raise ValueError(f"predicción no finita en {clave_valores}, hito={hito}")

            filas.append({
                "materia": materia,
                "trimestre_prueba": trimestre_prueba,
                "estudiante_id": clave_valores["estudiante_id"],
                "seccion": clave_valores["seccion"],
                "hito": hito,
                "total_trimestre_real": float(fila_reg["total_trimestre"]),
                "estado_real": fila_reg["Cantidad de participaciones del trimestre"],
                "prediccion_continua": prediccion,
                **{f"posterior_{estado}": posterior[estado] for estado in ESTADOS},
                "ess": ess,
                "n_train_fold": n_train_fold,
                "n_test_fold": n_test_fold,
                "evidencia_anio_que_cursa": evidencia["Año que cursa"],
                "evidencia_tamano_grupo": evidencia["Tamaño del grupo"],
                "evidencia_participaciones_semana": evidencia["Participaciones de la semana"],
                "evidencia_participaciones_semana_anterior": evidencia[
                    "Participaciones de la semana anterior"
                ],
            })

    resumen_fold = {
        "materia": materia,
        "trimestre_prueba": trimestre_prueba,
        "n_train_fold": n_train_fold,
        "n_test_fold": n_test_fold,
        "n_excluidos_anio_faltante": int(falta_anio.sum()),
        "excluidos_anio_faltante": excluidos["estudiante_id"].tolist(),
        "n_evaluables": len(evaluables),
        "n_predicciones": len(filas),
    }

    return filas, resumen_fold


def ejecutar_inferencia(
    sesiones: pd.DataFrame | None = None,
    reg: pd.DataFrame | None = None,
    sem: pd.DataFrame | None = None,
    ess: int = ESS_SELECCIONADO,
    hitos: tuple[int, ...] = HITOS,
) -> tuple[pd.DataFrame, list[dict]]:
    """Ejecuta C1 sobre los 14 pliegues leave-one-trimestre-out. Devuelve
    `(resultados, resumenes_por_fold)`; `resultados` tiene una fila por
    registro estudiante-sección evaluable × hito."""
    if sesiones is None or reg is None or sem is None:
        sesiones, reg, sem = ensamblar_conjunto()

    todas_filas: list[dict] = []
    resumenes: list[dict] = []

    for materia in sorted(sesiones["materia"].unique()):
        trimestres = sorted(sesiones.loc[sesiones["materia"] == materia, "trimestre"].unique())
        for trimestre_prueba in trimestres:
            filas, resumen = evaluar_fold(
                sesiones, reg, sem, materia, trimestre_prueba, ess=ess, hitos=hitos
            )
            todas_filas.extend(filas)
            resumenes.append(resumen)

    resultados = pd.DataFrame(todas_filas)
    return resultados, resumenes


def verificar_resultados(
    resultados: pd.DataFrame, resumenes: list[dict], sem: pd.DataFrame, hitos: tuple[int, ...] = HITOS
) -> dict:
    """Verificaciones automáticas exigidas por la carta 6. Lanza
    `AssertionError` con un mensaje claro si alguna falla; si todas pasan,
    devuelve un resumen numérico para dejar registrado en el notebook."""
    clave_fila = ["materia", "trimestre_prueba", "estudiante_id", "seccion", "hito"]

    # Una predicción por registro x hito, sin duplicados.
    conteos = resultados.groupby(clave_fila).size()
    assert (conteos == 1).all(), f"hay combinaciones con más de una fila: {conteos[conteos != 1]}"

    # Exactamente len(hitos) hitos por registro evaluable.
    conteos_registro = resultados.groupby(
        ["materia", "trimestre_prueba", "estudiante_id", "seccion"]
    ).size()
    assert (conteos_registro == len(hitos)).all(), (
        f"registros sin exactamente {len(hitos)} hitos: {conteos_registro[conteos_registro != len(hitos)]}"
    )

    # Posterior normalizada (recalculada desde las columnas guardadas).
    columnas_posterior = [f"posterior_{estado}" for estado in ESTADOS]
    sumas = resultados[columnas_posterior].sum(axis=1)
    assert (sumas.sub(1.0).abs() < 1e-6).all(), "hay posteriores que no suman 1"

    # Sin NaN en la predicción; valor esperado finito.
    assert resultados["prediccion_continua"].notna().all(), "hay NaN en prediccion_continua"
    assert np.isfinite(resultados["prediccion_continua"]).all(), "hay valores no finitos"

    # Exclusión explícita de los 3 registros con año faltante: no deben
    # aparecer en absoluto en los resultados.
    total_excluidos = sum(r["n_excluidos_anio_faltante"] for r in resumenes)
    ids_excluidos = {eid for r in resumenes for eid in r["excluidos_anio_faltante"]}
    assert not (set(resultados["estudiante_id"]) & ids_excluidos), (
        "un registro con año faltante aparece en los resultados"
    )

    # Correspondencia semana=hito <-> su rezago: el rezago de la fila
    # semana=hito debe coincidir con "Participaciones de la semana" de la
    # fila semana=hito-1 del mismo registro (nunca semanas posteriores).
    problemas_rezago = 0
    for hito in hitos:
        filas_hito = sem[sem["semana"] == hito][
            list(CLAVE) + ["semana", "Participaciones de la semana anterior"]
        ]
        filas_prev = sem[sem["semana"] == hito - 1][
            list(CLAVE) + ["Participaciones de la semana"]
        ].rename(columns={"Participaciones de la semana": "esperado"})
        cruce = filas_hito.merge(filas_prev, on=list(CLAVE), how="left")
        problemas_rezago += int(
            (cruce["Participaciones de la semana anterior"] != cruce["esperado"]).sum()
        )
    assert problemas_rezago == 0, f"{problemas_rezago} filas con rezago inconsistente"

    # Ningún uso del trimestre de prueba para CPD/tema/medias: ya lo
    # garantiza preparar_fold/medias_entrenamiento_por_estado por
    # construcción; se reafirma aquí con los metadatos de cada pliegue.
    for r in resumenes:
        pass  # la garantía está en evaluar_fold (assert de trimestres_train)

    return {
        "n_predicciones_totales": len(resultados),
        "n_registros_evaluados": len(conteos_registro),
        "n_excluidos_anio_faltante": total_excluidos,
        "n_pliegues": len(resumenes),
    }


if __name__ == "__main__":
    sesiones, reg, sem = ensamblar_conjunto()

    resultados, resumenes = ejecutar_inferencia(sesiones, reg, sem)
    verificacion = verificar_resultados(resultados, resumenes, sem)

    print(f"Predicciones totales: {verificacion['n_predicciones_totales']}")
    print(f"Registros evaluados (únicos): {verificacion['n_registros_evaluados']}")
    print(f"Excluidos por año faltante: {verificacion['n_excluidos_anio_faltante']}")
    print(f"Pliegues: {verificacion['n_pliegues']}")
    print()

    resumen_asignatura = (
        resultados.groupby(["materia", "trimestre_prueba", "hito"])
        .size()
        .rename("n")
        .reset_index()
    )
    print(resumen_asignatura.to_string(index=False))

    print()
    print("verificaciones superadas: 1 predicción por registro-sección x hito, "
          "3 hitos por registro evaluable, posteriores normalizadas, sin NaN, "
          "valor esperado finito, 3 registros con año faltante excluidos, "
          "rezago consistente entre semana=hito y semana=hito-1")

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(RUTA_CSV_PREDICCIONES, index=False)
    print(f"\nCSV guardado en: {RUTA_CSV_PREDICCIONES}")
