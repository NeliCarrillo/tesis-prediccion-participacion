"""Comparación controlada: NaN conservado vs. dropna() global durante el
ajuste de CPD — tarjeta del tutor "Evaluar con y sin datos faltantes en el
entrenamiento".

NO es la implementación principal ni la reemplaza. La implementación
principal (carta 6, ya cerrada) sigue siendo: CPD ajustadas conservando NaN
por nodo (`ajuste_cpd.ajustar_cpd` sobre `preparar_fold`, sin dropna()
global) e inferencia con evidencia parcial para año académico faltante.
Este módulo solo instrumenta, en paralelo, la variante "dropna" (el
comportamiento previo a esa carta) para poder comparar ambas sin tocar
ningún archivo de la implementación principal.

Todo lo demás se mantiene idéntico entre las dos condiciones: misma
estructura (`red_bayesiana.construir_modelo_manual`), misma separación
train/test y mismo mapeo de tema train-only (`ajuste_cpd.preparar_fold`),
mismo ESS=5, misma evidencia C1 (se reutiliza literalmente
`inferencia._evidencia_c1`, sin reimplementarla), mismas medias train-only
(`valor_esperado.medias_entrenamiento_por_estado`) y la misma población de
evaluación (los 524 registros de prueba en los 14 pliegues, incluidos los 3
de año faltante, evaluados con evidencia parcial en AMBAS condiciones). La
única diferencia entre "nan" y "dropna" es qué filas ve
`DiscreteBayesianEstimator` al ajustar las CPD de entrenamiento.

Importar este módulo no ejecuta nada; `ejecutar_comparacion()` corre los
14 pliegues × 2 condiciones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ensamblado import ensamblar_conjunto, CLAVE, COLUMNAS_BN
from ajuste_cpd import preparar_fold, ajustar_cpd, verificar_cpd, ESS_SELECCIONADO
from red_bayesiana import construir_modelo_manual
from valor_esperado import ESTADOS, medias_entrenamiento_por_estado, valor_esperado
from inferencia import _evidencia_c1, NODO_OBJETIVO, HITOS

TRATAMIENTOS: tuple[str, ...] = ("nan", "dropna")


def _preparar_train_para_ajuste(train_bn: pd.DataFrame, tratamiento: str) -> tuple[pd.DataFrame, dict]:
    """Devuelve `(datos_para_ajustar_cpd, filas_utilizables_por_nodo)` para
    la condición pedida, a partir del `train_bn` que ya devuelve
    `ajuste_cpd.preparar_fold` (10 columnas en texto, NaN real preservado
    en «Año que cursa» y «Participaciones de la semana anterior»)."""
    if tratamiento == "nan":
        from ajuste_cpd import filas_utilizables_por_nodo
        return train_bn, filas_utilizables_por_nodo(train_bn)
    if tratamiento == "dropna":
        completo = train_bn.dropna()
        n = len(completo)
        return completo, {nodo: n for nodo in COLUMNAS_BN}
    raise ValueError(f"tratamiento desconocido: {tratamiento!r}")


def evaluar_fold_variante(
    sesiones: pd.DataFrame,
    reg: pd.DataFrame,
    sem: pd.DataFrame,
    materia: str,
    trimestre_prueba: str,
    tratamiento: str,
    ess: int = ESS_SELECCIONADO,
    hitos: tuple[int, ...] = HITOS,
) -> tuple[list[dict], dict]:
    """Igual que `inferencia.evaluar_fold`, salvo por qué filas ve el ajuste
    de CPD (`tratamiento`: "nan" o "dropna"). Reutiliza
    `ajuste_cpd.preparar_fold` para la separación train/test y el mapeo de
    tema (idéntico en ambas condiciones), y `inferencia._evidencia_c1` para
    la evidencia C1 (sin reimplementarla)."""
    train_bn, _test_bn, reporte_fold = preparar_fold(sesiones, materia, trimestre_prueba)

    assert trimestre_prueba not in reporte_fold["trimestres_train"], (
        f"{materia}/{trimestre_prueba}: fuga de partición"
    )

    train_para_ajuste, filas_por_nodo = _preparar_train_para_ajuste(train_bn, tratamiento)

    modelo = construir_modelo_manual()
    modelo = ajustar_cpd(modelo, train_para_ajuste, ess)
    verificacion = verificar_cpd(modelo)
    ve = VariableElimination(modelo)

    # Medias train-only: no dependen de las columnas de sesión ni del
    # tratamiento de NaN durante el ajuste de CPD — idénticas en ambas
    # condiciones, por construcción.
    medias = medias_entrenamiento_por_estado(reg, materia, trimestre_prueba)

    registros_test = reg[(reg["materia"] == materia) & (reg["trimestre"] == trimestre_prueba)].copy()

    filas = []
    for _, fila_reg in registros_test.iterrows():
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
                    f"{materia}/{trimestre_prueba}/hito={hito}: se esperaba 1 fila "
                    f"semana={hito} para {clave_valores}, hay {len(fila_semana)}"
                )
            fila_semana = fila_semana.iloc[0]

            # Misma evidencia C1 que la implementación principal — la misma
            # función, no una copia.
            evidencia = _evidencia_c1(fila_reg, fila_semana)

            resultado = ve.query(variables=[NODO_OBJETIVO], evidence=evidencia, show_progress=False)
            estados_orden = resultado.state_names[NODO_OBJETIVO]
            posterior = {estado: float(p) for estado, p in zip(estados_orden, resultado.values)}

            suma = sum(posterior.values())
            if abs(suma - 1.0) > 1e-6:
                raise ValueError(f"posterior no normalizada ({suma}) en {clave_valores}, hito={hito}")

            prediccion = valor_esperado(posterior, medias)
            if not np.isfinite(prediccion):
                raise ValueError(f"predicción no finita en {clave_valores}, hito={hito}")

            filas.append({
                "tratamiento": tratamiento,
                "materia": materia,
                "trimestre_prueba": trimestre_prueba,
                "estudiante_id": clave_valores["estudiante_id"],
                "seccion": clave_valores["seccion"],
                "hito": hito,
                "total_trimestre_real": float(fila_reg["total_trimestre"]),
                "prediccion_continua": prediccion,
                "evidencia_completa": "Año que cursa" in evidencia,
            })

    resumen_fold = {
        "tratamiento": tratamiento,
        "materia": materia,
        "trimestre_prueba": trimestre_prueba,
        "n_filas_ajuste_objetivo": filas_por_nodo["Cantidad de participaciones del trimestre"],
        "filas_utilizables_por_nodo": filas_por_nodo,
        "min_probabilidad_cpd": verificacion["min_probabilidad_global"],
        "max_error_normalizacion_cpd": verificacion["max_error_normalizacion"],
    }

    return filas, resumen_fold


def ejecutar_comparacion(
    sesiones: pd.DataFrame | None = None,
    reg: pd.DataFrame | None = None,
    sem: pd.DataFrame | None = None,
    ess: int = ESS_SELECCIONADO,
    hitos: tuple[int, ...] = HITOS,
) -> tuple[pd.DataFrame, list[dict]]:
    """Ejecuta los 14 pliegues bajo las dos condiciones ("nan", "dropna").
    Devuelve `(resultados, resumenes)`; `resultados` tiene una fila por
    tratamiento × registro de prueba × hito (1.572 × 2 = 3.144 filas)."""
    if sesiones is None or reg is None or sem is None:
        sesiones, reg, sem = ensamblar_conjunto()

    todas_filas: list[dict] = []
    resumenes: list[dict] = []

    for materia in sorted(sesiones["materia"].unique()):
        trimestres = sorted(sesiones.loc[sesiones["materia"] == materia, "trimestre"].unique())
        for trimestre_prueba in trimestres:
            for tratamiento in TRATAMIENTOS:
                filas, resumen = evaluar_fold_variante(
                    sesiones, reg, sem, materia, trimestre_prueba, tratamiento, ess=ess, hitos=hitos
                )
                todas_filas.extend(filas)
                resumenes.append(resumen)

    resultados = pd.DataFrame(todas_filas)
    return resultados, resumenes


def metricas_por_materia_hito(resultados_tratamiento: pd.DataFrame) -> pd.DataFrame:
    """RMSE/MAE/R² por materia × hito, agregando las predicciones de todos
    los pliegues de esa materia — mismo criterio de agregación que ya usa
    la LSTM (`metricas_adaptacion.csv`, una fila por materia × hito)."""
    filas = []
    for (materia, hito), g in resultados_tratamiento.groupby(["materia", "hito"]):
        real, pred = g["total_trimestre_real"], g["prediccion_continua"]
        filas.append({
            "materia": materia,
            "hito": hito,
            "rmse": float(np.sqrt(mean_squared_error(real, pred))),
            "mae": float(mean_absolute_error(real, pred)),
            "r2": float(r2_score(real, pred)),
            "n": len(g),
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    sesiones, reg, sem = ensamblar_conjunto()

    resultados, resumenes = ejecutar_comparacion(sesiones, reg, sem)

    assert len(resultados) == 1572 * 2, f"se esperaban {1572*2} filas, hay {len(resultados)}"
    for tratamiento in TRATAMIENTOS:
        n = (resultados["tratamiento"] == tratamiento).sum()
        assert n == 1572, f"{tratamiento}: {n} filas, se esperaban 1572"

    print(f"filas totales: {len(resultados)} (2 tratamientos x 1.572)")
    print("verificaciones superadas: misma población (524 registros x 3 hitos) "
          "en ambos tratamientos, incluidos los 3 registros de año faltante "
          "con evidencia parcial en ambos")
