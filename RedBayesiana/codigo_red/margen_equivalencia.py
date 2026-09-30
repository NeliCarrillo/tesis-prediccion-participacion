"""Margen de la LSTM y posibilidad de que la red bayesiana quede dentro de él
(comentario del tutor sobre "no se definió un margen estadístico de
equivalencia").

Análisis complementario: NO modifica ningún resultado del informe. Las
métricas oficiales siguen siendo las de `comparacion_bayes_lstm_extrapolacion.csv`.

1. Margen: para cada asignatura × hito, compara el RMSE y el R² de la red
   bayesiana con la media de la LSTM ± una y dos desviaciones estándar
   entre las cinco semillas (Tabla 16).
2. Otra regla de suavizado: repite la validación cruzada de 14 pliegues de
   la red bayesiana cambiando solo el prior de las CPD (BDeu con ESS 1, 5,
   10 y 50, y K2, que suma una pseudocuenta a cada celda). Todo lo demás es
   idéntico a la implementación principal (`comparacion_faltantes`,
   tratamiento "nan"). Con ESS=5 se exige reproducir exactamente las
   predicciones oficiales.
3. Más datos: cota optimista. Se ajusta la red con TODOS los trimestres,
   incluido el que se evalúa (los modelos del prototipo), y se mide su
   error sobre esos mismos estudiantes. Es lo mejor que podría lograr esta
   estructura si tuviera datos de toda la población; si ni así entra en el
   margen de la LSTM, más datos del mismo tipo no bastarían.

Correr el archivo genera `resultados/margen_equivalencia.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from sklearn.metrics import mean_squared_error, r2_score

import comparacion_faltantes as cf
from ensamblado import ESTADOS_BN
from ensamblado import ensamblar_conjunto

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_COMPARACION = RAIZ / "RedBayesiana" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"
RUTA_OFICIAL = RAIZ / "RedBayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_PROTOTIPO = RAIZ / "LSTM" / "nuevo" / "explicabilidad_lstm_casos.csv"
RUTA_SALIDA = RAIZ / "RedBayesiana" / "resultados" / "margen_equivalencia.csv"
LLAVE = ["materia", "trimestre", "estudiante_id", "seccion", "hito"]

VARIANTES = (("BDeu", 1), ("BDeu", 5), ("BDeu", 10), ("BDeu", 50), ("K2", None))


def _ajustar_con_prior(prior: str, ess):
    def ajustar(modelo, datos_bn, _ess):
        opciones = dict(state_names=ESTADOS_BN, prior_type=prior)
        if prior == "BDeu":
            opciones["equivalent_sample_size"] = ess
        modelo.fit(datos_bn, estimator=DiscreteBayesianEstimator(**opciones))
        return modelo
    return ajustar


def validar_variante(prior: str, ess, sesiones, reg, sem) -> pd.DataFrame:
    cf.ajustar_cpd = _ajustar_con_prior(prior, ess)
    filas = []
    for materia in sorted(sesiones["materia"].unique()):
        for trimestre in sorted(sesiones.loc[sesiones["materia"] == materia, "trimestre"].unique()):
            f, _ = cf.evaluar_fold_variante(sesiones, reg, sem, materia, trimestre, "nan")
            filas.extend(f)
    return pd.DataFrame(filas).rename(columns={"trimestre_prueba": "trimestre"})


def metricas(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"materia": m, "hito": h,
         "rmse": float(np.sqrt(mean_squared_error(g["total_trimestre_real"], g[columna]))),
         "r2": float(r2_score(g["total_trimestre_real"], g[columna]))}
        for (m, h), g in df.groupby(["materia", "hito"])
    ])


if __name__ == "__main__":
    base = pd.read_csv(RUTA_COMPARACION)
    tabla = base[["materia", "hito", "rmse_bayes", "r2_bayes", "rmse_lstm", "rmse_lstm_desv",
                  "r2_lstm", "r2_lstm_desv"]].copy()
    tabla["brecha_rmse_en_desv"] = (tabla.rmse_bayes - tabla.rmse_lstm) / tabla.rmse_lstm_desv
    tabla["brecha_r2_en_desv"] = (tabla.r2_lstm - tabla.r2_bayes) / tabla.r2_lstm_desv

    oficial = pd.read_csv(RUTA_OFICIAL).rename(columns={"trimestre_prueba": "trimestre"})
    sesiones, reg, sem = ensamblar_conjunto()
    for prior, ess in VARIANTES:
        nombre = f"{prior}{ess or ''}"
        print("Validación cruzada con", nombre, flush=True)
        pred = validar_variante(prior, ess, sesiones, reg, sem)
        assert len(pred) == 1572
        if (prior, ess) == ("BDeu", 5):
            junto = pred.merge(oficial, on=LLAVE, suffixes=("", "_oficial"))
            assert len(junto) == 1572
            assert np.allclose(junto.prediccion_continua, junto.prediccion_continua_oficial, atol=1e-9), \
                "ESS=5 no reproduce las predicciones oficiales"
        m = metricas(pred, "prediccion_continua").rename(
            columns={"rmse": f"rmse_{nombre}", "r2": f"r2_{nombre}"})
        tabla = tabla.merge(m, on=["materia", "hito"])

    prototipo = pd.read_csv(RUTA_PROTOTIPO).merge(
        oficial[LLAVE + ["total_trimestre_real"]], on=LLAVE)
    assert len(prototipo) == 1572
    m = metricas(prototipo, "pred_bayes").rename(columns={"rmse": "rmse_todos_los_datos", "r2": "r2_todos_los_datos"})
    tabla = tabla.merge(m, on=["materia", "hito"])

    columnas_rmse = [c for c in tabla.columns if c.startswith("rmse_") and c not in ("rmse_lstm", "rmse_lstm_desv")]
    tabla["mejor_rmse_bayes"] = tabla[columnas_rmse].min(axis=1)
    tabla["dentro_2desv_alguna_variante"] = tabla.mejor_rmse_bayes <= tabla.rmse_lstm + 2 * tabla.rmse_lstm_desv
    tabla.to_csv(RUTA_SALIDA, index=False)
    pd.set_option("display.width", 250)
    print(tabla.round(3).to_string(index=False))
    print("Guardado en", RUTA_SALIDA)
