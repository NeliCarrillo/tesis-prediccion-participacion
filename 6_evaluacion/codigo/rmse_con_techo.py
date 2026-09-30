"""RMSE de la red bayesiana ajustado al techo del valor esperado
(comentario del tutor sobre el techo de 12,0 a 22,4 participaciones).

Análisis complementario: NO modifica ningún resultado del informe.

El valor esperado de la red no puede superar la media del estado "12 o más"
en el entrenamiento de cada pliegue (el techo). Aquí, cuando el total real
supera ese techo, se toma el techo como valor correcto, es decir, se evalúa
contra min(real, techo). Así se mide solo el error que la red podía evitar
con su representación. La LSTM no se ajusta, porque sí puede predecir por
encima del techo.

Correr el archivo genera `resultados/rmse_con_techo.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ensamblado import ensamblar_conjunto
from valor_esperado import medias_entrenamiento_por_estado

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_COMPARACION = RAIZ / "6_evaluacion" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"
RUTA_SALIDA = RAIZ / "6_evaluacion" / "resultados" / "rmse_con_techo.csv"


if __name__ == "__main__":
    _sesiones, reg, _sem = ensamblar_conjunto()
    pred = pd.read_csv(RUTA_BAYES)
    assert len(pred) == 1572
    techos = {
        (m, t): float(medias_entrenamiento_por_estado(reg, m, t)["12 o más"])
        for m, t in pred[["materia", "trimestre_prueba"]].drop_duplicates().itertuples(index=False)
    }
    pred["techo"] = [techos[(m, t)] for m, t in zip(pred.materia, pred.trimestre_prueba)]
    pred["real_ajustado"] = np.minimum(pred.total_trimestre_real, pred.techo)
    assert pred.prediccion_continua.max() <= pred.techo.max()

    comparacion = pd.read_csv(RUTA_COMPARACION).set_index(["materia", "hito"])
    filas = []
    for (materia, hito), g in pred.groupby(["materia", "hito"]):
        filas.append({
            "materia": materia, "hito": hito,
            "casos_sobre_techo": int((g.total_trimestre_real > g.techo).sum()),
            "rmse_bayes": float(np.sqrt(mean_squared_error(g.total_trimestre_real, g.prediccion_continua))),
            "rmse_bayes_ajustado_techo": float(np.sqrt(mean_squared_error(g.real_ajustado, g.prediccion_continua))),
            "r2_bayes_ajustado_techo": float(r2_score(g.real_ajustado, g.prediccion_continua)),
            "rmse_lstm": float(comparacion.loc[(materia, hito), "rmse_lstm"]),
        })
    tabla = pd.DataFrame(filas)
    assert np.allclose(tabla.rmse_bayes, comparacion.loc[list(zip(tabla.materia, tabla.hito)), "rmse_bayes"].values)
    tabla.to_csv(RUTA_SALIDA, index=False)
    print(f"Techo entre {pred.techo.min():.1f} y {pred.techo.max():.1f}; "
          f"{int((pred.total_trimestre_real > pred.techo).sum())} predicciones con total real sobre el techo")
    print(tabla.round(3).to_string(index=False))
    print("Guardado en", RUTA_SALIDA)
