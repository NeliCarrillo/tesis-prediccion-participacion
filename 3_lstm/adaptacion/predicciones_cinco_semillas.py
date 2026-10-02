"""Predicciones por estudiante de la LSTM con las cinco semillas del barrido.

El cuaderno `adaptacion_lstm_participaciones.ipynb` solo conservó las
predicciones individuales de la semilla 42; de las otras cuatro guardó métricas
agregadas (`metricas_semillas.csv`). Este script ejecuta las mismas celdas del
cuaderno que preparan los datos y definen `validacion_cruzada` (sin copiar ni
modificar su código) y repite la validación cruzada con cada semilla, guardando
las predicciones por estudiante con el mismo orden de identificadores de la
celda de exportación.

Comprobaciones: la semilla 42 debe reproducir
`predicciones_lstm_validacion_cruzada.csv` y cada semilla debe reproducir su
RMSE y R² de `metricas_semillas.csv`. No modifica ninguna métrica publicada.

Correr el archivo genera `3_lstm/resultados/predicciones_lstm_cinco_semillas.csv`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
CUADERNO = AQUI / "adaptacion_lstm_participaciones.ipynb"
RESULTADOS = RAIZ / "3_lstm" / "resultados"
SALIDA = RESULTADOS / "predicciones_lstm_cinco_semillas.csv"
SEMILLAS = [42, 7, 123, 2024, 31]
CELDA_ENTRENAMIENTO = 34  # primera celda que entrena; las anteriores solo preparan


def cargar_funciones_del_cuaderno() -> dict:
    """Ejecuta las celdas de código anteriores al entrenamiento."""
    os.chdir(AQUI)
    celdas = json.loads(CUADERNO.read_text())["cells"]
    espacio: dict = {}
    for i, celda in enumerate(celdas[:CELDA_ENTRENAMIENTO]):
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        fuente = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
        exec(compile(fuente, f"celda_{i}", "exec"), espacio)
    for nombre in ("validacion_cruzada", "calcular_metricas", "estatica", "X", "X_temas", "y", "HITOS", "CLAVES"):
        assert nombre in espacio, f"el cuaderno no definió {nombre}"
    return espacio


def predecir(espacio: dict):
    import numpy as np
    import pandas as pd
    estatica, X, X_temas, y = espacio["estatica"], espacio["X"], espacio["X_temas"], espacio["y"]
    filas_metricas, bloques = [], []
    for semilla in SEMILLAS:
        for materia in sorted(estatica.materia.unique()):
            filas = (estatica.materia == materia).to_numpy()
            estatica_materia = estatica[filas].reset_index(drop=True)
            trimestres = estatica_materia.trimestre.to_numpy()
            for hito in espacio["HITOS"]:
                salida = espacio["validacion_cruzada"](X[filas], X_temas[filas], y[filas], trimestres, hito,
                                                       semilla=semilla)
                ids = []
                for periodo in sorted(np.unique(trimestres)):
                    prueba = trimestres == periodo
                    if prueba.sum() < 10 or (~prueba).sum() < 30:
                        continue
                    ids.append(estatica_materia.loc[prueba, espacio["CLAVES"]])
                bloque = pd.concat(ids, ignore_index=True)
                assert len(bloque) == len(salida["real"])
                bloque["hito"], bloque["semilla"] = hito, semilla
                bloque["total_trimestre_real"] = salida["real"]
                bloque["prediccion_lstm"] = salida["modelo"]
                bloques.append(bloque)
                filas_metricas.append({"semilla": semilla, "materia": materia, "hito": hito,
                                       **espacio["calcular_metricas"](salida["real"], salida["modelo"])})
        print(f"semilla {semilla} terminada", flush=True)
    return pd.concat(bloques, ignore_index=True), pd.DataFrame(filas_metricas)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    espacio = cargar_funciones_del_cuaderno()
    predicciones, metricas = predecir(espacio)
    assert len(predicciones) == 1572 * len(SEMILLAS)

    publicadas = pd.read_csv(RESULTADOS / "metricas_semillas.csv")
    comparacion = metricas.merge(publicadas, on=["semilla", "materia", "hito"], suffixes=("", "_publicado"))
    assert len(comparacion) == 60
    dif_rmse = (comparacion.rmse.round(4) - comparacion.rmse_publicado).abs().max()
    dif_r2 = (comparacion.r2.round(4) - comparacion.r2_publicado).abs().max()
    print(f"diferencia máxima con metricas_semillas.csv: RMSE {dif_rmse:.4f}, R² {dif_r2:.4f}")
    assert dif_rmse <= 1e-4 and dif_r2 <= 1e-4, "las métricas no reproducen las publicadas"

    oficial = pd.read_csv(RESULTADOS / "predicciones_lstm_validacion_cruzada.csv")
    llave = ["materia", "trimestre", "seccion", "estudiante_id", "hito"]
    s42 = predicciones[predicciones.semilla == 42].merge(oficial[llave + ["prediccion_lstm"]], on=llave,
                                                         suffixes=("", "_oficial"))
    assert len(s42) == 1572
    assert np.allclose(s42.prediccion_lstm, s42.prediccion_lstm_oficial, atol=1e-4), \
        "la semilla 42 no reproduce las predicciones oficiales"
    predicciones.to_csv(SALIDA, index=False)
    print("Reproducción verificada. Guardado en", SALIDA)
