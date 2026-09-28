"""Comparación cuantitativa de los 3 enfoques — carta 7 de Sprint 6 ("Evaluar
con RMSE y R², y comparar contra la LSTM y la extrapolación proporcional").

No re-ejecuta ninguna validación cruzada ni reentrena nada: lee las salidas
ya persistidas de cada pipeline (Sprint 2 para LSTM/extrapolación, Sprint 4
para la red bayesiana) y las agrega por materia × hito, reuniendo las
predicciones de los pliegues de cada materia.

Las columnas de la LSTM son la media y la desviación estándar de las cinco
semillas (`LSTM/nuevo/metricas_semillas.csv`), las mismas que reporta la
Tabla 16. No se usa la fila "modelo" de `metricas_adaptacion.csv`, que
corresponde a una sola semilla (42).

Importar este módulo no ejecuta nada; correr el archivo genera y guarda la
tabla comparativa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_PREDICCIONES_BAYES = RAIZ / "RedBayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_METRICAS_ADAPTACION = RAIZ / "LSTM" / "nuevo" / "metricas_adaptacion.csv"
RUTA_PREDICCIONES_LSTM = RAIZ / "LSTM" / "nuevo" / "predicciones_lstm_validacion_cruzada.csv"
RUTA_METRICAS_SEMILLAS = RAIZ / "LSTM" / "nuevo" / "metricas_semillas.csv"
RUTA_RESUMEN_SEMILLAS = RAIZ / "LSTM" / "nuevo" / "metricas_semillas_resumen.csv"
RUTA_METRICAS_NAN_DROPNA = RAIZ / "RedBayesiana" / "resultados" / "comparacion_nan_vs_dropna_metricas.csv"
RUTA_SALIDA = RAIZ / "RedBayesiana" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"

N_SEMILLAS = 5


def metricas_por_materia_hito(predicciones: pd.DataFrame, columna: str, sufijo: str) -> pd.DataFrame:
    """RMSE/MAE/R² por materia × hito de la columna de predicción dada,
    calculados sobre las predicciones individuales (sin redondeo previo)."""
    filas = []
    for (materia, hito), grupo in predicciones.groupby(["materia", "hito"]):
        real = grupo["total_trimestre_real"]
        pred = grupo[columna]
        filas.append({
            "materia": materia,
            "hito": hito,
            f"rmse_{sufijo}": float(np.sqrt(mean_squared_error(real, pred))),
            f"mae_{sufijo}": float(mean_absolute_error(real, pred)),
            f"r2_{sufijo}": float(r2_score(real, pred)),
        })
    return pd.DataFrame(filas)


def metricas_lstm_cinco_semillas(metricas_semillas: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación estándar muestral de RMSE/MAE/R² entre las cinco
    semillas, por materia × hito (mismo criterio que la Tabla 16)."""
    assert metricas_semillas["semilla"].nunique() == N_SEMILLAS, "se esperaban cinco semillas"
    agregado = metricas_semillas.groupby(["materia", "hito"]).agg(
        rmse_lstm=("rmse", "mean"), rmse_lstm_desv=("rmse", "std"),
        mae_lstm=("mae", "mean"), mae_lstm_desv=("mae", "std"),
        r2_lstm=("r2", "mean"), r2_lstm_desv=("r2", "std"),
    )
    return agregado.reset_index()


def construir_comparacion() -> pd.DataFrame:
    """Tabla ancha materia × hito con RMSE/MAE/R² de los 3 enfoques."""
    bayes = metricas_por_materia_hito(pd.read_csv(RUTA_PREDICCIONES_BAYES), "prediccion_continua", "bayes")
    lstm = metricas_lstm_cinco_semillas(pd.read_csv(RUTA_METRICAS_SEMILLAS))
    # La extrapolación es determinista: se calcula desde las predicciones
    # individuales, no desde metricas_adaptacion.csv (redondeado a 4 decimales).
    extrapolacion = metricas_por_materia_hito(
        pd.read_csv(RUTA_PREDICCIONES_LSTM), "prediccion_extrapolacion", "extrapolacion"
    )

    comparacion = bayes.merge(lstm, on=["materia", "hito"], how="inner")
    comparacion = comparacion.merge(extrapolacion, on=["materia", "hito"], how="inner")
    return comparacion.sort_values(["materia", "hito"]).reset_index(drop=True)


if __name__ == "__main__":
    comparacion = construir_comparacion()

    assert len(comparacion) == 12, f"se esperaban 12 filas (4 materias x 3 hitos), hay {len(comparacion)}"

    # Bayes: debe coincidir con rmse_nan (tratamiento principal) ya persistido.
    referencia = pd.read_csv(RUTA_METRICAS_NAN_DROPNA)
    cruce = comparacion.merge(referencia, on=["materia", "hito"])
    assert len(cruce) == 12, "no se pudo cruzar con comparacion_nan_vs_dropna_metricas.csv"
    assert np.allclose(cruce["rmse_bayes"], cruce["rmse_nan"], atol=1e-6), "RMSE de Bayes no coincide con rmse_nan"

    # LSTM: la media y la desviación muestral (n-1) de cinco semillas deben
    # coincidir con el resumen que alimenta la Tabla 16, que guarda 3 decimales.
    resumen = pd.read_csv(RUTA_RESUMEN_SEMILLAS)
    cruce_lstm = comparacion.merge(resumen, on=["materia", "hito"])
    assert np.allclose(cruce_lstm["rmse_lstm"], cruce_lstm["rmse_media"], atol=6e-4), "RMSE LSTM no coincide con rmse_media"
    assert np.allclose(cruce_lstm["rmse_lstm_desv"], cruce_lstm["rmse_desv"], atol=6e-4), "desviación LSTM no coincide con rmse_desv"
    assert np.allclose(cruce_lstm["r2_lstm"], cruce_lstm["r2_media"], atol=6e-4), "R² LSTM no coincide con r2_media"

    # Extrapolación: debe coincidir con metricas_adaptacion.csv (4 decimales).
    adaptacion = pd.read_csv(RUTA_METRICAS_ADAPTACION)
    ref_ext = adaptacion[adaptacion["fuente"] == "extrapolación"].merge(comparacion, on=["materia", "hito"])
    assert len(ref_ext) == 12, "no se pudo cruzar la extrapolación con metricas_adaptacion.csv"
    assert np.allclose(ref_ext["rmse"], ref_ext["rmse_extrapolacion"], atol=6e-5), "RMSE extrapolación no coincide"
    assert np.allclose(ref_ext["r2"], ref_ext["r2_extrapolacion"], atol=6e-5), "R² extrapolación no coincide"

    print(comparacion.round(4).to_string(index=False))
    print()
    print("verificaciones superadas: 12 filas; RMSE de Bayes coincide con rmse_nan; "
          "RMSE, desviación y R² de LSTM coinciden con metricas_semillas_resumen.csv (Tabla 16); "
          "extrapolación coincide con metricas_adaptacion.csv")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    comparacion.to_csv(RUTA_SALIDA, index=False)
    print(f"\nCSV guardado en: {RUTA_SALIDA}")
