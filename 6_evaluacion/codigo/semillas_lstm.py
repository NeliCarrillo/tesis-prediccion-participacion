"""Robustez de los análisis por estudiante frente a la semilla de la LSTM
(comentario del tutor: "idealmente, correrlo con varias semillas").

Análisis complementario: NO modifica ningún resultado del informe. Repite, con
las predicciones por estudiante de las cinco semillas
(`3_lstm/resultados/predicciones_lstm_cinco_semillas.csv`, generadas por
`3_lstm/adaptacion/predicciones_cinco_semillas.py`), los cálculos que en el
informe usan solo la semilla 42:

1. Posición de cada semilla y su distancia a la media de las cinco, por
   asignatura e hito (respaldo de la elección de la semilla 42, Apéndice C).
2. Cambio de pendiente del RMSE entre hitos con intervalo bootstrap del 95%
   (filas de la LSTM de la Tabla J2).
3. RMSE de la LSTM en los casos sin participación en las dos semanas
   observadas (Algoritmos y Programación-S6 y Estructura de Datos-S8).
4. RMSE de la LSTM en los casos con respaldo (brecha de Comparación LSTM y Bayes).
5. Diferencia promedio entre el RMSE de la LSTM (media de las cinco semillas) y el de
   la red bayesiana y la extrapolación proporcional, comparada con la variación del
   RMSE entre semillas (comentario del tutor: "reportar la diferencia promedio").
6. Variación entre semillas de la predicción individual (mayor menos menor predicción de
   un mismo estudiante e hito), que el RMSE agregado no muestra porque los errores se compensan.

Correr el archivo genera `6_evaluacion/resultados/semillas_lstm_*.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_SEMILLAS = RAIZ / "3_lstm" / "resultados" / "predicciones_lstm_cinco_semillas.csv"
RUTA_METRICAS = RAIZ / "3_lstm" / "resultados" / "metricas_semillas.csv"
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_COMPARACION = RAIZ / "6_evaluacion" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"
RUTA_RESPALDO = RAIZ / "6_evaluacion" / "resultados" / "respaldo_evidencia_casos.csv"
CARPETA = RAIZ / "6_evaluacion" / "resultados"
LLAVE = ["materia", "trimestre", "seccion", "estudiante_id", "hito"]
SEMILLAS = [42, 7, 123, 2024, 31]
REMUESTREOS = 4000


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def posiciones(metricas: pd.DataFrame) -> pd.DataFrame:
    m = metricas.copy()
    m["puesto_rmse"] = m.groupby(["materia", "hito"]).rmse.rank()
    m["media_cinco"] = m.groupby(["materia", "hito"]).rmse.transform("mean")
    m["distancia_a_la_media"] = (m.rmse - m.media_cinco).abs()
    return m


def brechas(metricas: pd.DataFrame) -> pd.DataFrame:
    """Variación del RMSE entre semillas (máximo menos mínimo) y diferencia de RMSE entre
    cada método y la media de las cinco semillas de la LSTM, por asignatura e hito."""
    variacion = (metricas.groupby(["materia", "hito"]).rmse.agg(["mean", "min", "max"])
                 .assign(rango_semillas=lambda t: t["max"] - t["min"]))
    comparacion = pd.read_csv(RUTA_COMPARACION).set_index(["materia", "hito"])
    t = variacion.join(comparacion[["rmse_bayes", "rmse_extrapolacion", "rmse_lstm"]])
    assert np.allclose(t["mean"], t.rmse_lstm, atol=1e-4), "la media de las semillas no coincide con la Tabla 16"
    t["dif_bayes_lstm"] = t.rmse_bayes - t["mean"]
    t["dif_extrapolacion_lstm"] = t.rmse_extrapolacion - t["mean"]
    t["veces_bayes"] = t.dif_bayes_lstm / t.rango_semillas
    t["extrapolacion_dentro_del_rango"] = t.dif_extrapolacion_lstm.abs() < t.rango_semillas
    # Valores publicados: Apéndice C (variación media 0,21; mínimo 0,09; máximo 0,38) y Tabla 21.
    assert round(t.rango_semillas.mean(), 2) == 0.21
    assert round(t.rango_semillas.min(), 2) == 0.09 and round(t.rango_semillas.max(), 2) == 0.38
    assert round(t.dif_bayes_lstm.min(), 2) == 0.26 and round(t.dif_bayes_lstm.max(), 2) == 4.35
    return t.reset_index()


def dispersion_individual(pred: pd.DataFrame) -> pd.DataFrame:
    """Rango entre las cinco semillas de la predicción de cada estudiante e hito."""
    g = pred.groupby(LLAVE).prediccion_lstm.agg(["count", "min", "max"])
    assert (g["count"] == len(SEMILLAS)).all() and len(g) == 1572
    g["rango"] = g["max"] - g["min"]
    return g.reset_index()


def pendientes(pred: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for semilla in SEMILLAS:
        rng = np.random.default_rng(0)
        for materia, g in pred[pred.semilla == semilla].groupby("materia"):
            ancho = g.pivot_table(index=["trimestre", "seccion", "estudiante_id"], columns="hito",
                                  values=["prediccion_lstm", "total_trimestre_real"])
            e2 = {h: ((ancho[("prediccion_lstm", h)] - ancho[("total_trimestre_real", h)]) ** 2).to_numpy()
                  for h in (4, 6, 8)}
            idx = rng.integers(0, len(ancho), (REMUESTREOS, len(ancho)))
            muestra = {h: np.sqrt(e2[h][idx].mean(axis=1)) for h in e2}
            puntual = {h: np.sqrt(e2[h].mean()) for h in e2}
            valor = puntual[8] - 2 * puntual[6] + puntual[4]
            bajo, alto = np.percentile(muestra[8] - 2 * muestra[6] + muestra[4], [2.5, 97.5])
            filas.append({"semilla": semilla, "materia": materia, "cambio_pendiente": valor,
                          "ic95_inferior": bajo, "ic95_superior": alto,
                          "distinto_de_cero": bool(bajo > 0 or alto < 0)})
    return pd.DataFrame(filas)


def casos_puntuales(pred: pd.DataFrame) -> pd.DataFrame:
    bayes = pd.read_csv(RUTA_BAYES).rename(columns={"trimestre_prueba": "trimestre"})
    bayes["evidencia_nula"] = ((bayes.evidencia_participaciones_semana.astype(str) == "0")
                               & (bayes.evidencia_participaciones_semana_anterior.astype(str) == "0"))
    respaldo = pd.read_csv(RUTA_RESPALDO)[LLAVE + ["respaldo"]]
    datos = pred.merge(bayes[LLAVE + ["evidencia_nula"]], on=LLAVE).merge(respaldo, on=LLAVE, how="left")
    filas = []
    for semilla, g in datos.groupby("semilla"):
        fila = {"semilla": semilla}
        for materia, hito, clave in (("Algoritmos y Programación", 6, "nula_ayp_s6"),
                                     ("Estructura de Datos", 8, "nula_ed_s8")):
            n = g[(g.materia == materia) & (g.hito == hito) & g.evidencia_nula]
            fila[f"rmse_{clave}"] = _rmse(n.prediccion_lstm, n.total_trimestre_real)
            fila[f"casos_{clave}"] = len(n)
        con = g[g.respaldo > 0]
        fila["rmse_con_respaldo"] = _rmse(con.prediccion_lstm, con.total_trimestre_real)
        fila["casos_con_respaldo"] = len(con)
        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pred = pd.read_csv(RUTA_SEMILLAS)
    assert len(pred) == 1572 * len(SEMILLAS)
    metricas = pd.read_csv(RUTA_METRICAS)
    tabla_posiciones = posiciones(metricas)
    tabla_pendientes = pendientes(pred)
    tabla_casos = casos_puntuales(pred)
    tabla_brechas = brechas(metricas)
    dispersion = dispersion_individual(pred)
    pd.set_option("display.width", 220)
    resumen = tabla_posiciones.groupby("semilla").agg(rmse_medio=("rmse", "mean"), puesto_medio=("puesto_rmse", "mean"),
                                                      distancia_media=("distancia_a_la_media", "mean"),
                                                      veces_mediana=("puesto_rmse", lambda x: int((x == 3).sum())),
                                                      veces_extremo=("puesto_rmse", lambda x: int(x.isin([1, 5]).sum())))
    print("Posición de cada semilla:\n", resumen.round(3).to_string())
    print("\nCambio de pendiente de la LSTM por semilla:\n", tabla_pendientes.round(2).to_string(index=False))
    print("\nCasos puntuales por semilla:\n", tabla_casos.round(2).to_string(index=False))
    print("\nDiferencia promedio frente a la variación entre semillas (RMSE, participaciones):")
    print("  variación entre semillas (media de las 12 combinaciones):", round(tabla_brechas.rango_semillas.mean(), 2))
    print("  red bayesiana menos LSTM:", round(tabla_brechas.dif_bayes_lstm.mean(), 2),
          "| mínimo en veces la variación:", round(tabla_brechas.veces_bayes.min(), 1))
    print("  extrapolación menos LSTM:", round(tabla_brechas.dif_extrapolacion_lstm.mean(), 2),
          "| combinaciones con diferencia menor que la variación:", int(tabla_brechas.extrapolacion_dentro_del_rango.sum()))
    print(tabla_brechas.round(2).to_string(index=False))
    print("\nRango entre semillas de la predicción individual (participaciones):")
    print(dispersion.rango.describe(percentiles=[.5, .9, .95]).round(2).to_string())
    print("  casos con rango mayor que 1:", round((dispersion.rango > 1).mean(), 3),
          "| mayor que 2:", round((dispersion.rango > 2).mean(), 3))
    dispersion.to_csv(CARPETA / "semillas_lstm_dispersion_individual.csv", index=False)
    tabla_brechas.to_csv(CARPETA / "semillas_lstm_brechas.csv", index=False)
    tabla_posiciones.to_csv(CARPETA / "semillas_lstm_posiciones.csv", index=False)
    resumen.reset_index().to_csv(CARPETA / "semillas_lstm_resumen.csv", index=False)
    tabla_pendientes.to_csv(CARPETA / "semillas_lstm_pendientes.csv", index=False)
    tabla_casos.to_csv(CARPETA / "semillas_lstm_casos.csv", index=False)
