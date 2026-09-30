"""Anomalías en la evolución del RMSE entre hitos (comentario del tutor:
"discutir todas las anomalías ... el comportamiento esperado es que, dentro
del margen de error, no haya cambio de pendiente").

Análisis complementario: NO modifica ningún resultado del informe. Usa las
predicciones oficiales de la red bayesiana y las de la LSTM con semilla 42
(las únicas guardadas por estudiante) junto con la extrapolación.

1. Margen de error: intervalo bootstrap del 95 % (4.000 remuestreos de
   estudiantes, pareados entre hitos) para el cambio de RMSE entre hitos y
   para el cambio de pendiente, (S8 − S6) − (S6 − S4).
2. Sesgo medio (predicción − real) por hito y proporción de casos con
   evidencia bayesiana nula (0 participaciones en la semana del hito y en
   la anterior).
3. Calendario: participación media por tipo de sesión, y en sesiones de
   contenido por bloque de semanas (hipótesis 2).
4. Hipótesis 1 ("ya acumularon el 10"): distribución de totales alrededor
   de 10 y participación relativa a la sección antes y después de alcanzar
   el umbral.

Correr el archivo genera `resultados/anomalias_pendiente_*.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ensamblado import cargar_datos_crudos

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "RedBayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_LSTM = RAIZ / "LSTM" / "nuevo" / "predicciones_lstm_validacion_cruzada.csv"
CARPETA = RAIZ / "RedBayesiana" / "resultados"
LLAVE = ["materia", "trimestre", "seccion", "estudiante_id", "hito"]
MODELOS = (("prediccion_continua", "Red bayesiana"), ("prediccion_lstm", "LSTM (semilla 42)"),
           ("prediccion_extrapolacion", "Extrapolación"))
REMUESTREOS = 4000


def cargar_predicciones() -> pd.DataFrame:
    bayes = pd.read_csv(RUTA_BAYES).rename(columns={"trimestre_prueba": "trimestre"})
    lstm = pd.read_csv(RUTA_LSTM)
    datos = bayes[LLAVE + ["total_trimestre_real", "prediccion_continua",
                           "evidencia_participaciones_semana", "evidencia_participaciones_semana_anterior"]].merge(
        lstm[LLAVE + ["prediccion_lstm", "prediccion_extrapolacion"]], on=LLAVE)
    assert len(datos) == 1572
    return datos


def pendientes(datos: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    filas = []
    for materia, g in datos.groupby("materia"):
        for columna, modelo in MODELOS:
            ancho = g.pivot_table(index=["trimestre", "seccion", "estudiante_id"], columns="hito",
                                  values=[columna, "total_trimestre_real"])
            e2 = {h: ((ancho[(columna, h)] - ancho[("total_trimestre_real", h)]) ** 2).to_numpy() for h in (4, 6, 8)}
            idx = rng.integers(0, len(ancho), (REMUESTREOS, len(ancho)))
            muestras = {h: np.sqrt(e2[h][idx].mean(axis=1)) for h in e2}
            puntual = {h: np.sqrt(e2[h].mean()) for h in e2}
            medidas = {
                "S4 a S6": (puntual[6] - puntual[4], muestras[6] - muestras[4]),
                "S6 a S8": (puntual[8] - puntual[6], muestras[8] - muestras[6]),
                "cambio de pendiente": (puntual[8] - 2 * puntual[6] + puntual[4],
                                        muestras[8] - 2 * muestras[6] + muestras[4]),
            }
            for medida, (valor, dist) in medidas.items():
                bajo, alto = np.percentile(dist, [2.5, 97.5])
                filas.append({"materia": materia, "modelo": modelo, "medida": medida, "valor": valor,
                              "ic95_inferior": bajo, "ic95_superior": alto,
                              "distinto_de_cero": bool(bajo > 0 or alto < 0)})
    return pd.DataFrame(filas)


def sesgos(datos: pd.DataFrame) -> pd.DataFrame:
    datos = datos.copy()
    for columna, modelo in MODELOS:
        datos[f"sesgo {modelo}"] = datos[columna] - datos.total_trimestre_real
    datos["evidencia bayesiana nula"] = (
        (datos.evidencia_participaciones_semana.astype(str) == "0")
        & (datos.evidencia_participaciones_semana_anterior.astype(str) == "0"))
    columnas = [f"sesgo {m}" for _, m in MODELOS] + ["evidencia bayesiana nula"]
    return datos.groupby(["materia", "hito"])[columnas].mean().reset_index()


def calendario(crudos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dictadas = crudos[crudos.tipo_sesion != "sin_clase"]
    por_tipo = dictadas.groupby("tipo_sesion").participaciones.agg(media="mean", sesiones_estudiante="size").reset_index()
    contenido = dictadas[dictadas.tipo_sesion == "contenido"].copy()
    contenido["bloque"] = pd.cut(contenido.semana, [0, 4, 8, 12], labels=["1-4", "5-8", "9-12"])
    por_bloque = contenido.groupby(["materia", "bloque"], observed=True).participaciones.mean().reset_index()
    return por_tipo, por_bloque


def umbral(crudos: pd.DataFrame) -> pd.DataFrame:
    registro = ["materia", "trimestre", "seccion", "estudiante_id"]
    semanal = crudos.groupby(registro + ["semana"]).participaciones.sum().reset_index()
    semanal["relativa"] = semanal.participaciones - semanal.groupby(
        ["materia", "trimestre", "seccion", "semana"]).participaciones.transform("mean")
    semanal = semanal.sort_values(registro + ["semana"])
    semanal["acumulada"] = semanal.groupby(registro).participaciones.cumsum()
    filas = []
    for limite in (5, 8, 10):
        casos = []
        for _, g in semanal.groupby(registro):
            g = g.set_index("semana")
            cruce = g.index[g.acumulada >= limite]
            if len(cruce) == 0 or not 4 <= cruce[0] <= 9:
                continue
            w = cruce[0]
            casos.append((g.loc[w - 3:w - 1, "relativa"].mean(), g.loc[w + 1:w + 3, "relativa"].mean(),
                          g.loc[w + 1:, "participaciones"].sum() == 0))
        casos = pd.DataFrame(casos, columns=["antes", "despues", "no_vuelve"])
        filas.append({"umbral": limite, "estudiantes": len(casos),
                      "relativa_3_semanas_antes": casos.antes.mean(),
                      "relativa_3_semanas_despues": casos.despues.mean(),
                      "no_vuelve_a_participar": casos.no_vuelve.mean()})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    datos = cargar_predicciones()
    crudos = cargar_datos_crudos()
    salidas = {"pendientes": pendientes(datos), "sesgos": sesgos(datos)}
    salidas["por_tipo_sesion"], salidas["contenido_por_bloque"] = calendario(crudos)
    salidas["umbral"] = umbral(crudos)
    totales = crudos.groupby(["materia", "trimestre", "seccion", "estudiante_id"]).participaciones.sum()
    print("Totales de 7 a 13:", totales.value_counts().sort_index().loc[7:13].to_dict())
    for nombre, tabla in salidas.items():
        tabla.to_csv(CARPETA / f"anomalias_pendiente_{nombre}.csv", index=False)
        print(f"\n== {nombre}\n{tabla.round(3).to_string(index=False)}")
