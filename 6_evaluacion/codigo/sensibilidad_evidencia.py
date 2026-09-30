"""Sensibilidad de la predicción de la red bayesiana ante cambios en la
evidencia (Tabla 20 del informe), por asignatura.

Análisis complementario: NO modifica ningún resultado del informe. Para cada
una de las 1.572 predicciones de validación cruzada, con el modelo del pliegue
(ESS = 5) y la evidencia C1, cada variable de evidencia observada se sustituye
por cada uno de sus estados y, además, se omite (la red la marginaliza),
manteniendo fijas las demás. La sensibilidad del caso a esa variable es el
rango entre la mayor y la menor predicción obtenidas. Se exige reproducir los
valores globales publicados en la Tabla 20.

Correr el archivo genera `resultados/sensibilidad_evidencia_casos.csv` y
`resultados/sensibilidad_evidencia_por_asignatura.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ajuste_cpd import ESS_SELECCIONADO, ajustar_cpd, preparar_fold
from ensamblado import ESTADOS_BN, ensamblar_conjunto
from red_bayesiana import construir_modelo_manual
from valor_esperado import medias_entrenamiento_por_estado, valor_esperado

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
CARPETA = RAIZ / "6_evaluacion" / "resultados"
OBJETIVO = "Cantidad de participaciones del trimestre"
EVIDENCIA = {
    "Participaciones de la semana": "evidencia_participaciones_semana",
    "Participaciones de la semana anterior": "evidencia_participaciones_semana_anterior",
    "Año que cursa": "evidencia_anio_que_cursa",
    "Tamaño del grupo": "evidencia_tamano_grupo",
}
# Valores publicados en la Tabla 20: (n, rango medio, mediana, rango máximo).
# El rango máximo del tamaño del grupo es 16,8547 y debe publicarse como 16,85
# (el informe muestra 16,86); por eso el máximo admite una diferencia de 0,01.
TABLA_20 = {
    "Participaciones de la semana": (1572, 7.55, 6.73, 21.33),
    "Participaciones de la semana anterior": (1572, 7.52, 6.73, 21.25),
    "Año que cursa": (1563, 4.63, 5.14, 18.79),
    "Tamaño del grupo": (1572, 4.51, 4.53, 16.86),
}


def calcular() -> pd.DataFrame:
    sesiones, reg, _sem = ensamblar_conjunto()
    casos = pd.read_csv(RUTA_BAYES, dtype=str)
    assert len(casos) == 1572
    filas = []
    for (materia, trimestre), g in casos.groupby(["materia", "trimestre_prueba"]):
        train_bn, _test, _rep = preparar_fold(sesiones, materia, trimestre)
        inferencia = VariableElimination(ajustar_cpd(construir_modelo_manual(), train_bn, ESS_SELECCIONADO))
        medias = medias_entrenamiento_por_estado(reg, materia, trimestre)
        cache: dict[tuple, float] = {}

        def predecir(evidencia: dict) -> float:
            clave = tuple(sorted(evidencia.items()))
            if clave not in cache:
                r = inferencia.query([OBJETIVO], evidence=evidencia, show_progress=False)
                cache[clave] = valor_esperado({e: float(p) for e, p in zip(r.state_names[OBJETIVO], r.values)}, medias)
            return cache[clave]

        for _, caso in g.iterrows():
            base = {n: caso[c] for n, c in EVIDENCIA.items() if isinstance(caso[c], str) and caso[c] != "nan"}
            fila = {"materia": materia, "trimestre": trimestre, "estudiante_id": caso.estudiante_id,
                    "seccion": caso.seccion, "hito": int(caso.hito)}
            for nodo in EVIDENCIA:
                if nodo not in base:
                    fila[nodo] = np.nan
                    continue
                predicciones = [predecir({**base, nodo: estado}) for estado in ESTADOS_BN[nodo]]
                sin_nodo = {k: v for k, v in base.items() if k != nodo}
                predicciones.append(predecir(sin_nodo))
                fila[nodo] = max(predicciones) - min(predicciones)
            filas.append(fila)
    return pd.DataFrame(filas)


def resumir(casos: pd.DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    grupos = [("Todas", casos)] if not por else list(casos.groupby(por))
    filas = []
    for clave, g in grupos:
        for nodo in EVIDENCIA:
            v = g[nodo].dropna()
            filas.append({"grupo": clave, "variable": nodo, "n": len(v), "rango_medio": v.mean(),
                          "mediana": v.median(), "q1": v.quantile(0.25), "q3": v.quantile(0.75), "maximo": v.max()})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    casos = calcular()
    glob = resumir(casos)
    for _, f in glob.iterrows():
        n, media, mediana, maximo = TABLA_20[f.variable]
        assert f.n == n and round(f.rango_medio, 2) == media and round(f.mediana, 2) == mediana \
            and abs(f.maximo - maximo) <= 0.01, (f.variable, f.n, f.rango_medio, f.mediana, f.maximo)
    print("Tabla 20 reproducida.\n", glob.round(2).to_string(index=False))
    por_materia = resumir(casos, ["materia"])
    print("\nPor asignatura:\n", por_materia.round(2).to_string(index=False))
    temporal = casos[["Participaciones de la semana", "Participaciones de la semana anterior"]].max(axis=1)
    contextual = casos[["Año que cursa", "Tamaño del grupo"]].max(axis=1)
    casos["contextual_supera_temporales"] = contextual > temporal
    print("\nCasos en que una variable contextual supera a ambas temporales:")
    print(casos.groupby("materia").contextual_supera_temporales.mean().round(3).to_string(),
          "| global", round(casos.contextual_supera_temporales.mean(), 3))
    casos.to_csv(CARPETA / "sensibilidad_evidencia_casos.csv", index=False)
    por_materia.to_csv(CARPETA / "sensibilidad_evidencia_por_asignatura.csv", index=False)


def figura(casos: pd.DataFrame) -> None:
    """Figura de sensibilidad por asignatura: un diagrama de caja por variable
    de evidencia (la caja cubre la mitad central de los casos y los bigotes, del
    percentil 5 al 95) y un rombo con el rango medio."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
    materias = ("Algoritmos y Programación", "Computación Emergente", "Estructura de Datos", "Matemáticas Discretas")
    variables = list(EVIDENCIA)
    etiquetas = ["Participaciones\nde la semana", "Participaciones de la\nsemana anterior", "Año que cursa", "Tamaño del grupo"]
    colores = {"Participaciones de la semana": "#1f4e79", "Participaciones de la semana anterior": "#1f4e79",
               "Año que cursa": "#b5651d", "Tamaño del grupo": "#b5651d"}
    figura_, ejes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True, sharey=True)
    for eje, materia in zip(ejes.flat, materias):
        g = casos[casos.materia == materia]
        datos = [g[v].dropna().to_numpy() for v in variables]
        posiciones = list(range(len(variables), 0, -1))
        cajas = eje.boxplot(datos, positions=posiciones, vert=False, widths=0.55, whis=(5, 95), showfliers=False,
                            patch_artist=True, medianprops=dict(color="white", linewidth=1.2))
        for caja, v in zip(cajas["boxes"], variables):
            caja.set(facecolor=colores[v], edgecolor=colores[v], alpha=0.85)
        for parte in ("whiskers", "caps"):
            for linea, v in zip(cajas[parte], [x for x in variables for _ in (0, 1)]):
                linea.set(color=colores[v], linewidth=0.8)
        for y, v in zip(posiciones, variables):
            media = g[v].mean()
            eje.plot(media, y, marker="D", color="black", markersize=4, zorder=5)
            eje.text(media, y + 0.36, f"{media:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=7)
        eje.set_title(materia, fontsize=9)
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas, fontsize=7.5)
        eje.set_xlim(0, 22)
        eje.grid(axis="x", color="0.9", linewidth=0.5)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
    for eje in ejes[1]:
        eje.set_xlabel("Rango de la predicción (participaciones)")
    figura_.tight_layout(h_pad=1.2, w_pad=1.0)
    salida = RAIZ / "figuras" / "resultados" / "figura_sensibilidad_por_asignatura.png"
    figura_.savefig(salida, dpi=300, bbox_inches="tight", facecolor="white")
    print("Figura guardada en", salida)


if __name__ == "__main__":
    figura(pd.read_csv(CARPETA / "sensibilidad_evidencia_casos.csv"))
