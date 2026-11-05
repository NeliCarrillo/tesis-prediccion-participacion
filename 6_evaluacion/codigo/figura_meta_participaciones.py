"""Apéndice J, hipótesis de la meta de participaciones: curva de supervivencia
empírica del total de participaciones del trimestre.

Para cada valor x se cuenta cuántos registros estudiante-sección hicieron al
menos x participaciones en el trimestre. Se usan los 524 registros una sola
vez (sin triplicarlos por los hitos S4, S6 y S8). Si los estudiantes dejaran de
participar al llegar a una meta (10 o 15), la curva mostraría una caída
abrupta justo después de ese valor; si no hay meta, desciende de forma gradual.

Para comprobar la caída se calcula, en cada valor x, la proporción de los
registros que llegaron a x y se quedaron exactamente en x (tasa de detención).
En 10 y 15 se compara con la tasa conjunta de los tres valores anteriores y los
tres siguientes, con una prueba binomial unilateral (¿se detienen más de lo
esperado?).

No modifica ningún resultado: lee `1_datos/estandarizados/` y comprueba las
cifras ya citadas en el informe (14 registros con 10, 15 con 9 y 11 con 11).

Correr el archivo genera
`figuras/apendice_J_anomalias/figura_supervivencia_participaciones.png` y
`6_evaluacion/resultados/meta_participaciones_supervivencia.csv`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ensamblado import cargar_datos_crudos

RAIZ = Path(__file__).resolve().parent.parent.parent
SALIDA_FIGURA = RAIZ / "figuras" / "apendice_J_anomalias" / "figura_supervivencia_participaciones.png"
SALIDA_CSV = RAIZ / "6_evaluacion" / "resultados" / "meta_participaciones_supervivencia.csv"
REGISTRO = ["materia", "trimestre", "seccion", "estudiante_id"]
UMBRALES = (10, 15)
VECINOS = 3
LINEA = "#3b5b92"


def _coma(valor: float, decimales: int = 2) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def supervivencia() -> tuple[pd.DataFrame, pd.DataFrame]:
    totales = cargar_datos_crudos().groupby(REGISTRO).participaciones.sum()
    assert len(totales) == 524 and totales.max() == 44
    conteo = totales.value_counts()
    assert (conteo[9], conteo[10], conteo[11]) == (15, 14, 11)   # cifras citadas en el Apéndice J

    x = np.arange(0, int(totales.max()) + 1)
    tabla = pd.DataFrame({"participaciones": x,
                          "registros_al_menos": [(totales >= v).sum() for v in x],
                          "registros_exactamente": [(totales == v).sum() for v in x]})
    tabla["proporcion_al_menos"] = tabla.registros_al_menos / len(totales)
    tabla["tasa_detencion"] = tabla.registros_exactamente / tabla.registros_al_menos
    assert tabla.registros_al_menos.iloc[0] == 524

    pruebas = []
    for u in UMBRALES:
        fila = tabla.set_index("participaciones").loc[u]
        vecinos = tabla[tabla.participaciones.between(u - VECINOS, u + VECINOS) & (tabla.participaciones != u)]
        tasa_vecinos = vecinos.registros_exactamente.sum() / vecinos.registros_al_menos.sum()
        prueba = binomtest(int(fila.registros_exactamente), int(fila.registros_al_menos), tasa_vecinos,
                           alternative="greater")
        pruebas.append({"umbral": u, "registros_al_menos": int(fila.registros_al_menos),
                        "registros_exactamente": int(fila.registros_exactamente),
                        "tasa_detencion": fila.tasa_detencion, "tasa_vecinos": tasa_vecinos,
                        "p_valor": prueba.pvalue})
    return tabla, pd.DataFrame(pruebas)


def graficar(tabla: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9.5})
    figura, ejes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    datos = tabla[tabla.registros_al_menos > 0]
    for eje, logaritmica in zip(ejes, (False, True)):
        eje.plot(datos.participaciones, datos.registros_al_menos, "-o", color=LINEA, linewidth=1.3,
                 markersize=2.8, zorder=3)
        for u in UMBRALES:
            n = int(tabla.set_index("participaciones").registros_al_menos[u])
            eje.axvline(u, color="0.45", linestyle="--", linewidth=0.9, zorder=1)
            eje.annotate(f"{n} registros", xy=(u, n), xytext=(4, 4), textcoords="offset points", fontsize=8.5,
                         color="0.2", zorder=4)
        if logaritmica:
            eje.set_yscale("log")
            eje.set_yticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ["1", "2", "5", "10", "20", "50", "100", "200", "500"])
            eje.minorticks_off()
            eje.set_ylim(0.8, 700)
        else:
            eje.set_ylim(0, 560)
        eje.set_xlim(-1, 45)
        eje.set_xticks([0, 5, 10, 15, 20, 30, 40])
        eje.grid(color="0.9", linewidth=0.6)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        eje.set_xlabel("Participaciones en el trimestre (x)")
    ejes[0].set_ylabel("Registros con al menos x participaciones")
    ejes[0].set_title(r"$\mathbf{(a)}$ Escala lineal", fontsize=9.5, loc="left")
    ejes[1].set_title(r"$\mathbf{(b)}$ Escala logarítmica", fontsize=9.5, loc="left")
    figura.tight_layout(w_pad=1.5)
    SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(SALIDA_FIGURA, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    tabla, pruebas = supervivencia()
    tabla.to_csv(SALIDA_CSV, index=False)
    print(tabla[tabla.participaciones <= 25].round(3).to_string(index=False))
    print("\nComparación con los tres valores anteriores y los tres siguientes:")
    print(pruebas.round(3).to_string(index=False))
    graficar(tabla)
    print("Figura guardada en", SALIDA_FIGURA)
