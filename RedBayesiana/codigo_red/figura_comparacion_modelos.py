"""Figura de Resultados (Sprint 6): RMSE de los tres enfoques (LSTM, red
bayesiana, extrapolación proporcional) por asignatura e hito.

La LSTM se representa con la media de las cinco semillas y barras de error
de ± una desviación estándar muestral entre semillas, las mismas cifras de
la Tabla 16. No recalcula ninguna métrica: lee
`comparacion_bayes_lstm_extrapolacion.csv` (generado por
`comparacion_modelos.py`).

El título va en el caption del informe, no dentro de la imagen. Ancho de
6,5 pulgadas para que el texto conserve al menos 9 pt al insertarse a ancho
de página.

Importar este módulo no ejecuta nada; correr el archivo genera la figura.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_COMPARACION = RAIZ / "RedBayesiana" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"
RUTA_SALIDA_FIGURA = RAIZ / "RedBayesiana" / "figuras" / "figura_comparacion_rmse_modelos.png"

MATERIAS: tuple[str, ...] = (
    "Algoritmos y Programación", "Computación Emergente",
    "Estructura de Datos", "Matemáticas Discretas",
)
HITOS: tuple[int, ...] = (4, 6, 8)
TAMANO_TEXTO = 9
TAMANO_VALOR = 7.5
SERIES = (
    ("rmse_lstm", "tab:blue"),
    ("rmse_bayes", "tab:red"),
    ("rmse_extrapolacion", "dimgray"),
)


def _con_coma(valor: float, decimales: int = 2) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


# Posiciones candidatas del valor respecto de su punto, en puntos
# tipográficos: (dx, dy, alineación horizontal, alineación vertical,
# penalización). Se prefieren justo arriba o justo abajo del punto.
CANDIDATOS = (
    (0, 5, "center", "bottom", 0.0), (0, -5, "center", "top", 0.0),
    (5, 4, "left", "bottom", 1.0), (-5, 4, "right", "bottom", 1.0),
    (5, -4, "left", "top", 1.0), (-5, -4, "right", "top", 1.0),
    (0, 12, "center", "bottom", 2.0), (0, -12, "center", "top", 2.0),
)


def _segmentos_en_pantalla(eje, g: pd.DataFrame) -> list:
    """Líneas, barras de error y marcadores del panel, en píxeles, que un
    valor no debe tapar."""
    import numpy as np
    obstaculos = []
    for columna, _ in SERIES:
        xy = eje.transData.transform(np.column_stack([g["hito"], g[columna]]))
        for a, b in zip(xy[:-1], xy[1:]):
            obstaculos.append(np.linspace(a, b, 60))
        obstaculos.append(xy)
    for _, fila in g.iterrows():
        extremos = [(fila["hito"], fila["rmse_lstm"] - fila["rmse_lstm_desv"]),
                    (fila["hito"], fila["rmse_lstm"] + fila["rmse_lstm_desv"])]
        a, b = eje.transData.transform(extremos)
        obstaculos.append(np.linspace(a, b, 10))
    return obstaculos


def rotular_valores(eje, g: pd.DataFrame, renderer) -> None:
    """Escribe el valor de cada punto arriba o abajo de él, en el color de su
    serie, eligiendo la posición que no toca líneas, marcadores ni otros
    valores. Se llama después de fijar la disposición final de la figura."""
    import numpy as np
    obstaculos = np.vstack(_segmentos_en_pantalla(eje, g))
    caja_eje = eje.get_window_extent(renderer)
    colocados = []
    puntos = []
    for hito in HITOS:
        fila = g[g["hito"] == hito].iloc[0]
        valores = [float(fila[c]) for c, _ in SERIES]
        for (columna, color), valor in zip(SERIES, valores):
            cercania = min(abs(valor - otro) for otro in valores if otro is not valor)
            puntos.append((cercania, hito, valor, color))
    for _, hito, valor, color in sorted(puntos):
        mejor = None
        for dx, dy, ha, va, penalizacion in CANDIDATOS:
            texto = eje.annotate(_con_coma(valor), xy=(hito, valor), xytext=(dx, dy),
                                 textcoords="offset points", fontsize=TAMANO_VALOR,
                                 color=color, ha=ha, va=va, zorder=5)
            caja = texto.get_window_extent(renderer).expanded(1.08, 1.15)
            dentro = ((obstaculos[:, 0] > caja.x0) & (obstaculos[:, 0] < caja.x1)
                      & (obstaculos[:, 1] > caja.y0) & (obstaculos[:, 1] < caja.y1)).sum()
            choques = sum(caja.overlaps(otra) for otra in colocados)
            fuera = not (caja_eje.x0 <= caja.x0 and caja.x1 <= caja_eje.x1
                         and caja_eje.y0 <= caja.y0 and caja.y1 <= caja_eje.y1)
            costo = dentro + 100 * choques + 100 * fuera + penalizacion
            if mejor is None or costo < mejor[0]:
                if mejor is not None:
                    mejor[1].remove()
                mejor = (costo, texto, caja)
            else:
                texto.remove()
        colocados.append(mejor[2])


if __name__ == "__main__":
    datos = pd.read_csv(RUTA_COMPARACION)
    assert len(datos) == 12, f"se esperaban 12 filas, hay {len(datos)}"
    assert datos["rmse_lstm_desv"].notna().all(), "falta la desviación entre semillas de la LSTM"

    plt.rcParams.update({"font.size": TAMANO_TEXTO})
    figura, ejes = plt.subplots(2, 2, figsize=(6.5, 5.6))
    for eje, materia in zip(ejes.flat, MATERIAS):
        g = datos[datos["materia"] == materia].sort_values("hito")
        eje.errorbar(g["hito"], g["rmse_lstm"], yerr=g["rmse_lstm_desv"], marker="o", markersize=4,
                     color="tab:blue", capsize=3, label="LSTM (media ± desviación entre cinco semillas)")
        eje.plot(g["hito"], g["rmse_bayes"], marker="s", markersize=4, color="tab:red", label="Red bayesiana")
        eje.plot(g["hito"], g["rmse_extrapolacion"], marker="^", markersize=4, color="tab:gray",
                 label="Extrapolación proporcional")
        eje.set_title(materia, fontsize=TAMANO_TEXTO + 1)
        eje.set_xticks(HITOS)
        eje.set_xticklabels([f"S{h}" for h in HITOS])
        eje.set_xlim(3.4, 8.6)
        tope = max(g["rmse_bayes"].max(), g["rmse_extrapolacion"].max(),
                   (g["rmse_lstm"] + g["rmse_lstm_desv"]).max())
        eje.set_ylim(0, tope * 1.2)
        eje.yaxis.set_major_formatter(lambda y, _pos: _con_coma(y, 0) if float(y).is_integer() else _con_coma(y, 1))
        eje.grid(alpha=0.3)
        eje.set_xlabel("Hito")
    for eje in ejes[:, 0]:
        eje.set_ylabel("RMSE (participaciones)")

    manejadores, etiquetas = ejes.flat[0].get_legend_handles_labels()
    orden = [etiquetas.index(e) for e in (
        "LSTM (media ± desviación entre cinco semillas)", "Red bayesiana", "Extrapolación proporcional")]
    manejadores, etiquetas = [manejadores[i] for i in orden], [etiquetas[i] for i in orden]
    figura.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.2)
    figura.legend(manejadores, etiquetas, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, frameon=False)
    renderer = figura.canvas.get_renderer()
    for eje, materia in zip(ejes.flat, MATERIAS):
        rotular_valores(eje, datos[datos["materia"] == materia].sort_values("hito"), renderer)

    RUTA_SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_SALIDA_FIGURA, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {RUTA_SALIDA_FIGURA}")
    print(datos[["materia", "hito", "rmse_lstm", "rmse_lstm_desv", "rmse_bayes", "rmse_extrapolacion"]].round(3).to_string(index=False))
