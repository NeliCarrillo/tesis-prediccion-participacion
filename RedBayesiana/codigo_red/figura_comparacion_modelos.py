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


if __name__ == "__main__":
    datos = pd.read_csv(RUTA_COMPARACION)
    assert len(datos) == 12, f"se esperaban 12 filas, hay {len(datos)}"
    assert datos["rmse_lstm_desv"].notna().all(), "falta la desviación entre semillas de la LSTM"

    plt.rcParams.update({"font.size": TAMANO_TEXTO})
    figura, ejes = plt.subplots(2, 2, figsize=(6.5, 5.2), sharex=True)
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
        eje.set_ylim(bottom=0)
        eje.grid(alpha=0.3)
    for eje in ejes[1]:
        eje.set_xlabel("Hito")
    for eje in ejes[:, 0]:
        eje.set_ylabel("RMSE (participaciones)")

    manejadores, etiquetas = ejes.flat[0].get_legend_handles_labels()
    orden = [etiquetas.index(e) for e in (
        "LSTM (media ± desviación entre cinco semillas)", "Red bayesiana", "Extrapolación proporcional")]
    manejadores, etiquetas = [manejadores[i] for i in orden], [etiquetas[i] for i in orden]
    figura.tight_layout(rect=(0, 0, 1, 0.9))
    figura.legend(manejadores, etiquetas, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, frameon=False)

    RUTA_SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_SALIDA_FIGURA, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {RUTA_SALIDA_FIGURA}")
    print(datos[["materia", "hito", "rmse_lstm", "rmse_lstm_desv", "rmse_bayes", "rmse_extrapolacion"]].round(3).to_string(index=False))
