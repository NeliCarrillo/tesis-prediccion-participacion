"""Figura de la Tabla 16: RMSE y R² de la red LSTM adaptada frente a la
extrapolación proporcional, por asignatura e hito.

Fila A: RMSE; fila B: R². Una columna por asignatura. La LSTM se muestra
con la media de cinco semillas y barras de error de ± una desviación
estándar muestral; la extrapolación es determinista y no lleva barras.

Los datos se leen de `comparacion_bayes_lstm_extrapolacion.csv` (generado
por `comparacion_modelos.py`) y se verifican contra los valores publicados
en la Tabla 16. Escala de grises, Times New Roman de 9 pt a 6,5 pulgadas de
ancho, sin título interno (va en el caption del informe).

Correr el archivo genera la figura en PNG y SVG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_DATOS = RAIZ / "RedBayesiana" / "resultados" / "comparacion_bayes_lstm_extrapolacion.csv"
CARPETA_SALIDA = RAIZ / "RedBayesiana" / "figuras"
NOMBRE = "figura_lstm_vs_extrapolacion_tabla16"

MATERIAS = ("Algoritmos y Programación", "Computación Emergente",
            "Estructura de Datos", "Matemáticas Discretas")
HITOS = (4, 6, 8)
TITULOS = {"Algoritmos y Programación": "Algoritmos y\nProgramación",
           "Computación Emergente": "Computación\nEmergente",
           "Estructura de Datos": "Estructura\nde Datos",
           "Matemáticas Discretas": "Matemáticas\nDiscretas"}

# Valores publicados en la Tabla 16 (redondeados como en el informe).
TABLA_16 = {
    "Algoritmos y Programación": dict(rmse=[3.38, 2.74, 1.38], rmse_sd=[0.05, 0.06, 0.04], rmse_ext=[5.70, 3.29, 2.80],
                                      r2=[0.665, 0.779, 0.944], r2_sd=[0.010, 0.010, 0.003], r2_ext=[0.044, 0.682, 0.769]),
    "Computación Emergente": dict(rmse=[2.58, 2.30, 1.36], rmse_sd=[0.06, 0.07, 0.09], rmse_ext=[3.88, 2.11, 1.47],
                                  r2=[0.770, 0.817, 0.936], r2_sd=[0.011, 0.010, 0.008], r2_ext=[0.480, 0.846, 0.925]),
    "Estructura de Datos": dict(rmse=[5.18, 3.84, 3.21], rmse_sd=[0.16, 0.10, 0.15], rmse_ext=[5.74, 3.89, 2.83],
                                r2=[0.553, 0.755, 0.829], r2_sd=[0.028, 0.012, 0.016], r2_ext=[0.452, 0.748, 0.867]),
    "Matemáticas Discretas": dict(rmse=[2.97, 2.17, 1.34], rmse_sd=[0.04, 0.14, 0.06], rmse_ext=[3.05, 1.92, 1.39],
                                  r2=[0.445, 0.705, 0.887], r2_sd=[0.014, 0.036, 0.010], r2_ext=[0.415, 0.769, 0.879]),
}

ESTILO_LSTM = dict(color="black", linestyle="-", marker="o", markersize=4.5, linewidth=1.2)
ESTILO_EXT = dict(color="0.45", linestyle="--", marker="^", markersize=5, linewidth=1.2,
                  markerfacecolor="white", markeredgecolor="0.45")


def cargar_y_verificar() -> pd.DataFrame:
    datos = pd.read_csv(RUTA_DATOS).set_index(["materia", "hito"])
    columnas = {"rmse": ("rmse_lstm", 2), "rmse_sd": ("rmse_lstm_desv", 2), "rmse_ext": ("rmse_extrapolacion", 2),
                "r2": ("r2_lstm", 3), "r2_sd": ("r2_lstm_desv", 3), "r2_ext": ("r2_extrapolacion", 3)}
    for materia, valores in TABLA_16.items():
        for clave, (columna, decimales) in columnas.items():
            calculado = [round(float(datos.loc[(materia, h), columna]), decimales) for h in HITOS]
            assert calculado == valores[clave], f"{materia}/{clave}: CSV {calculado} vs Tabla 16 {valores[clave]}"
    return datos


if __name__ == "__main__":
    datos = cargar_y_verificar()

    fuente = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if fuente.exists():
        font_manager.fontManager.addfont(str(fuente))
        plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "svg.fonttype": "none"})

    figura, ejes = plt.subplots(2, 4, figsize=(6.5, 4.6))
    filas = [
        ("rmse_lstm", "rmse_lstm_desv", "rmse_extrapolacion", "RMSE (participaciones)", (0, 6.5), "A"),
        ("r2_lstm", "r2_lstm_desv", "r2_extrapolacion", "R²", (0, 1.0), "B"),
    ]
    for fila, (media, desv, ext, etiqueta, limites, letra) in enumerate(filas):
        for columna, materia in enumerate(MATERIAS):
            eje = ejes[fila, columna]
            g = datos.loc[materia].loc[list(HITOS)]
            eje.errorbar(HITOS, g[media], yerr=g[desv], capsize=2.5, elinewidth=0.8, **ESTILO_LSTM)
            eje.plot(HITOS, g[ext], **ESTILO_EXT)
            eje.set_ylim(*limites)
            eje.set_xticks(HITOS)
            eje.set_xticklabels([f"S{h}" for h in HITOS])
            eje.set_xlim(3.4, 8.6)
            eje.grid(axis="y", color="0.88", linewidth=0.5)
            eje.set_axisbelow(True)
            for lado in ("top", "right"):
                eje.spines[lado].set_visible(False)
            if fila == 0:
                eje.set_title(TITULOS[materia], fontsize=9)
            if columna == 0:
                eje.set_ylabel(etiqueta)
            else:
                eje.tick_params(labelleft=False)
        for columna in range(4):
            ejes[fila, columna].set_xlabel("Hito")

    leyenda = [
        Line2D([0], [0], **ESTILO_LSTM, label="Red LSTM (media ± desviación estándar entre cinco semillas)"),
        Line2D([0], [0], **ESTILO_EXT, label="Extrapolación proporcional"),
    ]
    figura.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8, h_pad=1.4)
    figura.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figura.savefig(CARPETA_SALIDA / f"{NOMBRE}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    print("Verificado contra la Tabla 16; figura guardada en", CARPETA_SALIDA / f"{NOMBRE}.png", "y .svg")
