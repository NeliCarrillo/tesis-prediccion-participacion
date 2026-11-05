"""Figura 4: distribución del total de participaciones del trimestre por asignatura, con los
cuatro histogramas en una fila.

Reproduce la figura de la celda 50 de `adaptacion_lstm_participaciones.ipynb` (mismos datos,
mismos intervalos de dos participaciones, mismos colores y misma línea de la media), pero en
una sola fila para ocupar menos alto en el informe. Se diseña a 7 pulgadas de ancho, con el área graficable de cada panel cuadrada, de modo que el texto conserva un tamaño de 9 a 10 pt al insertarla sin escalar,
como el de la versión anterior escalada a ese mismo ancho. La información que antes iba en el
título de cada panel (registros, porcentaje con cero y máximo) se muestra dentro del panel.

Los datos salen de `5_prototipo/services/lstm_service._preparar_datos()`, la misma preparación
del cuaderno; se comprueba que se obtienen los 524 registros y los valores de la figura anterior.

Correr el archivo genera `figuras/metodo_datos_y_exploracion/figura_distribucion_objetivo.png`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "5_prototipo" / "services"))
import lstm_service  # noqa: E402

SALIDA = RAIZ / "figuras" / "metodo_datos_y_exploracion" / "figura_distribucion_objetivo.png"
NOMBRES = {"Algoritmos y Programación": "Algoritmos y\nProgramación",
           "Computación Emergente": "Computación\nEmergente",
           "Estructura de Datos": "Estructura de\nDatos",
           "Matemáticas Discretas": "Matemáticas\nDiscretas"}
# Valores de la versión anterior de la figura: (registros, % con cero, máximo, media).
ESPERADO = {"Algoritmos y Programación": (147, 20, 24, 5.15), "Computación Emergente": (182, 42, 30, 3.18),
            "Estructura de Datos": (108, 38, 44, 4.91), "Matemáticas Discretas": (87, 21, 13, 4.16)}


def _coma(valor: float, decimales: int = 2) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def main() -> None:
    datos = lstm_service._preparar_datos()
    estatica, y = datos["estatica"], datos["y"]
    assert len(y) == 524
    materias = sorted(estatica.materia.unique())
    colores = dict(zip(materias, plt.cm.tab10.colors))  # los mismos colores de la figura anterior
    bins = np.arange(0, y.max() + 2, 2)

    plt.rcParams.update({"font.size": 9})
    figura, ejes = plt.subplots(1, 4, figsize=(7.0, 2.9), sharex=True, sharey=True)
    for eje, materia in zip(ejes, materias):
        valores = y[(estatica.materia == materia).to_numpy()]
        n, cero, maximo, media = len(valores), (valores == 0).mean(), valores.max(), valores.mean()
        assert (n, round(cero * 100), int(maximo), round(media, 2)) == ESPERADO[materia], materia
        eje.hist(valores, bins=bins, color=colores[materia], alpha=0.85, edgecolor="white", linewidth=0.4)
        eje.axvline(media, color="k", linestyle="--", linewidth=1.0)
        eje.text(0.97, 0.97, f"{n} registros\n{cero:.0%} con cero\nmáximo {int(maximo)}\nmedia {_coma(media)}",
                 transform=eje.transAxes, ha="right", va="top", fontsize=8.5, linespacing=1.25)
        eje.set_title(NOMBRES[materia], fontsize=9.5)
        eje.tick_params(labelsize=9)
        eje.set_xticks([0, 20, 40])
        eje.grid(alpha=0.3)
        eje.set_box_aspect(1)  # área graficable cuadrada
    ejes[0].set_ylabel("Registros estudiante-sección", fontsize=9)
    figura.supxlabel("Total de participaciones del trimestre", fontsize=9.5, y=0.11)
    figura.legend([plt.Line2D([0], [0], color="k", linestyle="--", linewidth=1.0)], ["Media"],
                  loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8.5)
    figura.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.6)
    figura.canvas.draw()
    for eje in ejes:  # comprobación: cada área graficable es un cuadrado
        caja = eje.get_window_extent()
        assert abs(caja.width - caja.height) < 1, (caja.width, caja.height)
    figura.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor="white")
    print("Figura guardada en", SALIDA)


if __name__ == "__main__":
    main()
