"""Figura 15: puntaje de Brier de la red bayesiana por asignatura e hito frente a
la referencia uniforme y la referencia marginal histórica.

No modifica ningún resultado del informe: calcula el puntaje de Brier a partir de
las distribuciones posteriores oficiales
(`4_red_bayesiana/resultados/predicciones_bayesiana_s4_s6_s8.csv`) y comprueba
que reproduce la Tabla 18 (dos decimales) antes de graficar. Cada asignatura usa
el mismo color con que se identifica en las tablas del informe: Algoritmos y
Programación en azul, Computación Emergente en verde, Estructura de Datos en
beige y Matemáticas Discretas en rojo.

Correr el archivo genera `figuras/resultados/figura_brier_referencias.png` y
`6_evaluacion/resultados/brier_por_asignatura_hito.csv`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
SALIDA_FIGURA = RAIZ / "figuras" / "resultados" / "figura_brier_referencias.png"
SALIDA_CSV = RAIZ / "6_evaluacion" / "resultados" / "brier_por_asignatura_hito.csv"
ESTADOS = ["0", "1-2", "3-5", "6-11", "12 o más"]
HITOS = [4, 6, 8]

# Colores de identificación de cada asignatura en las tablas del informe.
COLORES = {
    "Algoritmos y Programación": "#2f6db5",
    "Computación Emergente": "#3a9a4a",
    "Estructura de Datos": "#c4a46b",
    "Matemáticas Discretas": "#c8372d",
}
MARCADORES = {
    "Algoritmos y Programación": "o",
    "Computación Emergente": "s",
    "Estructura de Datos": "^",
    "Matemáticas Discretas": "D",
}

# Tabla 18 del informe (puntaje de Brier, dos decimales).
TABLA_18 = {
    ("Algoritmos y Programación", 4): 0.86, ("Algoritmos y Programación", 6): 0.86,
    ("Algoritmos y Programación", 8): 0.80, ("Computación Emergente", 4): 0.63,
    ("Computación Emergente", 6): 0.67, ("Computación Emergente", 8): 0.65,
    ("Estructura de Datos", 4): 0.78, ("Estructura de Datos", 6): 0.73,
    ("Estructura de Datos", 8): 0.82, ("Matemáticas Discretas", 4): 0.97,
    ("Matemáticas Discretas", 6): 1.02, ("Matemáticas Discretas", 8): 0.98,
}


def brier(probabilidades: np.ndarray, observado: np.ndarray) -> np.ndarray:
    """Puntaje de Brier multicategoría de cada caso (entre 0 y 2)."""
    indicador = (np.asarray(ESTADOS)[None, :] == observado[:, None]).astype(float)
    return ((probabilidades - indicador) ** 2).sum(axis=1)


def calcular() -> tuple[pd.DataFrame, float, float]:
    datos = pd.read_csv(RUTA_BAYES, dtype={"estado_real": str})
    assert len(datos) == 1572
    posterior = datos[[f"posterior_{e}" for e in ESTADOS]].to_numpy()
    observado = datos.estado_real.to_numpy()
    datos["brier"] = brier(posterior, observado)

    # Referencia marginal histórica: proporción de cada estado en los 524 registros.
    registros = datos.drop_duplicates(["materia", "trimestre_prueba", "seccion", "estudiante_id"])
    assert len(registros) == 524
    marginal = registros.estado_real.value_counts(normalize=True).reindex(ESTADOS).fillna(0).to_numpy()
    ref_uniforme = float(brier(np.full((len(datos), 5), 0.2), observado).mean())
    ref_marginal = float(brier(np.tile(marginal, (len(datos), 1)), observado).mean())

    tabla = datos.groupby(["materia", "hito"]).brier.mean().reset_index()
    for fila in tabla.itertuples():
        assert round(fila.brier, 2) == TABLA_18[(fila.materia, fila.hito)], (fila.materia, fila.hito, fila.brier)
    assert round(datos.brier.mean(), 2) == 0.79
    assert round(ref_uniforme, 2) == 0.80 and round(ref_marginal, 2) == 0.77
    return tabla, ref_uniforme, ref_marginal


def graficar(tabla: pd.DataFrame, ref_uniforme: float, ref_marginal: float) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    figura, eje = plt.subplots(figsize=(6.5, 4.0))
    eje.axhline(ref_uniforme, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    eje.axhline(ref_marginal, color="0.35", linestyle=":", linewidth=1.4, zorder=1)
    eje.text(8.3, ref_uniforme + 0.004, f"Referencia uniforme ({ref_uniforme:.2f})".replace(".", ","),
             fontsize=8.5, color="0.25", va="bottom", ha="left")
    eje.text(8.3, ref_marginal - 0.004, f"Referencia marginal histórica ({ref_marginal:.2f})".replace(".", ","),
             fontsize=8.5, color="0.25", va="top", ha="left")

    for materia, color in COLORES.items():
        g = tabla[tabla.materia == materia].sort_values("hito")
        eje.plot(g.hito, g.brier, color=color, marker=MARCADORES[materia], markersize=7, linewidth=2,
                 markeredgecolor="white", markeredgewidth=1, label=materia, zorder=3)

    eje.set_xticks(HITOS, [f"S{h}" for h in HITOS])
    eje.set_xlim(3.5, 10.3)
    eje.set_ylim(0.58, 1.08)
    eje.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    eje.set_xlabel("Hito (semana)")
    eje.set_ylabel("Puntaje de Brier")
    eje.grid(axis="y", color="0.9", linewidth=0.6)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9)
    figura.tight_layout()
    SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(SALIDA_FIGURA, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    tabla, ref_uniforme, ref_marginal = calcular()
    print(tabla.round(4).to_string(index=False))
    print(f"Referencia uniforme {ref_uniforme:.4f}; referencia marginal histórica {ref_marginal:.4f}")
    tabla.to_csv(SALIDA_CSV, index=False)
    graficar(tabla, ref_uniforme, ref_marginal)
    print("Tabla 18 reproducida. Figura guardada en", SALIDA_FIGURA)
