"""Re-observación de los 3 casos atípicos ya identificados en Sprint 2
(Figura 7 del notebook LSTM, `anon_145`/`anon_370`/`anon_322` — relación
acumulado↔total) a través de las predicciones finales de ambos modelos.

No se reinventan los IDs ni se recalcula nada de las validaciones cruzadas:
se extraen las filas ya existentes de esos 3 estudiantes en los 2 CSV
oficiales (predicciones_bayesiana_s4_s6_s8.csv de Sprint 4,
predicciones_lstm_validacion_cruzada.csv de Sprint 2) y se comparan lado a
lado con el valor real.

Importar este módulo no ejecuta nada; correr el archivo genera y guarda la
tabla y la figura.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_PREDICCIONES_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_PREDICCIONES_LSTM = RAIZ / "3_lstm" / "resultados" / "predicciones_lstm_validacion_cruzada.csv"
RUTA_SALIDA_CSV = RAIZ / "6_evaluacion" / "resultados" / "casos_atipicos_predicciones.csv"
RUTA_SALIDA_FIGURA = RAIZ / "figuras" / "resultados" / "figura_atipicos_predicciones.png"

# Mismos 3 IDs ya destacados en la Figura 7 del notebook LSTM (Sprint 2).
IDS_ATIPICOS: tuple[str, ...] = ("anon_145", "anon_370", "anon_322")
HITOS: tuple[int, ...] = (4, 6, 8)


def construir_tabla_atipicos() -> pd.DataFrame:
    bayes = pd.read_csv(RUTA_PREDICCIONES_BAYES)
    bayes = bayes[bayes["estudiante_id"].isin(IDS_ATIPICOS)][
        ["materia", "estudiante_id", "hito", "total_trimestre_real", "prediccion_continua"]
    ].rename(columns={"prediccion_continua": "prediccion_bayes"})

    lstm = pd.read_csv(RUTA_PREDICCIONES_LSTM)
    lstm = lstm[lstm["estudiante_id"].isin(IDS_ATIPICOS)][
        ["materia", "estudiante_id", "hito", "total_trimestre_real",
         "prediccion_lstm", "prediccion_extrapolacion"]
    ]

    tabla = bayes.merge(
        lstm, on=["materia", "estudiante_id", "hito", "total_trimestre_real"], how="inner"
    )
    tabla = tabla.sort_values(["estudiante_id", "hito"]).reset_index(drop=True)
    return tabla[[
        "materia", "estudiante_id", "hito", "total_trimestre_real",
        "prediccion_bayes", "prediccion_lstm", "prediccion_extrapolacion",
    ]]


def graficar_atipicos(tabla: pd.DataFrame) -> None:
    """Un panel por hito: predicción de cada enfoque (eje y) frente al total
    real (eje x) para los tres casos de la Figura 7, con la identidad
    (y = x) de referencia. Las predicciones de la LSTM son de la semilla 42,
    la única con salida por caso. Sin título interno (va en el caption);
    ancho de 6,5 pulgadas para conservar al menos 9 pt."""
    plt.rcParams.update({"font.size": 9})
    colores = {"prediccion_lstm": "tab:blue", "prediccion_bayes": "tab:red",
               "prediccion_extrapolacion": "tab:gray"}
    etiquetas = {"prediccion_lstm": "LSTM (semilla 42)", "prediccion_bayes": "Red bayesiana",
                 "prediccion_extrapolacion": "Extrapolación proporcional"}
    marcadores = {"anon_145": "o", "anon_370": "s", "anon_322": "^"}

    figura, ejes = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True)
    tope = max(tabla["total_trimestre_real"].max(),
               tabla[list(colores)].max().max()) * 1.05
    for eje, hito in zip(ejes, HITOS):
        grupo = tabla[tabla["hito"] == hito]
        for _, fila in grupo.iterrows():
            for columna in colores:
                eje.scatter(
                    fila["total_trimestre_real"], fila[columna],
                    color=colores[columna], marker=marcadores[fila["estudiante_id"]],
                    s=36, alpha=0.9, edgecolor="black", linewidth=0.5, zorder=3,
                )
        eje.plot([0, tope], [0, tope], color="gray", linewidth=1, alpha=0.4)
        eje.set_xlim(0, tope)
        eje.set_ylim(0, tope)
        eje.set_title(f"S{hito}", fontsize=10)
        eje.set_xlabel("Total real")
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("Predicción")

    manejadores_modelo = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colores[c],
                   markeredgecolor="black", markersize=6, label=etiquetas[c])
        for c in colores
    ]
    manejadores_estudiante = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=6, label=e)
        for e, m in marcadores.items()
    ]
    figura.tight_layout(rect=(0, 0, 1, 0.8))
    # matplotlib llena la leyenda por columnas: se intercalan para que la
    # primera fila muestre los enfoques y la segunda los estudiantes.
    intercalados = [h for par in zip(manejadores_modelo, manejadores_estudiante) for h in par]
    figura.legend(handles=intercalados,
                  loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    RUTA_SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_SALIDA_FIGURA, dpi=300, bbox_inches="tight")
    plt.close(figura)


if __name__ == "__main__":
    tabla = construir_tabla_atipicos()

    assert len(tabla) == len(IDS_ATIPICOS) * len(HITOS), (
        f"se esperaban {len(IDS_ATIPICOS) * len(HITOS)} filas (3 estudiantes x 3 hitos), hay {len(tabla)}"
    )
    for estudiante_id in IDS_ATIPICOS:
        n = (tabla["estudiante_id"] == estudiante_id).sum()
        assert n == len(HITOS), f"{estudiante_id}: {n} filas, se esperaban {len(HITOS)}"

    print(tabla.to_string(index=False))
    print()
    print("verificaciones superadas: 9 filas (3 estudiantes atípicos x 3 hitos), "
          "mismo valor real en ambos CSV de origen (join exitoso)")

    graficar_atipicos(tabla)

    RUTA_SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_SALIDA_CSV, index=False)
    print(f"\nCSV guardado en: {RUTA_SALIDA_CSV}")
    print(f"Figura guardada en: {RUTA_SALIDA_FIGURA}")
