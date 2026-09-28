"""Figura de Resultados (Sprint 6): distribución posterior completa de la
red bayesiana para los tres casos de la Figura 7, por hito.

Cada panel indica el respaldo de la combinación de evidencia consultada en
el entrenamiento del pliegue, expresado en registros estudiante-sección
distintos. El respaldo se contó en filas de sesión (la unidad con la que se
ajusta la tabla de probabilidad condicional del objetivo); ambos conteos se
recalculan aquí con `preparar_fold` y las mismas funciones del pipeline.

No se recalcula ninguna posterior: se leen las columnas `posterior_*` de
`predicciones_bayesiana_s4_s6_s8.csv`. Sin título interno (va en el
caption); ancho de 6,5 pulgadas para conservar al menos 9 pt.

Importar este módulo no ejecuta nada; correr el archivo genera la figura.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensamblado import ensamblar_conjunto
from inferencia import _evidencia_c1

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_PREDICCIONES_BAYES = RAIZ / "RedBayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_SALIDA_FIGURA = RAIZ / "RedBayesiana" / "figuras" / "figura_posterior_atipicos.png"

IDS_ATIPICOS: tuple[str, ...] = ("anon_145", "anon_370", "anon_322")
HITOS: tuple[int, ...] = (4, 6, 8)
ESTADOS: tuple[str, ...] = ("0", "1-2", "3-5", "6-11", "12 o más")
COLUMNAS_POSTERIOR = [f"posterior_{estado}" for estado in ESTADOS]
PADRES = ("Año que cursa", "Tamaño del grupo", "Participaciones de la semana",
          "Participaciones de la semana anterior")
NODO_OBJETIVO = "Cantidad de participaciones del trimestre"


def respaldo_en_entrenamiento(estudiante_id: str) -> dict[int, tuple[int, int]]:
    """{hito: (filas de sesión, registros estudiante-sección)} de entrenamiento
    del pliegue cuyo valor en los cuatro padres del objetivo coincide
    exactamente con la evidencia del estudiante en ese hito."""
    sesiones, registros, semanas = ensamblar_conjunto()
    fila_reg = registros[registros["estudiante_id"] == estudiante_id]
    assert len(fila_reg) == 1, f"{estudiante_id}: se esperaba un único registro"
    fila_reg = fila_reg.iloc[0]
    materia, trimestre = fila_reg["materia"], fila_reg["trimestre"]

    entrenamiento = sesiones[(sesiones["materia"] == materia) & (sesiones["trimestre"] != trimestre)]
    utilizables = entrenamiento.dropna(subset=list(PADRES) + [NODO_OBJETIVO])

    resultado = {}
    for hito in HITOS:
        fila_semana = semanas[
            (semanas["estudiante_id"] == estudiante_id) & (semanas["materia"] == materia)
            & (semanas["trimestre"] == trimestre) & (semanas["seccion"] == fila_reg["seccion"])
            & (semanas["semana"] == hito)
        ].iloc[0]
        evidencia = _evidencia_c1(fila_reg, fila_semana)
        mascara = np.ones(len(utilizables), dtype=bool)
        for padre in PADRES:
            mascara &= (utilizables[padre] == evidencia[padre]).to_numpy()
        coincidentes = utilizables[mascara]
        registros_distintos = len(coincidentes[["estudiante_id", "trimestre", "seccion"]].drop_duplicates())
        resultado[hito] = (len(coincidentes), registros_distintos)
    return resultado


def etiqueta_respaldo(respaldo: dict[int, tuple[int, int]]) -> str:
    registros = [respaldo[h][1] for h in HITOS]
    if all(r == 0 for r in registros):
        return "Sin respaldo"
    return "Respaldo: " + "/".join(str(r) for r in registros) + " registros"


if __name__ == "__main__":
    predicciones = pd.read_csv(RUTA_PREDICCIONES_BAYES)
    datos = predicciones[predicciones["estudiante_id"].isin(IDS_ATIPICOS)]
    assert len(datos) == len(IDS_ATIPICOS) * len(HITOS), f"se esperaban 9 filas, hay {len(datos)}"

    respaldos = {e: respaldo_en_entrenamiento(e) for e in IDS_ATIPICOS}
    for estudiante_id, respaldo in respaldos.items():
        print(estudiante_id, {h: f"{f} filas de sesión / {r} registros" for h, (f, r) in respaldo.items()})

    plt.rcParams.update({"font.size": 9})
    colores_hito = {4: "tab:blue", 6: "tab:orange", 8: "tab:green"}
    x = np.arange(len(ESTADOS))
    ancho = 0.27

    figura, ejes = plt.subplots(1, 3, figsize=(6.5, 3.1), sharey=True)
    for eje, estudiante_id in zip(ejes, IDS_ATIPICOS):
        grupo = datos[datos["estudiante_id"] == estudiante_id]
        for i, hito in enumerate(HITOS):
            fila = grupo[grupo["hito"] == hito].iloc[0]
            eje.bar(x + (i - 1) * ancho, [fila[c] for c in COLUMNAS_POSTERIOR], width=ancho,
                    color=colores_hito[hito], label=f"S{hito}")
        real = grupo["total_trimestre_real"].iloc[0]
        eje.set_title(f"{estudiante_id} (real = {real:.0f})\n{etiqueta_respaldo(respaldos[estudiante_id])}",
                      fontsize=9)
        eje.set_xticks(x)
        eje.set_xticklabels(ESTADOS, rotation=30)
        eje.set_ylim(0, 1.05)
        eje.grid(axis="y", alpha=0.3)
    ejes[0].set_ylabel("Probabilidad posterior")
    ejes[1].set_xlabel("Estado del objetivo")

    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    figura.tight_layout(rect=(0, 0, 1, 0.88))
    figura.legend(manejadores, etiquetas, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, frameon=False)

    RUTA_SALIDA_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_SALIDA_FIGURA, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {RUTA_SALIDA_FIGURA}")
