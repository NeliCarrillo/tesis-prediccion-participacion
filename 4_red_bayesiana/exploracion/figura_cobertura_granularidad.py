"""Figura del Apéndice M: cobertura y granularidad de los datos cuantitativos
(exploración inicial, Sprint 2).

Panel (a): distribución de las 384 sesiones registradas según el campo
`tipo_sesion` de la plantilla estándar. Las sesiones de contenido se separan
de las demás; las demás se desglosan con los cuatro tipos que aparecen en los
datos (evaluación, administrativa, sin clase y repaso), sin crear categorías
nuevas. Una sesión es una combinación única de asignatura, trimestre, sección,
semana y día de sesión.

Panel (b): conteos de cada unidad de análisis. Son granularidades distintas de
los mismos datos, no partes de un total, por eso se muestran como indicadores
separados y no como un gráfico de partes.

Lee `1_datos/estandarizados/` directamente y comprueba todas las cifras antes de
graficar. No modifica ningún resultado.

Uso:  python3 4_red_bayesiana/exploracion/figura_cobertura_granularidad.py
Salida: figuras/apendice_M_exploracion_datos/figura_cobertura_granularidad.png (300 dpi)
"""
from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
DESTINO = RAIZ / "figuras" / "apendice_M_exploracion_datos" / "figura_cobertura_granularidad.png"
ANCHO_IN = 6.5
AZUL = "#8fa8d8"        # mismo azul de las figuras de exploración (Figuras 3, 8 y 10)
GRIS = "#d4d4d4"
TINTA = "0.15"
TINTA_SECUNDARIA = "0.35"

NOMBRES_TIPO = {"evaluacion": "Evaluación", "administrativa": "Administrativa",
                "sin_clase": "Sin clase", "repaso": "Repaso"}


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _porcentaje(p: float) -> str:
    return f"{p:.2f}".replace(".", ",") + "%"


def cifras() -> dict:
    archivos = glob.glob(str(RAIZ / "1_datos" / "estandarizados" / "**" / "*.csv"), recursive=True)
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    sesiones = df.drop_duplicates(["materia", "trimestre", "seccion", "semana", "dia_sesion"])
    tipos = sesiones.tipo_sesion.value_counts()
    c = {
        "filas": len(df),
        "estudiante_semana": len(df.drop_duplicates(["materia", "trimestre", "seccion", "estudiante_id", "semana"])),
        "registros": len(df.drop_duplicates(["materia", "trimestre", "seccion", "estudiante_id"])),
        "secciones": len(df.drop_duplicates(["materia", "trimestre", "seccion"])),
        "asignaturas": df.materia.nunique(),
        "semanas": df.semana.nunique(),
        "sesiones": len(sesiones),
        "tipos": tipos,
    }
    assert set(tipos.index) == {"contenido", *NOMBRES_TIPO}, tipos.index.tolist()
    assert (c["filas"], c["estudiante_semana"], c["registros"], c["secciones"]) == (11700, 6288, 524, 17)
    assert (c["asignaturas"], c["semanas"]) == (4, 12)
    assert c["estudiante_semana"] == c["registros"] * c["semanas"]
    assert c["sesiones"] == 384 and tipos["contenido"] == 264
    assert tipos.drop("contenido").to_dict() == {"evaluacion": 80, "administrativa": 21, "sin_clase": 10, "repaso": 9}
    return c


def graficar(c: dict) -> None:
    plt.rcParams["font.family"] = "Arial"
    figura = plt.figure(figsize=(ANCHO_IN, 3.2))
    eje_a = figura.add_axes([0.0, 0.0, 0.47, 0.88])
    eje_b = figura.add_axes([0.50, 0.0, 0.50, 0.88])
    for x, letra, titulo in ((0.0, "a", f"Sesiones registradas ({c['sesiones']})"),
                             (0.505, "b", "Unidades de análisis")):
        figura.text(x, 0.97, rf"$\mathbf{{({letra})}}$ " + titulo, fontsize=10.5, color=TINTA,
                    ha="left", va="top")

    # Panel (a): sesiones de contenido frente a los otros tipos. La porción de contenido
    # empieza arriba y avanza en sentido antihorario; la de los otros tipos queda a la derecha.
    tipos = c["tipos"]
    contenido, resto = int(tipos["contenido"]), int(tipos.drop("contenido").sum())
    total = c["sesiones"]
    centro = (-0.55, 0.0)
    eje_a.pie([contenido, resto], colors=[AZUL, GRIS], startangle=90, counterclock=True, radius=1.0,
              center=centro, wedgeprops={"edgecolor": "white", "linewidth": 2})
    angulo_contenido = np.deg2rad(90 + 180 * contenido / total)
    angulo_resto = np.deg2rad(90 + 360 * contenido / total + 180 * resto / total)
    for angulo, r, texto in (
            (angulo_contenido, 0.45, f"Contenido\n{contenido}\n({_porcentaje(100 * contenido / total)})"),
            (angulo_resto, 0.62, f"{resto}\n({_porcentaje(100 * resto / total)})")):
        eje_a.text(centro[0] + r * np.cos(angulo), centro[1] + r * np.sin(angulo), texto, ha="center",
                   va="center", fontsize=9.5, color=TINTA, linespacing=1.2)
    borde = (centro[0] + np.cos(angulo_resto), centro[1] + np.sin(angulo_resto))
    desglose = "\n".join(f"{NOMBRES_TIPO[t]}: {int(tipos[t])}" for t in NOMBRES_TIPO)
    eje_a.annotate(f"Otros tipos: {resto}\n" + desglose, xy=borde, xytext=(0.62, 0.55), ha="left",
                   va="top", fontsize=9, color=TINTA, linespacing=1.3,
                   arrowprops={"arrowstyle": "-", "color": "0.55", "linewidth": 0.8,
                               "shrinkA": 0, "shrinkB": 0, "relpos": (0, 0.93)})
    eje_a.set_xlim(-1.6, 1.75)
    eje_a.set_ylim(-1.08, 1.08)
    eje_a.set_aspect("equal")
    eje_a.axis("off")

    # Panel (b): una tarjeta por unidad de análisis, de la más fina a la más gruesa.
    indicadores = [
        (c["filas"], "filas estudiante-sesión", "una por estudiante en cada sesión"),
        (c["estudiante_semana"], "observaciones estudiante-semana",
         f"{c['registros']} registros por {c['semanas']} semanas"),
        (c["registros"], "registros estudiante-sección", "un estudiante en una sección"),
        (c["secciones"], "secciones", f"de {c['asignaturas']} asignaturas"),
    ]
    eje_b.set_xlim(0, 1)
    eje_b.set_ylim(0, 1)
    eje_b.axis("off")
    alto, separacion = 0.22, 0.035
    for i, (valor, unidad, detalle) in enumerate(indicadores):
        y0 = 1 - (i + 1) * alto - i * separacion
        eje_b.add_patch(FancyBboxPatch((0.01, y0), 0.98, alto, boxstyle="round,pad=0,rounding_size=0.025",
                                       facecolor="#eef2fa", edgecolor=AZUL, linewidth=0.9))
        medio = y0 + alto / 2
        eje_b.text(0.04, medio, _miles(valor), ha="left", va="center", fontsize=14, fontweight="bold",
                   color=TINTA)
        eje_b.text(0.31, medio + 0.04, unidad, ha="left", va="center", fontsize=9, color=TINTA)
        eje_b.text(0.31, medio - 0.045, detalle, ha="left", va="center", fontsize=8.5,
                   color=TINTA_SECUNDARIA)

    figura.canvas.draw()
    limite = eje_b.transAxes.transform((0.985, 0))[0]
    for texto in eje_b.texts:  # comprobación: ningún texto se sale de su tarjeta
        assert texto.get_window_extent().x1 < limite, texto.get_text()
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(DESTINO, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    c = cifras()
    print(f"filas estudiante-sesión {c['filas']}, estudiante-semana {c['estudiante_semana']}, "
          f"registros {c['registros']}, secciones {c['secciones']}, sesiones {c['sesiones']}")
    print(c["tipos"].to_string())
    graficar(c)
    print("guardado:", DESTINO)
