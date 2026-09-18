"""
Compara la estructura de la red bayesiana diseñada a mano (con conocimiento
de dominio) contra la que aprendería un algoritmo de búsqueda por puntaje
(hill-climbing + BIC, pgmpy), a partir de los datos.

Además, genera figuras de comparación de ambas estructuras para cada
asignatura y una figura consolidada para su inclusión en el informe.

NO toca el notebook de la LSTM ni ningún otro archivo del proyecto; es un
análisis aparte, autocontenido. Lee «Datos Tesis Downstream/» directamente.

Corre el análisis por separado para cada asignatura (el modelo se entrena por
asignatura, igual que la LSTM y la red bayesiana manual), construyendo un
registro por estudiante-sesión (la unidad más fina disponible: 9.612 filas en
total) con las variables discretizadas según la Tabla 12 y el Apéndice A
(Tabla A4) del informe.

Para cada asignatura:
    (a) HillClimbSearch sin restricciones (bic-d).
    (b) HillClimbSearch con restricciones temporales (Tabla 13).

Reporta los arcos aprendidos, los compara con el grafo manual, calcula el
puntaje BIC del grafo manual y del aprendido con restricciones y genera
figuras de comparación.

Uso:
    python3 RedBayesiana/estructura_aprendida.py
"""

from pathlib import Path
import glob
import warnings
import re

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork


# ======================================================================
# 0. CONFIGURACIÓN
# ======================================================================

RAIZ = Path(__file__).resolve().parent.parent.parent

CLAVE = [
    "estudiante_id",
    "materia",
    "trimestre",
    "seccion"
]

# Carpeta donde se guardarán las figuras
CARPETA_FIGURAS = RAIZ / "RedBayesiana" / "figuras"
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1. CARGA DE DATOS
# ======================================================================

frames = [
    pd.read_csv(f)
    for f in glob.glob(
        str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"),
        recursive=True
    )
]

df = pd.concat(frames, ignore_index=True)

df["participaciones"] = pd.to_numeric(
    df["participaciones"],
    errors="coerce"
)

df["numero_lista"] = pd.to_numeric(
    df["numero_lista"],
    errors="coerce"
)

df["tema"] = pd.to_numeric(
    df["tema"],
    errors="coerce"
)

df["anio_academico"] = pd.to_numeric(
    df["anio_academico"],
    errors="coerce"
)

assert len(df) == 9612, (
    f"se esperaban 9.612 filas estudiante-sesión, hay {len(df)}"
)


# ======================================================================
# 2. VARIABLES POR REGISTRO ESTUDIANTE-SECCIÓN
# ======================================================================

tam_sec = (
    df.groupby(
        ["materia", "trimestre", "seccion"]
    )["estudiante_id"]
    .nunique()
    .rename("tamano_grupo")
)

reg = (
    df.drop_duplicates(subset=CLAVE)[
        CLAVE + ["numero_lista", "anio_academico"]
    ]
    .copy()
)

reg = reg.merge(
    tam_sec,
    on=["materia", "trimestre", "seccion"]
)

reg["posicion"] = (
    reg["numero_lista"] /
    reg["tamano_grupo"]
)

tot = (
    df.groupby(CLAVE)["participaciones"]
    .sum()
    .rename("total_trimestre")
    .reset_index()
)

reg = reg.merge(
    tot,
    on=CLAVE
)

assert len(reg) == 437


# ======================================================================
# 3. VARIABLES POR SEMANA
# ======================================================================

sem = (
    df.groupby(CLAVE + ["semana"])["participaciones"]
    .sum()
    .rename("participaciones_semana")
    .reset_index()
)

n_sesiones = (
    df.groupby(CLAVE + ["semana"])
    .size()
    .rename("n_sesiones_semana")
    .reset_index()
)

sem = sem.merge(
    n_sesiones,
    on=CLAVE + ["semana"]
)

n_eval = (
    df[df["tipo_sesion"] == "evaluacion"]
    .groupby(CLAVE + ["semana"])
    .size()
    .rename("n_eval_semana")
)

sem = sem.merge(
    n_eval,
    on=CLAVE + ["semana"],
    how="left"
)

sem["n_eval_semana"] = (
    sem["n_eval_semana"]
    .fillna(0)
    .astype(int)
)

assert len(sem) == 5244

sem = sem.sort_values(
    CLAVE + ["semana"]
)

# Participaciones de la semana anterior
sem["participaciones_semana_anterior"] = (
    sem.groupby(CLAVE)["participaciones_semana"]
    .shift(1)
)


# ======================================================================
# 4. DISCRETIZACIONES
# ======================================================================

def bin_anio(a):
    return str(int(a)) if a < 5 else "5 o más"


def bin_tamano(t):
    if t <= 28:
        return "pequeño"

    if t == 30:
        return "mediano"

    return "grande"


def bin_posicion(p):
    if p <= 0.25:
        return "≤0,25"

    if p <= 0.50:
        return "0,25–0,50"

    if p <= 0.75:
        return "0,50–0,75"

    return ">0,75"


def bin_partsem(p):
    if pd.isna(p):
        return None

    if p == 0:
        return "0"

    if p == 1:
        return "1"

    if p == 2:
        return "2"

    return "3 o más"


def bin_nses(n):
    return "1" if n == 1 else "2"


def bin_neval(n):
    return str(int(n))


def bin_trimestre(t):
    if t == 0:
        return "0"

    if t <= 2:
        return "1-2"

    if t <= 5:
        return "3-5"

    if t <= 11:
        return "6-11"

    return "12 o más"


reg["Año que cursa"] = (
    reg["anio_academico"]
    .apply(bin_anio)
)

reg["Tamaño del grupo"] = (
    reg["tamano_grupo"]
    .apply(bin_tamano)
)

reg["Posición relativa en la lista"] = (
    reg["posicion"]
    .apply(bin_posicion)
)

reg["Cantidad de participaciones del trimestre"] = (
    reg["total_trimestre"]
    .apply(bin_trimestre)
)

sem["Participaciones de la semana"] = (
    sem["participaciones_semana"]
    .apply(bin_partsem)
)

sem["Participaciones de la semana anterior"] = (
    sem["participaciones_semana_anterior"]
    .apply(bin_partsem)
)

sem["Número de sesiones de la semana"] = (
    sem["n_sesiones_semana"]
    .apply(bin_nses)
)

sem["Sesiones de evaluación de la semana"] = (
    sem["n_eval_semana"]
    .apply(bin_neval)
)


# ======================================================================
# 4.1. DISCRETIZACIÓN DEL TEMA
# ======================================================================

temas = (
    df[df["tema"] != 0]
    .groupby(["materia", "tema"])
    .agg(
        n=("participaciones", "size"),
        media=("participaciones", "mean")
    )
    .reset_index()
)


def tercios(g):
    g = g.sort_values("media").copy()

    g["acum"] = g["n"].cumsum()

    total = g["n"].sum()

    g["categoria"] = pd.cut(
        g["acum"],
        bins=[
            0,
            total / 3,
            2 * total / 3,
            total
        ],
        labels=[
            "baja",
            "media",
            "alta"
        ]
    )

    return g


_tercios_por_materia = [
    tercios(g)
    for _, g in temas.groupby("materia")
]

temas = pd.concat(
    _tercios_por_materia,
    ignore_index=True
)

mapa_tema = {}

for _, row in temas.iterrows():

    mapa_tema[
        (row["materia"], row["tema"])
    ] = row["categoria"]


def bin_tema(materia, tema):

    if tema == 0:
        return "0"

    return mapa_tema.get(
        (materia, tema),
        None
    )


df["Tema de la sesión"] = [
    bin_tema(m, t)
    for m, t in zip(
        df["materia"],
        df["tema"]
    )
]


# ======================================================================
# 4.2. VERIFICACIÓN
# ======================================================================

print(
    "=== Verificación de conteos "
    "(deben coincidir con Tabla 12 / Tabla A4) ==="
)

print(
    "Tamaño del grupo:",
    reg.drop_duplicates(
        subset=[
            "materia",
            "trimestre",
            "seccion"
        ]
    )[
        "Tamaño del grupo"
    ].value_counts().to_dict()
)

print(
    "Posición relativa:",
    reg[
        "Posición relativa en la lista"
    ].value_counts().to_dict()
)

print(
    "Participaciones semana:",
    sem[
        "Participaciones de la semana"
    ].value_counts(
        dropna=False
    ).to_dict()
)

print(
    "Participaciones trimestre:",
    reg[
        "Cantidad de participaciones del trimestre"
    ].value_counts().to_dict()
)

print(
    "Tema (todas las asignaturas, código 0 aparte):",
    df[
        "Tema de la sesión"
    ].value_counts(
        dropna=False
    ).to_dict()
)

print()


# ======================================================================
# 5. ENSAMBLAR REGISTRO ESTUDIANTE-SESIÓN
# ======================================================================

sesiones = df[
    [
        "estudiante_id",
        "materia",
        "trimestre",
        "seccion",
        "semana",
        "dia_sesion",
        "Tema de la sesión"
    ]
].copy()

sesiones = sesiones.merge(
    reg[
        CLAVE + [
            "Año que cursa",
            "Tamaño del grupo",
            "Posición relativa en la lista",
            "Cantidad de participaciones del trimestre"
        ]
    ],
    on=CLAVE
)

sesiones = sesiones.merge(
    sem[
        CLAVE + [
            "semana",
            "Participaciones de la semana",
            "Participaciones de la semana anterior",
            "Número de sesiones de la semana",
            "Sesiones de evaluación de la semana"
        ]
    ],
    on=CLAVE + ["semana"]
)


# ======================================================================
# 6. VARIABLES DE LA RED BAYESIANA
# ======================================================================

COLUMNAS_BN = [
    "Año que cursa",
    "Sección",
    "Tamaño del grupo",
    "Posición relativa en la lista",
    "Tema de la sesión",
    "Número de sesiones de la semana",
    "Sesiones de evaluación de la semana",
    "Participaciones de la semana anterior",
    "Participaciones de la semana",
    "Cantidad de participaciones del trimestre",
]

sesiones["Sección"] = (
    sesiones["seccion"]
    .astype(str)
)


# ======================================================================
# 7. CAPAS TEMPORALES
# ======================================================================

INICIO = [
    "Año que cursa",
    "Sección",
    "Tamaño del grupo",
    "Posición relativa en la lista"
]

DURANTE_ANTERIOR = [
    "Participaciones de la semana anterior"
]

DURANTE_ACTUAL = [
    "Tema de la sesión",
    "Número de sesiones de la semana",
    "Sesiones de evaluación de la semana",
    "Participaciones de la semana"
]

FIN = [
    "Cantidad de participaciones del trimestre"
]


# ======================================================================
# 8. ESTRUCTURA MANUAL
# ======================================================================

ARCOS_MANUALES = [

    # Estructura del contexto
    (
        "Sección",
        "Tamaño del grupo"
    ),

    # Relaciones dentro de la semana
    (
        "Número de sesiones de la semana",
        "Sesiones de evaluación de la semana"
    ),

    (
        "Sesiones de evaluación de la semana",
        "Participaciones de la semana"
    ),

    (
        "Posición relativa en la lista",
        "Participaciones de la semana"
    ),

    (
        "Tema de la sesión",
        "Participaciones de la semana"
    ),

    # Dependencia temporal
    (
        "Participaciones de la semana anterior",
        "Participaciones de la semana"
    ),

    # Dependencias con el objetivo
    (
        "Participaciones de la semana",
        "Cantidad de participaciones del trimestre"
    ),

    (
        "Participaciones de la semana anterior",
        "Cantidad de participaciones del trimestre"
    ),

    (
        "Tamaño del grupo",
        "Cantidad de participaciones del trimestre"
    ),

    (
        "Año que cursa",
        "Cantidad de participaciones del trimestre"
    ),
]


# ======================================================================
# 9. FUNCIONES PARA GENERAR LAS FIGURAS
# ======================================================================

def nombre_seguro(texto):
    """
    Convierte el nombre de una asignatura en un nombre válido
    para un archivo.
    """

    texto = str(texto)

    texto = re.sub(
        r"[^A-Za-z0-9áéíóúÁÉÍÓÚñÑ_-]+",
        "_",
        texto
    )

    return texto.strip("_")


def construir_posiciones():
    """
    Define una distribución manual de los nodos para mantener
    una lectura temporal de izquierda a derecha.

    El orden es:
        Inicio → Semana anterior → Semana actual → Fin
    """

    posiciones = {

        # Inicio
        "Año que cursa": (-3.8, 2.4),
        "Sección": (-3.8, 0.8),
        "Tamaño del grupo": (-3.8, -0.8),
        "Posición relativa en la lista": (-3.8, -2.4),

        # Semana anterior
        "Participaciones de la semana anterior": (-1.3, 0.0),

        # Semana actual
        "Tema de la sesión": (1.3, 2.2),
        "Número de sesiones de la semana": (1.3, 0.7),
        "Sesiones de evaluación de la semana": (1.3, -0.8),
        "Participaciones de la semana": (1.3, -2.3),

        # Fin
        "Cantidad de participaciones del trimestre": (4.0, 0.0),
    }

    return posiciones


def crear_grafo_comparativo(
    materia,
    arcos_manual,
    arcos_aprendidos,
    ruta_salida
):
    """
    Genera un grafo comparativo:

        - Arco manual solamente:
              línea continua

        - Arco aprendido solamente:
              línea discontinua

        - Arco presente en ambos:
              línea continua y resaltada

    El gráfico compara específicamente el grafo manual con
    el grafo aprendido bajo restricciones temporales.
    """

    manual_set = set(arcos_manual)
    aprendido_set = set(arcos_aprendidos)

    comunes = manual_set & aprendido_set
    solo_manual = manual_set - aprendido_set
    solo_aprendido = aprendido_set - manual_set

    G = nx.DiGraph()

    G.add_nodes_from(COLUMNAS_BN)

    # --------------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    posiciones = construir_posiciones()

    # --------------------------------------------------------------
    # Nodos
    # --------------------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        posiciones,
        node_size=5000,
        node_color="white",
        edgecolors="black",
        linewidths=1.5,
        node_shape="s",
        ax=ax
    )

    # --------------------------------------------------------------
    # Etiquetas
    # --------------------------------------------------------------

    etiquetas = {

        "Año que cursa":
            "Año que\ncursa",

        "Sección":
            "Sección",

        "Tamaño del grupo":
            "Tamaño del\ngrupo",

        "Posición relativa en la lista":
            "Posición relativa\nen la lista",

        "Tema de la sesión":
            "Tema de la\nsesión",

        "Número de sesiones de la semana":
            "N.º de sesiones\nde la semana",

        "Sesiones de evaluación de la semana":
            "Sesiones de evaluación\nde la semana",

        "Participaciones de la semana anterior":
            "Participaciones de\nla semana anterior",

        "Participaciones de la semana":
            "Participaciones de\nla semana",

        "Cantidad de participaciones del trimestre":
            "Cantidad de participaciones\ndel trimestre",
    }

    nx.draw_networkx_labels(
        G,
        posiciones,
        labels=etiquetas,
        font_size=8,
        font_weight="normal",
        ax=ax
    )

    # --------------------------------------------------------------
    # Arcos SOLO MANUAL
    # --------------------------------------------------------------

    if solo_manual:

        nx.draw_networkx_edges(
            G,
            posiciones,
            edgelist=list(solo_manual),
            edge_color="black",
            style="solid",
            width=1.8,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.04",
            min_source_margin=15,
            min_target_margin=15,
            ax=ax
        )

    # --------------------------------------------------------------
    # Arcos SOLO APRENDIDOS
    # --------------------------------------------------------------

    if solo_aprendido:

        nx.draw_networkx_edges(
            G,
            posiciones,
            edgelist=list(solo_aprendido),
            edge_color="dimgray",
            style="dashed",
            width=1.8,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.04",
            min_source_margin=15,
            min_target_margin=15,
            ax=ax
        )

    # --------------------------------------------------------------
    # Arcos COMUNES
    # --------------------------------------------------------------

    if comunes:

        nx.draw_networkx_edges(
            G,
            posiciones,
            edgelist=list(comunes),
            edge_color="black",
            style="solid",
            width=2.8,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.0",
            min_source_margin=15,
            min_target_margin=15,
            ax=ax
        )

    # --------------------------------------------------------------
    # Etiquetas de capas
    # --------------------------------------------------------------

    ax.text(
        -3.8,
        3.5,
        "INICIO",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.text(
        -1.3,
        3.5,
        "SEMANA ANTERIOR",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.text(
        1.3,
        3.5,
        "SEMANA ACTUAL",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.text(
        4.0,
        3.5,
        "FIN",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    # --------------------------------------------------------------
    # Leyenda
    # --------------------------------------------------------------

    from matplotlib.lines import Line2D

    elementos_leyenda = [

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.8,
            linestyle="-",
            label="Arco presente en ambas estructuras"
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            linestyle="-",
            label="Arco solo en el grafo manual"
        ),

        Line2D(
            [0],
            [0],
            color="dimgray",
            linewidth=1.8,
            linestyle="--",
            label="Arco solo en el grafo aprendido"
        ),
    ]

    ax.legend(
        handles=elementos_leyenda,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    # --------------------------------------------------------------
    # Información de comparación
    # --------------------------------------------------------------

    ax.text(
        0.01,
        0.01,
        (
            f"Asignatura: {materia}\n"
            f"Arcos manuales: {len(manual_set)}   |   "
            f"Arcos aprendidos: {len(aprendido_set)}   |   "
            f"Coincidentes: {len(comunes)}"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom"
    )

    # --------------------------------------------------------------
    # Configuración final
    # --------------------------------------------------------------

    ax.set_xlim(-5.0, 5.2)
    ax.set_ylim(-3.5, 4.0)

    ax.axis("off")

    plt.tight_layout()

    # Guardar en PNG de alta resolución
    fig.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"Figura generada: {ruta_salida}"
    )


def crear_figura_consolidada(
    resultados_graficos,
    ruta_salida
):
    """
    Genera una figura consolidada con un panel por asignatura.

    Cada panel muestra:
        - estructura manual
        - estructura aprendida
        - coincidencias entre ambas

    La estructura de cada panel mantiene la misma distribución
    temporal para facilitar la comparación.
    """

    materias = list(resultados_graficos.keys())

    if not materias:
        return

    # --------------------------------------------------------------
    # Figura
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        len(materias),
        1,
        figsize=(15, 7 * len(materias))
    )

    if len(materias) == 1:
        axes = [axes]

    posiciones = construir_posiciones()

    etiquetas = {

        "Año que cursa":
            "Año que\ncursa",

        "Sección":
            "Sección",

        "Tamaño del grupo":
            "Tamaño del\ngrupo",

        "Posición relativa en la lista":
            "Posición relativa\nen la lista",

        "Tema de la sesión":
            "Tema de la\nsesión",

        "Número de sesiones de la semana":
            "N.º de sesiones\nde la semana",

        "Sesiones de evaluación de la semana":
            "Sesiones de evaluación\nde la semana",

        "Participaciones de la semana anterior":
            "Participaciones de\nla semana anterior",

        "Participaciones de la semana":
            "Participaciones de\nla semana",

        "Cantidad de participaciones del trimestre":
            "Cantidad de participaciones\ndel trimestre",
    }

    # --------------------------------------------------------------
    # Dibujar cada asignatura
    # --------------------------------------------------------------

    for ax, materia in zip(
        axes,
        materias
    ):

        arcos_manual = set(
            resultados_graficos[materia]["manual"]
        )

        arcos_aprendidos = set(
            resultados_graficos[materia]["aprendido"]
        )

        comunes = (
            arcos_manual &
            arcos_aprendidos
        )

        solo_manual = (
            arcos_manual -
            arcos_aprendidos
        )

        solo_aprendido = (
            arcos_aprendidos -
            arcos_manual
        )

        G = nx.DiGraph()

        G.add_nodes_from(
            COLUMNAS_BN
        )

        # ----------------------------------------------------------
        # Nodos
        # ----------------------------------------------------------

        nx.draw_networkx_nodes(
            G,
            posiciones,
            node_size=4200,
            node_color="white",
            edgecolors="black",
            linewidths=1.3,
            node_shape="s",
            ax=ax
        )

        # ----------------------------------------------------------
        # Etiquetas
        # ----------------------------------------------------------

        nx.draw_networkx_labels(
            G,
            posiciones,
            labels=etiquetas,
            font_size=7,
            ax=ax
        )

        # ----------------------------------------------------------
        # Solo manual
        # ----------------------------------------------------------

        if solo_manual:

            nx.draw_networkx_edges(
                G,
                posiciones,
                edgelist=list(solo_manual),
                edge_color="black",
                style="solid",
                width=1.5,
                arrows=True,
                arrowsize=16,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.04",
                min_source_margin=15,
                min_target_margin=15,
                ax=ax
            )

        # ----------------------------------------------------------
        # Solo aprendido
        # ----------------------------------------------------------

        if solo_aprendido:

            nx.draw_networkx_edges(
                G,
                posiciones,
                edgelist=list(solo_aprendido),
                edge_color="dimgray",
                style="dashed",
                width=1.5,
                arrows=True,
                arrowsize=16,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-0.04",
                min_source_margin=15,
                min_target_margin=15,
                ax=ax
            )

        # ----------------------------------------------------------
        # Comunes
        # ----------------------------------------------------------

        if comunes:

            nx.draw_networkx_edges(
                G,
                posiciones,
                edgelist=list(comunes),
                edge_color="black",
                style="solid",
                width=2.5,
                arrows=True,
                arrowsize=18,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.0",
                min_source_margin=15,
                min_target_margin=15,
                ax=ax
            )

        # ----------------------------------------------------------
        # Encabezado del panel
        # ----------------------------------------------------------

        ax.text(
            0.5,
            0.98,
            str(materia),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold"
        )

        # ----------------------------------------------------------
        # Capas temporales
        # ----------------------------------------------------------

        ax.text(
            -3.8,
            3.5,
            "INICIO",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

        ax.text(
            -1.3,
            3.5,
            "SEMANA ANTERIOR",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

        ax.text(
            1.3,
            3.5,
            "SEMANA ACTUAL",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

        ax.text(
            4.0,
            3.5,
            "FIN",
            ha="center",
            fontsize=8,
            fontweight="bold"
        )

        # ----------------------------------------------------------
        # Estadísticas
        # ----------------------------------------------------------

        ax.text(
            0.5,
            0.02,
            (
                f"Manual: {len(arcos_manual)} arcos   |   "
                f"Aprendido: {len(arcos_aprendidos)} arcos   |   "
                f"Coincidentes: {len(comunes)}"
            ),
            transform=ax.transAxes,
            ha="center",
            fontsize=7
        )

        ax.set_xlim(-5.0, 5.2)
        ax.set_ylim(-3.5, 4.0)

        ax.axis("off")

    # --------------------------------------------------------------
    # Leyenda global
    # --------------------------------------------------------------

    from matplotlib.lines import Line2D

    elementos_leyenda = [

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.5,
            linestyle="-",
            label="Arco presente en ambas estructuras"
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            linestyle="-",
            label="Arco solo en el grafo manual"
        ),

        Line2D(
            [0],
            [0],
            color="dimgray",
            linewidth=1.5,
            linestyle="--",
            label="Arco solo en el grafo aprendido"
        ),
    ]

    fig.legend(
        handles=elementos_leyenda,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9
    )

    fig.subplots_adjust(
        hspace=0.10,
        bottom=0.04
    )

    fig.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"\nFigura consolidada generada: {ruta_salida}"
    )


# ======================================================================
# 10. APRENDIZAJE Y COMPARACIÓN
# ======================================================================

resultados = {}

resultados_graficos = {}


for materia in sorted(
    sesiones["materia"].unique()
):

    print("=" * 70)
    print(materia)
    print("=" * 70)

    datos = (
        sesiones[
            sesiones["materia"] == materia
        ][COLUMNAS_BN]
        .copy()
    )

    # La primera semana no tiene "semana anterior"
    datos = datos.dropna()

    n = len(datos)

    print(
        "filas estudiante-sesión disponibles "
        f"(con semana anterior definida): {n}"
    )

    # --------------------------------------------------------------
    # Verificación de estados
    # --------------------------------------------------------------

    advertencias = []

    for col in COLUMNAS_BN:

        conteo = datos[col].value_counts()

        chicos = conteo[
            conteo < 5
        ]

        if len(chicos):

            advertencias.append(
                f"  {col}: estados con menos de 5 casos -> "
                f"{chicos.to_dict()}"
            )

        if datos[col].nunique() == 1:

            advertencias.append(
                f"  {col}: es CONSTANTE en esta asignatura "
                f"(único valor: {datos[col].iloc[0]!r}) "
                "-> no puede aportar ningún arco, ni al "
                "algoritmo ni de forma informativa al grafo manual"
            )

    if advertencias:

        print("ADVERTENCIA:")

        print(
            "\n".join(advertencias)
        )

    # --------------------------------------------------------------
    # Convertir a categórico/string
    # --------------------------------------------------------------

    for col in COLUMNAS_BN:

        datos[col] = (
            datos[col]
            .astype(str)
        )

    variante_resultados = {}

    modelo_b_completo = None

    # ==============================================================
    # (A) HILL CLIMBING SIN RESTRICCIONES
    # ==============================================================

    try:

        hc = HillClimbSearch(
            datos
        )

        modelo_a = hc.estimate(
            scoring_method="bic-d",
            show_progress=False
        )

        arcos_a = sorted(
            modelo_a.edges()
        )

        print(
            f"\n(a) Sin restricciones — "
            f"{len(arcos_a)} arcos aprendidos:"
        )

        for u, v in arcos_a:

            print(
                f"    {u} → {v}"
            )

        variante_resultados[
            "libre"
        ] = arcos_a

    except Exception as e:

        print(
            f"\n(a) FALLÓ: "
            f"{type(e).__name__}: {e}"
        )

        variante_resultados[
            "libre"
        ] = None

    # ==============================================================
    # (B) HILL CLIMBING CON RESTRICCIONES TEMPORALES
    # ==============================================================

    try:

        ek = ExpertKnowledge(
            temporal_order=[
                INICIO,
                DURANTE_ANTERIOR,
                DURANTE_ACTUAL,
                FIN
            ]
        )

        hc2 = HillClimbSearch(
            datos
        )

        modelo_b = hc2.estimate(
            scoring_method="bic-d",
            expert_knowledge=ek,
            show_progress=False
        )

        arcos_b = sorted(
            modelo_b.edges()
        )

        print(
            f"\n(b) Con restricciones temporales "
            f"(Tabla 13) — "
            f"{len(arcos_b)} arcos aprendidos:"
        )

        for u, v in arcos_b:

            print(
                f"    {u} → {v}"
            )

        variante_resultados[
            "restringido"
        ] = arcos_b

        modelo_b_completo = modelo_b

    except Exception as e:

        print(
            f"\n(b) FALLÓ: "
            f"{type(e).__name__}: {e}"
        )

        variante_resultados[
            "restringido"
        ] = None

        modelo_b_completo = None

    # ==============================================================
    # COMPARACIÓN CON EL GRAFO MANUAL
    # ==============================================================

    manual_set = set(
        ARCOS_MANUALES
    )

    if variante_resultados[
        "restringido"
    ] is not None:

        aprendido_set = set(
            variante_resultados[
                "restringido"
            ]
        )

        comunes = (
            manual_set &
            aprendido_set
        )

        solo_aprendido = (
            aprendido_set -
            manual_set
        )

        solo_manual = (
            manual_set -
            aprendido_set
        )

        print(
            "\nComparación "
            "(contra la variante b, con restricciones):"
        )

        print(
            f"  En ambos ({len(comunes)}):"
        )

        for u, v in sorted(comunes):

            print(
                f"    {u} → {v}"
            )

        print(
            f"  Solo en el aprendido "
            f"({len(solo_aprendido)}):"
        )

        for u, v in sorted(
            solo_aprendido
        ):

            print(
                f"    {u} → {v}"
            )

        print(
            f"  Solo en el manual "
            f"({len(solo_manual)}):"
        )

        for u, v in sorted(
            solo_manual
        ):

            print(
                f"    {u} → {v}"
            )

    # ==============================================================
    # PUNTAJES BIC
    # ==============================================================

    bic = BIC(
        datos
    )

    modelo_manual = (
        DiscreteBayesianNetwork()
    )

    modelo_manual.add_nodes_from(
        COLUMNAS_BN
    )

    modelo_manual.add_edges_from(
        ARCOS_MANUALES
    )

    score_manual = None
    score_aprendido = None

    try:

        score_manual = bic.score(
            modelo_manual
        )

        print(
            f"\nBIC del grafo manual: "
            f"{score_manual:.1f}"
        )

    except Exception as e:

        print(
            f"\nBIC del grafo manual "
            f"FALLÓ: "
            f"{type(e).__name__}: {e}"
        )

    if modelo_b_completo is not None:

        try:

            score_aprendido = bic.score(
                modelo_b_completo
            )

            print(
                f"BIC del grafo aprendido "
                f"(b, restringido): "
                f"{score_aprendido:.1f}"
            )

            if score_manual is not None:

                diferencia = (
                    score_aprendido -
                    score_manual
                )

                print(
                    f"Diferencia "
                    f"(aprendido − manual): "
                    f"{diferencia:.1f}"
                )

        except Exception as e:

            print(
                f"BIC del grafo aprendido "
                f"FALLÓ: "
                f"{type(e).__name__}: {e}"
            )

    # ==============================================================
    # GUARDAR RESULTADOS
    # ==============================================================

    resultados[materia] = {
        "variantes": variante_resultados,
        "bic_manual": score_manual,
        "bic_aprendido": score_aprendido
    }

    # ==============================================================
    # GENERAR FIGURA DE LA ASIGNATURA
    # ==============================================================

    if variante_resultados[
        "restringido"
    ] is not None:

        nombre_archivo = (
            "estructura_comparada_"
            + nombre_seguro(materia)
            + ".png"
        )

        ruta_figura = (
            CARPETA_FIGURAS /
            nombre_archivo
        )

        crear_grafo_comparativo(
            materia=materia,
            arcos_manual=ARCOS_MANUALES,
            arcos_aprendidos=(
                variante_resultados[
                    "restringido"
                ]
            ),
            ruta_salida=ruta_figura
        )

        resultados_graficos[
            materia
        ] = {
            "manual": ARCOS_MANUALES,
            "aprendido": (
                variante_resultados[
                    "restringido"
                ]
            )
        }

    print()


# ======================================================================
# 11. FIGURA CONSOLIDADA
# ======================================================================

if resultados_graficos:

    ruta_consolidada = (
        CARPETA_FIGURAS /
        "estructura_comparada_todas_asignaturas.png"
    )

    crear_figura_consolidada(
        resultados_graficos,
        ruta_consolidada
    )


# ======================================================================
# 12. FINAL
# ======================================================================

print("=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)

print(
    f"Las figuras se guardaron en:\n"
    f"{CARPETA_FIGURAS}"
)

print()
print(
    "Se generó una figura individual por asignatura "
    "y una figura consolidada cuando el aprendizaje "
    "restringido fue exitoso."
)