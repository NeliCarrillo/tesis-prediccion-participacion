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
registro por estudiante-sesión (la unidad más fina disponible: 11.700 filas en
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
import os
import warnings
import re
import sys

# pgmpy resuelve algunos empates de HillClimbSearch iterando conjuntos. Fijar
# el hash antes de importar la biblioteca hace reproducible esa elección.
if os.environ.get("PYTHONHASHSEED") != "0":
    entorno = os.environ.copy()
    entorno["PYTHONHASHSEED"] = "0"
    os.execve(sys.executable, [sys.executable, *sys.argv], entorno)

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.estimators import HillClimbSearch, BIC
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

from ensamblado import ensamblar_conjunto, CLAVE, COLUMNAS_BN, ESTADOS_BN, RAIZ
from discretizacion import ajustar_mapa_temas, aplicar_mapa_temas
from red_bayesiana import ARCOS_MANUALES


# ======================================================================
# 0. CONFIGURACIÓN
# ======================================================================

# Carpeta donde se guardarán las figuras
CARPETA_FIGURAS = RAIZ / "RedBayesiana" / "figuras"
CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 1-6. ENSAMBLADO DEL CONJUNTO DISCRETIZADO (Sprint 4, carta 2)
# ======================================================================
# La carga de datos, la construcción de `reg` (524 registros
# estudiante-sección) y `sem` (6.288 filas estudiante-semana), y la
# discretización de las 9 variables no supervisadas ahora viven en
# `ensamblado.py` y `discretizacion.py`, reusadas aquí en vez de
# duplicadas. Ver `ensamblado.py` para el detalle de la unidad de ensamblado
# (estudiante-sesión) y por qué no se colapsa a estudiante-semana.

sesiones, reg, sem = ensamblar_conjunto()

# Mapa descriptivo del conjunto completo, para el Apéndice A (Tabla A4) y la
# verificación de más abajo. La evaluación por fold, definida más adelante
# en `preparar_fold_tema`, nunca reutiliza este mapa: ajusta uno nuevo solo
# con los trimestres de entrenamiento de cada pliegue.
mapa_tema_global, temas = ajustar_mapa_temas(sesiones)

sesiones["Tema de la sesión"] = [
    *aplicar_mapa_temas(
        sesiones,
        mapa_tema_global,
        permitir_no_visto=False
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
    sesiones[
        "Tema de la sesión"
    ].value_counts(
        dropna=False
    ).to_dict()
)

print()


# ======================================================================
# 6. HIPERPARÁMETROS DE ESTIMACIÓN
# ======================================================================
# COLUMNAS_BN y ESTADOS_BN ahora vienen de `ensamblado.py` (import arriba);
# `sesiones` ya incluye "Sección" discretizada desde `ensamblar_conjunto`.

ESS_EVALUADOS = [1, 5, 10]
ESS_SELECCIONADO = 5


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
# ARCOS_MANUALES ahora vive en red_bayesiana.py (Sprint 4, carta 3), la
# fuente única de verdad de la estructura final (10 arcos, Figura 12); se
# importa arriba junto con ensamblado/discretizacion para no mantener dos
# copias de la misma lista.


# ======================================================================
# 8.1. VALIDACIÓN LEAVE-ONE-TRIMESTER-OUT Y ESTIMACIÓN DE CPT
# ======================================================================

def preparar_fold_tema(materia, trimestre_prueba):
    """Prepara train/test y ajusta el mapa de tema solo con train."""
    datos_materia = sesiones[
        sesiones["materia"] == materia
    ].copy()

    train = datos_materia[
        datos_materia["trimestre"] != trimestre_prueba
    ].copy()
    test = datos_materia[
        datos_materia["trimestre"] == trimestre_prueba
    ].copy()

    mapa, tabla_temas = ajustar_mapa_temas(train)

    train["Tema de la sesión"] = aplicar_mapa_temas(
        train,
        mapa,
        permitir_no_visto=False
    )
    test["Tema de la sesión"] = aplicar_mapa_temas(
        test,
        mapa,
        permitir_no_visto=True
    )

    temas_no_vistos = sorted(
        test.loc[
            test["Tema de la sesión"] == "tema_no_visto",
            "tema"
        ].astype(int).unique().tolist()
    )
    filas_no_vistas = int(
        (test["Tema de la sesión"] == "tema_no_visto").sum()
    )

    categorias_presentes = set(
        tabla_temas["categoria"].dropna().astype(str)
    )
    categorias_vacias = [
        categoria
        for categoria in ["baja", "media", "alta"]
        if categoria not in categorias_presentes
    ]

    train_bn = train[COLUMNAS_BN].dropna().astype(str)
    test_bn = test[COLUMNAS_BN].dropna().astype(str)

    for columna, estados in ESTADOS_BN.items():
        valores_invalidos = set(train_bn[columna]) - set(estados)
        valores_invalidos |= set(test_bn[columna]) - set(estados)
        if valores_invalidos:
            raise ValueError(
                f"Estados fuera del dominio en {columna}: "
                f"{sorted(valores_invalidos)}"
            )

    return {
        "materia": materia,
        "trimestre_prueba": trimestre_prueba,
        "train": train_bn,
        "test": test_bn,
        "temas_no_vistos": temas_no_vistos,
        "filas_no_vistas": filas_no_vistas,
        "filas_test": len(test),
        "categorias_vacias_train": categorias_vacias
    }


def ajustar_cpt_bdeu(datos_train, ess):
    """Ajusta las CPT del grafo manual con BDeu y dominio completo."""
    modelo = DiscreteBayesianNetwork()
    modelo.add_nodes_from(COLUMNAS_BN)
    modelo.add_edges_from(ARCOS_MANUALES)

    estimador = DiscreteBayesianEstimator(
        state_names=ESTADOS_BN,
        prior_type="BDeu",
        equivalent_sample_size=ess
    )
    modelo.fit(datos_train, estimator=estimador)

    min_probabilidad = 1.0
    max_error_normalizacion = 0.0

    for cpd in modelo.get_cpds():
        valores = np.asarray(cpd.values, dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(
                f"La CPT de {cpd.variable} contiene valores no finitos"
            )

        min_probabilidad = min(min_probabilidad, float(valores.min()))
        sumas = valores.sum(axis=0)
        max_error_normalizacion = max(
            max_error_normalizacion,
            float(np.max(np.abs(sumas - 1.0)))
        )

    return modelo, min_probabilidad, max_error_normalizacion


print("=" * 70)
print("VALIDACIÓN LEAVE-ONE-TRIMESTER-OUT DEL TEMA")
print("=" * 70)
print(
    "El mapa de cada fold se ajusta solo con train. "
    "ESS evaluados con train:",
    ESS_EVALUADOS
)

auditoria_folds = []

for materia in sorted(sesiones["materia"].unique()):
    trimestres = sorted(
        sesiones.loc[
            sesiones["materia"] == materia,
            "trimestre"
        ].unique()
    )

    for trimestre_prueba in trimestres:
        fold = preparar_fold_tema(materia, trimestre_prueba)
        sensibilidad = {}

        for ess in ESS_EVALUADOS:
            modelo_cpt, minimo, error = ajustar_cpt_bdeu(
                fold["train"],
                ess
            )
            sensibilidad[ess] = {
                "min_probabilidad": minimo,
                "max_error_normalizacion": error
            }

            # La CPT hija conserva explícitamente tema_no_visto como estado
            # del padre, aunque no haya aparecido en train.
            cpd_participacion = modelo_cpt.get_cpds(
                "Participaciones de la semana"
            )
            if "tema_no_visto" not in cpd_participacion.state_names[
                "Tema de la sesión"
            ]:
                raise AssertionError(
                    "tema_no_visto no fue conservado en la CPT"
                )

        fold["sensibilidad_ess"] = sensibilidad
        auditoria_folds.append(fold)

        print(
            f"{materia} | test={trimestre_prueba} | "
            f"no vistos={fold['temas_no_vistos'] or 'ninguno'} | "
            f"filas={fold['filas_no_vistas']}/{fold['filas_test']} | "
            f"categorías vacías en train="
            f"{fold['categorias_vacias_train'] or 'ninguna'}"
        )

print("\nSensibilidad BDeu (solo train; todas las CPT):")
for ess in ESS_EVALUADOS:
    minimo = min(
        fold["sensibilidad_ess"][ess]["min_probabilidad"]
        for fold in auditoria_folds
    )
    error = max(
        fold["sensibilidad_ess"][ess]["max_error_normalizacion"]
        for fold in auditoria_folds
    )
    print(
        f"  ESS={ess}: probabilidad mínima={minimo:.12g}; "
        f"error máximo de normalización={error:.3g}"
    )

print(
    "ESS seleccionado para la implementación: "
    f"{ESS_SELECCIONADO} (valor predeterminado de pgmpy y punto medio "
    "del análisis 1/5/10; se fija sin usar test)."
)

param_participacion_antes = (4 - 1) * 4 * 3 * 4 * 4
param_participacion_despues = (4 - 1) * 4 * 3 * 5 * 4
param_tema_antes = 4 - 1
param_tema_despues = 5 - 1
param_objetivo = (5 - 1) * 4 * 4 * 5 * 3

assert param_participacion_antes == 576
assert param_participacion_despues == 720
assert param_tema_antes == 3
assert param_tema_despues == 4
assert param_objetivo == 960

print(
    "Parámetros libres: participación semanal "
    f"{param_participacion_antes} -> {param_participacion_despues}; "
    f"tema {param_tema_antes} -> {param_tema_despues}; "
    f"objetivo={param_objetivo}."
)
print()


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


ETIQUETAS_NODOS = {
    "Año que cursa": "Año",
    "Sección": "Sección",
    "Tamaño del grupo": "Tamaño\ngrupo",
    "Posición relativa en la lista": "Posición\nlista",
    "Tema de la sesión": "Tema",
    "Número de sesiones de la semana": "Sesiones",
    "Sesiones de evaluación de la semana": "Sesiones\neval.",
    "Participaciones de la semana anterior": "Part.\nprevia",
    "Participaciones de la semana": "Part.\nsemanal",
    "Cantidad de participaciones del trimestre": "Part.\ntrimestral",
}

NOMBRES_CORTOS_MATERIA = {
    "Algoritmos y Programación": "Algoritmos y Programación",
    "Computación Emergente": "Computación Emergente",
    "Estructura de Datos": "Estructura de Datos",
    "Matemáticas Discretas": "Matemáticas Discretas",
}


def _curvatura_arco(origen, destino, indice):
    """Separa arcos paralelos y verticales sin alterar el grafo."""
    posiciones = construir_posiciones()
    x1, y1 = posiciones[origen]
    x2, y2 = posiciones[destino]

    if x1 == x2:
        return 0.22 if y2 > y1 else -0.22

    signo = -1 if indice % 2 else 1
    distancia = abs(x2 - x1)
    return signo * (0.045 if distancia <= 3 else 0.085)


def dibujar_grafo_aprendido(
    ax,
    nodos,
    arcos,
    materia,
    font_nodos,
    font_panel,
    font_capas,
    node_size
):
    """Dibuja exclusivamente un DAG aprendido con orden temporal fijo."""
    posiciones = construir_posiciones()
    nodos = list(nodos)
    arcos = list(arcos)

    G = nx.DiGraph()
    G.add_nodes_from(nodos)
    G.add_edges_from(arcos)

    for indice, arco in enumerate(sorted(arcos)):
        nx.draw_networkx_edges(
            G,
            posiciones,
            edgelist=[arco],
            edge_color="#303030",
            width=2.4,
            arrows=True,
            arrowsize=30,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={_curvatura_arco(*arco, indice)}",
            min_source_margin=28,
            min_target_margin=28,
            ax=ax
        )

    # Los nodos y sus rótulos se dibujan después de los arcos para que
    # ninguna flecha atraviese el texto.
    nx.draw_networkx_nodes(
        G,
        posiciones,
        nodelist=nodos,
        node_size=node_size,
        node_color="#f7f7f7",
        edgecolors="#303030",
        linewidths=2.2,
        node_shape="s",
        ax=ax
    )

    nx.draw_networkx_labels(
        G,
        posiciones,
        labels={n: ETIQUETAS_NODOS[n] for n in nodos},
        font_family="Arial",
        font_size=font_nodos,
        font_weight="normal",
        ax=ax
    )

    titulo = NOMBRES_CORTOS_MATERIA.get(materia, materia)
    ax.set_title(
        f"{titulo} ({len(arcos)} arcos)",
        fontsize=font_panel,
        fontweight="bold",
        family="Arial",
        pad=16
    )
    ax.set_xlim(-5.15, 5.35)
    ax.set_ylim(-3.55, 3.55)
    ax.axis("off")


def crear_grafo_aprendido(materia, nodos, arcos, ruta_salida):
    """Genera la figura individual de la estructura restringida aprendida."""
    fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
    dibujar_grafo_aprendido(
        ax=ax,
        nodos=nodos,
        arcos=arcos,
        materia=materia,
        font_nodos=22,
        font_panel=28,
        font_capas=21,
        node_size=6000
    )
    fig.savefig(ruta_salida, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Figura aprendida generada: {ruta_salida}")


def crear_figura_consolidada(resultados_graficos, ruta_salida):
    """Genera cuatro paneles con solo las estructuras restringidas aprendidas."""
    orden = [
        "Algoritmos y Programación",
        "Computación Emergente",
        "Estructura de Datos",
        "Matemáticas Discretas",
    ]
    materias = [m for m in orden if m in resultados_graficos]

    if not materias:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 11),
        constrained_layout=True
    )

    for ax, materia in zip(axes.flat, materias):
        resultado = resultados_graficos[materia]
        dibujar_grafo_aprendido(
            ax=ax,
            nodos=resultado["nodos"],
            arcos=resultado["aprendido"],
            materia=materia,
            font_nodos=22,
            font_panel=24,
            font_capas=20,
            node_size=5500
        )

    for ax in axes.flat[len(materias):]:
        ax.axis("off")

    fig.suptitle(
        "Estructuras aprendidas automáticamente\n"
        "para contraste metodológico",
        fontsize=28,
        fontweight="bold",
        family="Arial"
    )
    fig.savefig(ruta_salida, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nFigura consolidada generada: {ruta_salida}")


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

    bic = BIC(
        datos,
        state_names=ESTADOS_BN
    )

    variante_resultados = {}

    modelo_b_completo = None

    # ==============================================================
    # (A) HILL CLIMBING SIN RESTRICCIONES
    # ==============================================================

    try:

        hc = HillClimbSearch(
            datos,
            state_names=ESTADOS_BN
        )

        modelo_a = hc.estimate(
            scoring_method=bic,
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
            datos,
            state_names=ESTADOS_BN
        )

        modelo_b = hc2.estimate(
            scoring_method=bic,
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

        crear_grafo_aprendido(
            materia=materia,
            nodos=list(modelo_b_completo.nodes()),
            arcos=(
                variante_resultados[
                    "restringido"
                ]
            ),
            ruta_salida=ruta_figura
        )

        resultados_graficos[
            materia
        ] = {
            "nodos": list(modelo_b_completo.nodes()),
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
