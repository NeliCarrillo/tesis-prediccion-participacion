"""Estructura final de la red bayesiana (Sprint 3, Figura 12), implementada
en `pgmpy.models.DiscreteBayesianNetwork` — Sprint 4, carta 3.

10 nodos, 10 arcos, sin CPD. `construir_modelo_manual()` es una fábrica: cada
llamada devuelve una instancia nueva e independiente, lista para que una
fase posterior (carta 4) la ajuste con los datos de una asignatura —
"materia" nunca es un nodo, solo segmenta qué datos se usan para ajustar
cada instancia.

Fuera de alcance de esta carta, deliberadamente no incluido aquí: estimación
de CPD/BDeu, inferencia, validación y métricas (cartas 4 a 7).

Importar este módulo no construye ningún modelo ni toca datos: solo declara
la estructura (listas/tuplas) y la función fábrica.
"""
from __future__ import annotations

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from ensamblado import COLUMNAS_BN

# Los 10 arcos de la Figura 12 (Método, Sprint 3), ya cerrados y verificados
# contra el informe actual — no se rediseñan aquí. Es la fuente de verdad de
# la estructura: qué es "un nodo" lo define este grafo, no el esquema de
# datos ensamblado.
ARCOS_MANUALES: tuple[tuple[str, str], ...] = (
    # Estructura del contexto
    ("Sección", "Tamaño del grupo"),

    # Relaciones dentro de la semana
    ("Número de sesiones de la semana", "Sesiones de evaluación de la semana"),
    ("Sesiones de evaluación de la semana", "Participaciones de la semana"),
    ("Posición relativa en la lista", "Participaciones de la semana"),
    ("Tema de la sesión", "Participaciones de la semana"),

    # Dependencia temporal
    ("Participaciones de la semana anterior", "Participaciones de la semana"),

    # Dependencias con el objetivo
    ("Participaciones de la semana", "Cantidad de participaciones del trimestre"),
    ("Participaciones de la semana anterior", "Cantidad de participaciones del trimestre"),
    ("Tamaño del grupo", "Cantidad de participaciones del trimestre"),
    ("Año que cursa", "Cantidad de participaciones del trimestre"),
)

# Los nodos son exactamente los que aparecen en ARCOS_MANUALES — se derivan
# del grafo, no del esquema de datos ensamblado. (En esta estructura ningún
# nodo queda aislado, así que el grafo por sí solo ya determina las 10
# variables.)
NODOS: tuple[str, ...] = tuple(sorted({nodo for arco in ARCOS_MANUALES for nodo in arco}))


def _verificar_consistencia_con_ensamblado() -> None:
    """Nodos del grafo y columnas del conjunto ensamblado son dos conceptos
    distintos que hoy deben coincidir (todo nodo debe tener su columna en
    los datos, y viceversa). Se verifica explícitamente en cada
    construcción, en vez de asumirlo por definición compartida, para
    detectar cualquier divergencia futura entre la estructura (Figura 12)
    y el esquema de datos de `ensamblado.py` (Sprint 4, carta 2)."""
    nodos = set(NODOS)
    columnas = set(COLUMNAS_BN)
    solo_en_grafo = nodos - columnas
    solo_en_datos = columnas - nodos
    if solo_en_grafo or solo_en_datos:
        raise ValueError(
            "red_bayesiana.NODOS y ensamblado.COLUMNAS_BN dejaron de "
            f"coincidir. Solo en el grafo={sorted(solo_en_grafo)}; "
            f"solo en los datos ensamblados={sorted(solo_en_datos)}"
        )


def construir_modelo_manual() -> DiscreteBayesianNetwork:
    """Construye la estructura final (Figura 12) sin CPD.

    Llamar una vez por asignatura, en la fase de ajuste (carta 4), para
    obtener una instancia propia por materia — esta función siempre
    devuelve un objeto nuevo, nunca uno compartido.

    Verifica programáticamente, antes de devolver el modelo, que:
      - es un DAG (sin ciclos);
      - tiene exactamente los 10 nodos de `NODOS`, ni más ni menos;
      - tiene exactamente los 10 arcos de `ARCOS_MANUALES`, ni más ni menos;
      - sus nodos siguen coincidiendo con las columnas que ensambla
        `ensamblado.py` (ver `_verificar_consistencia_con_ensamblado`).
    """
    _verificar_consistencia_con_ensamblado()

    modelo = DiscreteBayesianNetwork()
    modelo.add_nodes_from(NODOS)
    modelo.add_edges_from(ARCOS_MANUALES)

    if not nx.is_directed_acyclic_graph(modelo):
        raise ValueError("La estructura construida no es un DAG válido")

    nodos_modelo = set(modelo.nodes())
    faltan_nodos = set(NODOS) - nodos_modelo
    sobran_nodos = nodos_modelo - set(NODOS)
    if faltan_nodos or sobran_nodos:
        raise ValueError(
            f"Nodos inesperados. faltan={sorted(faltan_nodos)} sobran={sorted(sobran_nodos)}"
        )
    if len(nodos_modelo) != 10:
        raise ValueError(f"Se esperaban 10 nodos, el modelo tiene {len(nodos_modelo)}")

    arcos_modelo = set(modelo.edges())
    arcos_esperados = set(ARCOS_MANUALES)
    faltan_arcos = arcos_esperados - arcos_modelo
    sobran_arcos = arcos_modelo - arcos_esperados
    if faltan_arcos or sobran_arcos:
        raise ValueError(
            f"Arcos inesperados. faltan={sorted(faltan_arcos)} sobran={sorted(sobran_arcos)}"
        )
    if len(arcos_modelo) != 10:
        raise ValueError(f"Se esperaban 10 arcos, el modelo tiene {len(arcos_modelo)}")

    return modelo


if __name__ == "__main__":
    modelo_1 = construir_modelo_manual()
    print(f"DAG válido: {nx.is_directed_acyclic_graph(modelo_1)}")
    print(f"Nodos ({len(modelo_1.nodes())}): {sorted(modelo_1.nodes())}")
    print(f"Arcos ({len(modelo_1.edges())}):")
    for origen, destino in sorted(modelo_1.edges()):
        print(f"    {origen} → {destino}")

    # Independencia entre instancias: cada llamada a construir_modelo_manual()
    # debe devolver un objeto propio, no uno compartido entre asignaturas.
    modelo_2 = construir_modelo_manual()
    assert modelo_1 is not modelo_2, "las instancias no deberían ser el mismo objeto"
    modelo_2.add_edge("Sección", "Año que cursa")
    assert len(modelo_1.edges()) == 10, (
        "modificar una instancia no debería afectar a otra ya construida"
    )
    assert len(modelo_2.edges()) == 11

    print()
    print("verificaciones superadas: DAG válido, 10 nodos, 10 arcos exactos, "
          "instancias independientes por llamada")
