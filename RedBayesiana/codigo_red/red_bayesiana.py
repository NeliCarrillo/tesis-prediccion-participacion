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

# Los 10 nodos son exactamente las columnas del conjunto que ensambla
# `ensamblado.ensamblar_conjunto()` (Sprint 4, carta 2) — una sola fuente de
# verdad para los nombres de nodo.
NODOS: tuple[str, ...] = tuple(COLUMNAS_BN)

# Los 10 arcos de la Figura 12 (Método, Sprint 3), ya cerrados y verificados
# contra el informe actual — no se rediseñan aquí.
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


def construir_modelo_manual() -> DiscreteBayesianNetwork:
    """Construye la estructura final (Figura 12) sin CPD.

    Llamar una vez por asignatura, en la fase de ajuste (carta 4), para
    obtener una instancia propia por materia — esta función siempre
    devuelve un objeto nuevo, nunca uno compartido.

    Verifica programáticamente, antes de devolver el modelo, que:
      - es un DAG (sin ciclos);
      - tiene exactamente los 10 nodos de `NODOS`, ni más ni menos;
      - tiene exactamente los 10 arcos de `ARCOS_MANUALES`, ni más ni menos.
    """
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
