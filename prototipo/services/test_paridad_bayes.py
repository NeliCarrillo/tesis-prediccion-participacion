"""Pruebas de paridad del servicio Bayes del prototipo — Sprint 5, tarjeta 3.

## Nivel A — evidencia (comparación directa contra el pipeline original)

Los modelos de Estrategia 2 (todo el histórico) son necesariamente
distintos de los de la validación cruzada de Sprint 4 (que excluyen un
trimestre) — sus CPD y sus predicciones no tienen por qué coincidir. Pero
la EVIDENCIA de un registro (año, tamaño de grupo, participaciones de la
semana y de la semana anterior) no depende de qué modelo la consulta,
solo del propio registro — por eso sí debe ser idéntica a la que ya quedó
grabada por Sprint 4 en
`RedBayesiana/resultados/predicciones_bayesiana_s4_s6_s8.csv`. Esta es la
comparación programática contra el procedimiento original que pide la
tarjeta: no es un mock, es el CSV que produjo `inferencia.py` real.

## Nivel B — inferencia funcional

S4/S6/S8, posterior normalizada, valor esperado finito, y un caso con año
académico faltante (evidencia parcial), verificado contra el modelo real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bayes_service
from caso import CasoPrediccion

REPO = Path(__file__).resolve().parent.parent.parent
CSV_SPRINT4 = REPO / "RedBayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"


def probar_paridad_evidencia() -> None:
    referencia = pd.read_csv(CSV_SPRINT4)

    casos = [
        ("Algoritmos y Programación", "2425-2", 1, "anon_001", 4),
        ("Computación Emergente", "2425-3", 1, "anon_148", 6),
        ("Matemáticas Discretas", "2526-2", 1, "anon_md_2526_2_1_001", 8),
        # anon_063: año académico faltante -- ejerce la evidencia parcial.
        ("Algoritmos y Programación", "2526-1", 1, "anon_063", 4),
    ]

    for materia, trimestre, seccion, estudiante_id, hito in casos:
        fila_ref = referencia[
            (referencia.materia == materia)
            & (referencia.trimestre_prueba == trimestre)
            & (referencia.seccion == seccion)
            & (referencia.estudiante_id == estudiante_id)
            & (referencia.hito == hito)
        ]
        assert len(fila_ref) == 1, f"no se encontró en el CSV de Sprint 4: {materia}/{estudiante_id}/h{hito}"
        fila_ref = fila_ref.iloc[0]

        caso = CasoPrediccion(materia, trimestre, str(seccion), estudiante_id, hito)
        evidencia, _fila_reg, _fila_semana = bayes_service.construir_evidencia_bayes(caso)

        # Tamaño del grupo y participaciones deben coincidir siempre.
        assert evidencia["Tamaño del grupo"] == fila_ref["evidencia_tamano_grupo"]
        assert evidencia["Participaciones de la semana"] == fila_ref["evidencia_participaciones_semana"]
        assert evidencia["Participaciones de la semana anterior"] == fila_ref["evidencia_participaciones_semana_anterior"]

        # Año que cursa: mismo criterio de omisión que Sprint 4.
        completa_ref = bool(fila_ref["evidencia_completa"])
        completa_nueva = "Año que cursa" in evidencia
        assert completa_nueva == completa_ref, (
            f"{caso}: evidencia_completa difiere (Sprint 4={completa_ref}, servicio={completa_nueva})"
        )
        if completa_ref:
            assert evidencia["Año que cursa"] == fila_ref["evidencia_anio_que_cursa"]

        print(f"Nivel A — {materia}/{estudiante_id}/h{hito}: evidencia idéntica a "
              f"predicciones_bayesiana_s4_s6_s8.csv (evidencia_completa={completa_ref})")


def probar_inferencia_funcional() -> None:
    casos = [
        CasoPrediccion("Algoritmos y Programación", "2425-2", "1", "anon_001", 4),
        CasoPrediccion("Algoritmos y Programación", "2425-2", "1", "anon_001", 6),
        CasoPrediccion("Algoritmos y Programación", "2425-2", "1", "anon_001", 8),
        CasoPrediccion("Computación Emergente", "2425-3", "1", "anon_148", 6),
        CasoPrediccion("Matemáticas Discretas", "2526-2", "1", "anon_md_2526_2_1_001", 8),
        # año académico faltante -> evidencia parcial
        CasoPrediccion("Algoritmos y Programación", "2526-1", "1", "anon_063", 4),
    ]

    for caso in casos:
        r = bayes_service.predict_bayes(caso)

        assert set(r.posterior.keys()) == set(bayes_service.ESTADOS)
        suma = sum(r.posterior.values())
        assert abs(suma - 1.0) < 1e-6, f"{caso}: posterior suma {suma}"
        assert np.isfinite(r.prediccion_continua), f"{caso}: predicción no finita"

        if caso.estudiante_id == "anon_063":
            assert r.evidencia_omitida == ("Año que cursa",), (
                f"se esperaba evidencia parcial (año omitido), llegó {r.evidencia_omitida}"
            )
            assert "Año que cursa" not in r.evidencia_utilizada
        else:
            assert r.evidencia_omitida == ()

        print(f"Nivel B — {caso.materia}/{caso.estudiante_id}/h{caso.hito}: "
              f"posterior_suma={suma:.6f}  valor_esperado={r.prediccion_continua:.3f}  "
              f"evidencia_omitida={r.evidencia_omitida}")


if __name__ == "__main__":
    probar_paridad_evidencia()
    print()
    probar_inferencia_funcional()
    print("\nTodas las pruebas de paridad Bayes superadas.")
