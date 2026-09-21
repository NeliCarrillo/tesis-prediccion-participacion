"""Orquesta la validación cruzada leave-one-trimester-out de la red
bayesiana — Sprint 4, carta 5.

Por cada asignatura, itera sobre TODOS sus trimestres, dejando uno fuera
como prueba y usando el resto como entrenamiento — la misma partición
categórica de trimestres que usa la LSTM
(`LSTM/nuevo/adaptacion_lstm_participaciones.ipynb`, función
`validacion_cruzada`, celdas 32/34), aunque a una granularidad distinta
(estudiante-sesión aquí; estudiante-sección en la LSTM).

Folds reales por asignatura, verificados contra los datos (NO son 4 para
las cuatro, a diferencia de lo que decía la descripción previa de esta
carta en Trello): Algoritmos y Programación 4, Computación Emergente 4,
Estructura de Datos 4, Matemáticas Discretas 2 — Matemáticas Discretas solo
tiene los trimestres 2526-2 y 2526-3. Total: 14 pliegues.

Por cada pliegue: separa train/test por trimestre antes de cualquier
transformación supervisada (`ajuste_cpd.preparar_fold`), ajusta las CPD con
BDeu — ESS=5 como configuración de referencia, más ESS=1 y 10 únicamente
como análisis de sensibilidad de las CPD (normalización y positividad) —
y reporta tamaños, exclusiones por faltantes y «tema_no_visto» en test.
Ninguna de las tres corridas de ESS usa el conjunto de prueba para nada:
ESS=5 no se declara "óptimo" ni se elige con resultados de test.

Fuera de alcance, deliberadamente no incluido aquí: inferencia
(VariableElimination), valor esperado, agregación de estudiante-sesión a
estudiante-sección, hitos S4/S6/S8, RMSE/R² y comparación con LSTM o con la
extrapolación — eso es la carta 6 en adelante. Este módulo no decide cómo
convertir las filas estudiante-sesión en una predicción estudiante-sección.

Importar este módulo no ejecuta nada; `ejecutar_validacion_cruzada()` corre
los 14 pliegues.
"""
from __future__ import annotations

import pandas as pd

from ensamblado import ensamblar_conjunto
from ajuste_cpd import preparar_fold, ajustar_cpd, verificar_cpd, ESS_EVALUADOS, ESS_SELECCIONADO
from red_bayesiana import construir_modelo_manual

FOLDS_ESPERADOS_POR_MATERIA = {
    "Algoritmos y Programación": 4,
    "Computación Emergente": 4,
    "Estructura de Datos": 4,
    "Matemáticas Discretas": 2,
}


def ejecutar_validacion_cruzada(sesiones: pd.DataFrame | None = None) -> list[dict]:
    """Ejecuta los 14 pliegues leave-one-trimestre-out (4+4+4+2). Para cada
    uno, separa train/test, ajusta las CPD con ESS=1/5/10 (BDeu) sobre
    train únicamente y verifica normalización/positividad. No hace
    inferencia: solo prepara y verifica cada pliegue.

    Devuelve la lista de reportes, uno por pliegue.
    """
    if sesiones is None:
        sesiones, _, _ = ensamblar_conjunto()

    resultados = []
    for materia in sorted(sesiones["materia"].unique()):
        trimestres = sorted(sesiones.loc[sesiones["materia"] == materia, "trimestre"].unique())

        for trimestre_prueba in trimestres:
            train_bn, test_bn, reporte = preparar_fold(sesiones, materia, trimestre_prueba)

            # Verificación automática de separación, usando los metadatos
            # que ya devolvió preparar_fold (que a su vez los verificó
            # sobre los datos crudos, antes de ajustar el mapa de tema).
            assert trimestre_prueba not in reporte["trimestres_train"], (
                f"{materia}/{trimestre_prueba}: el trimestre de prueba aparece en "
                "los trimestres de entrenamiento reportados"
            )

            reporte["verificacion_cpd_por_ess"] = {}
            for ess in ESS_EVALUADOS:
                modelo = construir_modelo_manual()
                modelo = ajustar_cpd(modelo, train_bn, ess)
                verificacion = verificar_cpd(modelo)
                reporte["verificacion_cpd_por_ess"][ess] = {
                    "min_probabilidad_global": verificacion["min_probabilidad_global"],
                    "max_error_normalizacion": verificacion["max_error_normalizacion"],
                }

            resultados.append(reporte)

    return resultados


def _imprimir_reporte(resultados: list[dict]) -> None:
    print(f"Total de pliegues: {len(resultados)} (esperados: 4+4+4+2=14)")
    print()

    for r in resultados:
        print("=" * 70)
        print(f"{r['materia']} | trimestre de prueba: {r['trimestre_prueba']}")
        print("=" * 70)
        print(f"  trimestres de train: {r['trimestres_train']}")
        print(
            f"  registros estudiante-sección: "
            f"train={r['registros_train']} test={r['registros_test']}"
        )
        print(
            f"  filas estudiante-sesión: "
            f"train bruto={r['sesiones_train_bruto']} útil={r['sesiones_train_util']} | "
            f"test bruto={r['sesiones_test_bruto']} útil={r['sesiones_test_util']}"
        )

        et, es = r["exclusiones_train"], r["exclusiones_test"]
        print(
            f"  exclusiones train: solo año={et['excluidas_solo_anio_faltante']} "
            f"solo rezago={et['excluidas_solo_rezago_faltante']} "
            f"ambos={et['excluidas_ambos_faltantes']}"
        )
        print(
            f"  exclusiones test:  solo año={es['excluidas_solo_anio_faltante']} "
            f"solo rezago={es['excluidas_solo_rezago_faltante']} "
            f"ambos={es['excluidas_ambos_faltantes']}"
        )
        print(f"  tema_no_visto en test: {r['tema_no_visto_en_test']}")

        for ess, v in r["verificacion_cpd_por_ess"].items():
            etiqueta = " (referencia, no elegido con test)" if ess == ESS_SELECCIONADO else ""
            print(
                f"  ESS={ess:>2}: min. probabilidad={v['min_probabilidad_global']:.3e} "
                f"| error máx. normalización={v['max_error_normalizacion']:.2e}{etiqueta}"
            )
        print()


if __name__ == "__main__":
    sesiones, _, _ = ensamblar_conjunto()

    resultados = ejecutar_validacion_cruzada(sesiones)

    # --- Verificaciones automáticas, sobre los 14 pliegues reales ---
    assert len(resultados) == 14, f"se esperaban 14 pliegues (4+4+4+2), hay {len(resultados)}"

    conteo_por_materia: dict[str, int] = {}
    for r in resultados:
        conteo_por_materia[r["materia"]] = conteo_por_materia.get(r["materia"], 0) + 1
    assert conteo_por_materia == FOLDS_ESPERADOS_POR_MATERIA, (
        f"conteo de pliegues por asignatura inesperado: {conteo_por_materia}"
    )

    for r in resultados:
        assert r["trimestre_prueba"] not in r["trimestres_train"], (
            f"{r['materia']}/{r['trimestre_prueba']}: fuga de partición"
        )
        for ess, v in r["verificacion_cpd_por_ess"].items():
            assert v["min_probabilidad_global"] > 0.0, (
                f"{r['materia']}/{r['trimestre_prueba']}/ESS={ess}: probabilidad no positiva"
            )
            assert v["max_error_normalizacion"] < 1e-9, (
                f"{r['materia']}/{r['trimestre_prueba']}/ESS={ess}: CPD mal normalizada"
            )

    _imprimir_reporte(resultados)

    print(
        "verificaciones superadas: 14 pliegues (4+4+4+2), ningún trimestre de "
        "prueba aparece en su propio train, mapa de tema ajustado solo con "
        "train en cada pliegue, CPD normalizadas y positivas para ESS=1, 5 y "
        "10 en los 14 pliegues. ESS=5 se mantiene como referencia, sin "
        "elegirse ni declararse óptimo a partir de ningún resultado de test."
    )
