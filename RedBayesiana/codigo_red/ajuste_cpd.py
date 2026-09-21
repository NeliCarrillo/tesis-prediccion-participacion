"""Estimación de las CPD de la estructura final (Figura 12) — Sprint 4,
carta 4.

Ajusta las tablas de probabilidad condicional de la estructura fija
(`red_bayesiana.construir_modelo_manual()`) sobre el conjunto
estudiante-sesión que ensambla `ensamblado.ensamblar_conjunto()`, con un
modelo independiente por asignatura. Usa el estimador bayesiano de pgmpy con
prior BDeu.

Nota sobre la clase de pgmpy: `pgmpy.estimators.BayesianEstimator` está
deprecada en esta versión instalada (1.1.2) — emite FutureWarning y su
propia documentación indica que será removida en 1.3.0, recomendando
`pgmpy.parameter_estimator.DiscreteBayesianEstimator` en su lugar (mismo
estimador bayesiano con prior BDeu, API vigente). Este módulo usa la
segunda, igual que ya hacía `estructura_aprendida.ajustar_cpt_bdeu`, para no
introducir una dependencia que pgmpy va a eliminar.

Alcance de esta carta: solo estimación de CPD y su verificación (CPD
normalizadas, probabilidades positivas para todo estado/configuración
válidos declarados, sensibilidad a ESS=1/5/10). NO incluye inferencia,
conversión del objetivo a valor continuo, validación leave-one-trimester-out
ni métricas — eso son las cartas 5 a 7.

Tema de la sesión y fuga de información: esta carta no hace validación
cruzada (esa es la carta 5), así que no existe todavía un conjunto de
prueba del que evitar fuga. `preparar_datos_asignatura` ajusta el mapa
baja/media/alta con el 100% de las sesiones de la asignatura y lo aplica a
esas mismas sesiones — es el mismo criterio que ya usa
`estructura_aprendida.py` para su mapa descriptivo global, documentado allí
como no reutilizado por la validación por pliegue. `ajustar_cpd`, la función
que realmente estima las CPD, no asume cómo se calculó "Tema de la sesión":
recibe los datos ya discretizados, así que la carta 5 podrá reutilizarla sin
cambios pasándole datos donde el mapa se ajustó solo con los trimestres de
entrenamiento de cada pliegue.

Año académico faltante: las filas cuyo año que cursa es faltante (3
registros estudiante-sección, propagados a todas sus sesiones) se excluyen
del ajuste porque el nodo "Año que cursa" participa en la red; nunca se
convierten en "5 o más" ni se imputan. Lo mismo para la primera semana de
cada registro (sin "Participaciones de la semana anterior"). Ninguna otra
de las 10 columnas debería tener faltantes — se verifica explícitamente en
vez de asumirlo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

from ensamblado import ensamblar_conjunto, COLUMNAS_BN, ESTADOS_BN, CLAVE
from discretizacion import ajustar_mapa_temas, aplicar_mapa_temas
from red_bayesiana import construir_modelo_manual

ESS_EVALUADOS: tuple[int, ...] = (1, 5, 10)
ESS_SELECCIONADO: int = 5

# Las únicas dos columnas de la Tabla 13 que pueden llegar faltantes.
COLUMNAS_CON_FALTANTES_ESPERADOS = (
    "Año que cursa",
    "Participaciones de la semana anterior",
)


def _excluir_incompletos(datos: pd.DataFrame, materia: str) -> tuple[pd.DataFrame, dict]:
    """A partir de sesiones de una asignatura con «Tema de la sesión» ya
    asignado, separa las filas completas de las excluidas por algún
    faltante entre las 10 columnas de la red — solo año académico y la
    primera semana de cada registro pueden estar faltantes; cualquier otro
    faltante se trata como un error, no como un caso a excluir en
    silencio. No imputa nada.

    Reusada por `preparar_datos_asignatura` (carta 4, sin fold) y
    `preparar_fold` (carta 5, con fold) para no duplicar este criterio.

    Devuelve `(datos_bn, reporte)`: `datos_bn` son las filas completas, con
    las 10 columnas ya en texto (listas para `ajustar_cpd`); `reporte` es un
    diccionario con el desglose de exclusiones.
    """
    n_inicial = len(datos)
    datos_bn = datos[COLUMNAS_BN]

    columnas_inesperadas_con_faltantes = [
        columna
        for columna in COLUMNAS_BN
        if columna not in COLUMNAS_CON_FALTANTES_ESPERADOS and datos_bn[columna].isna().any()
    ]
    if columnas_inesperadas_con_faltantes:
        raise ValueError(
            f"Faltantes inesperados en {materia}, fuera de "
            f"{COLUMNAS_CON_FALTANTES_ESPERADOS}: {columnas_inesperadas_con_faltantes}"
        )

    falta_anio = datos_bn["Año que cursa"].isna()
    falta_rezago = datos_bn["Participaciones de la semana anterior"].isna()

    reporte = {
        "materia": materia,
        "n_inicial": n_inicial,
        "excluidas_solo_anio_faltante": int((falta_anio & ~falta_rezago).sum()),
        "excluidas_solo_rezago_faltante": int((falta_rezago & ~falta_anio).sum()),
        "excluidas_ambos_faltantes": int((falta_anio & falta_rezago).sum()),
    }
    reporte["n_excluido_total"] = (
        reporte["excluidas_solo_anio_faltante"]
        + reporte["excluidas_solo_rezago_faltante"]
        + reporte["excluidas_ambos_faltantes"]
    )

    completos = datos_bn.dropna().astype(str)
    reporte["n_utilizado"] = len(completos)

    assert reporte["n_utilizado"] == n_inicial - reporte["n_excluido_total"], (
        "el desglose de exclusiones no cuadra con las filas efectivamente usadas"
    )

    return completos, reporte


def preparar_datos_asignatura(sesiones: pd.DataFrame, materia: str) -> tuple[pd.DataFrame, dict]:
    """Prepara los datos de una asignatura para el ajuste de CPD.

    Ajusta "Tema de la sesión" con el 100% de las sesiones de `materia` (sin
    conjunto de prueba en esta carta, ver docstring del módulo) y excluye
    las filas incompletas con `_excluir_incompletos`.

    Devuelve `(datos_bn, reporte)` — ver `_excluir_incompletos`.
    """
    datos = sesiones[sesiones["materia"] == materia].copy()

    mapa, _ = ajustar_mapa_temas(datos)
    datos["Tema de la sesión"] = aplicar_mapa_temas(datos, mapa, permitir_no_visto=False)

    return _excluir_incompletos(datos, materia)


def preparar_fold(
    sesiones: pd.DataFrame, materia: str, trimestre_prueba: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prepara un pliegue leave-one-trimestre-out de una asignatura —
    Sprint 4, carta 5.

    Separa train/test por `trimestre` ANTES de cualquier transformación
    supervisada; ajusta el mapa de «Tema de la sesión» únicamente con las
    sesiones de train y lo aplica después a ambos conjuntos (en test, un
    código de tema ausente de train se etiqueta «tema_no_visto» sin
    consultar su participación); excluye de cada lado, por separado, las
    filas incompletas (año académico o rezago faltante), con el mismo
    criterio que `preparar_datos_asignatura` — no imputa nada ni cambia los
    estados o cortes fijos.

    Devuelve `(train_bn, test_bn, reporte)`. `reporte` incluye tamaños en
    registros estudiante-sección y en filas estudiante-sesión (brutas y
    útiles) para train y test, el desglose de exclusiones de cada lado, el
    número de «tema_no_visto» en test, y los trimestres de entrenamiento
    usados — para que quien llame pueda verificar por sí mismo que
    `trimestre_prueba` no está entre ellos.
    """
    datos_materia = sesiones[sesiones["materia"] == materia].copy()

    es_prueba = datos_materia["trimestre"] == trimestre_prueba
    if not es_prueba.any():
        raise ValueError(f"{materia!r} no tiene el trimestre {trimestre_prueba!r}")

    train_crudo = datos_materia[~es_prueba].copy()
    test_crudo = datos_materia[es_prueba].copy()

    # Verificación explícita de separación, sobre los datos crudos, antes
    # de ajustar ningún mapa: ningún trimestre de prueba puede aparecer en
    # entrenamiento, y el conjunto de prueba solo puede contener el
    # trimestre excluido.
    assert not (train_crudo["trimestre"] == trimestre_prueba).any(), (
        f"fuga de partición: {trimestre_prueba!r} aparece en el train de {materia!r}"
    )
    assert (test_crudo["trimestre"] == trimestre_prueba).all(), (
        f"el test de {materia!r} contiene trimestres distintos de {trimestre_prueba!r}"
    )

    # El mapa de tema se ajusta únicamente con las sesiones de train.
    mapa, _ = ajustar_mapa_temas(train_crudo)
    train_crudo["Tema de la sesión"] = aplicar_mapa_temas(train_crudo, mapa, permitir_no_visto=False)
    test_crudo["Tema de la sesión"] = aplicar_mapa_temas(test_crudo, mapa, permitir_no_visto=True)

    train_bn, reporte_train = _excluir_incompletos(train_crudo, materia)
    test_bn, reporte_test = _excluir_incompletos(test_crudo, materia)

    reporte = {
        "materia": materia,
        "trimestre_prueba": trimestre_prueba,
        "trimestres_train": sorted(train_crudo["trimestre"].unique()),
        "registros_train": int(train_crudo[CLAVE].drop_duplicates().shape[0]),
        "registros_test": int(test_crudo[CLAVE].drop_duplicates().shape[0]),
        "sesiones_train_bruto": len(train_crudo),
        "sesiones_train_util": reporte_train["n_utilizado"],
        "sesiones_test_bruto": len(test_crudo),
        "sesiones_test_util": reporte_test["n_utilizado"],
        "exclusiones_train": reporte_train,
        "exclusiones_test": reporte_test,
        "tema_no_visto_en_test": int((test_crudo["Tema de la sesión"] == "tema_no_visto").sum()),
    }

    return train_bn, test_bn, reporte


def ajustar_cpd(modelo: DiscreteBayesianNetwork, datos_bn: pd.DataFrame, ess: int) -> DiscreteBayesianNetwork:
    """Ajusta las CPD de `modelo` (estructura ya construida, sin CPD) sobre
    `datos_bn` (10 columnas, texto, sin faltantes) con el estimador
    bayesiano de pgmpy, prior BDeu y el `ess` (equivalent sample size)
    dado. Devuelve el mismo objeto `modelo`, ahora con CPD ajustadas — no
    modifica la estructura (nodos/arcos)."""
    estimador = DiscreteBayesianEstimator(
        state_names=ESTADOS_BN,
        prior_type="BDeu",
        equivalent_sample_size=ess,
    )
    modelo.fit(datos_bn, estimator=estimador)
    return modelo


def verificar_cpd(modelo: DiscreteBayesianNetwork) -> dict:
    """Verifica, para cada CPD del modelo ajustado, que está normalizada
    (cada configuración de padres suma 1 sobre los estados del nodo) y que
    todas sus probabilidades son positivas — incluidos los estados
    declarados en el dominio que no se observaron en los datos (como
    «tema_no_visto» en esta carta), que BDeu debe seguir asignando con
    masa positiva por el suavizado del prior.

    Devuelve un resumen con el mínimo global de probabilidad, el máximo
    error de normalización, y el detalle por nodo (forma de la CPD y
    parámetros libres = (estados del nodo − 1) × combinaciones de padres)."""
    detalle = []
    min_global = float("inf")
    max_error_normalizacion = 0.0

    for cpd in modelo.get_cpds():
        valores = np.asarray(cpd.values, dtype=float)

        if not np.isfinite(valores).all():
            raise ValueError(f"La CPD de {cpd.variable} contiene valores no finitos")

        minimo = float(valores.min())
        if minimo <= 0.0:
            raise ValueError(
                f"La CPD de {cpd.variable} tiene una probabilidad no positiva ({minimo})"
            )

        sumas_por_configuracion = valores.sum(axis=0)
        error_normalizacion = float(np.max(np.abs(sumas_por_configuracion - 1.0)))

        min_global = min(min_global, minimo)
        max_error_normalizacion = max(max_error_normalizacion, error_normalizacion)

        parametros_libres = int((valores.shape[0] - 1) * int(np.prod(valores.shape[1:])))

        detalle.append({
            "nodo": cpd.variable,
            "forma": valores.shape,
            "parametros_libres": parametros_libres,
            "min_probabilidad": minimo,
            "error_normalizacion": error_normalizacion,
        })

    return {
        "min_probabilidad_global": min_global,
        "max_error_normalizacion": max_error_normalizacion,
        "detalle_por_nodo": detalle,
    }


if __name__ == "__main__":
    sesiones, reg, sem = ensamblar_conjunto()

    for materia in sorted(sesiones["materia"].unique()):
        print("=" * 70)
        print(materia)
        print("=" * 70)

        datos_bn, reporte = preparar_datos_asignatura(sesiones, materia)
        print(
            f"filas iniciales: {reporte['n_inicial']} | "
            f"utilizadas: {reporte['n_utilizado']} | "
            f"excluidas: {reporte['n_excluido_total']} "
            f"(solo año faltante={reporte['excluidas_solo_anio_faltante']}, "
            f"solo rezago faltante={reporte['excluidas_solo_rezago_faltante']}, "
            f"ambos={reporte['excluidas_ambos_faltantes']})"
        )

        for ess in ESS_EVALUADOS:
            modelo = construir_modelo_manual()
            modelo = ajustar_cpd(modelo, datos_bn, ess)
            verificacion = verificar_cpd(modelo)
            marca = " <- seleccionado" if ess == ESS_SELECCIONADO else ""
            print(
                f"  ESS={ess:>2}: min. probabilidad={verificacion['min_probabilidad_global']:.3e} "
                f"| error máx. normalización={verificacion['max_error_normalizacion']:.2e}{marca}"
            )

            if ess == ESS_SELECCIONADO:
                for nodo in verificacion["detalle_por_nodo"]:
                    if nodo["nodo"] in ("Participaciones de la semana", "Cantidad de participaciones del trimestre"):
                        print(
                            f"      {nodo['nodo']}: forma={nodo['forma']}, "
                            f"parámetros libres={nodo['parametros_libres']}"
                        )
        print()

    print("verificaciones superadas: CPD normalizadas y con probabilidad positiva "
          "en todos los estados/configuraciones declarados, para ESS=1, 5 y 10, "
          "en las cuatro asignaturas")
