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

Año académico y rezago faltantes — tratamiento por CPD, no por fila
completa: `pgmpy.parameter_estimator.DiscreteBayesianEstimator` calcula
cada CPD de forma independiente, usando únicamente las filas donde ese nodo
y sus padres directos tienen valor (verificado empíricamente antes de este
cambio, incluso con la estructura y los datos reales del proyecto: un nodo
sin relación con «Año que cursa» usa el 100% de las filas aunque esa
columna tenga NaN). Por eso este módulo ya NO descarta la fila completa
solo porque «Año que cursa» o «Participaciones de la semana anterior» sea
NaN — eso era más agresivo de lo necesario y le quitaba información a los
8 nodos que no dependen de ninguna de esas dos variables. El único nodo que
genuinamente necesita ambas presentes es el objetivo («Cantidad de
participaciones del trimestre», que tiene a los dos como padres directos):
para ese nodo, pgmpy ya excluye por sí solo las filas con cualquiera de los
dos faltantes, sin que este módulo tenga que hacerlo de antemano.

Ninguna de las dos se convierte nunca en "5 o más", en "0" ni en ningún
otro estado — el NaN real se preserva hasta que llega a pgmpy. Ninguna otra
de las 10 columnas debería tener faltantes — se verifica explícitamente en
vez de asumirlo. El reporte de exclusiones (`excluidas_solo_anio_faltante`,
etc.) se conserva por trazabilidad y para comparar contra el comportamiento
anterior, aunque ya no describe filas realmente descartadas del ajuste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

from ensamblado import ensamblar_conjunto, COLUMNAS_BN, ESTADOS_BN, CLAVE
from discretizacion import ajustar_mapa_temas, aplicar_mapa_temas
from red_bayesiana import construir_modelo_manual, ARCOS_MANUALES

ESS_EVALUADOS: tuple[int, ...] = (1, 5, 10)
ESS_SELECCIONADO: int = 5

# Las únicas dos columnas de la Tabla 13 que pueden llegar faltantes.
COLUMNAS_CON_FALTANTES_ESPERADOS = (
    "Año que cursa",
    "Participaciones de la semana anterior",
)


def _padres_por_nodo() -> dict[str, list[str]]:
    """Padres directos de cada nodo, según `red_bayesiana.ARCOS_MANUALES`."""
    padres: dict[str, list[str]] = {nodo: [] for nodo in COLUMNAS_BN}
    for origen, destino in ARCOS_MANUALES:
        padres[destino].append(origen)
    return padres


def filas_utilizables_por_nodo(datos_bn: pd.DataFrame) -> dict[str, int]:
    """Cuenta, para cada uno de los 10 nodos, cuántas filas de `datos_bn`
    tienen valor no nulo tanto en el nodo como en todos sus padres directos
    — es decir, cuántas filas puede usar realmente la CPD de ese nodo al
    ajustarse con `DiscreteBayesianEstimator`. Antes de esta carta, las 10
    CPD usaban el mismo número de filas (el de `dropna()` sobre las 10
    columnas); ahora cada una usa el máximo disponible según su propio
    subgrafo — solo «Cantidad de participaciones del trimestre» (el
    objetivo, con «Año que cursa» y «Participaciones de la semana
    anterior» como padres) sigue necesitando ambas presentes."""
    padres = _padres_por_nodo()
    return {
        nodo: int(datos_bn[[nodo, *padres[nodo]]].dropna().shape[0]) if padres[nodo] else
        int(datos_bn[[nodo]].dropna().shape[0])
        for nodo in COLUMNAS_BN
    }


def _a_texto_preservando_faltantes(datos_bn: pd.DataFrame) -> pd.DataFrame:
    """Convierte las columnas a texto para pgmpy, preservando los NaN
    reales — nunca los convierte en la cadena literal 'nan' (que
    `DiscreteBayesianEstimator` trataría como un estado válido más)."""
    resultado = datos_bn.copy()
    for columna in resultado.columns:
        valores = resultado[columna]
        no_nulos = valores.notna()
        resultado[columna] = valores.astype(object)
        resultado.loc[no_nulos, columna] = valores.loc[no_nulos].astype(str)
    return resultado


def _reportar_faltantes(datos: pd.DataFrame, materia: str) -> tuple[pd.DataFrame, dict]:
    """A partir de sesiones de una asignatura con «Tema de la sesión» ya
    asignado, calcula el reporte de faltantes de las 10 columnas de la red
    — solo año académico y la primera semana de cada registro pueden estar
    faltantes; cualquier otro faltante se trata como un error, no como un
    caso a excluir en silencio. No imputa nada.

    A diferencia de la versión anterior de esta función, YA NO excluye
    ninguna fila: `DiscreteBayesianEstimator` calcula cada CPD con las
    filas que esa CPD específica puede usar (ver docstring del módulo), así
    que aquí se devuelven las filas completas de `datos` (con los NaN reales
    preservados, no descartados), listas para `ajustar_cpd`.

    Reusada por `preparar_datos_asignatura` (carta 4, sin fold) y
    `preparar_fold` (carta 5, con fold) para no duplicar este criterio.

    Devuelve `(datos_bn, reporte)`: `datos_bn` tiene las 10 columnas en
    texto, con los faltantes reales preservados; `reporte` trae el
    desglose de faltantes (por trazabilidad y comparación con el
    comportamiento anterior) y `filas_utilizables_por_nodo`.
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
    # Se conserva por comparabilidad con el comportamiento anterior: cuántas
    # filas serían "completas en las 10 columnas" — ya no es lo que se pasa
    # a ajustar_cpd, que ahora recibe las n_inicial filas con NaN intactos.
    reporte["n_utilizado"] = n_inicial - reporte["n_excluido_total"]

    datos_bn_final = _a_texto_preservando_faltantes(datos_bn)
    reporte["filas_utilizables_por_nodo"] = filas_utilizables_por_nodo(datos_bn_final)

    return datos_bn_final, reporte


def preparar_datos_asignatura(sesiones: pd.DataFrame, materia: str) -> tuple[pd.DataFrame, dict]:
    """Prepara los datos de una asignatura para el ajuste de CPD.

    Ajusta "Tema de la sesión" con el 100% de las sesiones de `materia` (sin
    conjunto de prueba en esta carta, ver docstring del módulo) y reporta
    los faltantes con `_reportar_faltantes` (ya no excluye filas por
    faltantes — ver docstring del módulo y de esa función).

    Devuelve `(datos_bn, reporte)` — ver `_reportar_faltantes`.
    """
    datos = sesiones[sesiones["materia"] == materia].copy()

    mapa, _ = ajustar_mapa_temas(datos)
    datos["Tema de la sesión"] = aplicar_mapa_temas(datos, mapa, permitir_no_visto=False)

    return _reportar_faltantes(datos, materia)


def preparar_fold(
    sesiones: pd.DataFrame, materia: str, trimestre_prueba: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prepara un pliegue leave-one-trimestre-out de una asignatura —
    Sprint 4, carta 5.

    Separa train/test por `trimestre` ANTES de cualquier transformación
    supervisada; ajusta el mapa de «Tema de la sesión» únicamente con las
    sesiones de train y lo aplica después a ambos conjuntos (en test, un
    código de tema ausente de train se etiqueta «tema_no_visto» sin
    consultar su participación); reporta, por separado para cada lado, los
    faltantes de año académico y rezago (`_reportar_faltantes`) — ya no
    excluye filas por esos faltantes, ver docstring del módulo — y no
    imputa nada ni cambia los estados o cortes fijos.

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

    train_bn, reporte_train = _reportar_faltantes(train_crudo, materia)
    test_bn, reporte_test = _reportar_faltantes(test_crudo, materia)

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
            f"«completas en las 10 columnas» (criterio anterior, solo informativo): "
            f"{reporte['n_utilizado']} "
            f"(solo año faltante={reporte['excluidas_solo_anio_faltante']}, "
            f"solo rezago faltante={reporte['excluidas_solo_rezago_faltante']}, "
            f"ambos={reporte['excluidas_ambos_faltantes']})"
        )
        print("filas realmente utilizables por nodo (tras eliminar el dropna() global):")
        for nodo, n in reporte["filas_utilizables_por_nodo"].items():
            recuperadas = n - reporte["n_utilizado"]
            marca = f" (+{recuperadas} respecto del criterio anterior)" if recuperadas else ""
            print(f"    {nodo}: {n}{marca}")

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
