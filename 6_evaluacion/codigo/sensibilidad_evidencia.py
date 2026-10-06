"""Sensibilidad de la predicción de la red bayesiana ante cambios en la
evidencia (Tabla 20 del informe), por asignatura.

Análisis complementario: NO modifica ningún resultado del informe. Para cada
una de las 1.572 predicciones de validación cruzada, con el modelo del pliegue
(ESS = 5) y la evidencia C1, cada variable de evidencia observada se sustituye
por cada uno de sus estados y, además, se omite (la red la marginaliza),
manteniendo fijas las demás. La sensibilidad del caso a esa variable es el
rango entre la mayor y la menor predicción obtenidas. Se exige reproducir los
valores globales publicados en la Tabla 20.

Además de esa sensibilidad local, el archivo calcula el aporte predictivo
observado de cada variable: el cambio de RMSE al omitirla de la evidencia
(ΔRMSE de omisión = RMSE sin la variable - RMSE con la evidencia completa), con
el mismo modelo del pliegue, que la marginaliza. Un ΔRMSE positivo indica que
conservar la variable reduce el error en estas predicciones; uno negativo, que
no lo reduce. Es una medida de este modelo y estos datos, no de importancia
global ni de causalidad. Como resultado secundario, resume la dirección del
cambio de la predicción al pasar de un estado al siguiente dentro de un mismo
caso, usando solo sustituciones cuya combinación de evidencia tiene respaldo.

Correr el archivo genera, en `resultados/`:
- `sensibilidad_evidencia_casos.csv` y `sensibilidad_evidencia_por_asignatura.csv`
  (rango de sensibilidad, sin cambios);
- `sensibilidad_evidencia_escenarios.csv`: la predicción de cada caso y variable
  con el valor observado, con cada estado posible y con la variable omitida;
- `sensibilidad_evidencia_aporte_omision.csv`: ΔRMSE de omisión, global y por
  asignatura;
- `sensibilidad_evidencia_direccion_estados.csv`: dirección del cambio entre
  estados consecutivos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ajuste_cpd import ESS_SELECCIONADO, ajustar_cpd, preparar_fold
from ensamblado import ESTADOS_BN, ensamblar_conjunto
from red_bayesiana import construir_modelo_manual
from valor_esperado import medias_entrenamiento_por_estado, valor_esperado

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
CARPETA = RAIZ / "6_evaluacion" / "resultados"
OBJETIVO = "Cantidad de participaciones del trimestre"
EVIDENCIA = {
    "Participaciones de la semana": "evidencia_participaciones_semana",
    "Participaciones de la semana anterior": "evidencia_participaciones_semana_anterior",
    "Año que cursa": "evidencia_anio_que_cursa",
    "Tamaño del grupo": "evidencia_tamano_grupo",
}
# Valores publicados en la Tabla 20: (n, rango medio, mediana, rango máximo).
# El rango máximo del tamaño del grupo es 16,8547 y debe publicarse como 16,85
# (el informe muestra 16,86); por eso el máximo admite una diferencia de 0,01.
TABLA_20 = {
    "Participaciones de la semana": (1572, 7.55, 6.73, 21.33),
    "Participaciones de la semana anterior": (1572, 7.52, 6.73, 21.25),
    "Año que cursa": (1563, 4.63, 5.14, 18.79),
    "Tamaño del grupo": (1572, 4.51, 4.53, 16.86),
}


LLAVE = ["materia", "trimestre", "seccion", "estudiante_id", "hito"]
NODOS = list(EVIDENCIA)


def calcular() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve el rango de sensibilidad por caso (procedimiento oficial) y, en
    formato largo, cada escenario del que sale ese rango: valor observado,
    sustitución por cada estado y omisión de la variable."""
    sesiones, reg, _sem = ensamblar_conjunto()
    casos = pd.read_csv(RUTA_BAYES, dtype=str)
    assert len(casos) == 1572
    filas = []
    escenarios = []
    for (materia, trimestre), g in casos.groupby(["materia", "trimestre_prueba"]):
        train_bn, _test, _rep = preparar_fold(sesiones, materia, trimestre)
        inferencia = VariableElimination(ajustar_cpd(construir_modelo_manual(), train_bn, ESS_SELECCIONADO))
        medias = medias_entrenamiento_por_estado(reg, materia, trimestre)
        cache: dict[tuple, float] = {}
        # Respaldo: filas estudiante-sesión del entrenamiento del pliegue con la misma
        # combinación de las cuatro variables (mismo criterio que respaldo_evidencia.py).
        conteo = train_bn.dropna(subset=NODOS + [OBJETIVO]).groupby(NODOS).size()

        def respaldo(evidencia: dict) -> float:
            if len(evidencia) < len(NODOS):
                return np.nan
            return float(conteo.get(tuple(evidencia[n] for n in NODOS), 0))

        def predecir(evidencia: dict) -> float:
            clave = tuple(sorted(evidencia.items()))
            if clave not in cache:
                r = inferencia.query([OBJETIVO], evidence=evidencia, show_progress=False)
                cache[clave] = valor_esperado({e: float(p) for e, p in zip(r.state_names[OBJETIVO], r.values)}, medias)
            return cache[clave]

        for _, caso in g.iterrows():
            base = {n: caso[c] for n, c in EVIDENCIA.items() if isinstance(caso[c], str) and caso[c] != "nan"}
            fila = {"materia": materia, "trimestre": trimestre, "estudiante_id": caso.estudiante_id,
                    "seccion": caso.seccion, "hito": int(caso.hito)}
            comun = {"materia": materia, "trimestre": trimestre, "seccion": caso.seccion,
                     "estudiante_id": caso.estudiante_id, "hito": int(caso.hito),
                     "total_trimestre_real": float(caso.total_trimestre_real),
                     "prediccion_oficial": float(caso.prediccion_continua),
                     "evidencia_completa": len(base) == len(NODOS)}
            prediccion_observada = predecir(base)
            for nodo in EVIDENCIA:
                if nodo not in base:
                    fila[nodo] = np.nan
                    continue
                predicciones = [predecir({**base, nodo: estado}) for estado in ESTADOS_BN[nodo]]
                sin_nodo = {k: v for k, v in base.items() if k != nodo}
                predicciones.append(predecir(sin_nodo))
                fila[nodo] = max(predicciones) - min(predicciones)
                escenarios.append({**comun, "variable": nodo, "valor_observado": base[nodo],
                                   "escenario": "observado", "estado": base[nodo],
                                   "prediccion": prediccion_observada, "respaldo": respaldo(base)})
                for estado, prediccion in zip(ESTADOS_BN[nodo], predicciones):
                    evidencia = {**base, nodo: estado}
                    escenarios.append({**comun, "variable": nodo, "valor_observado": base[nodo],
                                       "escenario": "sustitucion", "estado": estado,
                                       "prediccion": prediccion, "respaldo": respaldo(evidencia)})
                escenarios.append({**comun, "variable": nodo, "valor_observado": base[nodo],
                                   "escenario": "omision", "estado": "",
                                   "prediccion": predicciones[-1], "respaldo": np.nan})
            filas.append(fila)
    return pd.DataFrame(filas), pd.DataFrame(escenarios)


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2).mean()))


def aporte_por_omision(escenarios: pd.DataFrame) -> pd.DataFrame:
    """ΔRMSE de omisión = RMSE con la variable omitida - RMSE con la evidencia completa,
    sobre los casos en que la variable fue observada (mismo modelo del pliegue)."""
    observado = escenarios[escenarios.escenario == "observado"].set_index(LLAVE + ["variable"])
    omitido = escenarios[escenarios.escenario == "omision"].set_index(LLAVE + ["variable"])
    datos = observado[["total_trimestre_real", "prediccion"]].join(
        omitido[["prediccion"]].rename(columns={"prediccion": "prediccion_sin"})).reset_index()
    assert len(datos) == len(observado) == len(omitido)
    filas = []
    for ambito, g in [("Global", datos)] + list(datos.groupby("materia")):
        for variable in NODOS:
            v = g[g.variable == variable]
            completa = _rmse(v.prediccion, v.total_trimestre_real)
            sin = _rmse(v.prediccion_sin, v.total_trimestre_real)
            filas.append({"variable": variable, "asignatura": ambito, "n": len(v),
                          "rmse_evidencia_completa": completa, "rmse_sin_variable": sin,
                          "delta_rmse_omision": sin - completa})
    return pd.DataFrame(filas)


def direccion_por_estados(escenarios: pd.DataFrame) -> pd.DataFrame:
    """Cambio de la predicción al pasar de un estado al siguiente (orden de ESTADOS_BN) dentro
    de un mismo caso con evidencia completa, solo cuando ambas combinaciones tienen respaldo.
    En el tamaño del grupo se añade la comparación pequeño a grande, porque cada asignatura
    observa solo dos de los tres estados (pequeño y mediano, o pequeño y grande).
    El denominador de los porcentajes es n_con_respaldo (casos en que ambos estados están
    respaldados); n_casos es el total de casos con evidencia completa en que se observó la variable."""
    sust = escenarios[(escenarios.escenario == "sustitucion") & escenarios.evidencia_completa]
    ancho_pred = sust.pivot_table(index=LLAVE + ["variable"], columns="estado", values="prediccion")
    ancho_resp = sust.pivot_table(index=LLAVE + ["variable"], columns="estado", values="respaldo")
    filas = []
    for variable in NODOS:
        estados = ESTADOS_BN[variable]
        pred = ancho_pred.xs(variable, level="variable")
        resp = ancho_resp.xs(variable, level="variable")
        transiciones = list(zip(estados[:-1], estados[1:]))
        if variable == "Tamaño del grupo":
            transiciones.append(("pequeño", "grande"))
        for desde, hacia in transiciones:
            cambio = pred[hacia] - pred[desde]
            con_respaldo = (resp[desde] > 0) & (resp[hacia] > 0)
            tabla = pd.DataFrame({"materia": cambio.index.get_level_values("materia"),
                                  "cambio": cambio.to_numpy(), "con_respaldo": con_respaldo.to_numpy()})
            for ambito, g in [("Global", tabla)] + list(tabla.groupby("materia")):
                c = g.cambio[g.con_respaldo]
                filas.append({"variable": variable, "transicion": f"{desde} a {hacia}", "asignatura": ambito,
                              "n_casos": len(g), "n_con_respaldo": len(c),
                              "mediana_cambio": c.median() if len(c) else np.nan,
                              "media_cambio": c.mean() if len(c) else np.nan,
                              "pct_aumenta": 100 * (c > 1e-9).mean() if len(c) else np.nan,
                              "pct_disminuye": 100 * (c < -1e-9).mean() if len(c) else np.nan})
    return pd.DataFrame(filas)


def resumir(casos: pd.DataFrame, por: list[str] | None = None) -> pd.DataFrame:
    grupos = [("Todas", casos)] if not por else list(casos.groupby(por))
    filas = []
    for clave, g in grupos:
        for nodo in EVIDENCIA:
            v = g[nodo].dropna()
            filas.append({"grupo": clave, "variable": nodo, "n": len(v), "rango_medio": v.mean(),
                          "mediana": v.median(), "q1": v.quantile(0.25), "q3": v.quantile(0.75), "maximo": v.max()})
    return pd.DataFrame(filas)


# Comprobación del cálculo preliminar del ΔRMSE de omisión (dos decimales).
APORTE_PRELIMINAR = {
    ("Global", "Participaciones de la semana"): 0.21, ("Global", "Participaciones de la semana anterior"): 0.26,
    ("Global", "Año que cursa"): -0.37, ("Global", "Tamaño del grupo"): -0.42,
    ("Algoritmos y Programación", "Participaciones de la semana"): 0.50,
    ("Computación Emergente", "Participaciones de la semana"): 0.07,
    ("Estructura de Datos", "Participaciones de la semana"): 0.11,
    ("Matemáticas Discretas", "Participaciones de la semana"): 0.19,
    ("Algoritmos y Programación", "Participaciones de la semana anterior"): 0.23,
    ("Computación Emergente", "Participaciones de la semana anterior"): 0.41,
    ("Estructura de Datos", "Participaciones de la semana anterior"): 0.19,
    ("Matemáticas Discretas", "Participaciones de la semana anterior"): 0.13,
    ("Algoritmos y Programación", "Año que cursa"): -0.27, ("Computación Emergente", "Año que cursa"): -0.07,
    ("Estructura de Datos", "Año que cursa"): -0.87, ("Matemáticas Discretas", "Año que cursa"): -0.37,
    ("Algoritmos y Programación", "Tamaño del grupo"): -0.37, ("Computación Emergente", "Tamaño del grupo"): -0.40,
    ("Estructura de Datos", "Tamaño del grupo"): -0.66, ("Matemáticas Discretas", "Tamaño del grupo"): -0.10,
}


def comprobar(casos: pd.DataFrame, escenarios: pd.DataFrame) -> None:
    """Comprobaciones de que los escenarios guardados son los del cálculo oficial."""
    obs = escenarios[escenarios.escenario == "observado"]
    # 1) La predicción con la evidencia observada reproduce la predicción oficial.
    assert np.allclose(obs.prediccion, obs.prediccion_oficial, rtol=0, atol=1e-9), "no reproduce la predicción oficial"
    # 2) La población: 1.572 casos; cada variable con su n publicado en la tabla de sensibilidad.
    assert obs.drop_duplicates(LLAVE).shape[0] == 1572
    for variable, (n, *_resto) in TABLA_20.items():
        assert (obs.variable == variable).sum() == n, variable
        # Por variable: el número de estados sustituidos y una omisión por caso.
        sust = escenarios[(escenarios.variable == variable) & (escenarios.escenario == "sustitucion")]
        assert len(sust) == n * len(ESTADOS_BN[variable])
        assert ((escenarios.variable == variable) & (escenarios.escenario == "omision")).sum() == n
    # 3) El rango reconstruido desde los escenarios coincide con el rango de sensibilidad.
    rango = (escenarios[escenarios.escenario != "observado"].groupby(LLAVE + ["variable"]).prediccion
             .agg(lambda x: x.max() - x.min()).unstack())
    oficial = casos.set_index(LLAVE)[NODOS]
    oficial.index = oficial.index.set_levels(oficial.index.levels[LLAVE.index("hito")].astype(int), level="hito")
    dif = (rango[NODOS] - oficial.reindex(rango.index)).abs()
    assert dif.max().max() < 1e-9, f"los rangos no coinciden: {dif.max().max()}"
    # 4) El respaldo de la evidencia observada coincide con el de respaldo_evidencia.py.
    ruta_respaldo = CARPETA / "respaldo_evidencia_casos.csv"
    if ruta_respaldo.exists():
        r = pd.read_csv(ruta_respaldo, dtype={"seccion": str})[LLAVE + ["respaldo"]]
        o = obs[obs.evidencia_completa].drop_duplicates(LLAVE)[LLAVE + ["respaldo"]].astype({"seccion": str})
        m = o.merge(r, on=LLAVE, suffixes=("", "_ref"))
        assert len(m) == 1563 and (m.respaldo == m.respaldo_ref).all(), "el respaldo no coincide con respaldo_evidencia.py"


if __name__ == "__main__":
    casos, escenarios = calcular()
    comprobar(casos, escenarios)
    print("Escenarios comprobados: reproducen las predicciones oficiales, los rangos y el respaldo.")
    glob = resumir(casos)
    for _, f in glob.iterrows():
        n, media, mediana, maximo = TABLA_20[f.variable]
        assert f.n == n and round(f.rango_medio, 2) == media and round(f.mediana, 2) == mediana \
            and abs(f.maximo - maximo) <= 0.01, (f.variable, f.n, f.rango_medio, f.mediana, f.maximo)
    print("Tabla 20 reproducida.\n", glob.round(2).to_string(index=False))
    por_materia = resumir(casos, ["materia"])
    print("\nPor asignatura:\n", por_materia.round(2).to_string(index=False))
    temporal = casos[["Participaciones de la semana", "Participaciones de la semana anterior"]].max(axis=1)
    contextual = casos[["Año que cursa", "Tamaño del grupo"]].max(axis=1)
    casos["contextual_supera_temporales"] = contextual > temporal
    print("\nCasos en que una variable contextual supera a ambas temporales:")
    print(casos.groupby("materia").contextual_supera_temporales.mean().round(3).to_string(),
          "| global", round(casos.contextual_supera_temporales.mean(), 3))
    casos.to_csv(CARPETA / "sensibilidad_evidencia_casos.csv", index=False)
    por_materia.to_csv(CARPETA / "sensibilidad_evidencia_por_asignatura.csv", index=False)

    aporte = aporte_por_omision(escenarios)
    for _, f in aporte.iterrows():
        esperado = APORTE_PRELIMINAR[(f.asignatura, f.variable)]
        assert round(f.delta_rmse_omision, 2) == esperado, (f.asignatura, f.variable, f.delta_rmse_omision, esperado)
    print("\nCambio de RMSE al omitir la variable (RMSE sin la variable - RMSE con la evidencia completa):")
    print(aporte.pivot(index="asignatura", columns="variable", values="delta_rmse_omision").round(2).to_string())
    print(aporte[aporte.asignatura == "Global"].round(3).to_string(index=False))
    direccion = direccion_por_estados(escenarios)
    print("\nDirección del cambio entre estados consecutivos (global, solo sustituciones con respaldo):")
    print(direccion[direccion.asignatura == "Global"].round(2).to_string(index=False))
    escenarios.to_csv(CARPETA / "sensibilidad_evidencia_escenarios.csv", index=False)
    aporte.to_csv(CARPETA / "sensibilidad_evidencia_aporte_omision.csv", index=False)
    direccion.to_csv(CARPETA / "sensibilidad_evidencia_direccion_estados.csv", index=False)


def figura(casos: pd.DataFrame) -> None:
    """Figura de sensibilidad por asignatura: un diagrama de caja por variable
    de evidencia (la caja cubre la mitad central de los casos y los bigotes, del
    percentil 5 al 95) y un rombo con el rango medio."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
    materias = ("Algoritmos y Programación", "Computación Emergente", "Estructura de Datos", "Matemáticas Discretas")
    variables = list(EVIDENCIA)
    etiquetas = ["Participaciones\nde la semana", "Participaciones de la\nsemana anterior", "Año que cursa", "Tamaño del grupo"]
    colores = {"Participaciones de la semana": "#1f4e79", "Participaciones de la semana anterior": "#1f4e79",
               "Año que cursa": "#b5651d", "Tamaño del grupo": "#b5651d"}
    figura_, ejes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True, sharey=True)
    for eje, materia in zip(ejes.flat, materias):
        g = casos[casos.materia == materia]
        datos = [g[v].dropna().to_numpy() for v in variables]
        posiciones = list(range(len(variables), 0, -1))
        cajas = eje.boxplot(datos, positions=posiciones, vert=False, widths=0.55, whis=(5, 95), showfliers=False,
                            patch_artist=True, medianprops=dict(color="white", linewidth=1.2))
        for caja, v in zip(cajas["boxes"], variables):
            caja.set(facecolor=colores[v], edgecolor=colores[v], alpha=0.85)
        for parte in ("whiskers", "caps"):
            for linea, v in zip(cajas[parte], [x for x in variables for _ in (0, 1)]):
                linea.set(color=colores[v], linewidth=0.8)
        for y, v in zip(posiciones, variables):
            media = g[v].mean()
            eje.plot(media, y, marker="D", color="black", markersize=4, zorder=5)
            eje.text(media, y + 0.36, f"{media:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=7)
        eje.set_title(materia, fontsize=9)
        eje.set_yticks(posiciones)
        eje.set_yticklabels(etiquetas, fontsize=7.5)
        eje.set_xlim(0, 22)
        eje.grid(axis="x", color="0.9", linewidth=0.5)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
    for eje in ejes[1]:
        eje.set_xlabel("Rango de la predicción (participaciones)")
    figura_.tight_layout(h_pad=1.2, w_pad=1.0)
    salida = RAIZ / "figuras" / "resultados" / "figura_sensibilidad_por_asignatura.png"
    figura_.savefig(salida, dpi=300, bbox_inches="tight", facecolor="white")
    print("Figura guardada en", salida)


if __name__ == "__main__":
    figura(pd.read_csv(CARPETA / "sensibilidad_evidencia_casos.csv"))
