"""Respaldo de la evidencia de la red bayesiana (Tabla 19) y combinación más
cercana con respaldo (comentarios del tutor en Resultados de los Datos
Atípicos).

Análisis complementario: NO modifica ningún resultado del informe. Usa las
predicciones oficiales de la red bayesiana, las de la LSTM con semilla 42 y
la extrapolación proporcional.

1. Respaldo de cada caso con evidencia completa: número de filas
   estudiante-sesión del entrenamiento de su pliegue con exactamente la misma
   combinación de las cuatro variables de evidencia. Se exige reproducir los
   conteos y el RMSE de la Tabla 19, y también las otras columnas de esa tabla:
   mediana de registros estudiante-sección que aportan esas filas, puntaje de
   Brier y concentración media de la distribución posterior.
2. Brecha frente a la LSTM según el respaldo (RMSE de los tres métodos por
   banda y en los casos con respaldo).
3. Causa de la falta de respaldo: qué valores de la evidencia no aparecen
   en ningún caso del entrenamiento del pliegue.
4. Combinación más cercana con respaldo: para los casos sin respaldo, se
   busca en el entrenamiento la combinación observada a menor distancia
   ordinal (suma de diferencias de posición de estado en las cuatro
   variables) y se predice con la distribución del objetivo en esas filas.
   Es una alternativa evaluada, no la implementación principal.

Correr el archivo genera `resultados/respaldo_evidencia_*.csv`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ajuste_cpd import preparar_fold
from ensamblado import ESTADOS_BN, ensamblar_conjunto
from valor_esperado import ESTADOS, medias_entrenamiento_por_estado, valor_esperado

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_LSTM = RAIZ / "3_lstm" / "resultados" / "predicciones_lstm_validacion_cruzada.csv"
CARPETA = RAIZ / "6_evaluacion" / "resultados"
LLAVE = ["materia", "trimestre", "seccion", "estudiante_id", "hito"]
OBJETIVO = "Cantidad de participaciones del trimestre"
EVIDENCIA = {  # columna del CSV oficial -> nodo de la red
    "evidencia_anio_que_cursa": "Año que cursa",
    "evidencia_tamano_grupo": "Tamaño del grupo",
    "evidencia_participaciones_semana": "Participaciones de la semana",
    "evidencia_participaciones_semana_anterior": "Participaciones de la semana anterior",
}
NODOS = list(EVIDENCIA.values())
BANDAS = [(-1, 0, "0"), (0, 10, "1–10"), (10, 50, "11–50"), (50, 200, "51–200"), (200, np.inf, ">200")]
# Valores publicados en la Tabla 19. El RMSE bayesiano de la banda 51–200 es
# 3,4246 y debe publicarse como 3,42 (el informe muestra 3,43); por eso la
# comprobación de ese valor admite una diferencia de 0,01.
TABLA_19 = {"0": (379, 7.49, 4.36), "1–10": (145, 8.76, 6.19), "11–50": (142, 5.86, 4.82),
            "51–200": (324, 3.43, 1.91), ">200": (573, 2.39, 1.65)}
# Otras columnas de la Tabla 19: (mediana de registros estudiante-sección, puntaje de Brier, concentración media).
TABLA_19_CALIDAD = {"0": (0, 0.80, 0.20), "1–10": (2, 1.08, 0.82), "11–50": (8, 0.87, 0.52),
                    "51–200": (8, 0.87, 0.57), ">200": (42, 0.64, 0.50)}


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def cargar() -> pd.DataFrame:
    bayes = pd.read_csv(RUTA_BAYES, dtype=str).rename(columns={"trimestre_prueba": "trimestre"})
    for c in ("hito", "seccion"):
        bayes[c] = bayes[c].astype(int)
    for c in ("total_trimestre_real", "prediccion_continua"):
        bayes[c] = bayes[c].astype(float)
    lstm = pd.read_csv(RUTA_LSTM)
    datos = bayes.merge(lstm[LLAVE + ["prediccion_lstm", "prediccion_extrapolacion"]], on=LLAVE)
    assert len(datos) == 1572
    return datos


def _posicion(nodo: str, estado: str) -> int:
    return ESTADOS_BN[nodo].index(estado)


def analizar(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sesiones, reg, _sem = ensamblar_conjunto()
    completos = datos[datos.evidencia_anio_que_cursa.notna() & (datos.evidencia_anio_que_cursa != "nan")].copy()
    filas = []
    for (materia, trimestre), g in completos.groupby(["materia", "trimestre"]):
        train_bn, _test, _rep = preparar_fold(sesiones, materia, trimestre)
        train = train_bn.dropna(subset=NODOS + [OBJETIVO])
        medias = medias_entrenamiento_por_estado(reg, materia, trimestre)
        conteo = train.groupby(NODOS).size()
        # Registros estudiante-sección distintos que aportan las filas de cada combinación.
        registros = (sesiones[(sesiones.materia == materia) & (sesiones.trimestre != trimestre)]
                     .dropna(subset=NODOS + [OBJETIVO])
                     .drop_duplicates(NODOS + ["estudiante_id", "seccion", "trimestre"])
                     .groupby(NODOS).size())
        combinaciones = pd.DataFrame(list(conteo.index), columns=NODOS)
        posiciones = np.column_stack([combinaciones[n].map(lambda e, n=n: _posicion(n, e)) for n in NODOS])
        for _, caso in g.iterrows():
            clave = tuple(caso[c] for c in EVIDENCIA)
            respaldo = int(conteo.get(clave, 0))
            ausentes = [n for n, v in zip(NODOS, clave) if not (train[n] == v).any()]
            posterior_caso = np.array([float(caso[f"posterior_{e}"]) for e in ESTADOS])
            observado = np.array([float(e == caso.estado_real) for e in ESTADOS])
            fila = {**{c: caso[c] for c in LLAVE}, **{c: caso[c] for c in EVIDENCIA},
                    "respaldo": respaldo, "registros_respaldo": int(registros.get(clave, 0)),
                    "brier": float(((posterior_caso - observado) ** 2).sum()),
                    "concentracion": float(posterior_caso.max()), "valores_ausentes": ", ".join(ausentes),
                    "total_trimestre_real": caso.total_trimestre_real, "prediccion_bayes": caso.prediccion_continua,
                    "prediccion_lstm": caso.prediccion_lstm, "prediccion_extrapolacion": caso.prediccion_extrapolacion}
            if respaldo == 0:
                objetivo = np.array([_posicion(n, v) for n, v in zip(NODOS, clave)])
                distancia = np.abs(posiciones - objetivo).sum(axis=1)
                minima = distancia.min()
                cercanas = combinaciones[distancia == minima]
                mascara = np.zeros(len(train), dtype=bool)
                for comb in cercanas.itertuples(index=False):
                    mascara |= (train[NODOS] == pd.Series(comb, index=NODOS)).all(axis=1).to_numpy()
                vecinas = train[mascara]
                frecuencias = vecinas[OBJETIVO].value_counts(normalize=True)
                posterior = {e: float(frecuencias.get(e, 0.0)) for e in ESTADOS}
                fila.update({"distancia_cercana": int(minima), "combinaciones_cercanas": len(cercanas),
                             "filas_cercanas": int(mascara.sum()),
                             "prediccion_cercana": valor_esperado(posterior, medias),
                             "cercana_descripcion": " | ".join(
                                 "; ".join(f"{n}={v}" for n, v in zip(NODOS, comb)) for comb in cercanas.itertuples(index=False))})
            filas.append(fila)
    casos = pd.DataFrame(filas)
    assert len(casos) == 1563, len(casos)

    resumen = []
    for bajo, alto, nombre in BANDAS:
        b = casos[(casos.respaldo > bajo) & (casos.respaldo <= alto)]
        n_esperado, rmse_bayes, rmse_ext = TABLA_19[nombre]
        assert len(b) == n_esperado, (nombre, len(b))
        assert abs(_rmse(b.prediccion_bayes, b.total_trimestre_real) - rmse_bayes) <= 0.01, nombre
        assert round(_rmse(b.prediccion_extrapolacion, b.total_trimestre_real), 2) == rmse_ext, nombre
        registros_mediana, brier, concentracion = TABLA_19_CALIDAD[nombre]
        assert b.registros_respaldo.median() == registros_mediana, nombre
        assert round(b.brier.mean(), 2) == brier, nombre
        assert round(b.concentracion.mean(), 2) == concentracion, nombre
        resumen.append({"banda": nombre, "casos": len(b),
                        "registros_mediana": b.registros_respaldo.median(), "brier": b.brier.mean(),
                        "concentracion": b.concentracion.mean(),
                        "rmse_bayes": _rmse(b.prediccion_bayes, b.total_trimestre_real),
                        "rmse_extrapolacion": _rmse(b.prediccion_extrapolacion, b.total_trimestre_real),
                        "rmse_lstm": _rmse(b.prediccion_lstm, b.total_trimestre_real)})
    con = casos[casos.respaldo > 0]
    sin = casos[casos.respaldo == 0]
    resumen.append({"banda": "con respaldo (>0)", "casos": len(con),
                    "rmse_bayes": _rmse(con.prediccion_bayes, con.total_trimestre_real),
                    "rmse_extrapolacion": _rmse(con.prediccion_extrapolacion, con.total_trimestre_real),
                    "rmse_lstm": _rmse(con.prediccion_lstm, con.total_trimestre_real)})
    resumen.append({"banda": "sin respaldo, combinación más cercana", "casos": len(sin),
                    "rmse_bayes": _rmse(sin.prediccion_cercana, sin.total_trimestre_real),
                    "rmse_extrapolacion": _rmse(sin.prediccion_extrapolacion, sin.total_trimestre_real),
                    "rmse_lstm": _rmse(sin.prediccion_lstm, sin.total_trimestre_real)})
    return casos, pd.DataFrame(resumen)


def omitir_valor_ausente(casos: pd.DataFrame) -> pd.DataFrame:
    """Alternativa a la combinación más cercana: en los casos sin respaldo,
    se retira de la evidencia el valor que no aparece en el entrenamiento del
    pliegue y la red lo marginaliza, igual que con el año faltante. Si todos
    los valores existen pero la combinación es inédita, no cambia nada."""
    from pgmpy.inference import VariableElimination
    from ajuste_cpd import ajustar_cpd, ESS_SELECCIONADO
    from red_bayesiana import construir_modelo_manual
    sesiones, reg, _sem = ensamblar_conjunto()
    sin = casos[casos.respaldo == 0].copy()
    predicciones = {}
    for (materia, trimestre), g in sin.groupby(["materia", "trimestre"]):
        train_bn, _test, _rep = preparar_fold(sesiones, materia, trimestre)
        inferencia = VariableElimination(ajustar_cpd(construir_modelo_manual(), train_bn, ESS_SELECCIONADO))
        medias = medias_entrenamiento_por_estado(reg, materia, trimestre)
        for i, caso in g.iterrows():
            ausentes = [a.strip() for a in str(caso.valores_ausentes).split(",") if a.strip() and a != "nan"]
            evidencia = {n: caso[f"evidencia_{c}"] for n, c in zip(NODOS, ("anio_que_cursa", "tamano_grupo",
                         "participaciones_semana", "participaciones_semana_anterior")) if n not in ausentes}
            r = inferencia.query([OBJETIVO], evidence=evidencia, show_progress=False)
            posterior = {e: float(p) for e, p in zip(r.state_names[OBJETIVO], r.values)}
            predicciones[i] = valor_esperado(posterior, medias)
    sin["prediccion_omitiendo_ausente"] = pd.Series(predicciones)
    return sin


if __name__ == "__main__":
    casos, resumen = analizar(cargar())
    pd.set_option("display.width", 220)
    print("Tabla 19 reproducida.\n", resumen.round(2).to_string(index=False))
    sin = casos[casos.respaldo == 0]
    print("\nValores ausentes del entrenamiento en los casos sin respaldo:")
    print(sin.valores_ausentes.replace("", "(ninguno: cada valor existe, pero no la combinación)").value_counts().to_string())
    print("\nSin respaldo por pliegue:")
    print(sin.groupby(["materia", "trimestre"]).size().to_string())
    print("\nDistancia a la combinación más cercana:", sin.distancia_cercana.value_counts().sort_index().to_dict())
    # Efecto de usar la combinación más cercana sobre el RMSE por asignatura e hito
    todos = casos.copy()
    todos["prediccion_alternativa"] = np.where(todos.respaldo == 0, todos.prediccion_cercana, todos.prediccion_bayes)
    efecto = todos.groupby(["materia", "hito"])[list(todos.columns)].apply(lambda g: pd.Series({
        "rmse_oficial": _rmse(g.prediccion_bayes, g.total_trimestre_real),
        "rmse_con_cercana": _rmse(g.prediccion_alternativa, g.total_trimestre_real),
        "rmse_lstm": _rmse(g.prediccion_lstm, g.total_trimestre_real)})).reset_index()
    print("\nRMSE por asignatura e hito (casos con evidencia completa):\n", efecto.round(2).to_string(index=False))
    atipicos = casos[casos.estudiante_id.isin(["anon_145", "anon_322", "anon_370"])]
    print("\nCasos atípicos:\n", atipicos[["estudiante_id", "hito", "respaldo", "valores_ausentes", "total_trimestre_real",
                                           "prediccion_bayes", "prediccion_cercana", "distancia_cercana",
                                           "filas_cercanas", "cercana_descripcion"]].to_string(index=False))
    sin = omitir_valor_ausente(casos)
    for etiqueta, g in (("sin respaldo", sin), ("con un valor ausente", sin[sin.valores_ausentes.fillna("") != ""]),
                        ("solo combinación inédita", sin[sin.valores_ausentes.fillna("") == ""])):
        print(f"{etiqueta} ({len(g)}): uniforme {_rmse(g.prediccion_bayes, g.total_trimestre_real):.2f}, "
              f"omitiendo el valor ausente {_rmse(g.prediccion_omitiendo_ausente, g.total_trimestre_real):.2f}, "
              f"combinación más cercana {_rmse(g.prediccion_cercana, g.total_trimestre_real):.2f}, "
              f"LSTM {_rmse(g.prediccion_lstm, g.total_trimestre_real):.2f}, "
              f"extrapolación {_rmse(g.prediccion_extrapolacion, g.total_trimestre_real):.2f}")
    casos = casos.join(sin[["prediccion_omitiendo_ausente"]])
    casos.to_csv(CARPETA / "respaldo_evidencia_casos.csv", index=False)
    resumen.to_csv(CARPETA / "respaldo_evidencia_bandas.csv", index=False)
    efecto.to_csv(CARPETA / "respaldo_evidencia_combinacion_cercana.csv", index=False)
