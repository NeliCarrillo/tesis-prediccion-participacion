"""Calendario académico, acontecimientos nacionales y anomalías del RMSE
entre hitos (comentario del tutor sobre la sección Comparación LSTM y
Bayes). Complementa `anomalias_pendiente.py`.

Análisis complementario: NO modifica ningún resultado del informe. Usa las
predicciones oficiales de la red bayesiana, las de la LSTM con semilla 42
(las únicas guardadas por estudiante), la extrapolación proporcional, los
datos estandarizados y los cronogramas de las 17 secciones.

1. Participación por semana y por tipo de sesión, y proporción del total
   ya observada en cada hito (explica la extrapolación y la LSTM).
2. Casos con evidencia bayesiana nula (0 participaciones en la semana del
   hito y en la anterior): relación entre el acumulado hasta el hito y el
   total real, y error de cada modelo en esos casos.
3. Semanas con acontecimientos nacionales: participación por sesión dictada,
   normalizada por el promedio de la sección, frente a la misma semana en
   los demás trimestres de la asignatura.
4. LSTM sin la marca de semana de evaluación: cambio de la predicción de los
   modelos finales del prototipo al poner en cero esa entrada. Describe el
   comportamiento del modelo, no su desempeño.
5. Figuras: calendario de participación (Figura 16) y mecanismo (Figura 17).

Correr el archivo genera `resultados/anomalias_calendario_*.csv` y las dos
figuras en `figuras/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rutas  # noqa: F401  (agrega 4_red_bayesiana/codigo a sys.path)
from ensamblado import cargar_datos_crudos

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_BAYES = RAIZ / "4_red_bayesiana" / "resultados" / "predicciones_bayesiana_s4_s6_s8.csv"
RUTA_LSTM = RAIZ / "3_lstm" / "resultados" / "predicciones_lstm_validacion_cruzada.csv"
CARPETA_RESULTADOS = RAIZ / "6_evaluacion" / "resultados"
CARPETA_FIGURAS = RAIZ / "figuras" / "resultados"
REGISTRO = ["materia", "trimestre", "seccion", "estudiante_id"]
LLAVE = REGISTRO + ["hito"]
MATERIAS = ("Algoritmos y Programación", "Computación Emergente",
            "Estructura de Datos", "Matemáticas Discretas")
HITOS = (4, 6, 8)
TAMANO_TEXTO = 9

# Lunes de la semana 1 de cada trimestre, según los cronogramas.
INICIO_TRIMESTRE = {"2425-2": "2025-01-06", "2425-3": "2025-04-21", "2526-1": "2025-09-08",
                    "2526-2": "2026-01-05", "2526-3": "2026-04-13"}

# Acontecimientos con fecha que caen dentro de una semana de clases
# (número, trimestre, semana, descripción, fuente). Los feriados salen de
# los cronogramas; los demás, de la prensa citada en el informe.
EVENTOS = (
    (1, "2425-2", 1, "9 y 10 de enero de 2025: protestas y juramentación presidencial", "CNN en Español (2025)"),
    (2, "2425-2", 9, "3 y 4 de marzo de 2025: Carnaval", "cronograma"),
    (3, "2425-3", 2, "1 de mayo de 2025: Día del Trabajador", "cronograma"),
    (4, "2425-3", 5, "22 y 23 de mayo de 2025: suspensión de clases escolares por las elecciones del 25 de mayo",
     "Efecto Cocuyo (2025)"),
    (5, "2425-3", 10, "24 de junio de 2025: Batalla de Carabobo", "cronograma"),
    (6, "2526-1", 7, "19 y 20 de octubre de 2025: días de júbilo nacional", "El Nacional (2025)"),
    (7, "2526-1", 12, "22 a 25 de noviembre de 2025: suspensión de vuelos internacionales", "Expansión (2025)"),
    (8, "2526-2", 1, "3 de enero de 2026: ataque de Estados Unidos; otras universidades dictaron clases "
                     "virtuales entre el 7 y el 10 de enero", "Efecto Cocuyo (2026)"),
    (9, "2526-2", 2, "12 de enero de 2026: reanudación de clases escolares en el país", "El Nacional (2026)"),
    (10, "2526-2", 7, "16 y 17 de febrero de 2026: Carnaval", "cronograma"),
    (11, "2526-3", 3, "1 de mayo de 2026: Día del Trabajador", "cronograma"),
    (12, "2526-3", 11, "24 de junio de 2026: Batalla de Carabobo y terremoto de magnitud 7,5", "CNN en Español (2026)"),
    (13, "2526-3", 12, "Semana posterior al terremoto", "CNN en Español (2026)"),
)


def _con_coma(valor: float, decimales: int = 2) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def cargar_predicciones(crudos: pd.DataFrame) -> pd.DataFrame:
    bayes = pd.read_csv(RUTA_BAYES).rename(columns={"trimestre_prueba": "trimestre"})
    lstm = pd.read_csv(RUTA_LSTM)
    datos = bayes[LLAVE + ["total_trimestre_real", "prediccion_continua",
                           "evidencia_participaciones_semana", "evidencia_participaciones_semana_anterior"]].merge(
        lstm[LLAVE + ["prediccion_lstm", "prediccion_extrapolacion"]], on=LLAVE)
    semanal = crudos.groupby(REGISTRO + ["semana"]).participaciones.sum().reset_index()
    acumulados = []
    for hito in HITOS:
        a = semanal[semanal.semana <= hito].groupby(REGISTRO).participaciones.sum().rename("acumulado").reset_index()
        acumulados.append(a.assign(hito=hito))
    datos = datos.merge(pd.concat(acumulados), on=LLAVE)
    datos["evidencia_nula"] = ((datos.evidencia_participaciones_semana.astype(str) == "0")
                               & (datos.evidencia_participaciones_semana_anterior.astype(str) == "0"))
    assert len(datos) == 1572
    return datos


def calendario(crudos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por sección y semana: tipos de sesión, participación media
    por estudiante y por sesión dictada."""
    sesiones = crudos.drop_duplicates(["materia", "trimestre", "seccion", "semana", "dia_sesion"])
    tipos = sesiones.groupby(["materia", "trimestre", "seccion", "semana"]).tipo_sesion.agg(list)
    por_estudiante = (crudos.groupby(REGISTRO + ["semana"]).participaciones.sum()
                      .groupby(["materia", "trimestre", "seccion", "semana"]).mean())
    dictadas = crudos[crudos.tipo_sesion != "sin_clase"]
    por_sesion = dictadas.groupby(["materia", "trimestre", "seccion", "semana"]).participaciones.mean()
    tabla = pd.DataFrame({"tipos": tipos, "participacion_estudiante": por_estudiante,
                          "participacion_sesion": por_sesion}).reset_index()
    tabla["evaluacion"] = tabla.tipos.apply(lambda t: "evaluacion" in t)
    tabla["sin_clase"] = tabla.tipos.apply(lambda t: "sin_clase" in t)
    tabla["participacion_relativa"] = tabla.participacion_sesion / tabla.groupby(
        ["materia", "trimestre", "seccion"]).participacion_sesion.transform("mean")
    tabla["tipos"] = tabla.tipos.apply(", ".join)
    return tabla


def proporcion_acumulada(crudos: pd.DataFrame) -> pd.DataFrame:
    semanal = crudos.groupby(["materia", "semana"]).participaciones.sum()
    total = semanal.groupby("materia").transform("sum")
    tabla = (semanal.groupby("materia").cumsum() / total).rename("proporcion_acumulada").reset_index()
    tabla["proporcion_uniforme"] = tabla.semana / 12
    return tabla


def evidencia_nula(datos: pd.DataFrame) -> pd.DataFrame:
    rmse = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
    filas = []
    for (materia, hito), g in datos.groupby(["materia", "hito"]):
        n = g[g.evidencia_nula]
        filas.append({"materia": materia, "hito": hito, "casos": len(g), "casos_evidencia_nula": len(n),
                      "proporcion_evidencia_nula": len(n) / len(g),
                      "correlacion_acumulado_total": float(n[["acumulado", "total_trimestre_real"]].corr().iloc[0, 1]),
                      "correlacion_prediccion_bayes": float(n[["prediccion_continua", "total_trimestre_real"]].corr().iloc[0, 1]),
                      "correlacion_prediccion_lstm": float(n[["prediccion_lstm", "total_trimestre_real"]].corr().iloc[0, 1]),
                      "rmse_bayes": rmse(n.prediccion_continua, n.total_trimestre_real),
                      "rmse_lstm": rmse(n.prediccion_lstm, n.total_trimestre_real),
                      "sesgo_bayes": float((n.prediccion_continua - n.total_trimestre_real).mean())})
    return pd.DataFrame(filas)


def semanas_con_eventos(cal: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for numero, trimestre, semana, descripcion, fuente in EVENTOS:
        for _, s in cal[(cal.trimestre == trimestre) & (cal.semana == semana)].iterrows():
            otros = cal[(cal.materia == s.materia) & (cal.trimestre != trimestre) & (cal.semana == semana)]
            filas.append({"evento": numero, "descripcion": descripcion, "fuente": fuente, "trimestre": trimestre,
                          "semana": semana, "materia": s.materia, "seccion": s.seccion, "tipos": s.tipos,
                          "participacion_relativa": s.participacion_relativa,
                          "misma_semana_otros_trimestres": otros.participacion_relativa.mean()})
    tabla = pd.DataFrame(filas)
    tabla["menor_que_otros"] = tabla.participacion_relativa < tabla.misma_semana_otros_trimestres
    return tabla


def lstm_sin_marca_evaluacion() -> pd.DataFrame:
    sys.path.insert(0, str(RAIZ / "3_lstm" / "adaptacion"))
    import explicabilidad_lstm as ex
    servicio = ex.lstm_service
    datos = servicio._preparar_datos()
    X, temas, estatica = datos["X"], datos["X_temas"], datos["estatica"]
    indice_eval = ex.N_EST + servicio.DINAMICAS.index("evaluaciones")
    filas = []
    for materia in MATERIAS:
        idx = np.where(estatica.materia.to_numpy() == materia)[0]
        for hito in HITOS:
            Xh, Th = X[idx, :hito].copy(), temas[idx, :hito].copy()
            base = ex.predecir_lote(materia, hito, Xh, Th)
            sin_marca = Xh.copy()
            sin_marca[:, :, indice_eval] = 0
            cambio = ex.predecir_lote(materia, hito, sin_marca, Th) - base
            filas.append({"materia": materia, "hito": hito, "cambio_medio": float(cambio.mean()),
                          "cambio_absoluto_medio": float(np.abs(cambio).mean()),
                          "evaluaciones_hasta_hito": float(Xh[:, :, indice_eval].sum(axis=1).mean())})
    return pd.DataFrame(filas)


def _estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": TAMANO_TEXTO, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    return plt


def figura_calendario(cal: pd.DataFrame) -> None:
    plt = _estilo()
    from matplotlib.colors import LinearSegmentedColormap
    filas = cal[["materia", "trimestre", "seccion"]].drop_duplicates()
    filas["orden"] = filas.materia.map({m: i for i, m in enumerate(MATERIAS)})
    filas = filas.sort_values(["orden", "trimestre", "seccion"]).reset_index(drop=True)
    matriz = np.full((len(filas), 12), np.nan)
    marcas = [["" for _ in range(12)] for _ in range(len(filas))]
    numeros = [["" for _ in range(12)] for _ in range(len(filas))]
    por_semana = {(n[1], n[2]): n[0] for n in EVENTOS}
    for i, f in filas.iterrows():
        g = cal[(cal.materia == f.materia) & (cal.trimestre == f.trimestre) & (cal.seccion == f.seccion)]
        for _, s in g.iterrows():
            j = int(s.semana) - 1
            matriz[i, j] = s.participacion_estudiante
            marcas[i][j] = ("E" if s.evaluacion else "") + ("F" if s.sin_clase else "")
            numeros[i][j] = str(por_semana.get((f.trimestre, int(s.semana)), ""))
    mapa = LinearSegmentedColormap.from_list("azul", ["#f4f7fb", "#1f4e79"])
    figura, eje = plt.subplots(figsize=(6.5, 5.4))
    imagen = eje.imshow(matriz, cmap=mapa, vmin=0, vmax=1.5, aspect="auto")
    for i in range(len(filas)):
        for j in range(12):
            oscuro = matriz[i, j] > 0.9
            if marcas[i][j]:
                eje.text(j, i, marcas[i][j], ha="center", va="center", fontsize=7.5, fontweight="bold",
                         color="white" if oscuro else "black")
            if numeros[i][j]:
                eje.text(j + 0.45, i - 0.42, numeros[i][j], ha="right", va="top", fontsize=6,
                         color="white" if oscuro else "#b22222")
    etiquetas = [f"{f.trimestre} (secc. {f.seccion})" for f in filas.itertuples()]
    eje.set_yticks(range(len(filas)))
    eje.set_yticklabels(etiquetas, fontsize=7.5)
    eje.set_xticks(range(12))
    eje.set_xticklabels([str(s) for s in range(1, 13)])
    eje.set_xlabel("Semana del trimestre")
    for hito in HITOS:
        eje.axvline(hito - 0.5, color="black", linewidth=1.1, linestyle="--")
        eje.text(hito - 0.5, -0.9, f"S{hito}", ha="center", va="bottom", fontsize=8)
    limites = filas.groupby("orden").size().cumsum().to_numpy()[:-1]
    for y in limites:
        eje.axhline(y - 0.5, color="black", linewidth=1.2)
    inicio = 0
    for orden, n in filas.groupby("orden").size().items():
        nombre = {"Algoritmos y Programación": "Algoritmos y\nProgramación",
                  "Computación Emergente": "Computación\nEmergente",
                  "Estructura de Datos": "Estructura\nde Datos",
                  "Matemáticas Discretas": "Matemáticas\nDiscretas"}[MATERIAS[orden]]
        eje.text(-4.1, inicio + (n - 1) / 2, nombre, ha="center", va="center", fontsize=7, rotation=90,
                 linespacing=1.0)
        inicio += n
    eje.set_xlim(-0.5, 11.5)
    for lado in ("top", "right", "left", "bottom"):
        eje.spines[lado].set_visible(False)
    barra = figura.colorbar(imagen, ax=eje, fraction=0.03, pad=0.02)
    barra.set_label("Participaciones por estudiante en la semana")
    barra.ax.yaxis.set_major_formatter(lambda y, _p: _con_coma(y, 1))
    figura.savefig(CARPETA_FIGURAS / "figura_calendario_participacion.png", dpi=300, bbox_inches="tight",
                   facecolor="white")


def figura_mecanismo(acumulada: pd.DataFrame, datos: pd.DataFrame) -> None:
    plt = _estilo()
    figura, ejes = plt.subplots(1, 3, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    estilos = {"Algoritmos y Programación": ("-", "o"), "Computación Emergente": ("--", "s"),
               "Estructura de Datos": ("-.", "^"), "Matemáticas Discretas": (":", "D")}
    eje = ejes[0]
    eje.plot([0, 12], [0, 1], color="0.6", linewidth=0.8, label="Reparto uniforme")
    for materia, (linea, marcador) in estilos.items():
        g = acumulada[acumulada.materia == materia]
        eje.plot(g.semana, g.proporcion_acumulada, linestyle=linea, marker=marcador, markersize=3,
                 color="black", linewidth=1, label=materia)
    for hito in HITOS:
        eje.axvline(hito, color="0.8", linewidth=0.6, zorder=0)
        eje.text(hito, 1.03, f"S{hito}", ha="center", va="bottom", fontsize=7)
    eje.set_xticks(range(1, 13))
    eje.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(1, 13)])
    eje.set_xlim(0, 12.3)
    eje.set_ylim(0, 1.02)
    eje.yaxis.set_major_formatter(lambda y, _p: f"{y * 100:.0f} %")
    eje.set_xlabel("Semana")
    eje.set_ylabel("Proporción del total")
    eje.set_title("a) Participación acumulada", fontsize=TAMANO_TEXTO, loc="left", pad=12)
    for eje, (materia, hito, letra) in zip(ejes[1:], (("Algoritmos y Programación", 6, "b"),
                                                     ("Estructura de Datos", 8, "c"))):
        g = datos[(datos.materia == materia) & (datos.hito == hito) & datos.evidencia_nula]
        tope = max(g.total_trimestre_real.max(), g.prediccion_lstm.max(), g.prediccion_continua.max()) + 1
        eje.plot([0, tope], [0, tope], color="0.6", linewidth=0.8)
        eje.scatter(g.total_trimestre_real, g.prediccion_continua, s=10, marker="s", color="tab:red",
                    alpha=0.7, linewidths=0, label="Red bayesiana")
        eje.scatter(g.total_trimestre_real, g.prediccion_lstm, s=10, marker="o", color="tab:blue",
                    alpha=0.7, linewidths=0, label="LSTM")
        eje.set_xlim(-0.5, tope)
        eje.set_ylim(-0.5, tope)
        eje.set_xlabel("Total real")
        eje.set_ylabel("Predicción")
        nombre = "Algoritmos y Prog." if materia.startswith("Algoritmos") else "Estructura de Datos"
        eje.set_title(f"{letra}) {nombre}, S{hito}", fontsize=TAMANO_TEXTO, loc="left", pad=12)
        paso = 5 if tope > 12 else 3
        eje.set_xticks(range(0, int(tope) + 1, paso))
        eje.set_yticks(range(0, int(tope) + 1, paso))
    for eje in ejes:
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
    manejadores, etiquetas = ejes[1].get_legend_handles_labels()
    figura.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.0)
    lineas, nombres = ejes[0].get_legend_handles_labels()
    figura.legend(lineas, nombres, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                  fontsize=6.5, handlelength=2.4, columnspacing=1.0,
                  title="a: proporción del total registrada hasta cada semana", title_fontsize=7.5)
    figura.legend(manejadores, etiquetas, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                  frameon=False, fontsize=7, markerscale=1.6,
                  title="b y c: cero participaciones en la semana\ndel hito y en la anterior",
                  title_fontsize=7.5)
    figura.savefig(CARPETA_FIGURAS / "figura_mecanismo_anomalias.png", dpi=300, bbox_inches="tight",
                   facecolor="white")


if __name__ == "__main__":
    crudos = cargar_datos_crudos()
    datos = cargar_predicciones(crudos)
    cal = calendario(crudos)
    acumulada = proporcion_acumulada(crudos)
    nula = evidencia_nula(datos)
    eventos = semanas_con_eventos(cal)
    marca = lstm_sin_marca_evaluacion()
    fechas = {t: pd.Timestamp(f) for t, f in INICIO_TRIMESTRE.items()}
    cal["lunes_semana"] = [fechas[t] + pd.Timedelta(weeks=int(s) - 1) for t, s in zip(cal.trimestre, cal.semana)]

    por_tipo = crudos[crudos.tipo_sesion != "sin_clase"].groupby("tipo_sesion").participaciones.mean()
    print("Participación por estudiante y sesión:", por_tipo.round(3).to_dict())
    print("Proporción acumulada en los hitos:")
    print(acumulada[acumulada.semana.isin(HITOS)].pivot(index="materia", columns="semana",
                                                          values="proporcion_acumulada").round(2).to_string())
    print("\nEvidencia bayesiana nula:\n", nula.round(2).to_string(index=False))
    print(f"\nSemanas con eventos: {len(eventos)} sección-semana; con participación menor que la misma semana "
          f"de otros trimestres: {int(eventos.menor_que_otros.sum())}")
    print("\nLSTM sin marca de evaluación:\n", marca.round(3).to_string(index=False))

    salidas = {"calendario": cal, "proporcion_acumulada": acumulada, "evidencia_nula": nula,
               "eventos": eventos, "lstm_sin_marca_evaluacion": marca}
    for nombre, tabla in salidas.items():
        tabla.to_csv(CARPETA_RESULTADOS / f"anomalias_calendario_{nombre}.csv", index=False)
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_calendario(cal)
    figura_mecanismo(acumulada, datos)
    print("Figuras y CSV guardados.")
