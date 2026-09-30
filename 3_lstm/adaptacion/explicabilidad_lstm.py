"""Intento de explicabilidad de la red LSTM (tarjeta "Documentar sensibilidad
ambos modelos", Sprint 5).

No reentrena nada ni modifica ninguna métrica reportada: analiza los 12
modelos finales ya entrenados del prototipo (`5_prototipo/artefactos_lstm/`,
uno por asignatura e hito, entrenados con todos los registros) y, para la
comparación, la red bayesiana final del prototipo. Tres análisis:

1. Pesos: qué puede leerse de los pesos de entrada de la primera capa y si
   esa lectura es estable entre los 12 modelos.
2. Combinaciones: sensibilidad de la predicción al cambiar las mismas cuatro
   variables de la Tabla 20 (participaciones de la semana del hito y de la
   anterior, año que cursa y tamaño del grupo), medida como rango máximo
   menos mínimo, comparable con la red bayesiana.
3. Acción: efecto de sumar una participación en la semana del hito, y si dos
   estudiantes con la misma evidencia reciben la misma respuesta.

Como se analizan los modelos finales sobre los mismos registros con que se
entrenaron, este análisis describe el comportamiento de los modelos y no es
una evaluación predictiva.

Correr el archivo genera los CSV en `3_lstm/resultados/` y la figura del Apéndice K
en `figuras/apendice_K_explicabilidad_lstm/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "5_prototipo"))
sys.path.insert(0, str(RAIZ / "5_prototipo" / "services"))
sys.path.insert(0, str(RAIZ / "4_red_bayesiana" / "codigo"))

import bayes_service  # noqa: E402
import lstm_service  # noqa: E402
from discretizacion import bin_anio, participaciones_semana, tamano_grupo  # noqa: E402

CARPETA = RAIZ / "3_lstm" / "resultados"
CARPETA_FIGURAS = RAIZ / "figuras" / "apendice_K_explicabilidad_lstm"
MATERIAS = ("Algoritmos y Programación", "Computación Emergente",
            "Estructura de Datos", "Matemáticas Discretas")
HITOS = (4, 6, 8)

N_EST = len(lstm_service.ESTATICAS)
I_ANIO = lstm_service.ESTATICAS.index("anio_academico")
I_TAMANO = lstm_service.ESTATICAS.index("tamano_grupo")
I_POSICION = lstm_service.ESTATICAS.index("posicion_lista")
I_PART = N_EST + lstm_service.DINAMICAS.index("participaciones")
RASGOS = lstm_service.ESTATICAS + lstm_service.DINAMICAS + [f"tema_{i}" for i in range(4)]

# Valores alternativos, alineados con los estados de la red bayesiana.
VALORES_PARTICIPACION = (0, 1, 2, 3)          # estados 0, 1, 2 y "3 o más"
VALORES_ANIO = (1, 2, 3, 4, 5)                # estados 1 a "5 o más"
VALORES_TAMANO = (27, 30, 39)                 # pequeño, mediano, grande (mediana de cada estado)


def predecir_lote(materia: str, hito: int, X: np.ndarray, X_temas: np.ndarray) -> np.ndarray:
    """Misma transformación que `lstm_service._predecir_desde_tensor`, en lote:
    imputación del año con la mediana, escalado, modelo y reconstrucción
    del total (acumulado hasta el hito + participaciones restantes)."""
    a = lstm_service._cargar_artefactos(materia, hito)
    X = X.copy()
    X[:, :, I_ANIO] = np.where(np.isnan(X[:, :, I_ANIO]), a["mediana_anio_academico"], X[:, :, I_ANIO])
    pasos, rasgos = X.shape[1], X.shape[2]
    Xe = ((X.reshape(-1, rasgos) - a["mean"]) / a["scale"]).reshape(-1, pasos, rasgos)
    z = a["modelo"].predict([Xe, X_temas], verbose=0).ravel()
    acumulado = X[:, :, I_PART].sum(axis=1)
    return acumulado + z * a["desviacion_objetivo"] + a["media_objetivo"]


def analisis_pesos() -> pd.DataFrame:
    filas = []
    for materia in MATERIAS:
        for hito in HITOS:
            modelo = lstm_service._cargar_artefactos(materia, hito)["modelo"]
            primera_lstm = next(capa for capa in modelo.layers if capa.__class__.__name__ == "LSTM")
            kernel = primera_lstm.get_weights()[0]
            assert kernel.shape == (len(RASGOS), 64), kernel.shape
            for i, rasgo in enumerate(RASGOS):
                w = kernel[i]
                filas.append({"materia": materia, "hito": hito, "rasgo": rasgo,
                              "peso_medio_abs": float(np.abs(w).mean()),
                              "prop_positivos": float((w > 0).mean())})
    return pd.DataFrame(filas)


def analisis_perturbaciones() -> pd.DataFrame:
    datos = lstm_service._preparar_datos()
    estatica = datos["estatica"]
    filas = []
    for materia in MATERIAS:
        idx = np.where((estatica.materia == materia).to_numpy())[0]
        entrada_bayes = bayes_service._cargar_modelo(materia)
        for hito in HITOS:
            X = datos["X"][idx, :hito].astype(np.float32).copy()
            Xt = datos["X_temas"][idx, :hito].copy()
            base = predecir_lote(materia, hito, X, Xt)
            # Control: el cálculo en lote reproduce el servicio del prototipo.
            from caso import CasoPrediccion
            for j in (0, len(idx) // 2, len(idx) - 1):
                reg = estatica.iloc[idx[j]]
                caso = CasoPrediccion(materia=materia, trimestre=reg.trimestre, seccion=reg.seccion,
                                      estudiante_id=reg.estudiante_id, hito=hito)
                esperado = lstm_service.predict_lstm(caso).prediccion_total
                assert abs(esperado - base[j]) < 1e-4, (materia, hito, esperado, base[j])

            def variante(modificar) -> np.ndarray:
                Xv = X.copy()
                modificar(Xv)
                return predecir_lote(materia, hito, Xv, Xt)

            escenarios = {
                "Participaciones de la semana": [variante(lambda M, v=v: M.__setitem__((slice(None), hito - 1, I_PART), v))
                                                 for v in VALORES_PARTICIPACION],
                "Participaciones de la semana anterior": [variante(lambda M, v=v: M.__setitem__((slice(None), hito - 2, I_PART), v))
                                                          for v in VALORES_PARTICIPACION],
                "Año que cursa": [variante(lambda M, v=v: M.__setitem__((slice(None), slice(None), I_ANIO), v))
                                  for v in VALORES_ANIO],
                "Tamaño del grupo": [variante(lambda M, v=v: M.__setitem__((slice(None), slice(None), I_TAMANO), v))
                                     for v in VALORES_TAMANO],
                "Posición relativa en la lista": [variante(lambda M, v=v: M.__setitem__((slice(None), slice(None), I_POSICION), v))
                                                  for v in (0.1, 0.5, 0.9)],
            }
            mas_uno = variante(lambda M: M.__setitem__((slice(None), hito - 1, I_PART), M[:, hito - 1, I_PART] + 1))
            # Interacción: efecto de +1 en la semana del hito con la anterior en 0 o en 3.
            def con_anterior(v_ant, extra):
                return variante(lambda M: (M.__setitem__((slice(None), hito - 2, I_PART), v_ant),
                                           M.__setitem__((slice(None), hito - 1, I_PART), M[:, hito - 1, I_PART] + extra)))
            efecto_ant0 = con_anterior(0, 1) - con_anterior(0, 0)
            efecto_ant3 = con_anterior(3, 1) - con_anterior(3, 0)

            for j, i in enumerate(idx):
                reg = estatica.iloc[i]
                semana = X[j, hito - 1, I_PART]
                anterior = X[j, hito - 2, I_PART]
                anio = X[j, 0, I_ANIO]
                evidencia = {"Tamaño del grupo": tamano_grupo(int(X[j, 0, I_TAMANO])),
                             "Participaciones de la semana": participaciones_semana(semana),
                             "Participaciones de la semana anterior": participaciones_semana(anterior)}
                estado_anio = None if np.isnan(anio) else bin_anio(anio)
                if estado_anio is not None:
                    evidencia["Año que cursa"] = estado_anio
                _, pred_b, _ = bayes_service.consultar_evidencia(materia, evidencia)
                ev_mas = dict(evidencia, **{"Participaciones de la semana": participaciones_semana(semana + 1)})
                _, pred_b_mas, _ = bayes_service.consultar_evidencia(materia, ev_mas)
                fila = {"materia": materia, "hito": hito, "estudiante_id": reg.estudiante_id,
                        "trimestre": reg.trimestre, "seccion": reg.seccion,
                        "config_bayes": "|".join(f"{k}={evidencia.get(k)}" for k in sorted(evidencia)),
                        "pred_lstm": base[j], "pred_bayes": pred_b,
                        "delta_lstm_mas1": mas_uno[j] - base[j],
                        "delta_restantes_lstm_mas1": mas_uno[j] - base[j] - 1.0,
                        "delta_bayes_mas1": pred_b_mas - pred_b,
                        "efecto_mas1_anterior0": efecto_ant0[j], "efecto_mas1_anterior3": efecto_ant3[j]}
                for variable, preds in escenarios.items():
                    valores = [p[j] for p in preds]
                    fila[f"rango_{variable}"] = max(valores) - min(valores)
                filas.append(fila)
    return pd.DataFrame(filas)


def figura(casos: pd.DataFrame, pesos: pd.DataFrame) -> None:
    """a) Predicciones de ambos modelos para estudiantes con la misma evidencia
    bayesiana (grupos de al menos 10 casos). b) Peso medio absoluto de cada
    entrada numérica en la primera capa LSTM de los 12 modelos finales."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
    coma = lambda y, _p: f"{y:g}".replace(".", ",")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    grupos = [g for _, g in casos.groupby(["materia", "hito", "config_bayes"]) if len(g) >= 10]
    grupos.sort(key=lambda g: g.pred_bayes.iloc[0])
    for k, g in enumerate(grupos):
        a1.scatter([k] * len(g), g.pred_lstm, s=5, color="tab:blue", alpha=0.45, linewidths=0,
                   label="LSTM" if k == 0 else None)
        a1.plot([k - 0.4, k + 0.4], [g.pred_bayes.iloc[0]] * 2, color="tab:red", linewidth=1.6,
                label="Red bayesiana" if k == 0 else None)
    a1.set_xlabel(f"Grupos con la misma evidencia bayesiana ({len(grupos)} grupos)")
    a1.set_ylabel("Predicción del total trimestral")
    a1.set_xticks([])
    a1.set_ylim(bottom=0)
    a1.yaxis.set_major_formatter(coma)
    a1.set_title("a) Misma evidencia, distintas predicciones", fontsize=9, loc="left")
    a1.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2)

    etiquetas = {"anio_academico": "Año", "seccion_num": "Sección", "tamano_grupo": "Tamaño",
                 "posicion_lista": "Posición", "participaciones": "Participaciones",
                 "sesiones": "Sesiones", "evaluaciones": "Evaluaciones"}
    orden = list(etiquetas)
    tabla = pesos[pesos.rasgo.isin(orden)].pivot_table(index=["materia", "hito"], columns="rasgo",
                                                       values="peso_medio_abs")[orden]
    for i, (_, fila) in enumerate(tabla.iterrows()):
        a2.plot(range(len(orden)), fila.to_numpy(), "-", color="0.7", linewidth=0.7,
                label="Cada modelo" if i == 0 else None)
    a2.plot(range(len(orden)), tabla.median().to_numpy(), "o-", color="black", linewidth=1.3,
            markersize=3.5, label="Mediana de los 12 modelos")
    a2.set_xticks(range(len(orden)))
    a2.set_xticklabels([etiquetas[r] for r in orden], rotation=35, ha="right")
    a2.set_ylabel("Peso medio absoluto")
    a2.set_ylim(0, 0.25)
    a2.yaxis.set_major_formatter(coma)
    a2.set_title("b) Pesos de la primera capa LSTM", fontsize=9, loc="left")
    a2.legend(frameon=False, fontsize=7.5, loc="lower left")
    for eje in (a1, a2):
        eje.grid(axis="y", color="0.9", linewidth=0.5)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARPETA_FIGURAS / "figura_explicabilidad_lstm.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    pesos = analisis_pesos()
    pesos.to_csv(CARPETA / "explicabilidad_lstm_pesos.csv", index=False)
    casos = analisis_perturbaciones()
    assert len(casos) == 1572, len(casos)
    casos.to_csv(CARPETA / "explicabilidad_lstm_casos.csv", index=False)
    figura(casos, pesos)
    print("CSV y figura guardados en", CARPETA)
