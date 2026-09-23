"""Genera los artefactos de `prototipo/artefactos_lstm/` — Sprint 5,
tarjeta 2 (Estrategia 2 aprobada): un modelo LSTM final por asignatura x
hito (12 en total), entrenado con TODOS los trimestres disponibles de esa
asignatura (sin dejar ninguno fuera) y semilla fija 42 — la misma
constante `SEMILLA` que usa por defecto la celda 34 del notebook
validado, no elegida por desempeño.

No es un pipeline alternativo. Reutiliza, sin copiarlas a mano:
  - las funciones de preparación de datos ya extraídas y verificadas en
    `services/lstm_service.py` (`cargar_datos`, `codificar_campos_cualitativos`,
    `agregar_por_semana`, `construir_tabla_estatica`, `construir_tensor_3d`,
    `truncar_a_hito`, y las constantes `CLAVES`/`ESTATICAS`/`DINAMICAS`/
    `N_SEMANAS`/`N_TEMAS`/`HITOS`);
  - `construir_modelo_lstm` y `escalar_objetivo`, extraídas aquí
    programáticamente (no retranscritas a mano) de las celdas 23 y 26 del
    notebook, porque `lstm_service.py` no las necesita para inferencia
    (solo carga un modelo ya entrenado) y por eso no viven ahí.

No recalcula métricas oficiales ni toca la validación cruzada: la única
entrada es el conjunto ya ensamblado por `lstm_service._preparar_datos()`
(idéntico al notebook — ver `services/test_paridad_lstm.py`, nivel A).

`escalar_caracteristicas` del notebook no se reutiliza tal cual porque
descarta el `StandardScaler` ya ajustado (pensada para el paradigma
train/test de la validación cruzada, donde nadie necesita conservar el
escalador después). Aquí sí hace falta conservarlo para persistir sus
parámetros, así que se replica su misma lógica interna (mismo reshape,
mismo `StandardScaler`) en la línea que lo usa — no es una reimplementación
distinta, es la única forma de obtener el objeto ajustado sin modificar
esa función en el notebook.

Uso:
    python3 entrenar_modelos_finales.py --salida artefactos_lstm
    python3 entrenar_modelos_finales.py --salida /ruta/de/prueba   # no toca los artefactos reales

El directorio de salida es obligatorio a propósito: así una ejecución de
prueba nunca puede sobrescribir `prototipo/artefactos_lstm/` por olvido.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))
import lstm_service  # noqa: E402 (ver sys.path arriba)

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "LSTM" / "nuevo" / "adaptacion_lstm_participaciones.ipynb"

SEMILLA = 42
UNIDADES = 16
EPOCAS_MAX = 150
LOTE = 16
VALIDATION_SPLIT = 0.15
PATIENCE = 20


def _extraer_del_notebook() -> tuple:
    """Extrae `construir_modelo_lstm` y `escalar_objetivo` de las celdas
    23 y 26 del notebook, verbatim (ejecutando su código fuente exacto,
    no una copia retranscrita), y confirma que la celda 2 sigue
    declarando `SEMILLA = 42` -- si el notebook cambiara ese valor en el
    futuro sin que este script se actualice, falla ruidosamente en vez de
    entrenar con una semilla distinta sin que nadie se dé cuenta."""
    nb = json.load(open(NOTEBOOK))
    celdas = nb["cells"][:56]  # nunca las 2 celdas aditivas de exportación

    fuente_semilla = "".join(celdas[2]["source"])
    assert "SEMILLA = 42" in fuente_semilla, (
        "la celda 2 del notebook ya no declara SEMILLA = 42 -- revisa este script "
        "antes de reentrenar, la semilla fija podría haber cambiado"
    )

    fuente_modelo = "".join(celdas[23]["source"])
    fuente_escalado = "".join(celdas[26]["source"])
    assert "def construir_modelo_lstm" in fuente_modelo
    assert "def escalar_objetivo" in fuente_escalado

    espacio_nombres = {"Model": tf.keras.models.Model, **{
        n: getattr(tf.keras.layers, n)
        for n in ("Input", "Embedding", "Reshape", "Concatenate", "LSTM", "Dense", "Dropout")
    }, "Adam": tf.keras.optimizers.Adam, "tf": tf, "np": np, "N_TEMAS": lstm_service.N_TEMAS}
    exec(fuente_modelo, espacio_nombres)
    exec(fuente_escalado, espacio_nombres)
    return espacio_nombres["construir_modelo_lstm"], espacio_nombres["escalar_objetivo"]


def entrenar(salida: Path) -> pd.DataFrame:
    construir_modelo_lstm, escalar_objetivo = _extraer_del_notebook()

    datos = lstm_service._preparar_datos()
    estatica, X, X_temas, y = datos["estatica"], datos["X"], datos["X_temas"], datos["y"]
    indice_anio = lstm_service.ESTATICAS.index("anio_academico")

    salida.mkdir(parents=True, exist_ok=True)
    resumen = []

    for materia in sorted(estatica.materia.unique()):
        filas = (estatica.materia == materia).to_numpy()
        X_m, X_temas_m, y_m = X[filas], X_temas[filas], y[filas]

        for hito in lstm_service.HITOS:
            t0 = time.time()

            X_hito = lstm_service.truncar_a_hito(X_m, hito)
            T_hito = lstm_service.truncar_a_hito(X_temas_m, hito)
            acumulado = X_hito[:, :, len(lstm_service.ESTATICAS)].sum(axis=1)
            restantes = y_m - acumulado

            columna_anio = X_hito[:, :, indice_anio].copy()
            mediana_anio = None
            if np.isnan(columna_anio).any():
                mediana_anio = float(np.nanmedian(columna_anio))
                faltan = np.isnan(X_hito[:, :, indice_anio])
                X_hito = X_hito.copy()
                X_hito[:, :, indice_anio] = np.where(faltan, mediana_anio, X_hito[:, :, indice_anio])

            pasos, caracteristicas = X_hito.shape[1], X_hito.shape[2]
            escalador = StandardScaler()
            X_escalado = escalador.fit_transform(X_hito.reshape(-1, caracteristicas)).reshape(-1, pasos, caracteristicas)

            media_obj, desviacion_obj = escalar_objetivo(restantes)

            tf.keras.utils.set_random_seed(SEMILLA)
            modelo = construir_modelo_lstm(hito, X.shape[2], unidades=UNIDADES)
            historia = modelo.fit(
                [X_escalado, T_hito], (restantes - media_obj) / desviacion_obj,
                epochs=EPOCAS_MAX, batch_size=LOTE, validation_split=VALIDATION_SPLIT, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
            )

            slug = lstm_service.SLUGS[materia]
            carpeta = salida / f"{slug}_h{hito}"
            carpeta.mkdir(parents=True, exist_ok=True)

            modelo.save(carpeta / "modelo.keras")
            json.dump({
                "mean": escalador.mean_.tolist(),
                "scale": escalador.scale_.tolist(),
                "columnas": lstm_service.ESTATICAS + lstm_service.DINAMICAS,
            }, open(carpeta / "escalador_x.json", "w"), indent=2)
            json.dump({
                "media_objetivo": media_obj,
                "desviacion_objetivo": desviacion_obj,
            }, open(carpeta / "objetivo.json", "w"), indent=2)
            json.dump({
                "mediana_anio_academico": mediana_anio,
            }, open(carpeta / "imputacion.json", "w"), indent=2)

            perdidas_val = historia.history["val_loss"]
            epocas_ejecutadas = len(perdidas_val)
            mejor_epoca = int(np.argmin(perdidas_val)) + 1  # 1-indexada
            activo_early_stopping = epocas_ejecutadas < EPOCAS_MAX

            resumen.append({
                "materia": materia, "hito": hito, "n_registros": int(filas.sum()),
                "epocas_ejecutadas": epocas_ejecutadas, "epoca_mejor_val_loss": mejor_epoca,
                "early_stopping_activo": activo_early_stopping,
                "segundos": round(time.time() - t0, 1),
                "mediana_anio_imputada": mediana_anio,
            })
            print(f"{materia} / hito {hito}: {epocas_ejecutadas} épocas, mejor en "
                  f"la {mejor_epoca} ({'EarlyStopping activo' if activo_early_stopping else 'agotó el máximo de épocas'}) "
                  f"-> {carpeta}")

    json.dump({
        "ESTATICAS": lstm_service.ESTATICAS, "DINAMICAS": lstm_service.DINAMICAS,
        "N_SEMANAS": lstm_service.N_SEMANAS, "N_TEMAS": lstm_service.N_TEMAS,
        "HITOS": list(lstm_service.HITOS), "SEMILLA": SEMILLA, "unidades": UNIDADES,
    }, open(salida / "config_compartida.json", "w"), indent=2)

    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(salida / "resumen_entrenamiento.csv", index=False)
    return resumen_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salida", required=True, type=Path,
        help="Directorio donde escribir los 12 artefactos (obligatorio, para no "
             "sobrescribir prototipo/artefactos_lstm/ por accidente).",
    )
    args = parser.parse_args()

    resumen_df = entrenar(args.salida.resolve())
    print(f"\n{len(resumen_df)} combinaciones entrenadas. Resumen en "
          f"{args.salida.resolve() / 'resumen_entrenamiento.csv'}")
