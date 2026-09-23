"""Pruebas de paridad del servicio LSTM del prototipo — Sprint 5, tarjeta 2.

No es una suite de pytest formal (el proyecto no usa pytest en ningún otro
lado): son dos comprobaciones ejecutables, en la misma línea que los
`if __name__ == "__main__":` de `RedBayesiana/codigo_red/`.

## Nivel A — preparación de datos (comparación directa y significativa)

Ejecuta las celdas 0-18 del notebook validado
(`LSTM/nuevo/adaptacion_lstm_participaciones.ipynb`, hasta construir
`X`/`X_temas`/`y`/`estatica`, sin entrenar nada) en un kernel real y
separado, y compara ese resultado, elemento por elemento, contra
`lstm_service._preparar_datos()`. Si algún valor difiere, el pipeline del
prototipo dejó de representar los datos exactamente igual que el notebook
validado.

## Nivel B — aritmética de inferencia (comprobación de plomería, no de paridad contra un modelo "original")

Los modelos que carga `predict_lstm` son los finales del prototipo
(Estrategia 2, entrenados con todo el histórico) — no existe un modelo
"original" equivalente en la validación cruzada con el que comparar
número contra número, porque la validación cruzada nunca entrena un
modelo con todos los trimestres. Por eso el nivel B no compara contra el
notebook: reimplementa de forma independiente la misma fórmula de
reconstrucción (escalar con mean_/scale_, predecir, desestandarizar, sumar
el acumulado) y verifica que `predict_lstm` coincide con esa
reimplementación — esto detecta errores de aritmética/indexado en el
servicio, no reemplaza una validación de desempeño.
"""
from __future__ import annotations

import json
import shutil
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lstm_service
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # caso

import lstm_service
from caso import CasoPrediccion

REPO = Path(__file__).resolve().parent.parent.parent
NOTEBOOK = REPO / "LSTM" / "nuevo" / "adaptacion_lstm_participaciones.ipynb"


def _ejecutar_preparacion_del_notebook() -> dict:
    """Ejecuta únicamente las celdas 0-18 del notebook (imports, carga,
    codificación, agregación semanal, tabla estática y construcción de
    tensores — ninguna celda de entrenamiento) en un kernel real, dentro
    de una carpeta temporal anidada bajo `LSTM/nuevo/` (para que el
    descubrimiento de `RAIZ` por `Path.cwd()` de la celda 6 encuentre
    igual `Datos Tesis Downstream/`). Devuelve `X`, `X_temas`, `y` y
    `estatica` tal como los produjo el notebook."""
    import nbformat
    from nbclient import NotebookClient

    nb = json.load(open(NOTEBOOK))
    assert len(nb["cells"]) >= 56, "el notebook tiene menos celdas de las esperadas"
    celdas_preparacion = copy.deepcopy(nb["cells"][0:19])  # 0..18 inclusive

    export = (
        "import numpy as _np, json as _json\n"
        "_np.save('X.npy', X)\n"
        "_np.save('X_temas.npy', X_temas)\n"
        "_np.save('y.npy', y)\n"
        "estatica.to_json('estatica.json', orient='split')\n"
    )
    celdas_preparacion.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": export,
    })

    # nbformat.from_dict espera "source" como una única cadena, no como
    # lista de líneas (a diferencia del JSON crudo de un .ipynb en disco,
    # que nbformat.read() normaliza automáticamente).
    for celda in celdas_preparacion:
        if isinstance(celda.get("source"), list):
            celda["source"] = "".join(celda["source"])

    scratch_nb = copy.deepcopy(nb)
    scratch_nb["cells"] = celdas_preparacion

    carpeta_temp = REPO / "LSTM" / "nuevo" / "_paridad_temp"
    carpeta_temp.mkdir(exist_ok=True)
    try:
        nb_obj = nbformat.from_dict(scratch_nb)
        NotebookClient(nb_obj, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(carpeta_temp)}}).execute()

        X = np.load(carpeta_temp / "X.npy")
        X_temas = np.load(carpeta_temp / "X_temas.npy")
        y = np.load(carpeta_temp / "y.npy")
        estatica = pd.read_json(carpeta_temp / "estatica.json", orient="split")
        return {"X": X, "X_temas": X_temas, "y": y, "estatica": estatica}
    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)


def probar_paridad_preparacion() -> None:
    referencia = _ejecutar_preparacion_del_notebook()
    servicio = lstm_service._preparar_datos()

    assert np.array_equal(referencia["X"], servicio["X"], equal_nan=True), "X difiere"
    assert np.array_equal(referencia["X_temas"], servicio["X_temas"]), "X_temas difiere"
    assert np.array_equal(referencia["y"], servicio["y"], equal_nan=True), "y difiere"

    cols = ["materia", "trimestre", "seccion", "estudiante_id"]
    a = referencia["estatica"][cols].sort_values(cols).reset_index(drop=True)
    b = servicio["estatica"][cols].sort_values(cols).reset_index(drop=True)
    assert a.equals(b), "las claves de estatica difieren en orden o contenido"

    print(f"Nivel A superado: X{referencia['X'].shape}, X_temas{referencia['X_temas'].shape}, "
          f"y{referencia['y'].shape} y estatica ({len(a)} registros) idénticos "
          "entre el notebook y lstm_service._preparar_datos().")


def probar_aritmetica_inferencia() -> None:
    casos = [
        CasoPrediccion("Algoritmos y Programación", "2425-2", "1", "anon_001", 4),
        CasoPrediccion("Computación Emergente", "2425-3", "1", "anon_148", 6),
        CasoPrediccion("Matemáticas Discretas", "2526-2", "1", "anon_md_2526_2_1_001", 8),
        # anon_063: uno de los 3 registros con año académico faltante -- ejerce
        # la ruta de imputación con la mediana congelada del modelo final.
        CasoPrediccion("Algoritmos y Programación", "2526-1", "1", "anon_063", 4),
    ]
    indice_anio = lstm_service.ESTATICAS.index("anio_academico")

    for caso in casos:
        resultado = lstm_service.predict_lstm(caso)

        # Reimplementación independiente de la misma fórmula, sin usar
        # ninguna función interna de lstm_service (más allá de construir
        # la entrada, que ya está cubierta por el nivel A).
        X_hito, X_temas_hito, acumulado = lstm_service.construir_entrada_lstm(caso)
        artefactos = lstm_service._cargar_artefactos(caso.materia, caso.hito)

        X_hito = X_hito.copy()
        if np.isnan(X_hito[:, :, indice_anio]).any():
            X_hito[:, :, indice_anio] = artefactos["mediana_anio_academico"]

        forma = X_hito.shape
        plano = X_hito.reshape(-1, forma[2])
        escalado = (plano - artefactos["mean"]) / artefactos["scale"]
        escalado = escalado.reshape(forma)

        pred_estandar = artefactos["modelo"].predict([escalado, X_temas_hito], verbose=0).ravel()[0]
        total_independiente = (
            acumulado + float(pred_estandar) * artefactos["desviacion_objetivo"] + artefactos["media_objetivo"]
        )

        assert np.isclose(resultado.prediccion_total, total_independiente, atol=1e-4), (
            f"{caso}: predict_lstm={resultado.prediccion_total} vs "
            f"reimplementación independiente={total_independiente}"
        )
        assert np.isfinite(resultado.prediccion_total)
        print(f"Nivel B — {caso.materia} / {caso.estudiante_id} / hito {caso.hito}: "
              f"predict_lstm={resultado.prediccion_total:.3f} == "
              f"reimplementación={total_independiente:.3f}")


if __name__ == "__main__":
    probar_paridad_preparacion()
    probar_aritmetica_inferencia()
    print("\nTodas las pruebas de paridad superadas.")
