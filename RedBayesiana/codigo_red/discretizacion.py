"""Discretización de las 10 variables de la red bayesiana según la Tabla 13
del informe («Fase de diseño del modelo de red bayesiana», Sprint 3).

Sección y las dos variables de sesión (número de sesiones, sesiones de
evaluación) ya llegan como enteros pequeños; se incluyen aquí igual, con una
función propia cada una, para centralizar toda la discretización en un solo
módulo reusable por `ensamblado.py` y `estructura_aprendida.py`.

Los cortes de este módulo son fijos y reproducibles: no se recalculan a
partir del conjunto de datos en cada ejecución (ni por asignatura ni por
pliegue de validación), para que un mismo valor discretice siempre al mismo
estado. Es la propuesta ya cerrada del informe (Tabla 13); no se rediseña
aquí ningún corte.

`año que cursa` y `participaciones de la semana anterior` pueden venir
faltantes (año académico desconocido; primera semana de un registro, sin
antecedente). En ambos casos la función correspondiente devuelve `None`
explícito: nunca se imputa ni se convierte un faltante en un estado válido
("5 o más" o "0").

`tema de la sesión` es la única discretización supervisada: usa la propia
participación para agrupar los códigos de cada asignatura en baja/media/alta,
por lo que debe ajustarse solo con los trimestres de entrenamiento de cada
pliegue de validación cruzada (`ajustar_mapa_temas`) y aplicarse después al
trimestre excluido (`aplicar_mapa_temas`, con `permitir_no_visto=True` para
que un código ausente del entrenamiento caiga en «tema_no_visto»).
"""
from __future__ import annotations

import pandas as pd


def tamano_grupo(n_estudiantes: int) -> str:
    """Tres estados, por el agrupamiento natural observado en las 14 secciones
    (18, 27, 28 | 30×7 | 35, 38, 40, 41 estudiantes)."""
    if n_estudiantes < 30:
        return "pequeño"
    if n_estudiantes == 30:
        return "mediano"
    return "grande"


def posicion_en_lista(numero_lista: int, tamano_grupo_seccion: int) -> str:
    """Cuatro estados por cuartiles (igualdad de frecuencia). La posición
    relativa (0 a 1) se calcula igual que en el LSTM: número de lista entre
    el tamaño del grupo."""
    posicion = numero_lista / tamano_grupo_seccion
    if posicion <= 0.25:
        return "≤0,25"
    if posicion <= 0.50:
        return "0,25–0,50"
    if posicion <= 0.75:
        return "0,50–0,75"
    return ">0,75"


def participaciones_semana(participaciones: int) -> str:
    """Cuatro estados: el 80% de las semanas registradas no tiene ninguna
    participación, por lo que el cero se aísla como estado propio."""
    if participaciones == 0:
        return "0"
    if participaciones == 1:
        return "1"
    if participaciones == 2:
        return "2"
    return "3 o más"


def participaciones_trimestre(total: int) -> str:
    """Cinco estados para la variable objetivo. El 34% de los registros no
    tiene ninguna participación en todo el trimestre (Figura 3)."""
    if total == 0:
        return "0"
    if total <= 2:
        return "1-2"
    if total <= 5:
        return "3-5"
    if total <= 11:
        return "6-11"
    return "12 o más"


def bin_anio(anio_academico) -> str | None:
    """Año que cursa: 1, 2, 3, 4, «5 o más». `generar_csv.py` ya limita el
    valor numérico a TOPE_ANIO=5; aquí solo se convierte a la etiqueta de
    texto del estado, preservando el faltante quien no tiene carnet
    identificable (3 registros estudiante-sección)."""
    if pd.isna(anio_academico):
        return None
    return str(int(anio_academico)) if anio_academico < 5 else "5 o más"


def bin_nses(n_sesiones_semana: int) -> str:
    """Número de sesiones de la semana: 1 o 2, sin transformación real."""
    return "1" if n_sesiones_semana == 1 else "2"


def bin_neval(n_eval_semana: int) -> str:
    """Sesiones de evaluación de la semana: 0, 1 o 2, sin transformación
    real."""
    return str(int(n_eval_semana))


def tercios(tabla_por_tema: "pd.DataFrame") -> "pd.DataFrame":
    """Agrupa los códigos de tema de una asignatura en baja/media/alta por
    igualdad de frecuencia de sesiones (percentiles acumulados 33,33% y
    66,67%), a partir de una tabla con una fila por (materia, tema) y las
    columnas «n» (sesiones) y «media» (participación promedio)."""
    tabla = tabla_por_tema.sort_values("media").copy()
    tabla["acum"] = tabla["n"].cumsum()
    total = tabla["n"].sum()
    tabla["categoria"] = pd.cut(
        tabla["acum"],
        bins=[0, total / 3, 2 * total / 3, total],
        labels=["baja", "media", "alta"],
    )
    return tabla


def ajustar_mapa_temas(datos_entrenamiento: "pd.DataFrame") -> tuple[dict, "pd.DataFrame"]:
    """Ajusta el mapa tema -> baja/media/alta usando solo las filas recibidas
    (para no filtrar información del trimestre excluido, debe llamarse con
    un `datos_entrenamiento` que ya excluya ese trimestre)."""
    temas = (
        datos_entrenamiento[datos_entrenamiento["tema"] != 0]
        .groupby(["materia", "tema"])
        .agg(n=("participaciones", "size"), media=("participaciones", "mean"))
        .reset_index()
    )
    tablas = [tercios(g) for _, g in temas.groupby("materia")]
    if not tablas:
        return {}, temas
    temas = pd.concat(tablas, ignore_index=True)
    mapa = {
        (row["materia"], int(row["tema"])): str(row["categoria"])
        for _, row in temas.iterrows()
    }
    return mapa, temas


def aplicar_mapa_temas(datos: "pd.DataFrame", mapa: dict, permitir_no_visto: bool) -> "pd.Series":
    """Aplica un mapa ya ajustado, sin consultar la participación de
    `datos` (evita la fuga de información del trimestre de prueba). Con
    `permitir_no_visto=True`, un código ausente del mapa se etiqueta
    «tema_no_visto» en vez de producir un faltante."""
    categorias = []
    for materia, tema in zip(datos["materia"], datos["tema"]):
        tema = int(tema)
        if tema == 0:
            categorias.append("0")
            continue
        categoria = mapa.get((materia, tema))
        if categoria is None and permitir_no_visto:
            categoria = "tema_no_visto"
        categorias.append(categoria)
    return pd.Series(categorias, index=datos.index, dtype="object")


if __name__ == "__main__":
    # Verificación rápida contra los tres casos usados como ejemplo en el informe
    assert tamano_grupo(18) == tamano_grupo(27) == tamano_grupo(28) == "pequeño"
    assert tamano_grupo(30) == "mediano"
    assert tamano_grupo(35) == tamano_grupo(41) == "grande"
    assert posicion_en_lista(1, 30) == "≤0,25"          # 0,03
    assert posicion_en_lista(20, 30) == "0,50–0,75"     # 0,67
    assert participaciones_semana(0) == "0"
    assert participaciones_semana(5) == "3 o más"
    assert participaciones_trimestre(0) == "0"
    assert participaciones_trimestre(44) == "12 o más"
    assert bin_anio(1) == "1" and bin_anio(5) == "5 o más"
    assert bin_anio(float("nan")) is None
    assert bin_nses(1) == "1" and bin_nses(2) == "2"
    assert bin_neval(0) == "0" and bin_neval(2) == "2"

    ejemplo = pd.DataFrame({
        "materia": ["X"] * 4,
        "tema": [1, 2, 3, 4],
        "participaciones": [0, 0, 5, 5],
    })
    ejemplo_train = pd.DataFrame({
        "materia": ["X", "X", "X", "X"] * 3,
        "tema": [1, 2, 3, 4] * 3,
        "participaciones": [0, 0, 5, 5] * 3,
    })
    mapa, _ = ajustar_mapa_temas(ejemplo_train)
    aplicado = aplicar_mapa_temas(ejemplo, mapa, permitir_no_visto=False)
    assert aplicado.iloc[0] == "baja" and aplicado.iloc[3] in ("media", "alta")
    no_visto = aplicar_mapa_temas(
        pd.DataFrame({"materia": ["X"], "tema": [99], "participaciones": [1]}),
        mapa,
        permitir_no_visto=True,
    )
    assert no_visto.iloc[0] == "tema_no_visto"

    print("verificaciones superadas")
