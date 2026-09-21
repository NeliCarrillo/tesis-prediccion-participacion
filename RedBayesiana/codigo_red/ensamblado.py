"""Ensambla el conjunto discretizado por asignatura para la red bayesiana,
a nivel estudiante-sesión, según la Tabla 13 del informe (Sprint 3).

Decisión de granularidad (Sprint 4, carta 2): la unidad final es
estudiante-sesión, no estudiante-semana. Una semana con dos sesiones produce
dos filas que comparten las variables semanales y difieren en el tema de
cada sesión — es intencional, no un defecto a corregir. La razón: «Tema de
la sesión» es un tema individual observado en una sesión, 975 de las 6.288
semanas (15,5%) tienen dos temas de contenido reales distintos, y no existe
en el informe, el código ni el historial del proyecto una regla para reducir
esos dos temas a un único estado semanal. El pipeline de Sprint 3
(sesiones → HillClimbSearch → BIC → CPT/BDeu) ya es consistente con esta
unidad; cambiarla exigiría una regla metodológica nueva y sin precedente.

Este módulo solo ensambla datos: no aprende estructura, no estima CPD y no
genera figuras. Importarlo no ejecuta nada por sí solo — cada función se
invoca explícitamente.

No hace dropna(): año académico y la primera semana de cada registro
(sin «participaciones de la semana anterior») quedan como faltantes
explícitos (`None`), para que la fase de estimación de CPD decida qué filas
usar.

«Tema de la sesión» no se incluye como columna fija en el conjunto que
devuelve `ensamblar_conjunto`: su discretización es supervisada y debe
ajustarse solo con los trimestres de entrenamiento de cada pliegue de
validación cruzada, nunca de forma global. Use `discretizacion.
ajustar_mapa_temas` / `aplicar_mapa_temas` por pliegue (ver
`estructura_aprendida.preparar_fold_tema` para un ejemplo ya en uso).
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from discretizacion import (
    tamano_grupo,
    posicion_en_lista,
    participaciones_semana,
    participaciones_trimestre,
    bin_anio,
    bin_nses,
    bin_neval,
)

RAIZ = Path(__file__).resolve().parent.parent.parent

CLAVE = ["estudiante_id", "materia", "trimestre", "seccion"]

# Las 10 variables de la Tabla 13 que recibe la red bayesiana. «Tema de la
# sesión» se agrega por pliegue (ver docstring del módulo), no aquí.
COLUMNAS_BN = [
    "Año que cursa",
    "Sección",
    "Tamaño del grupo",
    "Posición relativa en la lista",
    "Tema de la sesión",
    "Número de sesiones de la semana",
    "Sesiones de evaluación de la semana",
    "Participaciones de la semana anterior",
    "Participaciones de la semana",
    "Cantidad de participaciones del trimestre",
]

# El dominio se declara de forma explícita para que pgmpy conserve estados
# válidos aunque tengan frecuencia cero en una asignatura o fold concreto.
ESTADOS_BN = {
    "Año que cursa": ["1", "2", "3", "4", "5 o más"],
    "Sección": ["1", "2"],
    "Tamaño del grupo": ["pequeño", "mediano", "grande"],
    "Posición relativa en la lista": ["≤0,25", "0,25–0,50", "0,50–0,75", ">0,75"],
    "Tema de la sesión": ["0", "baja", "media", "alta", "tema_no_visto"],
    "Número de sesiones de la semana": ["1", "2"],
    "Sesiones de evaluación de la semana": ["0", "1", "2"],
    "Participaciones de la semana anterior": ["0", "1", "2", "3 o más"],
    "Participaciones de la semana": ["0", "1", "2", "3 o más"],
    "Cantidad de participaciones del trimestre": ["0", "1-2", "3-5", "6-11", "12 o más"],
}


def cargar_datos_crudos(raiz: Path = RAIZ) -> pd.DataFrame:
    """Carga y concatena los 17 CSV de «Datos Tesis Downstream/» (una fila
    por estudiante y sesión)."""
    archivos = glob.glob(str(raiz / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True)
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)

    for columna in ("participaciones", "numero_lista", "tema", "anio_academico"):
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    assert len(df) == 11700, (
        f"se esperaban 11.700 filas estudiante-sesión antes de cualquier filtro, hay {len(df)}"
    )
    return df


def _construir_registros(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por registro estudiante-sección (524), con las variables
    estáticas ya discretizadas."""
    tamano_seccion = (
        df.groupby(["materia", "trimestre", "seccion"])["estudiante_id"]
        .nunique()
        .rename("tamano_grupo")
    )

    reg = df.drop_duplicates(subset=CLAVE)[CLAVE + ["numero_lista", "anio_academico"]].copy()
    reg = reg.merge(tamano_seccion, on=["materia", "trimestre", "seccion"])
    reg["posicion"] = reg["numero_lista"] / reg["tamano_grupo"]

    total_trimestre = (
        df.groupby(CLAVE)["participaciones"].sum().rename("total_trimestre").reset_index()
    )
    reg = reg.merge(total_trimestre, on=CLAVE)

    assert len(reg) == 524, f"se esperaban 524 registros estudiante-sección, hay {len(reg)}"

    faltantes_anio = int(reg["anio_academico"].isna().sum())
    assert faltantes_anio == 3, (
        f"se esperaban 3 registros estudiante-sección con año faltante, hay {faltantes_anio}"
    )

    reg["Año que cursa"] = reg["anio_academico"].apply(bin_anio)
    reg["Tamaño del grupo"] = reg["tamano_grupo"].apply(tamano_grupo)
    reg["Posición relativa en la lista"] = reg.apply(
        lambda fila: posicion_en_lista(fila["numero_lista"], fila["tamano_grupo"]), axis=1
    )
    reg["Cantidad de participaciones del trimestre"] = (
        reg["total_trimestre"].apply(participaciones_trimestre)
    )
    return reg


def _construir_semanas(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por estudiante-semana (6.288), con las variables semanales
    ya discretizadas, incluido el rezago de «participaciones de la semana
    anterior» dentro de cada registro."""
    sem = (
        df.groupby(CLAVE + ["semana"])["participaciones"]
        .sum()
        .rename("participaciones_semana")
        .reset_index()
    )

    n_sesiones = df.groupby(CLAVE + ["semana"]).size().rename("n_sesiones_semana").reset_index()
    sem = sem.merge(n_sesiones, on=CLAVE + ["semana"])

    n_eval = (
        df[df["tipo_sesion"] == "evaluacion"]
        .groupby(CLAVE + ["semana"])
        .size()
        .rename("n_eval_semana")
    )
    sem = sem.merge(n_eval, on=CLAVE + ["semana"], how="left")
    sem["n_eval_semana"] = sem["n_eval_semana"].fillna(0).astype(int)

    assert len(sem) == 6288, f"se esperaban 6.288 filas estudiante-semana, hay {len(sem)}"

    # El rezago se calcula dentro de cada registro (CLAVE), nunca entre
    # estudiantes o secciones distintos; la primera semana de cada registro
    # queda sin antecedente (NaN), no se rellena.
    sem = sem.sort_values(CLAVE + ["semana"])
    sem["participaciones_semana_anterior"] = (
        sem.groupby(CLAVE)["participaciones_semana"].shift(1)
    )

    con_rezago = int(sem["participaciones_semana_anterior"].notna().sum())
    assert con_rezago == 5764, (
        f"se esperaban 5.764 estudiante-semana con rezago definido, hay {con_rezago}"
    )

    sem["Participaciones de la semana"] = sem["participaciones_semana"].apply(participaciones_semana)
    sem["Participaciones de la semana anterior"] = (
        sem["participaciones_semana_anterior"].apply(
            lambda p: None if pd.isna(p) else participaciones_semana(p)
        )
    )
    sem["Número de sesiones de la semana"] = sem["n_sesiones_semana"].apply(bin_nses)
    sem["Sesiones de evaluación de la semana"] = sem["n_eval_semana"].apply(bin_neval)
    return sem


def ensamblar_conjunto(df: pd.DataFrame | None = None, raiz: Path = RAIZ):
    """Ensambla el conjunto discretizado a nivel estudiante-sesión.

    Devuelve `(sesiones, reg, sem)`:
      - `sesiones`: 11.700 filas estudiante-sesión, con las 9 variables no
        supervisadas de la Tabla 13 ya discretizadas, más «materia»
        (segmentación, no nodo) y «tema» (código crudo, sin mapear — la
        categoría baja/media/alta se agrega por pliegue, ver docstring del
        módulo). Sin dropna(): los faltantes de año académico y de la
        primera semana de cada registro quedan explícitos.
      - `reg`: 524 registros estudiante-sección (variables estáticas).
      - `sem`: 6.288 filas estudiante-semana (variables semanales), la
        unidad de referencia del informe, distinta de `sesiones`.
    """
    if df is None:
        df = cargar_datos_crudos(raiz)

    reg = _construir_registros(df)
    sem = _construir_semanas(df)

    sesiones = df[
        ["estudiante_id", "materia", "trimestre", "seccion", "semana", "dia_sesion", "tema", "participaciones"]
    ].copy()
    sesiones = sesiones.merge(
        reg[CLAVE + ["Año que cursa", "Tamaño del grupo", "Posición relativa en la lista",
                     "Cantidad de participaciones del trimestre"]],
        on=CLAVE,
    )
    sesiones = sesiones.merge(
        sem[CLAVE + ["semana", "Participaciones de la semana", "Participaciones de la semana anterior",
                     "Número de sesiones de la semana", "Sesiones de evaluación de la semana"]],
        on=CLAVE + ["semana"],
    )
    sesiones["Sección"] = sesiones["seccion"].astype(str)

    assert len(sesiones) == 11700, (
        f"se esperaban 11.700 filas estudiante-sesión ensambladas, hay {len(sesiones)}"
    )

    return sesiones, reg, sem


if __name__ == "__main__":
    sesiones, reg, sem = ensamblar_conjunto()

    print("=== Verificación de la estructura base ===")
    print(f"filas estudiante-sesión (crudas, antes de filtros): {len(sesiones)}")
    print(f"registros estudiante-sección: {len(reg)}")
    print(f"filas estudiante-semana: {len(sem)}")
    print(f"estudiante-semana con rezago definido: {int(sem['Participaciones de la semana anterior'].notna().sum())}")
    print(f"registros estudiante-sección con año faltante: {int(reg['Año que cursa'].isna().sum())}")
    print()

    print("=== Trazabilidad: conjunto estudiante-sesión ensamblado, por asignatura "
          "(antes de eliminar filas por faltantes) ===")
    for materia in sorted(sesiones["materia"].unique()):
        n = int((sesiones["materia"] == materia).sum())
        print(f"  {materia}: {n} filas estudiante-sesión")

    print()
    print("verificaciones superadas")
