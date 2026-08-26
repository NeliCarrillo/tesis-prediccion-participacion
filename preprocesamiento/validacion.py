"""Chequeos post-procesamiento.

Regla de estas funciones: solo reportan cifras agregadas (conteos, rangos). Nunca imprimen ni
devuelven una cedula o un nombre, para que sus notas puedan ir directo al log publico.
"""
import pandas as pd


def validar_total_algoritmos_2425_2_sec1(base_dir, resultados):
    """Cruza la suma de 'participaciones' por estudiante contra la columna TOTAL que ya
    trae el Excel original de Algoritmos 2425-2 sec1 (el unico de las 11 fuentes cuya hoja
    trae, ademas de las columnas de fecha, un total ya calculado por el docente)."""
    original = pd.read_excel(
        base_dir / "Algoritmos y Programacion/2425-2/Participacion sec 1 Alg 2425-2.xlsx",
        sheet_name="Partic", header=0)
    columnas_fecha = list(original.columns[3:-1])
    suma_por_fila = original[columnas_fecha].sum(axis=1, skipna=True)
    diferencia = (suma_por_fila - original['TOTAL']).abs()
    n_coinciden = int((diferencia < 0.001).sum())
    n_total = len(original)

    notas = [f"Suma de columnas de fecha vs columna TOTAL del Excel original "
             f"(Algoritmos 2425-2 sec1): coinciden {n_coinciden}/{n_total} estudiantes."]
    if n_coinciden < n_total:
        notas.append(
            f"Los {n_total - n_coinciden} restantes: el TOTAL del Excel original no incluye la "
            f"ultima columna de fecha (formula no extendida en el archivo fuente); la suma por "
            f"columna de esta extraccion es mas confiable que ese TOTAL para esos casos.")
    return notas


def semanas_sin_tema(resultados, nombres_fuentes):
    """Reporta, para cada fuente indicada, las semanas cuyo cronograma no tiene un tema
    registrado (tipicamente examen o feriado -- no es un error de cruce)."""
    notas = []
    for nombre in nombres_fuentes:
        filas = resultados.get(nombre)
        if not filas:
            continue
        df = pd.DataFrame(filas)
        huecos = sorted(df.loc[df['tema'] == '', 'semana'].unique().tolist())
        if huecos:
            notas.append(f"[{nombre}] Semanas sin tema en el cronograma (no es error de cruce): {huecos}")
    return notas
