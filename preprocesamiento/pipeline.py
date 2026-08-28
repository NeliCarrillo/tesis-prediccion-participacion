"""Orquesta la Fase 2 completa: cronogramas -> extraccion -> anonimizacion -> CSV + logs -> validacion.

Se puede correr de dos formas equivalentes:
  - Como script:  python3 pipeline.py          (desde la carpeta preprocesamiento/)
  - Importado:    from pipeline import ejecutar  (usado por el notebook 01_estandarizacion_y_anonimizacion.ipynb)
"""
from pathlib import Path

import pandas as pd

from modulos.entrada.cronogramas import cargar_cronogramas
from modulos.procesamiento.fuentes import procesar_todas_las_fuentes
from modulos.salida.anonimizacion import construir_mapa, aplicar_mapa
from modulos.salida.validacion import validar_total_algoritmos_2425_2_sec1, semanas_sin_tema

COLUMNAS_SALIDA = [
    'estudiante_id', 'numero_lista', 'materia', 'trimestre', 'seccion',
    'fecha', 'semana', 'tema', 'participaciones', 'tipo_participacion', 'asistencia',
]

FUENTES_A_REVISAR_HUECOS_DE_TEMA = ['algoritmos_2526-1', 'computacion_emergente_2526-2']


def ejecutar(directorio_notebook=None):
    """Corre el pipeline completo y devuelve un resumen (ver claves al final de esta funcion).

    directorio_notebook: carpeta desde la que se calculan las rutas relativas (por defecto,
    el directorio de trabajo actual). Al abrir el notebook con Jupyter/VS Code, el directorio
    de trabajo ya es 'preprocesamiento/', asi que no hace falta pasar nada.
    """
    directorio_notebook = Path(directorio_notebook) if directorio_notebook else Path.cwd()
    raiz_repo = directorio_notebook.parent
    base_dir = raiz_repo / "Datos Tesis Upstream"
    out_dir = base_dir / "_procesado"
    conf_dir = out_dir / "_confidencial"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"No se encontro la carpeta de datos crudos en {base_dir}. "
            f"Corre este pipeline desde la carpeta 'preprocesamiento/' del repositorio.")

    cronogramas = cargar_cronogramas(base_dir)
    resultados, log_publico, log_confidencial = procesar_todas_las_fuentes(base_dir, cronogramas)

    cedulas = {fila['cedula_raw'] for filas in resultados.values() for fila in filas}
    mapa = construir_mapa(cedulas)

    log_publico.append("")
    log_publico.append("=== Mapeo de anonimizacion ===")
    log_publico.append(f"Total de estudiantes unicos detectados en las {len(resultados)} fuentes: {len(mapa)}.")
    log_publico.append(
        "IDs anon_001..anon_%03d asignados ordenando cedulas de forma ascendente (deterministico); "
        "el mismo estudiante recibe el mismo estudiante_id en todos los CSV donde aparezca." % len(mapa))

    pd.DataFrame({
        'cedula': sorted(mapa, key=int),
        'estudiante_id': [mapa[c] for c in sorted(mapa, key=int)],
    }).to_csv(conf_dir / "mapeo_estudiantes.csv", index=False)

    archivos_generados = []
    for nombre, filas in resultados.items():
        filas_anon = aplicar_mapa(filas, mapa)
        if not filas_anon:
            log_publico.append(f"[{nombre}] ADVERTENCIA: no se genero ninguna fila.")
            continue
        df = pd.DataFrame(filas_anon)[COLUMNAS_SALIDA]
        ruta_csv = out_dir / f"{nombre}_participaciones.csv"
        df.to_csv(ruta_csv, index=False)
        archivos_generados.append((nombre, ruta_csv, len(df)))

    log_publico.append("")
    log_publico.append("=== Archivos CSV generados ===")
    for nombre, ruta_csv, n_filas in archivos_generados:
        log_publico.append(f"{ruta_csv.name}: {n_filas} filas")

    log_publico.append("")
    log_publico.append("=== Validacion post-procesamiento ===")
    log_publico.extend(validar_total_algoritmos_2425_2_sec1(base_dir, resultados))
    log_publico.extend(semanas_sin_tema(resultados, FUENTES_A_REVISAR_HUECOS_DE_TEMA))

    (out_dir / "log_limpieza.txt").write_text("\n".join(log_publico), encoding="utf-8")
    (conf_dir / "log_limpieza_detalle.txt").write_text(
        "\n".join(log_confidencial) if log_confidencial else "(sin observaciones confidenciales)",
        encoding="utf-8")

    return {
        'resultados': resultados,
        'archivos_generados': archivos_generados,
        'n_estudiantes': len(mapa),
        'out_dir': out_dir,
        'conf_dir': conf_dir,
        'log_publico': log_publico,
    }


if __name__ == "__main__":
    resumen = ejecutar()
    print(f"{len(resumen['archivos_generados'])} archivos CSV generados en {resumen['out_dir']}")
    print(f"{resumen['n_estudiantes']} estudiantes unicos anonimizados")
    for nombre, ruta_csv, n_filas in resumen['archivos_generados']:
        print(f"  {ruta_csv.name}: {n_filas} filas")
