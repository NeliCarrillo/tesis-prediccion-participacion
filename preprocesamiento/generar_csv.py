#!/usr/bin/env python3
"""Convierte las hojas «Estandar» y «Cronograma» de cada archivo de participaciones
en un CSV por sección, siguiendo la estructura estándar definida en el informe.

Uso:   python3 preprocesamiento/generar_csv.py
Salida: un .csv por sección en «Datos Tesis Downstream», con la misma estructura
de carpetas que «Datos Tesis Upstream».
"""
from __future__ import annotations
import csv, math, re, unicodedata, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
UPSTREAM = RAIZ / "Datos Tesis Upstream"
DOWNSTREAM = RAIZ / "Datos Tesis Downstream"
MAPEO = UPSTREAM / "_procesado" / "_confidencial" / "mapeo_estudiantes.csv"
CATALOGO = RAIZ / "CATALOGO_Temas.xlsx"

COLUMNAS = ["numero_lista", "estudiante_id", "materia", "trimestre", "seccion",
            "semana", "dia_sesion", "tema", "tipo_sesion",
            "participaciones", "asistencia", "anio_academico"]

DIAS = ("lunes", "martes", "miercoles", "miércoles", "jueves")
CODIGOS_PRESENTE = {"P", "p", "T", "F", "J"}      # presente sin participar
CODIGOS_AUSENTE = {"⚕️", "⚖️"}                     # ausencia justificada


# --------------------------------------------------------------------- lectura
def _leer_xml(path: Path, hoja: str):
    """Respaldo para los archivos que openpyxl no abre (sheetId inválido)."""
    with zipfile.ZipFile(path) as z:
        wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
        rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8", "replace")
        destino = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
        ruta = None
        for m in re.finditer(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
            if m.group(1).strip().lower() == hoja.lower():
                t = destino.get(m.group(2), "").lstrip("/")
                ruta = t if t.startswith("xl/") else "xl/" + t
        if ruta is None:
            return None
        textos = []
        if "xl/sharedStrings.xml" in z.namelist():
            sx = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
            for si in re.findall(r"<si>(.*?)</si>", sx, re.S):
                textos.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
        xml = z.read(ruta).decode("utf-8", "replace")

    celdas = {}
    # las celdas autocerradas <c .../> se consumen aparte: de lo contrario la
    # expresión se traga la celda siguiente y desplaza la fila una columna
    for m in re.finditer(r"<c\b([^>]*?)(?:/>|>(.*?)</c>)", xml, re.S):
        attrs, dentro = m.group(1), m.group(2) or ""
        ref = re.search(r'\br="([A-Z]+)(\d+)"', attrs)
        if not ref:
            continue
        col = 0
        for ch in ref.group(1):
            col = col * 26 + ord(ch) - 64
        v = re.search(r"<v>(.*?)</v>", dentro, re.S)
        t = re.search(r"<t[^>]*>(.*?)</t>", dentro, re.S)
        if 't="s"' in attrs and v:
            i = int(v.group(1)); valor = textos[i] if i < len(textos) else None
        elif t:
            valor = t.group(1)
        elif v:
            valor = v.group(1)
        else:
            continue
        celdas[(int(ref.group(2)), col)] = valor
    if not celdas:
        return None
    filas = max(f for f, _ in celdas); cols = max(c for _, c in celdas)
    return pd.DataFrame([[celdas.get((f, c)) for c in range(1, cols + 1)]
                         for f in range(1, filas + 1)])


def leer(path: Path, hoja: str):
    try:
        return pd.read_excel(path, sheet_name=hoja, header=None)
    except Exception:
        return _leer_xml(path, hoja)


def sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def canonicas() -> dict:
    """Nombres canónicos de las asignaturas, tomados de las hojas del catálogo."""
    hojas = [h for h in pd.ExcelFile(CATALOGO).sheet_names if h.lower() != "tipos de sesion"]
    canon = {}
    for h in hojas:
        clave = sin_tildes(h).replace("estructuras", "estructura")
        canon[clave] = h
    return canon


def normalizar_materia(nombre: str, canon: dict) -> str:
    clave = sin_tildes(nombre).replace("estructuras", "estructura")
    return canon.get(clave, nombre)


def texto(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


# ------------------------------------------------------------------ conversión
def interpretar(valor: str, con_asistencia: bool):
    """Traduce una celda de la rejilla a (participaciones, asistencia)."""
    if not valor:
        return 0, (0 if con_asistencia else "")
    if valor in CODIGOS_PRESENTE:
        return 0, 1
    if valor in CODIGOS_AUSENTE:
        return 0, 0
    try:
        # las medias participaciones se redondean hacia arriba
        return math.ceil(float(valor.replace(",", "."))), 1
    except ValueError:
        return "", ""


def anio_academico(carnet: str, trimestre: str):
    digitos = re.sub(r"\D", "", carnet)
    m = re.match(r"(\d{2})(\d{2})-?(\d)", trimestre.strip())
    if len(digitos) < 4 or not m:
        return ""
    ingreso = int(digitos[:4])
    calendario = 2000 + int(m.group(1) if m.group(3) == "1" else m.group(2))
    return calendario - ingreso + 1


def leer_cronograma(path: Path):
    df = leer(path, "Cronograma")
    sesiones = {}
    for i in range(1, len(df)):
        fila = df.iloc[i].tolist()
        if not texto(fila[0]).isdigit():
            continue
        clave = (int(float(fila[0])), int(float(fila[1])))
        tema = texto(fila[2]) if len(fila) > 2 else ""
        tipo = texto(fila[3]) if len(fila) > 3 else ""
        sesiones[clave] = (tema, tipo or ("contenido" if tema not in ("", "0") else ""))
    return sesiones


def procesar(path: Path, anonimo, canon: dict, avisos: list):
    est = leer(path, "Estandar")
    if est is None:
        avisos.append(f"{path.name}: no se pudo leer la hoja «Estandar»")
        return []
    cronograma = leer_cronograma(path)

    def meta(fila):
        for v in est.iloc[fila].tolist()[1:]:
            if texto(v):
                return texto(v)
        return ""

    materia, trimestre, seccion = normalizar_materia(meta(0), canon), meta(1), meta(2)
    if materia != meta(0):
        avisos.append(f"{path.name}: asignatura {meta(0)!r} normalizada a {materia!r}")
    encabezado = [texto(v) for v in est.iloc[4].tolist()]
    dias = [texto(v).lower() for v in est.iloc[5].tolist()]

    # cada semana ocupa dos subcolumnas; son sesiones reales las que llevan día
    columnas = {}
    for j, v in enumerate(encabezado):
        m = re.match(r"Semana (\d+)$", v)
        if m:
            usadas = [k for k in (j, j + 1) if k < len(dias) and dias[k] in DIAS]
            for n, k in enumerate(usadas, 1):
                columnas[(int(m.group(1)), n)] = (k, sin_tildes(dias[k]))

    rejilla = [texto(v) for i in range(6, len(est)) for v in est.iloc[i].tolist()[5:]]
    con_asistencia = any(v in CODIGOS_PRESENTE for v in rejilla)

    filas = []
    for i in range(6, len(est)):
        datos = est.iloc[i].tolist()
        cedula = re.sub(r"\D", "", texto(datos[3]))
        if not cedula:
            continue
        anio = anio_academico(texto(datos[4]), trimestre)
        for (semana, sesion), (col, dia) in sorted(columnas.items()):
            tema, tipo = cronograma.get((semana, sesion), ("", ""))
            valor = texto(datos[col]) if col < len(datos) else ""
            if tipo == "sin_clase":
                # la sesión no se dictó: no hubo oportunidad de participar
                participaciones, asistencia = "", ""
                if valor:
                    avisos.append(f"{path.name}: sem {semana} ses {sesion} marcada sin clase "
                                  f"pero la rejilla trae {valor!r}")
            else:
                participaciones, asistencia = interpretar(valor, con_asistencia)
                if participaciones == "" and valor:
                    avisos.append(f"{path.name}: código no reconocido {valor!r} "
                                  f"(sem {semana}, ses {sesion})")
            filas.append({
                "numero_lista": texto(datos[0]), "estudiante_id": anonimo(cedula),
                "materia": materia, "trimestre": trimestre, "seccion": seccion,
                "semana": semana, "dia_sesion": dia, "tema": tema, "tipo_sesion": tipo,
                "participaciones": participaciones, "asistencia": asistencia,
                "anio_academico": anio,
            })
    return filas


# ------------------------------------------------------------------ ejecución
def main():
    MAPEO.parent.mkdir(parents=True, exist_ok=True)
    mapa = {}
    if MAPEO.exists():
        mapa = {r["cedula"]: r["estudiante_id"] for r in csv.DictReader(MAPEO.open(encoding="utf-8"))}

    def anonimo(cedula):
        if cedula not in mapa:
            mapa[cedula] = f"anon_{len(mapa) + 1:03d}"
        return mapa[cedula]

    canon = canonicas()
    archivos = sorted(p for p in UPSTREAM.rglob("*.xlsx")
                      if "cronograma" not in p.name.lower() and not p.name.startswith("~$"))
    avisos, total, generados = [], 0, set()
    for path in archivos:
        filas = procesar(path, anonimo, canon, avisos)
        if not filas:
            continue
        destino = DOWNSTREAM / path.relative_to(UPSTREAM).parent / (path.stem + ".csv")
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS)
            w.writeheader(); w.writerows(filas)
        total += len(filas)
        generados.add(destino.resolve())
        print(f"  {destino.relative_to(DOWNSTREAM)}  ->  {len(filas)} filas")

    # los CSV que ya no corresponden a ningún archivo de origen se eliminan, de modo
    # que la carpeta refleje siempre el estado actual y no acumule versiones viejas
    sobrantes = [c for c in DOWNSTREAM.rglob("*.csv") if c.resolve() not in generados]
    for c in sobrantes:
        c.unlink()
        print(f"  eliminado (ya no tiene origen): {c.relative_to(DOWNSTREAM)}")

    with MAPEO.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cedula", "estudiante_id"]); w.writerows(sorted(mapa.items()))

    print(f"\n{len(archivos)} secciones, {total} filas, {len(mapa)} estudiantes distintos"
          + (f", {len(sobrantes)} csv eliminado(s)" if sobrantes else ""))
    if avisos:
        print(f"\n{len(avisos)} aviso(s):")
        for a in avisos:
            print("  - " + a)


if __name__ == "__main__":
    main()
