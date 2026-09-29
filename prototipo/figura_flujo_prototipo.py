"""Figura 13 del informe: flujo de selección, inferencia y visualización del
prototipo, tal como está implementado en `app.py`.

Ancho de 6,5 pulgadas y Times New Roman de 9 pt, para insertarse a ancho de
página sin que el texto quede por debajo de ese tamaño. Sin título interno:
el título va en el caption del informe.

Correr el archivo genera `figura_flujo_prototipo.png` en esta carpeta.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.patches import Polygon, Rectangle

RUTA_SALIDA = Path(__file__).resolve().parent / "figura_flujo_prototipo.png"
CARPETA_FUENTES = Path("/System/Library/Fonts/Supplemental")

ANCHO = 6.5
TAMANO = 9
INTERLINEA = TAMANO * 1.25 / 72
RELLENO = 0.07
SEP = 0.13
GROSOR = 0.8

for nombre in ("Times New Roman.ttf", "Times New Roman Bold.ttf"):
    if (CARPETA_FUENTES / nombre).exists():
        font_manager.fontManager.addfont(str(CARPETA_FUENTES / nombre))
plt.rcParams.update({"font.family": "Times New Roman", "font.size": TAMANO})

figura = plt.figure(figsize=(ANCHO, 12))
eje = figura.add_axes((0, 0, 1, 1))
eje.axis("off")


_MEDIDOR = TextToPath()


def _ancho_pulgadas(texto: str, negrita: bool = False) -> float:
    fuente = FontProperties(family="Times New Roman", size=TAMANO, weight="bold" if negrita else "normal")
    ancho, _, _ = _MEDIDOR.get_text_width_height_descent(texto, fuente, ismath=False)
    return ancho / 72


def _envolver(texto: str, ancho: float, negrita: bool = False) -> list[str]:
    """Ajuste de línea midiendo el ancho real de cada renglón en la fuente."""
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = f"{actual} {palabra}".strip()
        if actual and _ancho_pulgadas(prueba, negrita) > ancho:
            lineas.append(actual)
            actual = palabra
        else:
            actual = prueba
    return lineas + [actual]


def caja(x: float, y_arriba: float, ancho: float, texto: str, *,
         encabezado: str | None = None, discontinua: bool = False) -> dict:
    """Caja centrada en x con el borde superior en y_arriba. `encabezado` se
    escribe en negrita en su propia línea, encima del texto."""
    lineas_cabeza = _envolver(encabezado, ancho - 0.16, negrita=True) if encabezado else []
    lineas = _envolver(texto, ancho - 0.16)
    n = len(lineas_cabeza) + len(lineas)
    alto = n * INTERLINEA + 2 * RELLENO
    y_abajo = y_arriba - alto
    eje.add_patch(Rectangle((x - ancho / 2, y_abajo), ancho, alto, facecolor="white",
                            edgecolor="black", linewidth=GROSOR,
                            linestyle=(0, (4, 2)) if discontinua else "-"))
    y = y_arriba - RELLENO - INTERLINEA / 2
    for linea in lineas_cabeza:
        eje.text(x, y, linea, ha="center", va="center", fontweight="bold")
        y -= INTERLINEA
    for linea in lineas:
        eje.text(x, y, linea, ha="center", va="center")
        y -= INTERLINEA
    return {"x": x, "arriba": y_arriba, "abajo": y_abajo,
            "izq": x - ancho / 2, "der": x + ancho / 2, "medio": (y_arriba + y_abajo) / 2}


def rombo(x: float, y_arriba: float, ancho: float, alto: float, texto: str) -> dict:
    y_c = y_arriba - alto / 2
    eje.add_patch(Polygon([(x, y_arriba), (x + ancho / 2, y_c), (x, y_arriba - alto), (x - ancho / 2, y_c)],
                          closed=True, facecolor="white", edgecolor="black", linewidth=GROSOR))
    eje.text(x, y_c, "\n".join(_envolver(texto, ancho * 0.5)), ha="center", va="center",
             linespacing=1.25)
    return {"x": x, "arriba": y_arriba, "abajo": y_arriba - alto}


def linea(*puntos) -> None:
    xs, ys = zip(*puntos)
    eje.plot(xs, ys, color="black", linewidth=GROSOR, solid_capstyle="butt")


def flecha(*puntos) -> None:
    """Polilínea ortogonal con punta de flecha al final."""
    if len(puntos) > 2:
        linea(*puntos[:-1])
    eje.annotate("", xy=puntos[-1], xytext=puntos[-2],
                 arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                 color="black", linewidth=GROSOR, shrinkA=0, shrinkB=0))


def bifurcar(x_origen: float, y_origen: float, destinos: list[tuple[float, float]]) -> None:
    """Desde (x_origen, y_origen) baja, se abre en una barra horizontal y
    llega con flecha a cada destino (x, y_arriba)."""
    y_barra = y_origen - 0.14
    linea((x_origen, y_origen), (x_origen, y_barra))
    xs = [d[0] for d in destinos]
    linea((min(xs + [x_origen]), y_barra), (max(xs + [x_origen]), y_barra))
    for x, y in destinos:
        flecha((x, y_barra), (x, y))


X_IZQ, W_IZQ = 1.30, 2.40
X_DER, W_DER = 4.55, 3.50
X_CENTRO = 3.25
W_TOTAL = 6.10

# 1. Tipo de caso
tipo = rombo(X_CENTRO, 11.9, 1.6, 0.42, "Tipo de caso")
y_ramas = tipo["abajo"] - 0.14 - 0.15

# 2. Estudiante ya registrado
l1 = caja(X_IZQ, y_ramas, W_IZQ, "El usuario selecciona asignatura, trimestre, sección, "
          "estudiante anonimizado e hito (S4, S6 o S8).", encabezado="Estudiante ya registrado")
l2 = caja(X_IZQ, l1["abajo"] - SEP, W_IZQ, "El sistema construye el caso con la "
          "información registrada del estudiante.")
flecha((X_IZQ, l1["abajo"]), (X_IZQ, l2["arriba"]))

# 3. Estudiante nuevo
cab_nuevo = rombo(X_DER, y_ramas, 3.1, 0.55, "Estudiante nuevo (hipotético): ¿cómo se crea?")
bifurcar(X_CENTRO, tipo["abajo"], [(X_IZQ, l1["arriba"]), (X_DER, cab_nuevo["arriba"])])

X_REAL, W_REAL = 3.76, 1.92
X_CERO, W_CERO = 5.56, 1.48
y_sub = cab_nuevo["abajo"] - 0.14 - 0.15
real = caja(X_REAL, y_sub, W_REAL, "Elige asignatura, trimestre, sección, un estudiante "
            "real e hito; el sistema precarga sus valores, que quedan editables.",
            encabezado="Partir de un estudiante real")
cero = caja(X_CERO, y_sub, W_CERO, "Elige asignatura, trimestre, sección e hito "
            "(S4, S6 o S8).", encabezado="Ingresar desde cero")
bifurcar(X_DER, cab_nuevo["abajo"], [(X_REAL, real["arriba"]), (X_CERO, cero["arriba"])])

seccion = caja(X_DER, min(real["abajo"], cero["abajo"]) - SEP, W_DER, "El sistema toma de la "
               "sección real el tamaño del grupo y, para cada semana hasta el hito, las "
               "sesiones, las sesiones de evaluación y el tema.")
flecha((X_REAL, real["abajo"]), (X_REAL, seccion["arriba"]))
flecha((X_CERO, cero["abajo"]), (X_CERO, seccion["arriba"]))
ingreso = caja(X_DER, seccion["abajo"] - SEP, W_DER, "El usuario ingresa o ajusta el año que "
               "cursa, la posición relativa en la lista y las participaciones de cada semana "
               "hasta el hito (0 si no participó o no hubo clase). Cualquiera de estos datos "
               "puede quedar sin dato.")
flecha((X_DER, seccion["abajo"]), (X_DER, ingreso["arriba"]))

# 4. Tronco común
boton = caja(X_CENTRO, ingreso["abajo"] - SEP - 0.05, W_TOTAL,
             "El usuario presiona el botón para generar la predicción.")
flecha((X_IZQ, l2["abajo"]), (X_IZQ, boton["arriba"]))
flecha((X_DER, ingreso["abajo"]), (X_DER, boton["arriba"]))

X_ENVIO, W_ENVIO = 2.15, 3.30
envio = caja(X_ENVIO, boton["abajo"] - SEP, W_ENVIO, "El mismo caso se envía a los dos modelos.")
flecha((X_ENVIO, boton["abajo"]), (X_ENVIO, envio["arriba"]))
resumen = caja(5.40, envio["arriba"] + 0.03, 2.00, "Solo estudiante nuevo: resumen de las "
               "variables del caso.", discontinua=True)
flecha((envio["der"], envio["medio"]), (resumen["izq"], envio["medio"]))

# 5. Modelos
y_modelos = min(envio["abajo"], resumen["abajo"]) - 0.14 - 0.15
lstm = caja(X_IZQ, y_modelos, W_IZQ, "Predicción puntual del total de participaciones "
            "del trimestre. Los datos sin dato se imputan con la mediana del histórico de la "
            "asignatura.", encabezado="Red LSTM")
b1 = caja(X_DER, y_modelos, W_DER, "Predicción continua del total de participaciones "
          "del trimestre.", encabezado="Red bayesiana")
bifurcar(X_ENVIO, envio["abajo"], [(X_IZQ, lstm["arriba"]), (X_DER, b1["arriba"])])
b2 = caja(X_DER, b1["abajo"] - SEP, W_DER, "Distribución posterior sobre los cinco estados "
          "del objetivo, con la evidencia utilizada y la no disponible, que la red "
          "marginaliza sin imputarla.")
b4 = caja(X_DER, b2["abajo"] - SEP, W_DER, "Simulador ¿Qué pasaría si...?: el usuario cambia "
          "una o varias variables de evidencia, o las marca como sin dato, y la predicción "
          "se recalcula con el mismo modelo.")
for a, b in ((b1, b2), (b2, b4)):
    flecha((X_DER, a["abajo"]), (X_DER, b["arriba"]))

# 6. Salidas
margen = caja(X_CENTRO, b4["abajo"] - SEP - 0.05, W_TOTAL, "En ambos modelos, la interfaz "
              "muestra el margen de error esperado para predicciones de esa magnitud.")
flecha((X_IZQ, lstm["abajo"]), (X_IZQ, margen["arriba"]))
flecha((X_DER, b4["abajo"]), (X_DER, margen["arriba"]))
total = caja(X_CENTRO, margen["abajo"] - SEP, W_TOTAL, "Solo para un estudiante ya registrado: "
             "la interfaz muestra además el total real de participaciones del trimestre.")
flecha((X_CENTRO, margen["abajo"]), (X_CENTRO, total["arriba"]))
final = caja(X_CENTRO, total["abajo"] - SEP, W_TOTAL, "Presentación diferenciada de los "
             "resultados de ambos modelos en la interfaz.")
flecha((X_CENTRO, total["abajo"]), (X_CENTRO, final["arriba"]))

# Lienzo ajustado al contenido, con 1 unidad = 1 pulgada.
y_min, y_max = final["abajo"] - 0.06, tipo["arriba"] + 0.06
eje.set_xlim(0, ANCHO)
eje.set_ylim(y_min, y_max)
figura.set_size_inches(ANCHO, y_max - y_min)
figura.savefig(RUTA_SALIDA, dpi=300, facecolor="white")
print(f"Figura guardada en: {RUTA_SALIDA}  ({ANCHO} x {y_max - y_min:.2f} pulgadas)")
