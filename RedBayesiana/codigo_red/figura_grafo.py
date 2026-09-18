"""Genera la Figura 7 del informe: estructura de dependencias del grafo (Sprint 3,
carta 4), reemplazando la Tabla 14. Mismo criterio de tamaño físico que la Figura 6
(figura_arquitectura.py): se guarda ya al ancho exacto de inserción (6,5 in / 16,51 cm,
márgenes APA 7 de 1 in), para que el tamaño de letra en puntos de matplotlib
corresponda 1:1 con el tamaño real en el documento.

Uso:  python3 RedBayesiana/figura_grafo.py
Salida: figura_grafo.png (300 dpi) y .pdf, junto a este script.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

plt.rcParams["font.family"] = "Arial"

ANCHO_IN = 6.5
ANCHO_CM = ANCHO_IN * 2.54  # 16.51 cm
ALTO_CM = 15.5
M = 0.6

GRIS, AZUL, BORDE = "#f0f0f0", "#dbe8f7", "#222222"
T_SIZE = 13

fig = plt.figure(figsize=(ANCHO_IN, ALTO_CM / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO_CM)
ax.set_ylim(0, ALTO_CM)
ax.axis("off")

cajas = {}

def caja(nombre, x, y, w, h, texto, fill=GRIS):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                 linewidth=1.2, edgecolor=BORDE, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=T_SIZE,
            linespacing=1.3)
    cajas[nombre] = {"x": x, "y": y, "w": w, "h": h}

def punto(nombre, lado, frac=0.5):
    c = cajas[nombre]
    if lado == "der":
        return (c["x"] + c["w"], c["y"] + c["h"] * frac)
    if lado == "izq":
        return (c["x"], c["y"] + c["h"] * frac)
    if lado == "abajo":
        return (c["x"] + c["w"] * frac, c["y"])
    if lado == "arriba":
        return (c["x"] + c["w"] * frac, c["y"] + c["h"])

def flecha(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.3, color=BORDE, shrinkA=0, shrinkB=0))

def flecha_curva(p0, p1, rad):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.3, color=BORDE, shrinkA=0, shrinkB=0,
                                  connectionstyle=f"arc3,rad={rad}"))

# ------------------------------------------------------------------ columnas y filas
W1, W2, W3 = 4.6, 4.6, 4.3
X1 = M
X2 = X1 + W1 + 0.7
X3 = X2 + W2 + 0.9
H = 1.9
GAP = 1.2

y_fila1 = ALTO_CM - M - H
y_fila2 = y_fila1 - H - GAP
y_fila3 = y_fila2 - H - GAP
y_fila4 = y_fila3 - H - GAP
y_fila5 = y_fila4 - H - GAP

caja("seccion", X1, y_fila1, W1, H, "Sección")
caja("tamano", X2, y_fila1, W2, H, "Tamaño del grupo")

caja("nsesiones", X1, y_fila2, W1, H, "Número de sesiones\nde la semana")
caja("evaluacion", X2, y_fila2, W2, H, "Sesiones de evaluación\nde la semana")

caja("posicion", X1, y_fila3, W1, H, "Posición relativa\nen la lista")
caja("partsemana", X2, y_fila3, W2, H, "Participaciones\nde la semana")
caja("objetivo", X3, y_fila3, W3, H, "Cantidad de\nparticipaciones\ndel trimestre\n(objetivo)", fill=AZUL)

caja("anio", X1, y_fila4, W1, H, "Año que cursa")
caja("tema", X1, y_fila5, W1, H, "Tema de la sesión")

flecha(punto("seccion", "der"), punto("tamano", "izq"))
flecha(punto("nsesiones", "der"), punto("evaluacion", "izq"))
# evaluación -> participaciones: entra por el lado izquierdo del borde superior,
# para dejar el lado derecho libre para el autobucle
flecha(punto("evaluacion", "abajo", frac=0.3), punto("partsemana", "arriba", frac=0.3))
flecha(punto("posicion", "der"), punto("partsemana", "izq"))
flecha(punto("partsemana", "der"), punto("objetivo", "izq"))

# año que cursa -> objetivo: recorre en codo por el carril vacío de su propia fila
# (las columnas 2 y 3 no tienen cajas a la altura de "año que cursa"), sin cruzar ninguna caja
p0 = punto("anio", "der")
esquina = (cajas["objetivo"]["x"] + cajas["objetivo"]["w"] / 2, p0[1])
p1 = punto("objetivo", "abajo", frac=0.5)
ax.plot([p0[0], esquina[0]], [p0[1], esquina[1]], color=BORDE, linewidth=1.3, zorder=1)
flecha(esquina, p1)

# tamaño del grupo -> objetivo: recorre en codo por el carril vacío de su propia fila
p0 = punto("tamano", "der")
esquina = (cajas["objetivo"]["x"] + cajas["objetivo"]["w"] / 2, p0[1])
p1 = punto("objetivo", "arriba", frac=0.5)
ax.plot([p0[0], esquina[0]], [p0[1], esquina[1]], color=BORDE, linewidth=1.3, zorder=1)
flecha(esquina, p1)

# tema de la sesión -> participaciones de la semana: recorre en codo por el carril
# vacío de su propia fila y cruza (sin tocar cajas) la línea de año que cursa
p0 = punto("tema", "der")
esquina = (cajas["partsemana"]["x"] + cajas["partsemana"]["w"] * 0.7, p0[1])
p1 = punto("partsemana", "abajo", frac=0.7)
ax.plot([p0[0], esquina[0]], [p0[1], esquina[1]], color=BORDE, linewidth=1.3, zorder=1)
flecha(esquina, p1)

# autobucle "semana anterior": sale y entra por el lado derecho del borde superior,
# lejos de la flecha de evaluación (izquierda) y de la salida hacia el objetivo (medio-derecha)
c = cajas["partsemana"]
x0 = c["x"] + c["w"] * 0.62
x1 = c["x"] + c["w"] * 0.92
y_top = c["y"] + c["h"]
centro_x = (x0 + x1) / 2
radio_x = (x1 - x0) / 2
arco = Arc((centro_x, y_top), radio_x * 2, 1.5, angle=0, theta1=20, theta2=160,
           linewidth=1.3, color=BORDE)
ax.add_patch(arco)
flecha((x0 + 0.04, y_top + 0.06), (x0, y_top))
ax.text(centro_x, y_top + 0.85, "semana anterior", ha="center", va="bottom",
        fontsize=T_SIZE - 3)

aqui = Path(__file__).resolve().parent
fig.savefig(aqui / "figura_grafo.png", dpi=300)
fig.savefig(aqui / "figura_grafo.pdf")
print("ok")
print(f"tamaño guardado: {ANCHO_IN} in x {ALTO_CM/2.54:.2f} in  ({ANCHO_CM:.2f} x {ALTO_CM:.2f} cm)")
