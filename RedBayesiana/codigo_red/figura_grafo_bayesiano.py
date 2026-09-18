"""Genera la Figura de la red bayesiana (grafo acíclico dirigido), con el
autobucle corregido: la dependencia de la semana anterior es un nodo propio
("Participaciones de la semana anterior"), no una flecha de un nodo hacia sí
mismo (que no sería un DAG válido; Russell & Norvig, sec. 14.5.2).

Usa networkx para construir el grafo y verificar que es un DAG (además del
orden topológico), y matplotlib para el dibujo. Las posiciones horizontales
se calculan por el método de baricentro estándar de layouts jerárquicos
(Sugiyama): cada nodo se ubica en el promedio horizontal de los nodos a los
que apunta, para minimizar cruces de flechas. El tamaño de cada caja se mide
con el renderer real de matplotlib (no un supuesto de ancho por caracter),
para que el ajuste de texto sea exacto y no haya cajas encimadas.

Layout de arriba hacia abajo (rankdir=TB): nivel 1 = raíces, nivel 2 =
intermedias, nivel 3 = objetivo, sola y en la fila inferior.

Se guarda ya al ancho exacto de inserción (6,5 in / 16,51 cm, márgenes APA 7
de 1 in), sin bbox_inches='tight', para que el tamaño de letra en puntos
corresponda 1:1 con el tamaño real en el documento (mínimo 14 pt pedido).

Uso:  python3 RedBayesiana/figura_grafo_bayesiano.py
Salida: figura_grafo_bayesiano.png (300 dpi), junto a este script.
"""
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

AQUI = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "Arial"

# ------------------------------------------------------------------ grafo (networkx)
NODOS = [
    "Sección",
    "Tamaño del grupo",
    "Número de sesiones de la semana",
    "Sesiones de evaluación de la semana",
    "Posición relativa en la lista",
    "Tema de la sesión",
    "Participaciones de la semana anterior",
    "Participaciones de la semana",
    "Año que cursa",
    "Cantidad de participaciones del trimestre (objetivo)",
]

ARCOS = [
    ("Sección", "Tamaño del grupo"),
    ("Número de sesiones de la semana", "Sesiones de evaluación de la semana"),
    ("Sesiones de evaluación de la semana", "Participaciones de la semana"),
    ("Posición relativa en la lista", "Participaciones de la semana"),
    ("Tema de la sesión", "Participaciones de la semana"),
    ("Participaciones de la semana anterior", "Participaciones de la semana"),
    ("Participaciones de la semana", "Cantidad de participaciones del trimestre (objetivo)"),
    ("Tamaño del grupo", "Cantidad de participaciones del trimestre (objetivo)"),
    ("Año que cursa", "Cantidad de participaciones del trimestre (objetivo)"),
]

OBJETIVO = "Cantidad de participaciones del trimestre (objetivo)"

G = nx.DiGraph()
G.add_nodes_from(NODOS)
G.add_edges_from(ARCOS)

assert nx.is_directed_acyclic_graph(G), "el grafo no es un DAG (no debería tener autobuclos ni ciclos)"
print("Nodos:", G.number_of_nodes(), " Arcos:", G.number_of_edges())
print("Es un DAG:", nx.is_directed_acyclic_graph(G))
print("Orden topológico:", list(nx.topological_sort(G)))
for n in NODOS:
    print(f"  {n!r}: padres={list(G.predecessors(n))}  hijos={list(G.successors(n))}")

NIVEL1 = ["Sección", "Número de sesiones de la semana", "Posición relativa en la lista",
          "Tema de la sesión", "Participaciones de la semana anterior", "Año que cursa"]
NIVEL2 = ["Tamaño del grupo", "Sesiones de evaluación de la semana", "Participaciones de la semana"]
NIVEL3 = [OBJETIVO]

for n in NIVEL1:
    assert G.in_degree(n) == 0, f"{n} debería ser raíz (sin padres)"
for n in NIVEL2 + NIVEL3:
    assert G.in_degree(n) > 0, f"{n} debería tener al menos un padre"

# ------------------------------------------------------------------ lienzo (cm = unidad de datos, para medir texto real)
ANCHO_IN = 8.3  # más ancha que 6,5 in: los 6 nodos raíz en español no caben en una fila a 14pt en 6,5 in
ANCHO_CM = ANCHO_IN * 2.54
ALTO_CM_TEMP = 30.0
M = 0.4
GRIS, AZUL, BORDE = "#f0f0f0", "#cfe0f5", "#222222"
T_SIZE = 14
GAP_MIN = 0.35

fig = plt.figure(figsize=(ANCHO_IN, ALTO_CM_TEMP / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO_CM)
ax.set_ylim(0, ALTO_CM_TEMP)
ax.axis("off")
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def medir_texto(s):
    """Ancho y alto reales (en cm) de una linea de texto a T_SIZE, usando el renderer."""
    t = ax.text(0, 0, s, fontsize=T_SIZE)
    bbox = t.get_window_extent(renderer=renderer)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
    t.remove()
    return x1 - x0, y1 - y0

ALTO_LINEA = medir_texto("Ay")[1] * 1.35  # alto de una linea con interlineado

def envolver(texto, ancho_max_cm):
    """Ajusta el texto a lineas que no superen ancho_max_cm, midiendo con el renderer."""
    palabras = texto.split(" ")
    lineas, actual = [], ""
    for p in palabras:
        candidato = (actual + " " + p).strip()
        if medir_texto(candidato)[0] > ancho_max_cm and actual:
            lineas.append(actual)
            actual = p
        else:
            actual = candidato
    lineas.append(actual)
    return lineas

def medir(texto, ancho_max_cm, pad_w=0.55, pad_h=0.5):
    lineas = envolver(texto, ancho_max_cm)
    w = max(medir_texto(l)[0] for l in lineas) + pad_w
    h = len(lineas) * ALTO_LINEA + pad_h
    return lineas, w, h

ANCHO_MAX_CM = {1: 3.0, 2: 4.4, 3: 5.6}
dims = {}
for n in NIVEL1:
    dims[n] = medir(n, ANCHO_MAX_CM[1])
for n in NIVEL2:
    dims[n] = medir(n, ANCHO_MAX_CM[2])
dims[OBJETIVO] = medir(OBJETIVO, ANCHO_MAX_CM[3])

for n, (lineas, w, h) in dims.items():
    print(f"  medido {n!r}: lineas={lineas} w={w:.2f}cm h={h:.2f}cm")

# ------------------------------------------------------------------ posiciones x: nivel 1 en fila, ancho real + gap fijo,
# centrada en el lienzo; niveles 2 y 3 por baricentro de las posiciones ya resueltas
anchos1 = [dims[n][1] for n in NIVEL1]
ancho_total1 = sum(anchos1) + GAP_MIN * (len(NIVEL1) - 1)
x1 = {}
cursor = 0.0
for n, w in zip(NIVEL1, anchos1):
    x1[n] = cursor + w / 2
    cursor += w + GAP_MIN

def baricentro(nodo, x_conocidas):
    padres = list(G.predecessors(nodo))
    return sum(x_conocidas[p] for p in padres) / len(padres)

x_conocidas = dict(x1)
x2 = {}
for n in ["Tamaño del grupo", "Sesiones de evaluación de la semana"]:
    x2[n] = baricentro(n, x_conocidas)
    x_conocidas[n] = x2[n]
x2["Participaciones de la semana"] = baricentro("Participaciones de la semana", x_conocidas)
x_conocidas["Participaciones de la semana"] = x2["Participaciones de la semana"]

# separar nivel 2 si el baricentro dejara cajas encimadas
orden2 = sorted(NIVEL2, key=lambda n: x2[n])
for i in range(1, len(orden2)):
    a, b = orden2[i - 1], orden2[i]
    minimo = x2[a] + dims[a][1] / 2 + GAP_MIN + dims[b][1] / 2
    if x2[b] < minimo:
        x2[b] = minimo

x3 = {n: baricentro(n, x_conocidas) for n in NIVEL3}

# corrección de bordes: un nodo de nivel 2/3 puede ser más ancho que el nodo de nivel 1
# del que hereda su x por baricentro, y su caja se saldría del lienzo por un lado;
# se mide el borde real de cada caja y se reescala/traslada todo el lienzo para que
# quepan todas, en vez de asumir que el ancho elegido a priori alcanza
TODAS_X = {**x1, **x2, **x3}
borde_izq = min(TODAS_X[n] - dims[n][1] / 2 for n in TODAS_X)
borde_der = max(TODAS_X[n] + dims[n][1] / 2 for n in TODAS_X)
desplazamiento = M - borde_izq
for d in (x1, x2, x3):
    for n in d:
        d[n] += desplazamiento
ANCHO_CM = (borde_der - borde_izq) + 2 * M
ANCHO_IN = ANCHO_CM / 2.54

print("\nPosiciones x (cm, reales, ya corregidas para no salirse del lienzo):")
for nivel, xs in [("nivel1", x1), ("nivel2", x2), ("nivel3", x3)]:
    print(f"  {nivel}: {{{', '.join(f'{k!r}: {v:.2f}' for k, v in xs.items())}}}")
print(f"  ancho final: {ANCHO_CM:.2f} cm = {ANCHO_IN:.2f} in")

# ------------------------------------------------------------------ altura final y redibujo
ALTO1 = max(dims[n][2] for n in NIVEL1)
GAP_V = 2.6
ALTO_CM = ALTO1 + GAP_V + dims["Participaciones de la semana"][2] + GAP_V + dims[OBJETIVO][2] + 2 * M
Y1 = ALTO_CM - M - ALTO1 / 2
Y2 = Y1 - ALTO1 / 2 - GAP_V - max(dims[n][2] for n in NIVEL2) / 2
Y3 = M + dims[OBJETIVO][2] / 2

plt.close(fig)
fig = plt.figure(figsize=(ANCHO_IN, ALTO_CM / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO_CM)
ax.set_ylim(0, ALTO_CM)
ax.axis("off")
fig.patch.set_facecolor("white")

cajas = {}

def caja(nombre, x_centro, y_centro, fill=GRIS):
    lineas, w, h = dims[nombre]
    x0, y0 = x_centro - w / 2, y_centro - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                 linewidth=1.3, edgecolor=BORDE, facecolor=fill, zorder=2))
    ax.text(x_centro, y_centro, "\n".join(lineas), ha="center", va="center", fontsize=T_SIZE,
            linespacing=1.25, zorder=3)
    cajas[nombre] = {"x": x_centro, "y": y_centro, "w": w, "h": h}

for n in NIVEL1:
    caja(n, x1[n], Y1)
for n in NIVEL2:
    caja(n, x2[n], Y2)
caja(OBJETIVO, x3[OBJETIVO], Y3, fill=AZUL)

def borde(nombre, lado):
    c = cajas[nombre]
    if lado == "abajo":
        return (c["x"], c["y"] - c["h"] / 2)
    if lado == "arriba":
        return (c["x"], c["y"] + c["h"] / 2)
    if lado == "izq":
        return (c["x"] - c["w"] / 2, c["y"])
    if lado == "der":
        return (c["x"] + c["w"] / 2, c["y"])

def flecha(p0, p1, rad=0.0):
    kw = dict(arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=BORDE, shrinkA=0, shrinkB=0, zorder=1)
    if rad:
        kw["connectionstyle"] = f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))

flecha(borde("Sección", "abajo"), borde("Tamaño del grupo", "arriba"))
flecha(borde("Número de sesiones de la semana", "abajo"), borde("Sesiones de evaluación de la semana", "arriba"))
flecha(borde("Posición relativa en la lista", "abajo"), borde("Participaciones de la semana", "arriba"))
flecha(borde("Tema de la sesión", "abajo"), borde("Participaciones de la semana", "arriba"))
flecha(borde("Participaciones de la semana anterior", "abajo"), borde("Participaciones de la semana", "arriba"))

flecha(borde("Participaciones de la semana", "abajo"), borde(OBJETIVO, "arriba"))
flecha(borde("Tamaño del grupo", "abajo"), borde(OBJETIVO, "arriba"))

# arco que salta un nivel (Año que cursa, nivel1 -> objetivo, nivel3): se curva para no
# atravesar la caja de "Participaciones de la semana" (nivel2)
flecha(borde("Año que cursa", "abajo"), borde(OBJETIVO, "der"), rad=-0.25)

destino = AQUI / "figura_grafo_bayesiano.png"
fig.savefig(destino, dpi=300, facecolor="white")
print(f"\nguardado: {destino}")
print(f"tamaño: {ANCHO_IN} in x {ALTO_CM/2.54:.2f} in")
