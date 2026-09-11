"""Genera la Figura de arquitectura del informe: red LSTM adaptada.

Diseñada para insertarse a exactamente 16,51 cm (6,5 in) de ancho, el ancho de
texto de una página carta con márgenes APA 7 de 1 in. A ese ancho, el texto
más pequeño de la figura queda en 8,5 pt Arial, por encima del mínimo de 8 pt.

Uso:  python3 LSTM/nuevo/figura_3_arquitectura.py
Salida: figura_arquitectura.png (300 dpi) y .pdf, junto a este script.

IMPORTANTE al insertarla: en Google Docs, clic derecho sobre la imagen ->
"Opciones de imagen" -> fijar el ancho en 16,51 cm (6,50 in) exactos. Si se
inserta a un ancho distinto, el tamaño de letra ya no corresponde a lo
calculado aquí.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Arial"

N_CODIGOS = 14
PARAMETROS = 3_949
PARAMETROS_ORIGINAL = 52_228

ANCHO_IN = 6.5
ANCHO_CM = ANCHO_IN * 2.54           # 16.51 cm; 1 unidad de dato = 1 cm
M = 0.6                              # margen lateral

GRIS, AZUL, BORDE = "#f2f2f2", "#e4e9f3", "#222222"

# ------------------------------------------------------------------ construcción "desde arriba"
# cursor: distancia acumulada desde el borde superior (crece hacia abajo)
cursor = 0.0
elementos = []   # (tipo, datos) en el orden en que se dibujan

def avanzar(alto):
    global cursor
    cursor += alto

def texto_libre(alto, dibujar):
    top = cursor; avanzar(alto)
    elementos.append(("texto", top, dibujar))

def caja(alto, w, x, titulo, sub=None, fill=GRIS, t_size=9.5, s_size=8.5, ancla=None, top=None):
    """Si `top` no se indica, la caja ocupa el cursor actual y lo avanza (caso normal,
    una caja por fila). Si se indica `top`, se usa ese techo sin tocar el cursor: así
    dos o más cajas de una misma fila comparten techo sin depender de corregir el
    cursor después."""
    global cursor
    if top is None:
        top = cursor
        cursor = top + alto
    elementos.append(("caja", top, alto, x, w, titulo, sub, fill, t_size, s_size))
    if ancla is not None: ancla["top"], ancla["bottom"], ancla["x"], ancla["w"] = top, top + alto, x, w
    return top, top + alto

# título
avanzar(0.5)
texto_libre(0.75, lambda ax, y: ax.text(M, y, "Arquitectura de la red LSTM adaptada",
            fontsize=12.5, fontweight="bold", va="top"))
texto_libre(1.15, lambda ax, y: ax.text(M, y, "Dos entradas concatenadas por semana, dos capas "
            "LSTM con regularización\ny una salida continua.", fontsize=8.5, va="top",
            color="#333333", linespacing=1.35))
avanzar(0.2)

ancho_total = ANCHO_CM - 2 * M
ANCHO_FILA = ancho_total - 1.8   # deja un carril de 1.8 cm libre a la derecha
ent_num, ent_tema, emb, resh, cat, l1, d1, l2, d2, dens, sal = ({} for _ in range(11))

caja(2.1, ANCHO_FILA, M, "Entrada numérica — 12 semanas × 7 rasgos",
     "año que cursa, sección, tamaño del grupo, posición en la lista,\nparticipaciones, sesiones, evaluaciones",
     ancla=ent_num)
avanzar(0.15)
texto_libre(0.55, lambda ax, y: ax.text(M, y, "se estandariza", fontsize=8.5, style="italic",
            color="#666666", va="top"))
avanzar(0.15)

caja(2.1, ANCHO_FILA, M, "Entrada de temas — 12 semanas × 2 códigos",
     "código de tema de la sesión 1 y de la sesión 2 de cada semana\n(0 = sin contenido)",
     ancla=ent_tema)
avanzar(0.15)
texto_libre(0.55, lambda ax, y: ax.text(M + 4.6, y, "no se estandariza: son códigos, no magnitudes",
            fontsize=8.5, style="italic", color="#666666", va="top"))
avanzar(0.15)

fila = cursor
caja(1.9, 6.6, M, "Embedding", f"{N_CODIGOS} códigos → vector de\n2 valores aprendidos", ancla=emb, top=fila)
caja(1.9, ANCHO_FILA - 6.6 - 0.5, M + 7.1, "Reshape", "12 × 4", ancla=resh, top=fila)
cursor = fila + 1.9
avanzar(0.15)

caja(1.9, ancho_total, M, "Concatenar", "12 semanas × 11 valores", fill=AZUL, ancla=cat)
avanzar(0.2)

mitad = (ancho_total - 0.5) / 2
fila = cursor
caja(2.0, mitad, M, "LSTM", "16 unidades", ancla=l1, top=fila)
caja(2.0, mitad, M + mitad + 0.5, "Dropout", "30 %", ancla=d1, top=fila)
cursor = fila + 2.0
avanzar(0.15)

fila = cursor
caja(2.0, mitad, M, "LSTM", "16 unidades", ancla=l2, top=fila)
caja(2.0, mitad, M + mitad + 0.5, "Dropout", "30 %", ancla=d2, top=fila)
cursor = fila + 2.0
avanzar(0.2)

fila = cursor
caja(2.3, 4.6, M, "Densa", "1 unidad, lineal", ancla=dens, top=fila)
caja(2.3, ancho_total - 4.6 - 0.5, M + 5.1, "Participaciones restantes\ndel trimestre",
     "total predicho = acumulado\nobservado + restantes", fill=AZUL, ancla=sal, top=fila)
cursor = fila + 2.3
avanzar(0.35)

texto_libre(0.5, lambda ax, y: ax.text(ANCHO_CM - M, y,
            f"{PARAMETROS:,} parámetros ajustables ({PARAMETROS_ORIGINAL:,} en el modelo original)".replace(",", "."),
            ha="right", va="top", fontsize=8.5, color="#333333"))
avanzar(0.15)

ALTO_CM = cursor

# ------------------------------------------------------------------ dibujo (y_mpl = ALTO_CM - y_desde_arriba)
fig = plt.figure(figsize=(ANCHO_IN, ALTO_CM / 2.54))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, ANCHO_CM); ax.set_ylim(0, ALTO_CM); ax.axis("off")

def y_mpl(y_arriba): return ALTO_CM - y_arriba

for el in elementos:
    if el[0] == "texto":
        _, top, dibujar = el
        dibujar(ax, y_mpl(top))
    else:
        _, top, alto, x, w, titulo, sub, fill, t_size, s_size = el
        y0 = y_mpl(top + alto)     # esquina inferior izquierda en coordenadas de matplotlib
        ax.add_patch(FancyBboxPatch((x, y0), w, alto, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    linewidth=1.1, edgecolor=BORDE, facecolor=fill))
        cy = y0 + alto / 2
        if sub:
            ax.text(x + w / 2, cy + alto * 0.19, titulo, ha="center", va="center", fontsize=t_size,
                    fontweight="bold", linespacing=1.25)
            ax.text(x + w / 2, cy - alto * 0.20, sub, ha="center", va="center", fontsize=s_size, linespacing=1.35)
        else:
            ax.text(x + w / 2, cy, titulo, ha="center", va="center", fontsize=t_size, fontweight="bold")

def flecha(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y_mpl(y0)), (x1, y_mpl(y1)), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color=BORDE, shrinkA=0, shrinkB=0))

cx = lambda d, frac=0.5: d["x"] + d["w"] * frac

# entrada de temas -> embedding
flecha(cx(ent_tema, 0.18), ent_tema["bottom"] + 0.05, cx(emb), emb["top"])
# embedding -> reshape
flecha(emb["x"] + emb["w"], (emb["top"] + emb["bottom"]) / 2, resh["x"], (resh["top"] + resh["bottom"]) / 2)
# reshape -> concatenar
flecha(cx(resh, 0.5), resh["bottom"], cx(resh, 0.5), cat["top"])
# entrada numérica -> concatenar (baja por la derecha)
x_der = ent_num["x"] + ANCHO_FILA + 0.9   # centro del carril libre
flecha(x_der, ent_num["bottom"], x_der, cat["top"])
# concatenar -> LSTM1
flecha(cx(cat), cat["bottom"], cx(l1), l1["top"])
# LSTM1 -> Dropout1
flecha(l1["x"] + l1["w"], (l1["top"] + l1["bottom"]) / 2, d1["x"], (d1["top"] + d1["bottom"]) / 2)
# Dropout1 -> LSTM2 (zigzag)
flecha(cx(d1), d1["bottom"], cx(l2), l2["top"])
# LSTM2 -> Dropout2
flecha(l2["x"] + l2["w"], (l2["top"] + l2["bottom"]) / 2, d2["x"], (d2["top"] + d2["bottom"]) / 2)
# Dropout2 -> Densa
flecha(cx(d2), d2["bottom"], cx(dens), dens["top"])
# Densa -> salida
flecha(dens["x"] + dens["w"], (dens["top"] + dens["bottom"]) / 2, sal["x"], (sal["top"] + sal["bottom"]) / 2)

aqui = Path(__file__).resolve().parent
fig.savefig(aqui / "figura_arquitectura.png", dpi=300)
fig.savefig(aqui / "figura_arquitectura.pdf")
print("ok")
print(f"tamaño guardado: {ANCHO_IN} in x {ALTO_CM/2.54:.2f} in  ({ANCHO_CM:.2f} x {ALTO_CM:.2f} cm)")
