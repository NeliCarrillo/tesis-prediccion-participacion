"""Genera la Figura 3 del informe: arquitectura de la red LSTM adaptada.

Uso:  python3 LSTM/nuevo/figura_3_arquitectura.py
Salida: figura_3_arquitectura.png (300 dpi) y .pdf, junto a este script.
Los recuentos de parámetros provienen de construir el modelo con Keras
(construir_modelo_lstm del cuaderno) con el catálogo de 14 códigos (0 a 13).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

N_CODIGOS = 14           # códigos 0 a 13 del catálogo (Estructura de Datos fija el máximo)
PARAMETROS = 3_949       # 1.792 + 2.112 + 28 + 17
PARAMETROS_ORIGINAL = 52_228

GRIS, AZUL, BORDE = "#f2f2f2", "#e4e9f3", "#222222"
fig, ax = plt.subplots(figsize=(14.6, 5.4))
ax.set_xlim(0, 30.2); ax.set_ylim(0, 11.2); ax.axis("off")


def caja(x, y, w, h, titulo, sub=None, fill=GRIS, t_size=11, s_size=8.5, t_dy=0.45, s_dy=-0.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                linewidth=1.3, edgecolor=BORDE, facecolor=fill))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + t_dy, titulo, ha="center", va="center", fontsize=t_size, fontweight="bold", linespacing=1.15)
        ax.text(x + w / 2, cy + s_dy, sub, ha="center", va="center", fontsize=s_size, linespacing=1.25)
    else:
        ax.text(x + w / 2, cy, titulo, ha="center", va="center", fontsize=t_size, fontweight="bold")


def flecha(puntos):
    for (x0, y0), (x1, y1) in zip(puntos[:-1], puntos[1:]):
        ultimo = (x1, y1) == puntos[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if ultimo else "-",
                                     mutation_scale=14, linewidth=1.3, color=BORDE, shrinkA=0, shrinkB=0))


# título
ax.text(0.4, 10.75, "Arquitectura de la red LSTM adaptada", fontsize=14, fontweight="bold", va="center")
ax.text(0.4, 10.1, "Dos entradas concatenadas por semana, dos capas LSTM con regularización y una salida continua.",
        fontsize=10.5, va="center", color="#333333")

# entradas
caja(0.4, 6.6, 6.9, 3.2, "Entrada numérica\n12 semanas × 7 rasgos",
     "año que cursa, sección,\ntamaño del grupo, posición en la lista,\nparticipaciones, sesiones,\nevaluaciones", t_size=11, s_size=9, t_dy=1.0, s_dy=-0.72)
ax.text(3.85, 6.38, "se estandariza", ha="center", va="top", fontsize=9, color="#666666", style="italic")
caja(0.4, 1.3, 6.9, 3.2, "Entrada de temas\n12 semanas × 2 códigos",
     "código de tema de la sesión 1 y\nde la sesión 2 de cada semana\n(0 = sin contenido)", t_size=11, s_size=9, t_dy=0.85, s_dy=-0.7)
ax.text(3.85, 1.1, "no se estandariza:\nson códigos, no magnitudes", ha="center", va="top", fontsize=9,
        color="#666666", style="italic", linespacing=1.15)

# rama de temas
caja(7.8, 1.7, 3.9, 1.9, "Embedding", f"{N_CODIGOS} códigos → vector\nde 2 valores aprendidos")
caja(12.2, 1.9, 2.4, 1.5, "Reshape", "12 × 4", t_dy=0.32, s_dy=-0.35)
caja(15.2, 4.2, 4.3, 1.9, "Concatenar", "12 semanas × 11 valores", fill=AZUL)

# cadena LSTM
x = 19.8
for titulo, sub, w in (("LSTM", "16 unidades", 2.2), ("Dropout", "30 %", 2.2), ("LSTM", "16 unidades", 2.2), ("Dropout", "30 %", 2.2)):
    caja(x, 4.2, w, 1.9, titulo, sub, t_size=10.5); x += w + 0.3
fin_cadena = x - 0.35

# salida
caja(19.8, 1.55, 2.6, 1.5, "Densa", "1 unidad, lineal", t_dy=0.32, s_dy=-0.35)
caja(23.0, 1.2, 6.4, 2.2, "Participaciones restantes\ndel trimestre",
     "total predicho =\nacumulado observado + restantes", fill=AZUL, t_size=11, s_size=9, t_dy=0.55, s_dy=-0.6)

# flechas
flecha([(7.3, 8.2), (17.2, 8.2), (17.2, 6.1)])
flecha([(7.3, 2.9), (7.8, 2.9)])
flecha([(11.7, 2.65), (12.2, 2.65)])
flecha([(14.6, 2.65), (17.2, 2.65), (17.2, 4.2)])
xs = [19.5, 19.8, 22.0, 22.3, 24.5, 24.8, 27.0, 27.3]
for a, b in zip(xs[0::2], xs[1::2]):
    flecha([(a, 5.15), (b, 5.15)])
flecha([(28.4, 4.2), (28.4, 3.55), (21.1, 3.55), (21.1, 3.05)])
flecha([(22.4, 2.3), (23.0, 2.3)])

ax.text(29.4, 0.45, f"{PARAMETROS:,} parámetros ajustables ({PARAMETROS_ORIGINAL:,} en el modelo original)".replace(",", "."),
        ha="right", va="center", fontsize=10, color="#333333")

aqui = Path(__file__).resolve().parent
fig.savefig(aqui / "figura_3_arquitectura.png", dpi=300, bbox_inches="tight")
fig.savefig(aqui / "figura_3_arquitectura.pdf", bbox_inches="tight")
print("ok")
