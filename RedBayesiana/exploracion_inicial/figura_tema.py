"""Genera la figura de exploración de tema de la sesión (participación empírica
por código, agrupada en terciles), a pedido del tutor separada de la Figura 7
(distribución de las otras cuatro variables discretizadas).

Lee «Datos Tesis Downstream/» directamente; no modifica el notebook.

Igual que las demás figuras: se guarda ya al ancho exacto de inserción (6,5 in /
16,51 cm), sin bbox_inches='tight', para que el tamaño de letra en puntos
corresponda 1:1 al tamaño real en el documento.

Uso:  python3 RedBayesiana/figura_tema.py
Salida: figura_tema.png (dpi=300), junto a este script.
"""
from pathlib import Path
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RAIZ = Path(__file__).resolve().parent.parent.parent
AQUI = Path(__file__).resolve().parent

ANCHO_IN = 6.5

T_TITULO = 13
T_EJES = 11
T_TICKS = 10
T_ANOTACION = 11

CORTE = "#b3352a"

# --------------------------------------------------------------- carga y cómputo
frames = [pd.read_csv(f) for f in glob.glob(str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True)]
df = pd.concat(frames, ignore_index=True)
df["participaciones"] = pd.to_numeric(df["participaciones"], errors="coerce")
df["tema"] = pd.to_numeric(df["tema"], errors="coerce")

temas = df[df["tema"] != 0].groupby(["materia", "tema"]).agg(
    n=("participaciones", "size"), media=("participaciones", "mean")
).reset_index()


def tercios(g):
    g = g.sort_values("media").copy()
    g["acum"] = g["n"].cumsum()
    total = g["n"].sum()
    g["categoria"] = pd.cut(g["acum"], bins=[0, total / 3, 2 * total / 3, total],
                             labels=["baja", "media", "alta"])
    return g


temas = temas.groupby("materia", group_keys=False).apply(tercios)

print("=== Verificación previa a graficar ===\n")
print("Tema de la sesión — códigos por asignatura y categoría (baja/media/alta):")
for materia, g in temas.groupby("materia"):
    n_codigos = g["tema"].nunique()
    conteo_cat = g.groupby("categoria", observed=True)["n"].sum()
    print(f"  {materia} ({n_codigos} códigos): {conteo_cat.to_dict()}")
print("\nGenerando figura...\n")

# --------------------------------------------------------------- figura
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

fig, ax = plt.subplots(figsize=(ANCHO_IN, 4.4), constrained_layout=True)

colores = {"Algoritmos y Programación": "#8fa8d8", "Estructura de Datos": "#e0a15c", "Computación Emergente": "#7fb37f"}
for materia, g in temas.groupby("materia"):
    g = g.sort_values("media")
    x = (g["acum"] - g["n"] / 2) / g["n"].sum() * 100
    ax.plot(x, g["media"], "o-", color=colores[materia], label=materia, markersize=5, linewidth=1.5)

ax.axvline(100 / 3, color=CORTE, linestyle="--", linewidth=1.4)
ax.axvline(200 / 3, color=CORTE, linestyle="--", linewidth=1.4)
trans = ax.get_xaxis_transform()
ax.text(100 / 6, 1.02, "baja", ha="center", va="bottom", fontsize=T_ANOTACION, transform=trans)
ax.text(50, 1.02, "media", ha="center", va="bottom", fontsize=T_ANOTACION, transform=trans)
ax.text(100 - 100 / 6, 1.02, "alta", ha="center", va="bottom", fontsize=T_ANOTACION, transform=trans)

ax.set_title("Tema de la sesión: participación promedio, agrupada en terciles",
             fontsize=T_TITULO, fontweight="bold", pad=16)
ax.set_xlabel("percentil acumulado de sesiones (por código de tema, dentro de la asignatura)", fontsize=T_EJES)
ax.set_ylabel("participaciones promedio por sesión", fontsize=T_EJES)
ax.tick_params(labelsize=T_TICKS)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.0f}%"))
ax.legend(fontsize=T_ANOTACION, loc="lower right", framealpha=0.9)

destino = AQUI / "figura_tema.png"
fig.savefig(destino, dpi=300)
print(f"guardado: {destino}")
