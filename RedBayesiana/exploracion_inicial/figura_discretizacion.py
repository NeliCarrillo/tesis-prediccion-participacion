"""Genera la figura de distribución de las cuatro variables discretizadas con
criterio (Tabla 12): posición relativa en la lista, tamaño del grupo,
participaciones de la semana y participaciones del trimestre. Marca los cortes
de estado sobre cada distribución.

La exploración de tema de la sesión vive aparte, en figura_tema.py (a pedido
del tutor: esta figura vuelve a su versión original de 4 paneles, y tema pasa
a ser una figura propia debajo).

Lee «Datos Tesis Downstream/» directamente; no modifica el notebook.

Igual que las Figuras 6 y 9: se guarda ya al ancho exacto de inserción (6,5 in /
16,51 cm), sin bbox_inches='tight', para que el tamaño de letra en puntos
corresponda 1:1 al tamaño real en el documento.

Uso:  python3 RedBayesiana/figura_discretizacion.py
Salida: figura_discretizacion.png (dpi=300), junto a este script.
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

T_TITULO_GENERAL = 14
T_TITULO_PANEL = 12
T_EJES = 10
T_TICKS = 9
T_ANOTACION = 10.5

CLAVE = ["estudiante_id", "materia", "trimestre", "seccion"]
CORTE = "#b3352a"
AZUL = "#8fa8d8"

# --------------------------------------------------------------- carga y cómputo
frames = [pd.read_csv(f) for f in glob.glob(str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True)]
df = pd.concat(frames, ignore_index=True)
df["participaciones"] = pd.to_numeric(df["participaciones"], errors="coerce")
df["numero_lista"] = pd.to_numeric(df["numero_lista"], errors="coerce")

tam_sec = df.groupby(["materia", "trimestre", "seccion"])["estudiante_id"].nunique().rename("tamano_grupo")

reg = df.drop_duplicates(subset=CLAVE)[CLAVE + ["numero_lista"]].copy()
reg = reg.merge(tam_sec, on=["materia", "trimestre", "seccion"])
reg["posicion"] = reg["numero_lista"] / reg["tamano_grupo"]
tot = df.groupby(CLAVE)["participaciones"].sum().rename("total").reset_index()
reg = reg.merge(tot, on=CLAVE)

sem = df.groupby(CLAVE + ["semana"])["participaciones"].sum().reset_index()

assert len(tam_sec) == 14
assert len(reg) == 437
assert len(sem) == 5244

# --------------------------------------------------------------- verificación (impresa)
print("=== Verificación previa a graficar ===\n")

print("Posición relativa en la lista — cuartiles observados:")
q = reg["posicion"].quantile([0.25, 0.5, 0.75])
print(f"  0,25 -> {q.loc[0.25]:.3f}   0,50 -> {q.loc[0.5]:.3f}   0,75 -> {q.loc[0.75]:.3f}")
assert abs(q.loc[0.25] - 0.27) < 0.01 and abs(q.loc[0.5] - 0.51) < 0.01 and abs(q.loc[0.75] - 0.77) < 0.01

print("\nTamaño del grupo — pequeño/mediano/grande:")
peq, med, gra = (tam_sec < 30).sum(), (tam_sec == 30).sum(), (tam_sec > 30).sum()
print(f"  pequeño(<30)={peq}  mediano(=30)={med}  grande(>30)={gra}  (de 14 secciones)")
assert (peq, med, gra) == (3, 7, 4)


def bin_semana(p):
    if p == 0: return "0"
    if p == 1: return "1"
    if p == 2: return "2"
    return "3 o más"


sem["bin"] = sem["participaciones"].apply(bin_semana)
conteo_semana = sem["bin"].value_counts().reindex(["0", "1", "2", "3 o más"])
print("\nParticipaciones de la semana:")
print(f"  {conteo_semana.to_dict()}  (de {len(sem)})")
assert conteo_semana.tolist() == [4201, 578, 268, 197]


def bin_trimestre(t):
    if t == 0: return "0"
    if t <= 2: return "1-2"
    if t <= 5: return "3-5"
    if t <= 11: return "6-11"
    return "12 o más"


reg["bin"] = reg["total"].apply(bin_trimestre)
conteo_trimestre = reg["bin"].value_counts().reindex(["0", "1-2", "3-5", "6-11", "12 o más"])
print("\nParticipaciones del trimestre:")
print(f"  {conteo_trimestre.to_dict()}  (de {len(reg)})")
assert conteo_trimestre.tolist() == [147, 106, 73, 62, 49]

print("\nTodo coincide con lo esperado. Generando figura...\n")

# --------------------------------------------------------------- figura
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

fig, ejes = plt.subplots(2, 2, figsize=(ANCHO_IN, 6.6), constrained_layout=True)
ax1, ax2 = ejes[0, 0], ejes[0, 1]
ax3, ax4 = ejes[1, 0], ejes[1, 1]

# ---- panel 1: posición relativa en la lista ----
ax = ax1
ax.hist(reg["posicion"], bins=20, range=(0, 1), color=AZUL, edgecolor="white")
ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
cortes = [0.27, 0.51, 0.77]
for c in cortes:
    ax.axvline(c, color=CORTE, linestyle="--", linewidth=1.3)
etiquetas = ["≤0,25", "0,25–\n0,50", "0,50–\n0,75", ">0,75"]
bordes = [0] + cortes + [1]
for i, et in enumerate(etiquetas):
    centro = (bordes[i] + bordes[i + 1]) / 2
    ax.text(centro, ax.get_ylim()[1] * 0.96, et, ha="center", va="top", fontsize=T_ANOTACION - 1, linespacing=1.0)
ax.set_title("Posición relativa en la lista", fontsize=T_TITULO_PANEL, fontweight="bold")
ax.set_xlabel("número de lista / tamaño del grupo", fontsize=T_EJES)
ax.set_ylabel("registros\nestudiante-sección", fontsize=T_EJES)
ax.tick_params(labelsize=T_TICKS)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.2f}".replace(".", ",")))

# ---- panel 2: tamaño del grupo ----
ax = ax2
valores = sorted(tam_sec.values)
conteo_valores = pd.Series(valores).value_counts().sort_index()
n_cat = len(conteo_valores)
ax.bar(range(n_cat), conteo_valores.values, color=AZUL, edgecolor="white", width=0.7)
ax.set_xticks(range(n_cat))
ax.set_xticklabels([str(v) for v in conteo_valores.index])
ax.set_ylim(0, conteo_valores.values.max() * 1.5)
for i, c in enumerate(conteo_valores.values):
    ax.text(i, c + 0.15, str(c), ha="center", va="bottom", fontsize=T_ANOTACION)
ax.axvspan(-0.5, 2.5, color=CORTE, alpha=0.08)
ax.axvspan(2.5, 3.5, color="#4c7a3f", alpha=0.08)
ax.axvspan(3.5, n_cat - 0.5, color=CORTE, alpha=0.08)
ax.text(1, ax.get_ylim()[1] * 0.96, "pequeño\n(≤28)", ha="center", va="top", fontsize=T_ANOTACION - 1, linespacing=1.0)
ax.text(3, ax.get_ylim()[1] * 0.96, "mediano\n(=30)", ha="center", va="top", fontsize=T_ANOTACION - 1, linespacing=1.0)
ax.text(5.5, ax.get_ylim()[1] * 0.96, "grande\n(≥35)", ha="center", va="top", fontsize=T_ANOTACION - 1, linespacing=1.0)
ax.set_title("Tamaño del grupo", fontsize=T_TITULO_PANEL, fontweight="bold")
ax.set_xlabel("estudiantes distintos en la sección", fontsize=T_EJES)
ax.set_ylabel("secciones (de 14)", fontsize=T_EJES)
ax.tick_params(labelsize=T_TICKS)

# ---- panel 3: participaciones de la semana ----
ax = ax3
orden_sem = ["0", "1", "2", "3 o más"]
valores_sem = conteo_semana.reindex(orden_sem)
ax.bar(range(4), valores_sem.values, color=AZUL, edgecolor="white", width=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(orden_sem)
ax.set_yscale("log")
for i, v in enumerate(valores_sem.values):
    ax.text(i, v * 1.2, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=T_ANOTACION)
ax.set_title("Participaciones de la semana", fontsize=T_TITULO_PANEL, fontweight="bold")
ax.set_xlabel("participaciones en la semana (estado)", fontsize=T_EJES)
ax.set_ylabel("filas estudiante-semana\n(escala log, de 5.244)", fontsize=T_EJES)
ax.tick_params(labelsize=T_TICKS)
ax.set_ylim(top=ax.get_ylim()[1] * 4)

# ---- panel 4: participaciones del trimestre ----
ax = ax4
ax.hist(reg["total"], bins=range(0, 46, 2), color=AZUL, edgecolor="white")
ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
cortes_tri = [0.5, 2.5, 5.5, 11.5]
for c in cortes_tri:
    ax.axvline(c, color=CORTE, linestyle="--", linewidth=1.3)
resumen = "\n".join([
    "0 (n=147)", "1–2 (n=106)", "3–5 (n=73)", "6–11 (n=62)", "12+ (n=49)",
])
ax.text(0.97, 0.95, resumen, transform=ax.transAxes, ha="right", va="top",
        fontsize=T_ANOTACION - 1, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"))
ax.set_title("Participaciones del trimestre", fontsize=T_TITULO_PANEL, fontweight="bold")
ax.set_xlabel("total de participaciones\n(registro estudiante-sección)", fontsize=T_EJES)
ax.set_ylabel("registros (de 437)", fontsize=T_EJES)
ax.tick_params(labelsize=T_TICKS)

fig.suptitle("Distribución de las variables discretizadas y sus cortes de estado",
             fontsize=T_TITULO_GENERAL, fontweight="bold")

destino = AQUI / "figura_discretizacion.png"
fig.savefig(destino, dpi=300)
print(f"guardado: {destino}")
