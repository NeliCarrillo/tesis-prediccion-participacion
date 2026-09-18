"""Figura para la sección «Codificación numérica de los campos cualitativos»
(Sprint 2): distribución de la posición relativa en la lista (número de lista
entre tamaño del grupo), sin cortes de estado (eso pertenece a la Tabla 12 del
Sprint 3, que ya requiere el concepto de discretización).

Incluye la verificación de que la variación entre barras es ruido muestral
(prueba chi-cuadrado contra la hipótesis de distribución uniforme), no un
patrón sistemático — la explicación que pide el comentario del tutor.

Lee «Datos Tesis Downstream/» directamente; no modifica el notebook.

Uso:  python3 RedBayesiana/figura_posicion_lista.py
Salida: figura_posicion_lista.png (dpi=300), junto a este script.
"""
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RAIZ = Path(__file__).resolve().parent.parent.parent
AQUI = Path(__file__).resolve().parent
ANCHO_IN = 6.5
AZUL = "#8fa8d8"
CLAVE = ["estudiante_id", "materia", "trimestre", "seccion"]

df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True)],
               ignore_index=True)
df["numero_lista"] = pd.to_numeric(df["numero_lista"], errors="coerce")
tam_sec = df.groupby(["materia", "trimestre", "seccion"])["estudiante_id"].nunique().rename("tamano_grupo")
reg = df.drop_duplicates(subset=CLAVE)[CLAVE + ["numero_lista"]].copy()
reg = reg.merge(tam_sec, on=["materia", "trimestre", "seccion"])
reg["posicion"] = reg["numero_lista"] / reg["tamano_grupo"]
assert len(reg) == 437

counts, edges = np.histogram(reg["posicion"], bins=20, range=(0, 1))
chi2, p = stats.chisquare(counts)
print(f"conteos por bin (20 bins, 437 registros): {counts.tolist()}")
print(f"prueba chi-cuadrado vs. uniforme: chi2={chi2:.2f}, p={p:.3f}")
assert p > 0.05, "se esperaba no rechazar uniformidad"
print(f"tamaños de grupo distintos que se combinan: {sorted(tam_sec.unique())}")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

fig, ax = plt.subplots(figsize=(ANCHO_IN, 3.6), constrained_layout=True)
ax.hist(reg["posicion"], bins=20, range=(0, 1), color=AZUL, edgecolor="white")
ax.set_xlabel("número de lista / tamaño del grupo", fontsize=11)
ax.set_ylabel("registros\nestudiante-sección", fontsize=11)
ax.tick_params(labelsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:.2f}".replace(".", ",")))

destino = AQUI / "figura_posicion_lista.png"
fig.savefig(destino, dpi=300)
print(f"guardado: {destino}")
