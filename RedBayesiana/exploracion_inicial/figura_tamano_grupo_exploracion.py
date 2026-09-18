"""Figura para la sección «Exploración inicial de los datos cuantitativos»
(Sprint 2): tamaño de las secciones, sin ninguna referencia a estados o
discretización (esos conceptos aún no se han introducido en esa parte del
informe). La versión con los tres estados (pequeño/mediano/grande) para la
red bayesiana sigue viviendo en la Tabla 12 del Sprint 3.

Lee «Datos Tesis Downstream/» directamente; no modifica el notebook.

Uso:  python3 RedBayesiana/figura_tamano_grupo_exploracion.py
Salida: figura_tamano_grupo_exploracion.png (dpi=300), junto a este script.
"""
from pathlib import Path
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent.parent
AQUI = Path(__file__).resolve().parent
ANCHO_IN = 6.5
AZUL = "#8fa8d8"

df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RAIZ / "Datos Tesis Downstream" / "**" / "*.csv"), recursive=True)],
               ignore_index=True)
tam_sec = df.groupby(["materia", "trimestre", "seccion"])["estudiante_id"].nunique()
assert len(tam_sec) == 17

conteo = tam_sec.value_counts().sort_index()
print(f"Tamaños de sección observados (n={len(tam_sec)} secciones):")
print(conteo.to_dict())
peq, med, gra = (tam_sec < 30).sum(), (tam_sec == 30).sum(), (tam_sec > 30).sum()
print(f"pequeño(<30)={peq}  igual a 30={med}  grande(>30)={gra}")
assert (peq, med, gra) == (4, 7, 6)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.axisbelow"] = True

fig, ax = plt.subplots(figsize=(ANCHO_IN, 3.6), constrained_layout=True)
ax.bar(range(len(conteo)), conteo.values, color=AZUL, edgecolor="white", width=0.6)
ax.set_xticks(range(len(conteo)))
ax.set_xticklabels([str(v) for v in conteo.index])
ax.set_ylim(0, conteo.values.max() * 1.3)
for i, c in enumerate(conteo.values):
    ax.text(i, c + 0.15, str(c), ha="center", va="bottom", fontsize=11)
ax.set_xlabel("estudiantes distintos en la sección", fontsize=11)
ax.set_ylabel(f"secciones (de {len(tam_sec)})", fontsize=11)
ax.tick_params(labelsize=10)

destino = AQUI / "figura_tamano_grupo_exploracion.png"
fig.savefig(destino, dpi=300)
print(f"guardado: {destino}")
