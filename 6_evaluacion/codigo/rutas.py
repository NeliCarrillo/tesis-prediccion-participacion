"""Rutas compartidas por los scripts de evaluación (Sprint 6).

Los análisis de esta carpeta reutilizan el código del modelo explicable, que
vive en `4_red_bayesiana/codigo/`. Importar este módulo agrega esa carpeta a
`sys.path`, de modo que los scripts puedan hacer `from ensamblado import ...`
igual que el código de la red, sin copiarlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
CODIGO_RED = RAIZ / "4_red_bayesiana" / "codigo"

if str(CODIGO_RED) not in sys.path:
    sys.path.insert(0, str(CODIGO_RED))
