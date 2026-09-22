"""Representación central del caso seleccionado en el prototipo — Sprint 5,
tarjeta 1.

`CasoPrediccion` es la única entrada que necesitarán, más adelante, las
tarjetas 2 y 3 (construir_entrada_lstm / construir_evidencia_bayes): ambas
rutas de transformación deben partir de este mismo objeto, para no duplicar
la lógica de "qué registro se seleccionó" en cada modelo.

`OpcionesCaso` reutiliza `ensamblado.ensamblar_conjunto()` (Sprint 4, ya
validado) como única fuente de verdad de qué registros existen realmente:
no vuelve a leer los CSV crudos ni reimplementa la deduplicación por
(materia, trimestre, sección, estudiante_id). Así, ninguna combinación
ofrecida en los selectores del prototipo puede ser una combinación
inexistente en los datos.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CODIGO_RED = RAIZ / "RedBayesiana" / "codigo_red"

# RedBayesiana/codigo_red/*.py usa imports "planos" (`from ensamblado import
# ...`), no relativos de paquete — así están hoy en el pipeline ya validado
# de Sprint 4, y no se tocan aquí. Para reutilizarlos desde `prototipo/` sin
# modificarlos, se agrega esa carpeta al sys.path antes de importar.
if str(RUTA_CODIGO_RED) not in sys.path:
    sys.path.insert(0, str(RUTA_CODIGO_RED))

from ensamblado import ensamblar_conjunto  # noqa: E402 (ver sys.path arriba)

HITOS: tuple[int, ...] = (4, 6, 8)


@dataclass(frozen=True)
class CasoPrediccion:
    """Un caso académico completo: un registro estudiante-sección y el hito
    en el que se quiere predecir. Inmutable una vez construido."""

    materia: str
    trimestre: str
    seccion: str
    estudiante_id: str
    hito: int

    def __post_init__(self) -> None:
        if self.hito not in HITOS:
            raise ValueError(f"hito debe ser uno de {HITOS}, llegó {self.hito!r}")


class OpcionesCaso:
    """Árbol de combinaciones válidas materia → trimestre → sección →
    estudiante, derivado de los 524 registros estudiante-sección reales.
    Es de solo lectura: no ajusta ni entrena nada, solo enumera qué caso
    puede seleccionarse."""

    def __init__(self, reg: pd.DataFrame | None = None) -> None:
        if reg is None:
            _, reg, _ = ensamblar_conjunto()
        self._reg = reg[["materia", "trimestre", "seccion", "estudiante_id"]].copy()
        self._reg["seccion"] = self._reg["seccion"].astype(str)

    def materias(self) -> list[str]:
        return sorted(self._reg["materia"].unique())

    def trimestres(self, materia: str) -> list[str]:
        filtro = self._reg["materia"] == materia
        return sorted(self._reg.loc[filtro, "trimestre"].unique())

    def secciones(self, materia: str, trimestre: str) -> list[str]:
        filtro = (self._reg["materia"] == materia) & (self._reg["trimestre"] == trimestre)
        return sorted(self._reg.loc[filtro, "seccion"].unique())

    def estudiantes(self, materia: str, trimestre: str, seccion: str) -> list[str]:
        filtro = (
            (self._reg["materia"] == materia)
            & (self._reg["trimestre"] == trimestre)
            & (self._reg["seccion"] == seccion)
        )
        return sorted(self._reg.loc[filtro, "estudiante_id"].unique())

    @staticmethod
    def hitos() -> tuple[int, ...]:
        return HITOS

    def existe(self, materia: str, trimestre: str, seccion: str, estudiante_id: str) -> bool:
        filtro = (
            (self._reg["materia"] == materia)
            & (self._reg["trimestre"] == trimestre)
            & (self._reg["seccion"] == seccion)
            & (self._reg["estudiante_id"] == estudiante_id)
        )
        return bool(filtro.any())
