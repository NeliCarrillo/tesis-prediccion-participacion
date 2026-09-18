"""Convierte la salida de la red bayesiana (una distribución de probabilidad sobre
los estados de «participaciones del trimestre», Tabla 12) en un valor continuo,
para poder evaluarla con las mismas métricas que la LSTM (RMSE, R²) sobre el
total de participaciones del trimestre.

Cada estado se representa por la media observada del total dentro de ese estado
en los 437 registros estudiante-sección (no por el punto medio del intervalo):
para los estados acotados ambas cosas casi coinciden, pero «12 o más» no tiene
punto medio propio, y la media empírica es el valor que minimiza el error
cuadrático dentro del estado, dado que no hay más información para distinguir
dentro de él.

Recalculado a partir de «Datos Tesis Downstream» el 2026-09-14:
  0         (n=147): 0,000
  1-2       (n=106): 1,396
  3-5       (n=73):  3,890
  6-11      (n=62):  8,274
  12 o más  (n=49):  18,776
"""
from __future__ import annotations

ESTADOS = ("0", "1-2", "3-5", "6-11", "12 o más")

MEDIA_POR_ESTADO = {
    "0": 0.0,
    "1-2": 1.396,
    "3-5": 3.890,
    "6-11": 8.274,
    "12 o más": 18.776,
}


def valor_esperado(posterior: dict) -> float:
    """`posterior`: {estado: probabilidad}, con las claves de ESTADOS.
    Devuelve la esperanza de la posterior, usando la media real de cada estado."""
    faltantes = set(ESTADOS) - posterior.keys()
    if faltantes:
        raise ValueError(f"faltan estados en la posterior: {faltantes}")
    return sum(MEDIA_POR_ESTADO[e] * p for e, p in posterior.items())


if __name__ == "__main__":
    # ejemplo: posterior muy concentrada en "0" debe devolver un valor cercano a 0
    assert abs(valor_esperado({"0": 0.9, "1-2": 0.1, "3-5": 0, "6-11": 0, "12 o más": 0}) - 0.1396) < 1e-6
    print("ok")
