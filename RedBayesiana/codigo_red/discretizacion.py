"""Discretización de las variables continuas y de alta cardinalidad para la red
bayesiana, según la Tabla 12 del informe («Fase de diseño del modelo de red
bayesiana», Sprint 3).

Cubre solo las variables que requieren una regla nueva. Las demás nodos de la
Tabla 11 no necesitan función propia:
  - año que cursa      -> ya se agrupa en 1, 2, 3, 4, «5 o más» en generar_csv.py
                          (TOPE_ANIO = 5), no se toca aquí.
  - sección            -> se usa tal cual (identificador entero de la Tabla 6).
  - sesiones/evaluaciones de la semana -> ya son enteros pequeños (0, 1 o 2),
                          se usan como estados sin transformación.
  - tema de la sesión  -> ya es el código del catálogo por asignatura
                          (CATALOGO_Temas.xlsx); su cardinalidad se documenta
                          en el Apéndice A.

Los cortes de este módulo son fijos y reproducibles: no se recalculan a partir
del conjunto de datos en cada ejecución, para que un mismo valor discretice
siempre al mismo estado, sin importar cuántos registros haya en el momento.
Es una primera propuesta (ver Tabla 12 y su nota metodológica); el número de
estados podrá ampliarse si el peso del nodo en la red final lo justifica.
"""
from __future__ import annotations


def tamano_grupo(n_estudiantes: int) -> str:
    """Tres estados, por el agrupamiento natural observado en las 14 secciones
    (18, 27, 28 | 30×7 | 35, 38, 40, 41 estudiantes)."""
    if n_estudiantes < 30:
        return "pequeño"
    if n_estudiantes == 30:
        return "mediano"
    return "grande"


def posicion_en_lista(numero_lista: int, tamano_grupo_seccion: int) -> str:
    """Cuatro estados por cuartiles (igualdad de frecuencia). La posición
    relativa (0 a 1) se calcula igual que en el LSTM: número de lista entre
    el tamaño del grupo."""
    posicion = numero_lista / tamano_grupo_seccion
    if posicion <= 0.25:
        return "≤0,25"
    if posicion <= 0.50:
        return "0,25–0,50"
    if posicion <= 0.75:
        return "0,50–0,75"
    return ">0,75"


def participaciones_semana(participaciones: int) -> str:
    """Cuatro estados: el 80% de las semanas registradas no tiene ninguna
    participación, por lo que el cero se aísla como estado propio."""
    if participaciones == 0:
        return "0"
    if participaciones == 1:
        return "1"
    if participaciones == 2:
        return "2"
    return "3 o más"


def participaciones_trimestre(total: int) -> str:
    """Cinco estados para la variable objetivo. El 34% de los registros no
    tiene ninguna participación en todo el trimestre (Figura 3)."""
    if total == 0:
        return "0"
    if total <= 2:
        return "1-2"
    if total <= 5:
        return "3-5"
    if total <= 11:
        return "6-11"
    return "12 o más"


if __name__ == "__main__":
    # Verificación rápida contra los tres casos usados como ejemplo en el informe
    assert tamano_grupo(18) == tamano_grupo(27) == tamano_grupo(28) == "pequeño"
    assert tamano_grupo(30) == "mediano"
    assert tamano_grupo(35) == tamano_grupo(41) == "grande"
    assert posicion_en_lista(1, 30) == "≤0,25"          # 0,03
    assert posicion_en_lista(20, 30) == "0,50–0,75"     # 0,67
    assert participaciones_semana(0) == "0"
    assert participaciones_semana(5) == "3 o más"
    assert participaciones_trimestre(0) == "0"
    assert participaciones_trimestre(44) == "12 o más"
    print("verificaciones superadas")
