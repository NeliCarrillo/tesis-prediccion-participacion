"""Prototipo — Sprint 5, tarjeta 1: "Definir los inputs y el flujo
funcional del prototipo".

Implementa únicamente el flujo de selección de caso (selectores
encadenados asignatura → trimestre → sección → estudiante → hito) y la
construcción de `CasoPrediccion`. NO ejecuta LSTM ni la red bayesiana
todavía: eso depende de congelar un modelo final por asignatura (tarjetas
2 y 3), que aún no existe en el repositorio (ver README de esta carpeta).

Ejecutar con:
    python3 app.py
"""
from __future__ import annotations

from nicegui import ui

from caso import CasoPrediccion, OpcionesCaso

opciones = OpcionesCaso()


@ui.page("/")
def pagina_principal() -> None:
    ui.label("Predicción explicable de participación estudiantil").classes(
        "text-2xl font-bold q-mt-md"
    )
    ui.label("Sistema de contraste: red bayesiana (explicable) vs. LSTM (no explicable)").classes(
        "text-subtitle1 text-grey-7 q-mb-md"
    )

    with ui.card().classes("w-full max-w-2xl"):
        ui.label("1. Seleccionar caso").classes("text-lg font-semibold")

        select_materia = ui.select(opciones.materias(), label="Asignatura").classes("w-full")
        select_trimestre = ui.select([], label="Trimestre").classes("w-full")
        select_seccion = ui.select([], label="Sección").classes("w-full")
        select_estudiante = ui.select([], label="Estudiante (anonimizado)").classes("w-full")
        select_hito = ui.select(
            {h: f"Semana {h}" for h in opciones.hitos()}, label="Hito"
        ).classes("w-full")

        for selector in (select_trimestre, select_seccion, select_estudiante, select_hito):
            selector.disable()

        boton_generar = ui.button("Generar predicción")
        boton_generar.disable()

        etiqueta_caso = ui.label().classes("text-caption text-grey-7 q-mt-sm")

        def _reiniciar(desde: str) -> None:
            """Limpia y deshabilita los selectores posteriores al nivel
            `desde` en la cadena asignatura→trimestre→sección→estudiante.
            Hito queda fuera de esta cadena a propósito: sus opciones
            (semana 4/6/8) no dependen del caso seleccionado, así que solo
            se le resetea el valor y se deshabilita, sin tocar sus
            opciones — si se limpiaran aquí, quedarían vacías para
            siempre, porque nada las vuelve a poblar más adelante."""
            widgets_con_opciones_dependientes = {
                "trimestre": select_trimestre,
                "seccion": select_seccion,
                "estudiante": select_estudiante,
            }
            niveles = list(widgets_con_opciones_dependientes)
            for nombre in niveles[niveles.index(desde):]:
                widget = widgets_con_opciones_dependientes[nombre]
                widget.set_options([])
                widget.set_value(None)
                widget.disable()
            select_hito.set_value(None)
            select_hito.disable()
            boton_generar.disable()
            etiqueta_caso.set_text("")

        def al_cambiar_materia() -> None:
            _reiniciar("trimestre")
            if select_materia.value:
                select_trimestre.set_options(opciones.trimestres(select_materia.value))
                select_trimestre.enable()

        def al_cambiar_trimestre() -> None:
            _reiniciar("seccion")
            if select_materia.value and select_trimestre.value:
                secciones = opciones.secciones(select_materia.value, select_trimestre.value)
                select_seccion.set_options(secciones)
                select_seccion.enable()

        def al_cambiar_seccion() -> None:
            _reiniciar("estudiante")
            if select_materia.value and select_trimestre.value and select_seccion.value:
                estudiantes = opciones.estudiantes(
                    select_materia.value, select_trimestre.value, select_seccion.value
                )
                select_estudiante.set_options(estudiantes)
                select_estudiante.enable()

        def al_cambiar_estudiante() -> None:
            select_hito.set_value(None)
            boton_generar.disable()
            etiqueta_caso.set_text("")
            if select_estudiante.value:
                select_hito.enable()

        def al_cambiar_hito() -> None:
            if select_hito.value:
                boton_generar.enable()
            else:
                boton_generar.disable()

        select_materia.on_value_change(al_cambiar_materia)
        select_trimestre.on_value_change(al_cambiar_trimestre)
        select_seccion.on_value_change(al_cambiar_seccion)
        select_estudiante.on_value_change(al_cambiar_estudiante)
        select_hito.on_value_change(al_cambiar_hito)

        def generar() -> None:
            caso = CasoPrediccion(
                materia=select_materia.value,
                trimestre=select_trimestre.value,
                seccion=select_seccion.value,
                estudiante_id=select_estudiante.value,
                hito=select_hito.value,
            )
            etiqueta_caso.set_text(
                f"Caso construido: {caso.materia} · {caso.trimestre} · "
                f"sección {caso.seccion} · {caso.estudiante_id} · semana {caso.hito}"
            )
            ui.notify(
                "Caso válido. La ejecución de LSTM y Bayes se habilita en las "
                "tarjetas 2 y 3, una vez congelados los modelos finales por "
                "asignatura (ver README de prototipo/).",
                type="info",
            )

        boton_generar.on_click(generar)

    with ui.card().classes("w-full max-w-2xl opacity-60 q-mt-md"):
        ui.label("2. Resultado LSTM").classes("text-lg font-semibold")
        ui.label(
            "Pendiente — tarjeta 2. Mostrará la predicción puntual del total "
            "trimestral para el hito seleccionado."
        ).classes("text-body2")

    with ui.card().classes("w-full max-w-2xl opacity-60 q-mt-md"):
        ui.label("3. Resultado red bayesiana").classes("text-lg font-semibold")
        ui.label(
            "Pendiente — tarjeta 3. Mostrará la predicción continua, la "
            "distribución posterior sobre los cinco estados, y la evidencia "
            "utilizada u omitida por ausencia."
        ).classes("text-body2")


ui.run(title="Prototipo — participación estudiantil", reload=False)
