"""Prototipo — Sprint 5.

Tarjeta 1: flujo de selección de caso (selectores encadenados asignatura →
trimestre → sección → estudiante → hito) y construcción de
`CasoPrediccion`.

Tarjeta 4 (esta iteración): visualización de la salida LSTM. La UI no
reimplementa nada del modelo — solo llama a
`services.lstm_service.predict_lstm(caso)` y presenta su resultado. Toda
la preparación de datos, imputación, escalado, carga del modelo y
reconstrucción del total siguen viviendo exclusivamente en
`services/lstm_service.py`.

Tarjeta 5 (esta iteración): visualización de la salida Bayes. Igual que
con la LSTM, la UI solo llama a
`services.bayes_service.predict_bayes(caso)` y presenta `ResultadoBayes`
— discretización, evidencia, CPD, inferencia y valor esperado siguen
viviendo exclusivamente en `services/bayes_service.py` (Sprint 4/tarjeta
3). Se conectó al mismo botón «Generar predicción» que ya ejecuta la
LSTM (ver `generar()`): es la integración incremental más pequeña posible
sin duplicar el flujo de selección. La integración conceptual/comparativa
de ambos modelos es la tarjeta 6, no esta.

Ejecutar con:
    python3 app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from nicegui import ui

from caso import CasoPrediccion, OpcionesCaso

sys.path.insert(0, str(Path(__file__).resolve().parent / "services"))
import bayes_service  # noqa: E402 (ver sys.path arriba)
import lstm_service  # noqa: E402 (ver sys.path arriba)

opciones = OpcionesCaso()


@ui.page("/")
def pagina_principal() -> None:
    ui.label("Predicción explicable de participación estudiantil").classes(
        "text-2xl font-bold q-mt-md"
    )
    ui.label("Predicciones mediante red bayesiana y LSTM").classes(
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
            _limpiar_resultado_lstm()
            _limpiar_resultado_bayes()

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
            _limpiar_resultado_lstm()
            _limpiar_resultado_bayes()
            if select_estudiante.value:
                select_hito.enable()

        def al_cambiar_hito() -> None:
            _limpiar_resultado_lstm()
            _limpiar_resultado_bayes()
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
            _ejecutar_lstm(caso)
            _ejecutar_bayes(caso)

        boton_generar.on_click(generar)

    with ui.card().classes("w-full max-w-2xl q-mt-md"):
        ui.label("2. Resultado LSTM").classes("text-lg font-semibold")
        ui.label("Modelo de contraste").classes("text-caption text-grey-7")

        estado_lstm = ui.label(
            "Selecciona un caso completo y presiona «Generar predicción»."
        ).classes("text-body2 text-grey-7 q-mt-sm")

        contenedor_prediccion_lstm = ui.column().classes("q-mt-sm")
        with contenedor_prediccion_lstm:
            etiqueta_prediccion_lstm = ui.label().classes("text-h4 text-weight-bold text-primary")
            etiqueta_frase_lstm = ui.label().classes("text-body1")
            etiqueta_contexto_lstm = ui.label().classes("text-caption text-grey-7")
        contenedor_prediccion_lstm.set_visibility(False)

        def _limpiar_resultado_lstm() -> None:
            estado_lstm.set_text("Selecciona un caso completo y presiona «Generar predicción».")
            estado_lstm.classes(replace="text-body2 text-grey-7 q-mt-sm")
            contenedor_prediccion_lstm.set_visibility(False)

        def _ejecutar_lstm(caso: CasoPrediccion) -> None:
            contenedor_prediccion_lstm.set_visibility(False)
            estado_lstm.set_text("Calculando predicción…")
            estado_lstm.classes(replace="text-body2 text-grey-7 q-mt-sm")
            try:
                resultado = lstm_service.predict_lstm(caso)
            except FileNotFoundError:
                estado_lstm.set_text(
                    f"No hay un modelo final entrenado para «{caso.materia}» / "
                    f"semana {caso.hito} todavía."
                )
                estado_lstm.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("Falta el artefacto del modelo para este caso.", type="negative")
                return
            except ValueError as error:
                estado_lstm.set_text(f"No se pudo calcular la predicción: {error}")
                estado_lstm.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("La LSTM no pudo evaluar este caso.", type="negative")
                return
            except Exception:  # noqa: BLE001 — error controlado, no se muestra el traceback
                estado_lstm.set_text(
                    "Ocurrió un error inesperado ejecutando la LSTM para este caso."
                )
                estado_lstm.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("Error inesperado en la inferencia LSTM.", type="negative")
                return

            estado_lstm.set_text("")
            etiqueta_prediccion_lstm.set_text(
                f"{resultado.prediccion_total:.1f} participaciones"
            )
            etiqueta_frase_lstm.set_text(
                f"Total trimestral previsto para la semana {resultado.hito} "
                "(no es una participación adicional ni el resultado de una sola semana)."
            )
            etiqueta_contexto_lstm.set_text(
                f"{resultado.materia} · {resultado.trimestre} · sección {resultado.seccion} · "
                f"{resultado.estudiante_id} · información disponible hasta la semana {resultado.hito}"
            )
            contenedor_prediccion_lstm.set_visibility(True)

    with ui.card().classes("w-full max-w-2xl q-mt-md"):
        ui.label("3. Resultado red bayesiana").classes("text-lg font-semibold")
        ui.label("Modelo principal").classes("text-caption text-grey-7")

        estado_bayes = ui.label(
            "Selecciona un caso completo y presiona «Generar predicción»."
        ).classes("text-body2 text-grey-7 q-mt-sm")

        contenedor_resultado_bayes = ui.column().classes("w-full q-mt-sm")
        with contenedor_resultado_bayes:
            etiqueta_prediccion_bayes = ui.label().classes("text-h4 text-weight-bold text-primary")
            etiqueta_frase_bayes = ui.label().classes("text-body1")
            etiqueta_contexto_bayes = ui.label().classes("text-caption text-grey-7")

            ui.label("Distribución posterior").classes("text-subtitle2 q-mt-md")
            ui.label(
                "Probabilidad posterior de cada estado del objetivo, dada la "
                "evidencia disponible para este caso."
            ).classes("text-caption text-grey-7")
            contenedor_posterior_bayes = ui.column().classes("w-full q-mt-xs")

            ui.label("Evidencia observada utilizada por la red").classes("text-subtitle2 q-mt-md")
            contenedor_evidencia_bayes = ui.column().classes("q-mt-xs")

            etiqueta_evidencia_omitida = ui.label().classes(
                "text-caption text-warning q-mt-sm"
            )
        contenedor_resultado_bayes.set_visibility(False)

        def _limpiar_resultado_bayes() -> None:
            estado_bayes.set_text("Selecciona un caso completo y presiona «Generar predicción».")
            estado_bayes.classes(replace="text-body2 text-grey-7 q-mt-sm")
            contenedor_resultado_bayes.set_visibility(False)

        def _ejecutar_bayes(caso: CasoPrediccion) -> None:
            contenedor_resultado_bayes.set_visibility(False)
            estado_bayes.set_text("Calculando predicción…")
            estado_bayes.classes(replace="text-body2 text-grey-7 q-mt-sm")
            try:
                resultado = bayes_service.predict_bayes(caso)
            except (FileNotFoundError, ValueError) as error:
                estado_bayes.set_text(f"No se pudo calcular la predicción: {error}")
                estado_bayes.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("La red bayesiana no pudo evaluar este caso.", type="negative")
                return
            except Exception:  # noqa: BLE001 — error controlado, no se muestra el traceback
                estado_bayes.set_text(
                    "Ocurrió un error inesperado ejecutando la red bayesiana para este caso."
                )
                estado_bayes.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("Error inesperado en la inferencia Bayes.", type="negative")
                return

            estado_bayes.set_text("")
            etiqueta_prediccion_bayes.set_text(
                f"{resultado.prediccion_continua:.1f} participaciones"
            )
            etiqueta_frase_bayes.set_text(
                f"Total trimestral estimado (valor esperado de la distribución posterior) "
                f"para la semana {resultado.hito}."
            )
            etiqueta_contexto_bayes.set_text(
                f"{resultado.materia} · {resultado.trimestre} · sección {resultado.seccion} · "
                f"{resultado.estudiante_id} · información disponible hasta la semana {resultado.hito}"
            )

            # Distribución posterior real, en el orden ya devuelto por el
            # servicio — no se renombran ni se inventan etiquetas
            # semánticas ("bajo"/"alto"): son los estados literales del
            # objetivo (Tabla 13, Sprint 3/4).
            contenedor_posterior_bayes.clear()
            with contenedor_posterior_bayes:
                for estado, probabilidad in resultado.posterior.items():
                    with ui.row().classes("w-full items-center no-wrap"):
                        ui.label(estado).style("width: 72px")
                        ui.linear_progress(value=probabilidad, show_value=False).classes("flex-grow")
                        ui.label(f"{probabilidad * 100:.1f}%").style("width: 56px; text-align: right")

            # Evidencia realmente usada por la consulta -- ni importancia
            # ni contribución, solo qué variables/valores entraron.
            contenedor_evidencia_bayes.clear()
            with contenedor_evidencia_bayes:
                for variable, valor in resultado.evidencia_utilizada.items():
                    ui.label(f"{variable}: {valor}").classes("text-body2")

            if resultado.evidencia_omitida:
                etiqueta_evidencia_omitida.set_text(
                    "Evidencia no disponible para esta inferencia: "
                    + ", ".join(resultado.evidencia_omitida)
                    + ". La red marginaliza esta variable; la inferencia es válida, no es un error."
                )
            else:
                etiqueta_evidencia_omitida.set_text("")

            contenedor_resultado_bayes.set_visibility(True)


ui.run(title="Prototipo — participación estudiantil", reload=False)
