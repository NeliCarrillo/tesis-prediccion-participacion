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

    modo = ui.toggle(
        ["Estudiante histórico", "Estudiante nuevo"], value="Estudiante histórico"
    ).classes("q-mb-md")

    tarjeta_seleccion_historica = ui.card().classes("w-full max-w-2xl")
    with tarjeta_seleccion_historica:
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

    tarjeta_lstm_historica = ui.card().classes("w-full max-w-2xl q-mt-md")
    with tarjeta_lstm_historica:
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
            etiqueta_valor_real_lstm = ui.label().classes("text-body2 text-weight-bold q-mt-xs")
            etiqueta_error_local_lstm = ui.label().classes("text-caption text-grey-7")
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

            # Valor real: solo existe para un estudiante histórico -- nunca
            # para uno hipotético. Es la verdad, no una predicción.
            try:
                real = lstm_service.valor_real(caso)
                etiqueta_valor_real_lstm.set_text(
                    f"Lo que este estudiante realmente hizo: {real:.0f} participaciones "
                    f"en el trimestre completo."
                )
            except Exception:  # noqa: BLE001 — no debe romper el resto de la tarjeta
                etiqueta_valor_real_lstm.set_text("")

            # Margen de error local: estimado sobre la validación cruzada
            # oficial (Sprint 2), NO el RMSE/R² global ya reportado.
            error = lstm_service.error_local(resultado.prediccion_total)
            if error["n"] > 0:
                etiqueta_error_local_lstm.set_text(
                    f"Margen de error esperado para predicciones de esta magnitud: "
                    f"± {error['mae_local']:.1f} participaciones en promedio "
                    f"(RMSE {error['rmse_local']:.1f}), estimado sobre {error['n']} casos "
                    f"similares de la validación cruzada — no es el RMSE/R² global del informe."
                )
            else:
                etiqueta_error_local_lstm.set_text("")

            contenedor_prediccion_lstm.set_visibility(True)

    tarjeta_bayes_historica = ui.card().classes("w-full max-w-2xl q-mt-md")
    with tarjeta_bayes_historica:
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
            etiqueta_valor_real_bayes = ui.label().classes("text-body2 text-weight-bold q-mt-xs")
            etiqueta_error_local_bayes = ui.label().classes("text-caption text-grey-7")

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

            ui.label("¿Qué pasaría si...?").classes("text-subtitle2 q-mt-md")
            ui.label(
                "Para cada variable observada, qué predeciría la red si "
                "tomara otro valor o no se conociera en absoluto. Es la "
                "misma consulta al mismo modelo ya ajustado, con una "
                "evidencia distinta — no una medida de importancia ni de "
                "causalidad entre variables."
            ).classes("text-caption text-grey-7")
            contenedor_sensibilidad_bayes = ui.column().classes("w-full q-mt-xs")
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

            try:
                real = bayes_service.valor_real(caso)
                etiqueta_valor_real_bayes.set_text(
                    f"Lo que este estudiante realmente hizo: {real:.0f} participaciones "
                    f"en el trimestre completo."
                )
            except Exception:  # noqa: BLE001 — no debe romper el resto de la tarjeta
                etiqueta_valor_real_bayes.set_text("")

            error = bayes_service.error_local(resultado.prediccion_continua)
            if error["n"] > 0:
                etiqueta_error_local_bayes.set_text(
                    f"Margen de error esperado para predicciones de esta magnitud: "
                    f"± {error['mae_local']:.1f} participaciones en promedio "
                    f"(RMSE {error['rmse_local']:.1f}), estimado sobre {error['n']} casos "
                    f"similares de la validación cruzada — no es el RMSE/R² global del informe."
                )
            else:
                etiqueta_error_local_bayes.set_text("")

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

            # Sensibilidad: misma consulta, evidencia distinta -- ver
            # bayes_service.explorar_sensibilidad. Solo tiene sentido para
            # variables que sí entraron a la consulta (no las omitidas).
            contenedor_sensibilidad_bayes.clear()
            with contenedor_sensibilidad_bayes:
                for variable in resultado.evidencia_utilizada:
                    with ui.expansion(f"Si cambiara «{variable}»").classes("w-full"):
                        try:
                            escenarios = bayes_service.explorar_sensibilidad(caso, variable)
                        except Exception:  # noqa: BLE001 — no debe romper el resto de la tarjeta
                            ui.label(
                                "No se pudo calcular la sensibilidad de esta variable."
                            ).classes("text-caption text-negative")
                            continue
                        for escenario in escenarios:
                            texto = (
                                f"{escenario.valor_alternativo}: "
                                f"{escenario.prediccion_continua:.1f} participaciones"
                            )
                            if escenario.es_el_valor_observado:
                                texto += "  (valor observado en este caso)"
                            ui.label(texto).classes(
                                "text-body2 text-weight-bold" if escenario.es_el_valor_observado
                                else "text-body2"
                            )

            contenedor_resultado_bayes.set_visibility(True)

    tarjeta_seleccion_nueva = ui.card().classes("w-full max-w-2xl q-mt-md")
    with tarjeta_seleccion_nueva:
        ui.label("1. Estudiante nuevo (hipotético)").classes("text-lg font-semibold")
        ui.label(
            "Simula un estudiante que no está en los datos históricos, "
            "uniéndose a una sección real ya existente. El tamaño del "
            "grupo y el calendario de sesiones/evaluaciones/temas de cada "
            "semana se toman de esa sección real (son iguales para todos "
            "sus estudiantes, nunca se inventan); lo único que se pide a "
            "mano es lo que sí depende del estudiante: año que cursa, "
            "posición en la lista y su propia participación semana a "
            "semana."
        ).classes("text-caption text-grey-7 q-mb-sm")

        ui.label("¿Usar un estudiante real como plantilla?").classes("text-subtitle2")
        ui.label(
            "Opcional: elige un estudiante real para precargar año, "
            "posición y participaciones semanales con sus valores reales "
            "— quedan editables, para modificarlos y crear tu variante "
            "hipotética. También coloca al estudiante nuevo en la misma "
            "asignatura, trimestre, sección e hito que la plantilla."
        ).classes("text-caption text-grey-7 q-mb-xs")

        select_materia_plantilla = ui.select(opciones.materias(), label="Asignatura (plantilla)").classes("w-full")
        select_trimestre_plantilla = ui.select([], label="Trimestre (plantilla)").classes("w-full")
        select_seccion_plantilla = ui.select([], label="Sección (plantilla)").classes("w-full")
        select_estudiante_plantilla = ui.select([], label="Estudiante (plantilla)").classes("w-full")
        select_hito_plantilla = ui.select(
            {h: f"Semana {h}" for h in opciones.hitos()}, label="Hito (plantilla)"
        ).classes("w-full")
        for selector in (
            select_trimestre_plantilla, select_seccion_plantilla,
            select_estudiante_plantilla, select_hito_plantilla,
        ):
            selector.disable()

        boton_rellenar_plantilla = ui.button("Rellenar con este estudiante").classes("q-mt-xs")
        boton_rellenar_plantilla.disable()

        def _reiniciar_plantilla(desde: str) -> None:
            widgets = {
                "trimestre": select_trimestre_plantilla,
                "seccion": select_seccion_plantilla,
                "estudiante": select_estudiante_plantilla,
            }
            niveles = list(widgets)
            for nombre in niveles[niveles.index(desde):]:
                widget = widgets[nombre]
                widget.set_options([])
                widget.set_value(None)
                widget.disable()
            select_hito_plantilla.set_value(None)
            select_hito_plantilla.disable()
            boton_rellenar_plantilla.disable()

        def al_cambiar_materia_plantilla() -> None:
            _reiniciar_plantilla("trimestre")
            if select_materia_plantilla.value:
                select_trimestre_plantilla.set_options(opciones.trimestres(select_materia_plantilla.value))
                select_trimestre_plantilla.enable()

        def al_cambiar_trimestre_plantilla() -> None:
            _reiniciar_plantilla("seccion")
            if select_materia_plantilla.value and select_trimestre_plantilla.value:
                secciones = opciones.secciones(select_materia_plantilla.value, select_trimestre_plantilla.value)
                select_seccion_plantilla.set_options(secciones)
                select_seccion_plantilla.enable()

        def al_cambiar_seccion_plantilla() -> None:
            select_estudiante_plantilla.set_options([])
            select_estudiante_plantilla.set_value(None)
            select_hito_plantilla.set_value(None)
            select_hito_plantilla.disable()
            boton_rellenar_plantilla.disable()
            if select_materia_plantilla.value and select_trimestre_plantilla.value and select_seccion_plantilla.value:
                estudiantes = opciones.estudiantes(
                    select_materia_plantilla.value, select_trimestre_plantilla.value, select_seccion_plantilla.value
                )
                select_estudiante_plantilla.set_options(estudiantes)
                select_estudiante_plantilla.enable()

        def al_cambiar_estudiante_plantilla() -> None:
            select_hito_plantilla.set_value(None)
            boton_rellenar_plantilla.disable()
            if select_estudiante_plantilla.value:
                select_hito_plantilla.enable()

        def al_cambiar_hito_plantilla() -> None:
            boton_rellenar_plantilla.set_enabled(bool(select_hito_plantilla.value))

        select_materia_plantilla.on_value_change(al_cambiar_materia_plantilla)
        select_trimestre_plantilla.on_value_change(al_cambiar_trimestre_plantilla)
        select_seccion_plantilla.on_value_change(al_cambiar_seccion_plantilla)
        select_estudiante_plantilla.on_value_change(al_cambiar_estudiante_plantilla)
        select_hito_plantilla.on_value_change(al_cambiar_hito_plantilla)

        ui.separator().classes("q-my-md")

        select_materia_n = ui.select(opciones.materias(), label="Asignatura").classes("w-full")
        select_trimestre_n = ui.select([], label="Trimestre").classes("w-full")
        select_seccion_n = ui.select([], label="Sección").classes("w-full")
        select_hito_n = ui.select(
            {h: f"Semana {h}" for h in opciones.hitos()}, label="Hito"
        ).classes("w-full")
        for selector in (select_trimestre_n, select_seccion_n, select_hito_n):
            selector.disable()

        numero_anio_n = ui.number(
            label="Año que cursa (dejar vacío si se desconoce)", min=1, max=6, step=1
        ).classes("w-full")
        numero_posicion_n = ui.number(
            label="Posición relativa en la lista (0 a 1)", min=0, max=1, step=0.01
        ).classes("w-full")

        ui.label("Participaciones por semana").classes("text-subtitle2 q-mt-sm")
        contenedor_semanas_n = ui.row().classes("w-full q-gutter-sm")
        campos_semanas_n: list = []

        def _reconstruir_semanas_n() -> None:
            contenedor_semanas_n.clear()
            campos_semanas_n.clear()
            if not select_hito_n.value:
                return
            with contenedor_semanas_n:
                for semana in range(1, select_hito_n.value + 1):
                    campo = ui.number(label=f"Semana {semana}", min=0, step=1, value=0).classes("w-24")
                    campos_semanas_n.append(campo)

        def rellenar_con_plantilla() -> None:
            caso_plantilla = CasoPrediccion(
                materia=select_materia_plantilla.value,
                trimestre=select_trimestre_plantilla.value,
                seccion=select_seccion_plantilla.value,
                estudiante_id=select_estudiante_plantilla.value,
                hito=select_hito_plantilla.value,
            )
            try:
                valores = lstm_service.valores_reales_estudiante(caso_plantilla)
            except Exception:  # noqa: BLE001
                ui.notify("No se pudo leer los valores reales de este estudiante.", type="negative")
                return

            # Coloca al estudiante nuevo en la misma asignatura/trimestre/
            # sección/hito de la plantilla, disparando la misma cadena de
            # selección que si el usuario la hubiera elegido a mano.
            select_materia_n.set_value(caso_plantilla.materia)
            al_cambiar_materia_n()
            select_trimestre_n.set_value(caso_plantilla.trimestre)
            al_cambiar_trimestre_n()
            select_seccion_n.set_value(caso_plantilla.seccion)
            al_cambiar_seccion_n()
            select_hito_n.set_value(caso_plantilla.hito)
            al_cambiar_hito_n()

            numero_anio_n.set_value(valores["anio_academico"])
            numero_posicion_n.set_value(round(valores["posicion_lista"], 3))
            for campo, valor in zip(campos_semanas_n, valores["participaciones_semanales"]):
                campo.set_value(valor)

            ui.notify(
                f"Campos rellenados con los valores reales de {caso_plantilla.estudiante_id} "
                f"— quedan editables.",
                type="positive",
            )

        boton_rellenar_plantilla.on_click(rellenar_con_plantilla)

        ui.label("Resumen de variables de entrada").classes("text-subtitle2 q-mt-md")
        ui.label(
            "Las 9 variables que recibe la red bayesiana (más el año, que "
            "puede quedar vacío): cuáles se fijan por la sección real "
            "elegida y cuáles ingresaste a mano."
        ).classes("text-caption text-grey-7")
        contenedor_resumen_variables_n = ui.column().classes("w-full q-mt-xs")
        contenedor_resumen_variables_n.set_visibility(False)

        boton_generar_n = ui.button("Generar predicción (estudiante nuevo)")
        boton_generar_n.disable()

        etiqueta_caso_n = ui.label().classes("text-caption text-grey-7 q-mt-sm")

        def _reiniciar_n(desde: str) -> None:
            widgets_con_opciones_dependientes = {
                "trimestre": select_trimestre_n,
                "seccion": select_seccion_n,
            }
            niveles = list(widgets_con_opciones_dependientes)
            for nombre in niveles[niveles.index(desde):]:
                widget = widgets_con_opciones_dependientes[nombre]
                widget.set_options([])
                widget.set_value(None)
                widget.disable()
            select_hito_n.set_value(None)
            select_hito_n.disable()
            boton_generar_n.disable()
            etiqueta_caso_n.set_text("")
            _reconstruir_semanas_n()
            _limpiar_resultado_lstm_n()
            _limpiar_resultado_bayes_n()

        def al_cambiar_materia_n() -> None:
            _reiniciar_n("trimestre")
            if select_materia_n.value:
                select_trimestre_n.set_options(opciones.trimestres(select_materia_n.value))
                select_trimestre_n.enable()

        def al_cambiar_trimestre_n() -> None:
            _reiniciar_n("seccion")
            if select_materia_n.value and select_trimestre_n.value:
                secciones = opciones.secciones(select_materia_n.value, select_trimestre_n.value)
                select_seccion_n.set_options(secciones)
                select_seccion_n.enable()

        def al_cambiar_seccion_n() -> None:
            select_hito_n.set_value(None)
            boton_generar_n.disable()
            etiqueta_caso_n.set_text("")
            _reconstruir_semanas_n()
            _limpiar_resultado_lstm_n()
            _limpiar_resultado_bayes_n()
            if select_seccion_n.value:
                select_hito_n.enable()

        def al_cambiar_hito_n() -> None:
            _reconstruir_semanas_n()
            _limpiar_resultado_lstm_n()
            _limpiar_resultado_bayes_n()
            boton_generar_n.set_enabled(bool(select_hito_n.value))

        select_materia_n.on_value_change(al_cambiar_materia_n)
        select_trimestre_n.on_value_change(al_cambiar_trimestre_n)
        select_seccion_n.on_value_change(al_cambiar_seccion_n)
        select_hito_n.on_value_change(al_cambiar_hito_n)

        def generar_n() -> None:
            if numero_posicion_n.value is None:
                ui.notify("Ingresa la posición relativa en la lista.", type="warning")
                return
            materia, trimestre, seccion, hito = (
                select_materia_n.value, select_trimestre_n.value,
                select_seccion_n.value, select_hito_n.value,
            )
            anio = numero_anio_n.value
            posicion = numero_posicion_n.value
            participaciones = [campo.value or 0 for campo in campos_semanas_n]
            etiqueta_caso_n.set_text(
                f"Estudiante nuevo en: {materia} · {trimestre} · sección {seccion} · "
                f"semana {hito}"
            )

            # Resumen de las 9 variables de entrada -- cuáles se fijan por
            # la sección real (tamaño de grupo, sesiones/evaluaciones/tema
            # de cada semana) y cuáles se ingresaron a mano.
            contenedor_resumen_variables_n.clear()
            try:
                horario = lstm_service.horario_real_seccion(materia, trimestre, seccion, hito)
                with contenedor_resumen_variables_n:
                    ui.label(
                        f"Fijado por la sección real ({materia} · {trimestre} · sección {seccion}):"
                    ).classes("text-body2 text-weight-bold")
                    ui.label(f"Tamaño del grupo: {horario['tamano_grupo']:.0f} estudiantes").classes("text-body2")
                    for semana_info in horario["semanas"]:
                        ui.label(
                            f"Semana {semana_info['semana']}: "
                            f"{semana_info['sesiones']:.0f} sesión(es), "
                            f"{semana_info['evaluaciones']:.0f} de evaluación, "
                            f"temas {semana_info['tema_1']}/{semana_info['tema_2']}"
                        ).classes("text-caption text-grey-7")

                    ui.label("Ingresado a mano:").classes("text-body2 text-weight-bold q-mt-xs")
                    ui.label(
                        f"Año que cursa: {anio if anio is not None else '(sin dato)'}"
                    ).classes("text-body2")
                    ui.label(f"Posición relativa en la lista: {posicion:.3f}").classes("text-body2")
                    ui.label(
                        "Participaciones por semana: "
                        + ", ".join(f"sem. {i + 1}: {p:.0f}" for i, p in enumerate(participaciones))
                    ).classes("text-body2")
                contenedor_resumen_variables_n.set_visibility(True)
            except Exception:  # noqa: BLE001 — el resumen no debe romper la generación
                contenedor_resumen_variables_n.set_visibility(False)

            _ejecutar_lstm_manual(materia, trimestre, seccion, hito, anio, posicion, participaciones)
            _ejecutar_bayes_manual(materia, trimestre, seccion, hito, anio, participaciones)

        boton_generar_n.on_click(generar_n)

    tarjeta_lstm_nueva = ui.card().classes("w-full max-w-2xl q-mt-md")
    with tarjeta_lstm_nueva:
        ui.label("2. Resultado LSTM (estudiante nuevo)").classes("text-lg font-semibold")
        ui.label("Modelo de contraste").classes("text-caption text-grey-7")

        estado_lstm_n = ui.label(
            "Completa los datos y presiona «Generar predicción (estudiante nuevo)»."
        ).classes("text-body2 text-grey-7 q-mt-sm")

        contenedor_prediccion_lstm_n = ui.column().classes("q-mt-sm")
        with contenedor_prediccion_lstm_n:
            etiqueta_prediccion_lstm_n = ui.label().classes("text-h4 text-weight-bold text-primary")
            etiqueta_frase_lstm_n = ui.label().classes("text-body1")
            etiqueta_contexto_lstm_n = ui.label().classes("text-caption text-grey-7")
            etiqueta_error_local_lstm_n = ui.label().classes("text-caption text-grey-7")
        contenedor_prediccion_lstm_n.set_visibility(False)

        def _limpiar_resultado_lstm_n() -> None:
            estado_lstm_n.set_text(
                "Completa los datos y presiona «Generar predicción (estudiante nuevo)»."
            )
            estado_lstm_n.classes(replace="text-body2 text-grey-7 q-mt-sm")
            contenedor_prediccion_lstm_n.set_visibility(False)

        def _ejecutar_lstm_manual(materia, trimestre, seccion, hito, anio, posicion, participaciones) -> None:
            contenedor_prediccion_lstm_n.set_visibility(False)
            estado_lstm_n.set_text("Calculando predicción…")
            estado_lstm_n.classes(replace="text-body2 text-grey-7 q-mt-sm")
            try:
                resultado = lstm_service.predict_lstm_manual(
                    materia, trimestre, seccion, hito, anio, posicion, participaciones
                )
            except (FileNotFoundError, ValueError) as error:
                estado_lstm_n.set_text(f"No se pudo calcular la predicción: {error}")
                estado_lstm_n.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("La LSTM no pudo evaluar este estudiante nuevo.", type="negative")
                return
            except Exception:  # noqa: BLE001 — error controlado, no se muestra el traceback
                estado_lstm_n.set_text("Ocurrió un error inesperado ejecutando la LSTM.")
                estado_lstm_n.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("Error inesperado en la inferencia LSTM.", type="negative")
                return

            estado_lstm_n.set_text("")
            etiqueta_prediccion_lstm_n.set_text(f"{resultado.prediccion_total:.1f} participaciones")
            etiqueta_frase_lstm_n.set_text(
                f"Total trimestral previsto para la semana {resultado.hito} "
                "(estudiante hipotético, no un registro histórico)."
            )
            etiqueta_contexto_lstm_n.set_text(
                f"{resultado.materia} · {resultado.trimestre} · sección {resultado.seccion} · "
                f"información hasta la semana {resultado.hito}"
            )

            error = lstm_service.error_local(resultado.prediccion_total)
            if error["n"] > 0:
                etiqueta_error_local_lstm_n.set_text(
                    f"Margen de error esperado para predicciones de esta magnitud: "
                    f"± {error['mae_local']:.1f} participaciones en promedio "
                    f"(RMSE {error['rmse_local']:.1f}), estimado sobre {error['n']} casos "
                    f"similares de la validación cruzada — no es el RMSE/R² global del informe."
                )
            else:
                etiqueta_error_local_lstm_n.set_text("")

            contenedor_prediccion_lstm_n.set_visibility(True)

    tarjeta_bayes_nueva = ui.card().classes("w-full max-w-2xl q-mt-md")
    with tarjeta_bayes_nueva:
        ui.label("3. Resultado red bayesiana (estudiante nuevo)").classes("text-lg font-semibold")
        ui.label("Modelo principal").classes("text-caption text-grey-7")

        estado_bayes_n = ui.label(
            "Completa los datos y presiona «Generar predicción (estudiante nuevo)»."
        ).classes("text-body2 text-grey-7 q-mt-sm")

        contenedor_resultado_bayes_n = ui.column().classes("w-full q-mt-sm")
        with contenedor_resultado_bayes_n:
            etiqueta_prediccion_bayes_n = ui.label().classes("text-h4 text-weight-bold text-primary")
            etiqueta_frase_bayes_n = ui.label().classes("text-body1")
            etiqueta_contexto_bayes_n = ui.label().classes("text-caption text-grey-7")
            etiqueta_error_local_bayes_n = ui.label().classes("text-caption text-grey-7")

            ui.label("Distribución posterior").classes("text-subtitle2 q-mt-md")
            contenedor_posterior_bayes_n = ui.column().classes("w-full q-mt-xs")

            ui.label("Evidencia observada utilizada por la red").classes("text-subtitle2 q-mt-md")
            contenedor_evidencia_bayes_n = ui.column().classes("q-mt-xs")

            etiqueta_evidencia_omitida_n = ui.label().classes("text-caption text-warning q-mt-sm")

            ui.label("¿Qué pasaría si...?").classes("text-subtitle2 q-mt-md")
            ui.label(
                "Igual que en la tarjeta anterior: la misma consulta al "
                "mismo modelo ya ajustado, con una evidencia distinta — no "
                "una medida de importancia ni de causalidad."
            ).classes("text-caption text-grey-7")
            contenedor_sensibilidad_bayes_n = ui.column().classes("w-full q-mt-xs")
        contenedor_resultado_bayes_n.set_visibility(False)

        def _limpiar_resultado_bayes_n() -> None:
            estado_bayes_n.set_text(
                "Completa los datos y presiona «Generar predicción (estudiante nuevo)»."
            )
            estado_bayes_n.classes(replace="text-body2 text-grey-7 q-mt-sm")
            contenedor_resultado_bayes_n.set_visibility(False)

        def _ejecutar_bayes_manual(materia, trimestre, seccion, hito, anio, participaciones) -> None:
            contenedor_resultado_bayes_n.set_visibility(False)
            estado_bayes_n.set_text("Calculando predicción…")
            estado_bayes_n.classes(replace="text-body2 text-grey-7 q-mt-sm")

            participacion_actual = participaciones[hito - 1]
            participacion_anterior = participaciones[hito - 2] if hito >= 2 else None
            try:
                resultado = bayes_service.predict_bayes_manual(
                    materia, trimestre, seccion, anio, participacion_actual, participacion_anterior
                )
            except (FileNotFoundError, ValueError) as error:
                estado_bayes_n.set_text(f"No se pudo calcular la predicción: {error}")
                estado_bayes_n.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("La red bayesiana no pudo evaluar este estudiante nuevo.", type="negative")
                return
            except Exception:  # noqa: BLE001 — error controlado, no se muestra el traceback
                estado_bayes_n.set_text(
                    "Ocurrió un error inesperado ejecutando la red bayesiana."
                )
                estado_bayes_n.classes(replace="text-body2 text-negative q-mt-sm")
                ui.notify("Error inesperado en la inferencia Bayes.", type="negative")
                return

            estado_bayes_n.set_text("")
            etiqueta_prediccion_bayes_n.set_text(f"{resultado.prediccion_continua:.1f} participaciones")
            etiqueta_frase_bayes_n.set_text(
                "Total trimestral estimado (valor esperado de la distribución posterior) "
                "para un estudiante hipotético."
            )
            etiqueta_contexto_bayes_n.set_text(
                f"{resultado.materia} · {resultado.trimestre} · sección {resultado.seccion}"
            )

            error = bayes_service.error_local(resultado.prediccion_continua)
            if error["n"] > 0:
                etiqueta_error_local_bayes_n.set_text(
                    f"Margen de error esperado para predicciones de esta magnitud: "
                    f"± {error['mae_local']:.1f} participaciones en promedio "
                    f"(RMSE {error['rmse_local']:.1f}), estimado sobre {error['n']} casos "
                    f"similares de la validación cruzada — no es el RMSE/R² global del informe."
                )
            else:
                etiqueta_error_local_bayes_n.set_text("")

            contenedor_posterior_bayes_n.clear()
            with contenedor_posterior_bayes_n:
                for estado, probabilidad in resultado.posterior.items():
                    with ui.row().classes("w-full items-center no-wrap"):
                        ui.label(estado).style("width: 72px")
                        ui.linear_progress(value=probabilidad, show_value=False).classes("flex-grow")
                        ui.label(f"{probabilidad * 100:.1f}%").style("width: 56px; text-align: right")

            contenedor_evidencia_bayes_n.clear()
            with contenedor_evidencia_bayes_n:
                for variable, valor in resultado.evidencia_utilizada.items():
                    ui.label(f"{variable}: {valor}").classes("text-body2")

            if resultado.evidencia_omitida:
                etiqueta_evidencia_omitida_n.set_text(
                    "Evidencia no disponible para esta inferencia: "
                    + ", ".join(resultado.evidencia_omitida)
                    + ". La red marginaliza esta variable; la inferencia es válida, no es un error."
                )
            else:
                etiqueta_evidencia_omitida_n.set_text("")

            contenedor_sensibilidad_bayes_n.clear()
            with contenedor_sensibilidad_bayes_n:
                for variable in resultado.evidencia_utilizada:
                    with ui.expansion(f"Si cambiara «{variable}»").classes("w-full"):
                        try:
                            escenarios = bayes_service.explorar_sensibilidad_manual(
                                resultado.materia, resultado.evidencia_utilizada, variable
                            )
                        except Exception:  # noqa: BLE001 — no debe romper el resto de la tarjeta
                            ui.label(
                                "No se pudo calcular la sensibilidad de esta variable."
                            ).classes("text-caption text-negative")
                            continue
                        for escenario in escenarios:
                            texto = (
                                f"{escenario.valor_alternativo}: "
                                f"{escenario.prediccion_continua:.1f} participaciones"
                            )
                            if escenario.es_el_valor_observado:
                                texto += "  (valor observado en este caso)"
                            ui.label(texto).classes(
                                "text-body2 text-weight-bold" if escenario.es_el_valor_observado
                                else "text-body2"
                            )

            contenedor_resultado_bayes_n.set_visibility(True)

    def al_cambiar_modo() -> None:
        es_historico = modo.value == "Estudiante histórico"
        for tarjeta in (tarjeta_seleccion_historica, tarjeta_lstm_historica, tarjeta_bayes_historica):
            tarjeta.set_visibility(es_historico)
        for tarjeta in (tarjeta_seleccion_nueva, tarjeta_lstm_nueva, tarjeta_bayes_nueva):
            tarjeta.set_visibility(not es_historico)

    modo.on_value_change(al_cambiar_modo)
    al_cambiar_modo()


ui.run(title="Prototipo — participación estudiantil", reload=False)
