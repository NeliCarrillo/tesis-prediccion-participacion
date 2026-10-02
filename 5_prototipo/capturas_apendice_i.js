// Capturas del Apéndice I (prototipo): recorta cada tarjeta de la interfaz en ejecución y las
// guarda en figuras/apendice_I_prototipo/ (a 2x, 672 px CSS de ancho, como las del informe).
//
// Uso (con el prototipo corriendo en http://localhost:8080):
//   cd 5_prototipo && python3 app.py            # en otra terminal
//   NODE_PATH=<ruta a node_modules con playwright> node capturas_apendice_i.js [carpeta_de_salida]
//
// Reproduce el flujo del apéndice: estudiante registrado anon_001 (Algoritmos y Programación,
// 2425-2, sección 1, semana 6) y estudiante nuevo precargado de anon_003 en la misma sección e hito.
// El desplegable de la LSTM con la predicción de cada semilla se deja abierto.
const { chromium } = require('playwright');
const SALIDA = process.argv[2] || "../figuras/apendice_I_prototipo";
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 900, height: 1600 }, deviceScaleFactor: 2 });
  await p.goto('http://localhost:8080/');
  await p.waitForTimeout(1500);
  await p.addStyleTag({ content: '.q-notifications__list, .q-notification { display: none !important; }' });

  const elegir = async (etiqueta, opcion) => {
    await p.locator('.q-field').filter({ hasText: etiqueta, visible: true }).first().click();
    await p.getByRole('option', { name: opcion, exact: true }).click();
    await p.waitForTimeout(400);
  };
  const tarjeta = (titulo) => p.locator('.q-card').filter({ hasText: titulo, visible: true }).first();
  const caja = async (loc) => {
    return await loc.evaluate(e => { const r = e.getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height }; });
  };
  // recorta [desde, hasta] (px CSS relativos a la tarjeta) y guarda
  const recorte = async (card, desde, hasta, nombre) => {
    const c = await caja(card);
    await p.screenshot({ path: `${SALIDA}/${nombre}.png`, fullPage: true,
      clip: { x: c.x, y: c.y + desde, width: c.w, height: hasta - desde } });
    console.log(nombre, Math.round(hasta - desde) * 2);
  };
  const rel = async (card, loc) => {  // posición vertical de un elemento relativa a la tarjeta
    const c = await caja(card); const e = await caja(loc);
    return { top: e.y - c.y, bottom: e.y + e.h - c.y };
  };
  const completa = async (card, nombre) => { const c = await caja(card); await recorte(card, 0, c.h, nombre); };
  const texto = (card, patron) => card.getByText(patron, { exact: false }).first();

  // ---------- Estudiante histórico ----------
  await elegir('Asignatura', 'Algoritmos y Programación');
  await elegir('Trimestre', '2425-2');
  await elegir('Sección', '1');
  await elegir('Estudiante (anonimizado)', 'anon_001');
  await elegir('Hito', 'Semana 6');
  await p.waitForTimeout(800);
  await p.getByRole('button', { name: 'Generar predicción', exact: true }).click();
  await p.getByText('Según la semilla de entrenamiento').first().waitFor({ timeout: 120000 });
  await p.getByText('Distribución posterior').first().waitFor({ timeout: 120000 });
  await p.getByText('Ver la predicción de cada semilla').filter({ visible: true }).first().click();
  await p.waitForTimeout(1500);
  await p.evaluate(() => window.scrollTo(0, 0));

  let c1 = tarjeta('1. Seleccionar caso');
  let s3 = await rel(c1, c1.getByText(/^Semana 3: /)); let s4 = await rel(c1, c1.getByText(/^Semana 4: /));
  const corte1 = (s3.bottom + s4.top) / 2;
  const alto1 = (await caja(c1)).h;
  await recorte(c1, 0, corte1, 'historico_1_seleccion_a');
  await recorte(c1, corte1, alto1, 'historico_1_seleccion_b');
  await completa(tarjeta('2. Resultado LSTM'), 'historico_2_lstm');
  let c3 = tarjeta('3. Resultado red bayesiana');
  let ev = await rel(c3, c3.getByText(/^Año que cursa: /).first());
  let sim = await rel(c3, c3.getByText('¿Qué pasaría si...?').first());
  const alto3 = (await caja(c3)).h;
  await recorte(c3, 0, ev.bottom + 8, 'historico_3_bayes_a');
  await recorte(c3, sim.top - 40, alto3, 'historico_3_bayes_b');

  // ---------- Estudiante nuevo ----------
  await p.getByRole('button', { name: 'Estudiante nuevo', exact: true }).click();
  await p.waitForTimeout(800);
  await p.getByRole('button', { name: 'Partir de un estudiante real' }).click();
  await p.waitForTimeout(800);
  await elegir('Asignatura', 'Algoritmos y Programación');
  await elegir('Trimestre', '2425-2');
  await elegir('Sección', '1');
  await elegir('Estudiante real de partida', 'anon_003');
  await elegir('Hito', 'Semana 6');
  await p.getByRole('button', { name: 'Precargar valores de este estudiante' }).click();
  await p.locator('.q-notification').first().waitFor({ state: 'detached', timeout: 20000 }).catch(() => {});
  await p.waitForTimeout(1000);
  await p.getByRole('button', { name: 'Generar predicción (estudiante nuevo)' }).click();
  await p.getByText('Según la semilla de entrenamiento').filter({ visible: true }).first().waitFor({ timeout: 120000 });
  await p.getByText('Distribución posterior').filter({ visible: true }).first().waitFor({ timeout: 120000 });
  await p.getByText('Ver la predicción de cada semilla').filter({ visible: true }).first().click();
  await p.waitForTimeout(1500);
  await p.evaluate(() => window.scrollTo(0, 0));

  let n1 = tarjeta('1. Estudiante nuevo');
  const alto_n1 = (await caja(n1)).h;
  let hito = await rel(n1, n1.locator('.q-field').filter({ hasText: 'Hito' }).first());
  let boton = await rel(n1, n1.getByRole('button', { name: 'Precargar valores de este estudiante' }));
  let w4 = await rel(n1, n1.getByText(/^Semana 4: \d/)); let w5 = await rel(n1, n1.getByText(/^Semana 5: \d/));
  await recorte(n1, 0, hito.bottom + 10, 'nuevo_1_formulario_a');
  await recorte(n1, boton.top - 10, (w4.bottom + w5.top) / 2, 'nuevo_1_formulario_b');
  await recorte(n1, (w4.bottom + w5.top) / 2, alto_n1, 'nuevo_1_formulario_c');
  await completa(tarjeta('2. Resultado LSTM (estudiante nuevo)'), 'nuevo_2_lstm');
  let n3 = tarjeta('3. Resultado red bayesiana (estudiante nuevo)');
  let simn = await rel(n3, n3.getByText('¿Qué pasaría si...?').first());
  let desc = await rel(n3, n3.getByText('Cambia uno o varios valores').first());
  const alto_n3 = (await caja(n3)).h;
  const corte3 = (simn.bottom + desc.top) / 2;
  await recorte(n3, 0, corte3, 'nuevo_3_bayes_a');
  await recorte(n3, corte3, alto_n3, 'nuevo_3_bayes_b');
  await b.close();
})();
