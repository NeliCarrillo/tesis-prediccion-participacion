# Prototipo de la tesis

Esta carpeta es la aplicación que permite escoger un estudiante real (asignatura,
trimestre, sección, estudiante e hito) y ver, para ese caso concreto, lo que
predicen los dos modelos de la tesis: la LSTM y la red bayesiana. Es un
prototipo académico para la defensa, no una aplicación en producción: una
sola pantalla, sin usuarios, sin base de datos, sin historial.

## Cómo levantarlo

```bash
cd prototipo
pip install -r requirements.txt   # solo la primera vez
python3 app.py
```

Abre `http://localhost:8080` en el navegador. Selecciona asignatura →
trimestre → sección → estudiante → hito (cada selector se habilita cuando
el anterior tiene un valor, así que nunca puedes armar una combinación que
no exista) y presiona "Generar predicción".

## Qué vas a ver

- **Tarjeta LSTM**: un solo número — el total de participaciones que la
  LSTM predice para todo el trimestre.
- **Tarjeta red bayesiana**: el mismo tipo de número (el "valor esperado"),
  más la distribución de probabilidad completa sobre los cinco estados
  posibles del objetivo, más qué información (evidencia) usó la red para
  llegar a esa predicción — y si le faltó algún dato, te lo dice
  explícitamente en vez de inventarlo.

## Qué es cada archivo (guía para entender la carpeta)

```
prototipo/
├── app.py                     la aplicación en sí (lo que ves en el navegador)
├── caso.py                    quién es "el estudiante seleccionado"
├── services/
│   ├── lstm_service.py        cómo se le pregunta a la LSTM
│   └── bayes_service.py       cómo se le pregunta a la red bayesiana
├── artefactos_lstm/           los 12 modelos LSTM ya entrenados, listos para usar
├── entrenar_modelos_finales.py   el script que generó esos 12 modelos
├── requirements.txt            lo mínimo para correr app.py
└── requirements-dev.txt        lo anterior + lo necesario solo para reentrenar
```

### `app.py`

Es la pantalla. Dibuja los selectores, arma el caso elegido y, cuando
presionas "Generar predicción", le pasa ese caso a `lstm_service.py` y a
`bayes_service.py` y muestra lo que cada uno responde. No hace ningún
cálculo por su cuenta: si mañana cambia cómo predice la LSTM o la red
bayesiana, este archivo no debería necesitar tocarse.

### `caso.py`

Define qué es "un caso": una asignatura + trimestre + sección + estudiante
+ hito (semana 4, 6 u 8). También sabe qué combinaciones existen de verdad
(consultando los datos reales), para que los selectores de `app.py` nunca
ofrezcan una opción que no tenga datos detrás.

### `services/lstm_service.py`

Es el "traductor" entre un caso elegido y una predicción de la LSTM. Toma
el caso, reconstruye exactamente la misma secuencia de datos que se usó
para entrenar el modelo (mismas semanas, mismo orden de variables, misma
forma de rellenar huecos), la pasa por el modelo ya entrenado, y devuelve
un solo número: el total de participaciones previsto para el trimestre.
Ninguna de estas transformaciones se inventó para el prototipo — son
exactamente las del notebook validado (`LSTM/nuevo/adaptacion_lstm_participaciones.ipynb`),
copiadas de ahí sin cambiarles ni una línea.

### `services/bayes_service.py`

El equivalente para la red bayesiana: toma el caso, arma la evidencia que
la red necesita (año del estudiante, tamaño del grupo, participaciones de
esa semana y de la anterior — si alguna falta, la omite en vez de
inventarla), consulta el modelo, y devuelve tanto la probabilidad de cada
uno de los cinco posibles resultados como el promedio esperado. Reutiliza
directamente el código de `RedBayesiana/codigo_red/` — no reimplementa
nada de la red bayesiana.

### `artefactos_lstm/`

Aquí viven los modelos LSTM **ya entrenados**, listos para responder al
instante sin tener que reentrenar nada cada vez que abres el prototipo.
Hay una carpeta por cada combinación de asignatura y semana (4 asignaturas
× 3 hitos = 12 carpetas, por ejemplo `algoritmos_y_programacion_h4/`).
Dentro de cada una:

| Archivo | Qué es, en palabras simples |
|---|---|
| `modelo.keras` | La red neuronal ya entrenada para esa asignatura y esa semana. |
| `escalador_x.json` | Los números que se usan para poner todas las variables en una escala comparable antes de dárselas al modelo (es la misma normalización que se usó al entrenar, guardada para poder repetirla exactamente). |
| `objetivo.json` | La media y la desviación que se usaron para "destraducir" la salida del modelo a un número real de participaciones. |
| `imputacion.json` | Qué valor se usa si a un estudiante le falta el año académico (le pasa a muy pocos casos). |

Y en la raíz de la carpeta, `config_compartida.json` anota con qué
variables, semillas y configuración se entrenaron los 12 modelos, para que
quede registrado cómo se hicieron.

**¿Por qué existen estos modelos y no son los mismos del informe?**
Los modelos que sustentan el RMSE/R² del informe se entrenaron
**excluyendo un trimestre a la vez** (así se mide qué tan bien predice
sobre datos que no vio) — esos nunca se guardan, son parte del proceso de
evaluación. Los modelos de esta carpeta son distintos a propósito: se
entrenaron **con todos los trimestres disponibles**, porque el prototipo
necesita responder sobre cualquier caso, incluidos los que en la
evaluación fueron "de prueba". Por esta razón, una predicción del
prototipo sobre un estudiante real **no debe leerse como una nueva medición
de desempeño** — esa medición ya está hecha y reportada en el informe.

**¿Por qué la red bayesiana no tiene una carpeta equivalente?** Porque no
lo necesita: ajustar la red bayesiana con todos los datos de una
asignatura tarda menos de un segundo (a diferencia de entrenar la LSTM,
que tarda varios segundos por combinación), así que el prototipo la
recalcula en el momento, cada vez que arranca, en vez de guardarla en
disco.

### `entrenar_modelos_finales.py`

Es el script que generó todo lo que hay en `artefactos_lstm/`. Si alguna
vez hace falta regenerarlos (por ejemplo, si se agrega una asignatura
nueva), se corre así:

```bash
python3 entrenar_modelos_finales.py --salida artefactos_lstm
```

El parámetro `--salida` es obligatorio a propósito, para que nunca se
sobrescriban los modelos actuales sin querer — se puede apuntar a una
carpeta de prueba primero para revisar el resultado.

### `requirements.txt` / `requirements-dev.txt`

`requirements.txt` es lo único que hace falta instalar para *usar* el
prototipo (`app.py`). `requirements-dev.txt` agrega lo que hace falta
además para *entrenar* los modelos finales (`entrenar_modelos_finales.py`)
— no lo necesitas si solo vas a mostrar el prototipo en la defensa.

## Cómo sabemos que esto funciona bien

Antes de conectar cada servicio a la interfaz, se comprobó, ejecutando
código real (nunca simulado):

- Que `lstm_service.py` prepara los datos exactamente igual que el
  notebook validado — comparando, celda por celda, los mismos números.
- Que la predicción final de `lstm_service.py` coincide, para varios
  estudiantes reales (incluido uno con año académico faltante), con una
  segunda implementación independiente de la misma fórmula.
- Que la evidencia que arma `bayes_service.py` para un estudiante es
  idéntica a la que ya quedó registrada en
  `RedBayesiana/resultados/predicciones_bayesiana_s4_s6_s8.csv` durante
  la validación de Sprint 4 — incluido el caso con año académico
  faltante, donde la red omite esa variable en vez de inventarla.
- Que la distribución de probabilidad que devuelve la red bayesiana
  siempre suma 1 y que el promedio esperado siempre es un número válido.

Esas comprobaciones se hicieron con scripts que ya no están en el
repositorio (eran para verificación puntual, no para uso diario) — queda
este párrafo como registro de que se hicieron y de qué confirmaron.
