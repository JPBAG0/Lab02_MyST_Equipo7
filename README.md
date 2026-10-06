# Lab02_MyST_Equipo7

**Integrantes:** Gian Carlo Campos Sayavedra (GCCS11) y Juan Pablo Barba González (JPBAG0)

**Nivel de alcance:** B (2 integrantes)

## Descripción

Estrategia sistemática sobre BTCUSDT con barras de 5 minutos (archivos publicados en Canvas, de los que se usan tres periodos sin huecos: entrenamiento de julio a noviembre de 2023, prueba en diciembre de 2023 y validación de mayo a junio de 2024). Combina tres indicadores de tres familias (Donchian como tendencia, ROC como momento y Keltner como volatilidad) con una regla de confirmación de 2 de 3, se evalúa en un motor de backtesting orientado a eventos con comisión de 0.125% por operación y ejecución en la apertura de la barra siguiente, y optimiza sus hiperparámetros con Optuna (TPE) maximizando el Calmar mediante walk-forward semanal (1 mes de entrenamiento y 1 semana de prueba). Como nivel B, detecta regímenes de mercado (reversión o rango, tendencia y crisis) con reglas sobre la volatilidad y el R² de una ventana de una semana, y compara un único conjunto de parámetros contra un conjunto por régimen.

## Regla de confirmación y protocolo

La regla de confirmación es: sea L el número de indicadores con señal +1 y S el número con señal -1; la señal es +1 si L >= 2, -1 si S >= 2 y 0 en otro caso.

Periodos de trabajo (UTC), acordados con el profesor porque los archivos de Canvas tienen huecos y estos tres periodos no los tienen:

| Conjunto | Periodo | Archivo de origen | Uso |
|---|---|---|---|
| Entrenamiento | 2023-07-01 a 2023-11-30 | `btc_project_train.csv` | optimización y walk-forward semanal |
| Prueba | 2023-12-01 a 2023-12-31 | `btc_project_train.csv` | evaluación con parámetros congelados |
| Validación | 2024-05-02 a 2024-06-03 | `btc_project_test.csv` | evaluación con parámetros congelados |

Los parámetros finales se optimizaron una sola vez sobre todo el entrenamiento, se guardaron en `docs/theta_frozen.json` y se commitearon (commit `28604b0`, subido a `main`) **antes** de evaluar en prueba y en validación, que se evaluaron una sola vez (resultados en el commit `dcef66d`). El orden de los commits demuestra la secuencia, no la intención. Además, ningún periodo es virgen: la versión anterior del proyecto (periodos con huecos, commit `8202c03`) ya había visto los tres, y el cambio de periodos se decidió después de ver esos resultados. Está declarado en `docs/EXPERIMENTS.md` y en el reporte.

## Resultados principales

- Walk-forward semanal sobre entrenamiento (17 semanas fuera de muestra, julio a noviembre de 2023): un solo conjunto de parámetros con filtro de calidad -7.0% (43 operaciones); un conjunto por régimen -9.0% (101 operaciones); comprar y mantener +29.1%.
- Parámetros congelados: el conjunto único no pasó el filtro de calidad (ninguna configuración con 30 o más operaciones tuvo Calmar positivo en el entrenamiento), así que no opera. El conjunto por régimen rinde -3.1% en prueba (19 operaciones; comprar y mantener +11.9%) y -3.0% en validación (18 operaciones; comprar y mantener +16.4%).
- No hay evidencia de ventaja: dentro de muestra el conjunto por régimen rinde +8.8%, y fuera de ella el retorno bruto por operación es nulo o negativo, por debajo del costo de ida y vuelta (0.25%).
- No hay diferencia significativa de retorno por operación entre reversión y tendencia en el walk-forward (p = 0.11).
- El detalle y la discusión están en `docs/reporte.pdf`.

## Instalación

Probado con Python 3.14.7.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los datos crudos congelados están en `data/` (`btc_project_train.csv` y `btc_project_test.csv`, tal como se descargaron de Canvas).

## Reproducir los resultados

```bash
python main.py
```

Corre todo el proyecto (unos 6 minutos; 365 s en la corrida registrada en `docs/run_log.txt`): audita los datos, recorta los tres periodos, hace el walk-forward semanal, recalcula los parámetros sobre todo el entrenamiento, los evalúa en prueba y en validación, calcula la robustez y el análisis de régimen, y deja las tablas en `docs/tables`, las figuras en `docs/figures` y el registro en la consola. Verifica además que los parámetros recalculados coinciden con los congelados en `docs/theta_frozen.json` (no los sobrescribe). Para comprobar solamente que todo corre (alrededor de 1 minuto; usa 8 pruebas por ventana, escribe en una carpeta temporal y no toca `docs/`):

```bash
python main.py --quick
```

Pruebas:

```bash
pytest
```

## Semilla aleatoria

Fijada en `42` (constante `SEED` en `src/optimize.py`): la usan el muestreador TPE de Optuna, el bootstrap y el silhouette. `main.py` además fija las semillas globales de `random` y `numpy`.

## Estructura

```
main.py            punto de entrada
data/              datos crudos congelados
src/
  data.py          carga, limpieza, auditoría y tramos continuos
  signals.py       indicadores y regla de confirmación 2 de 3
  backtest.py      motor orientado a eventos con costos
  metrics.py       Sharpe, Sortino, Calmar, drawdown, win rate, retornos periódicos
  optimize.py      optimización, walk-forward, congelado de parámetros y robustez
  regimes.py       detección de régimen
  plots.py         figuras
tests/             pruebas con pytest
notebooks/         análisis y figuras (sin lógica nueva)
docs/              reporte, presentación, parámetros congelados, bitácora, tablas y figuras
```

## Notas

- La exploración de otros métodos de régimen (K-means y un HMM con probabilidades filtradas) se retiró del código al final del proyecto; se conserva en la etiqueta de git `exploracion-regimenes` y está resumida en `docs/EXPERIMENTS.md`.
- Los archivos de Canvas tienen huecos (una ventana diaria sin cotización, cortes de hasta 2 días en el archivo train y uno de 122 días en el archivo test). Por eso se usan solo tres periodos sin cortes mayores a 6 horas (`docs/tables/periodos.csv`); `main.py` falla si alguno tuviera uno. No se rellenó ningún precio: dentro de cada periodo solo faltan entre 0.2% y 2.0% de las barras, en huecos cortos. El tratamiento está descrito en el reporte.

## Uso de asistencia de IA

[REVISAR Y AJUSTAR ANTES DE ENTREGAR: debe describir con exactitud lo que hicieron ustedes dos.]

Se usó Claude (Anthropic) como asistente a lo largo del proyecto, en estas partes:

- Discusión de decisiones de diseño: elección de indicadores por correlación entre señales, política de salida, método de régimen y protocolo para congelar parámetros antes de evaluar en prueba.
- Redacción inicial del código de `src/`, de `main.py` y de las pruebas (incluidas dos pruebas cuyo equity se calculó a mano), que los autores revisaron, ejecutaron en su equipo y modificaron.
- Diagnósticos de los resultados y borradores del reporte, de este README, del notebook y de la presentación.

Los autores ejecutaron todo el código, verificaron los resultados y son responsables de cada línea del repositorio.
