# Bitácora de experimentos

> **Versión anterior (hasta el commit `8202c03`).** Todo lo de esta sección, hasta "Nueva versión", se hizo con el archivo train completo (2022-06 a 2023-12, con huecos) y usa solo `btc_project_train.csv`. Se conserva como bitácora de cómo se tomaron las decisiones de diseño.

## Ejecuciones

| # | Qué se corrió | Configuraciones evaluadas | Tiempo | Resultado |
|---|---|---|---|---|
| A | m=5, r=2 con signal_exit_after en {0, 48, 288, nunca} | 4 | segundos | 6,169 a 2,390 operaciones, retorno entre -100% y -99.3% |
| B | m en {10, 20, 40}, sin salida por señal | 3 | segundos | retorno -83%, -49%, -19%; bruto por operación 0.07%, 0.14%, 0.16% |
| 1 | Walk-forward, 65 ventanas x 150 pruebas TPE, un solo theta, sin gate | 9,750 | 161 s | OOS -98.1%, 2,142 operaciones, Sharpe -12.5 |
| 2 | Mismas optimizaciones de la ejecución 1, operando solo si el mejor Calmar de entrenamiento es mayor a 0 | 9,750 (las mismas) | 162 s | OOS +5.1%, 354 operaciones, Sharpe 0.31, 51 de 65 semanas operadas |
| 3 | Walk-forward con un theta por régimen (reglas), 150 pruebas por régimen y ventana, gate 0 | 21,450 | 294 s | OOS -35.1%, 430 operaciones, Sharpe -1.55, drawdown -39.4% |

Referencia en las mismas semanas: comprar y mantener +90.1%, Sharpe 1.51, drawdown -37.9%.

## Análisis del régimen (no evalúa la estrategia)
- Variables: volatilidad, R^2 de tendencia y autocorrelación horaria. La autocorrelación se descartó de la clasificación porque con ella el silhouette de K-means cae de 0.47 a 0.29, las rachas se acortan y el cluster de crisis pasa de 9% a 25% del tiempo.
- Métodos comparados en train: reglas (silhouette 0.45), K-means de 2 variables (0.47) y HMM filtrado (0.25). Se eligieron las reglas.
- Detalles del hallazgo: en datos de 5 minutos la autocorrelación de rezago 1 es positiva (0.09) y se desvanece desde el rezago 2, un efecto de cómo se construyen las barras.

## Decisiones y cuándo se tomaron
- Antes de ver resultados de la estrategia: ranuras de búsqueda del SPEC, mínimo de operaciones por ventana, variables y método de régimen (reglas, umbrales cuantil 90 de volatilidad y R^2 de 0.5), actualización horaria, cierre de la posición al cambiar el régimen, no operar un régimen sin datos suficientes.
- Después de ver resultados: ampliar el rango de m a 10-60 y de max_hold a 10 días y agregar signal_exit_after (tras A y B); agregar el gate de Calmar mayor a 0 (tras la ejecución 1). El efecto del gate medido en train es optimista.

## Limitaciones conocidas del procedimiento
- Cada semana de prueba cierra a la fuerza lo que tenga abierto (56 de 430 salidas en la ejecución 3), lo que castiga el holding largo.
- Con ventanas de un mes la crisis tiene datos suficientes (288 barras o más) en solo 13 de 65 ventanas (14 tienen alguna barra de crisis).

## Hecho después de esta bitácora
- Prueba de diferencias entre regímenes (Welch y bootstrap), sensibilidad de ±20% y curva de costos: ver docs/tables.
- Parámetros congelados en el commit 43aa031 y evaluados una sola vez en prueba: docs/tables/metricas_test.csv.

## Código de la exploración de régimen
El código de K-means y del HMM con probabilidades filtradas, usado para comparar métodos de régimen, se retiró del repositorio al final del proyecto. Se conserva en la etiqueta de git `exploracion-regimenes`.

---

# Nueva versión: periodos sin huecos (commits `28604b0` y `dcef66d`)

## Por qué se cambió
Los archivos de Canvas tienen huecos (una ventana diaria sin cotización, cortes de hasta 2 días en el archivo train y de 122 días en el archivo test). Con el profesor se acordó trabajar solo con periodos sin huecos: **train** Jul-Nov 2023 y **test** Dic 2023 (ambos del archivo train) y **validación** May-Jun 2024 (archivo test). Se mantienen sin cambios el espacio de búsqueda, las 150 pruebas por ventana, los mínimos de operaciones (10 por ventana, 30 al ajustar sobre todo el train), el filtro de calidad (Calmar mayor a 0), las reglas de régimen (cuantil 90 de la volatilidad y R^2 de 0.5) y la semilla.

## Qué se decidió antes y después de ver resultados
- **Después de ver resultados:** el cambio de periodos mismo. Los tres periodos ya habían sido vistos por la versión anterior (Jul-Nov y Dic como parte del entrenamiento; May-Jun como prueba), así que ninguno es virgen. Un resultado distinto no se puede leer como una prueba ciega.
- **Antes de evaluar en test y validación:** θ se optimizó una sola vez sobre Jul-Nov y se commiteó (`28604b0`, subido a `main`) antes de correr la evaluación (`dcef66d`).
- **Durante la implementación:** como el θ único no pasó el filtro de calidad y no opera, las preguntas 1, 3 y 4 y las figuras 4 y 5 se calculan con la estrategia por régimen (`run_regime_strategy`), dentro de muestra.

## Ejecuciones

| # | Qué se corrió | Configuraciones evaluadas | Tiempo | Resultado |
|---|---|---|---|---|
| 4 | Walk-forward, 17 ventanas x 150 pruebas TPE, un solo theta con filtro de calidad | 2,550 | 96 s | OOS -7.0%, 43 operaciones, Sharpe -2.73; opera 7 de 17 semanas |
| 5 | Walk-forward con un theta por régimen, 50 optimizaciones de régimen x 150 pruebas | 7,500 | 199 s | OOS -9.0%, 101 operaciones, Sharpe -1.80, drawdown -13.1% |
| 6 | Congelado de θ sobre todo Jul-Nov (4 optimizaciones x 150 pruebas) | 600 | 52 s | θ único sin parámetros (mejor Calmar -1.0); los tres regímenes con parámetros |
| 7 | Evaluación única en test (Dic 2023) y validación (May-Jun 2024) | 0 | 1 s | θ por régimen -3.1% (19 operaciones) y -3.0% (18 operaciones); θ único sin operar |

Referencia: comprar y mantener rinde +29.1% (walk-forward), +11.9% (test) y +16.4% (validación). Total: 10,650 configuraciones y 365 s en la corrida completa (`docs/run_log.txt`); el θ recalculado coincidió con el congelado.

## Hallazgo sobre el Calmar
Cuando ninguna configuración gana dinero, el Calmar premia la ruina total: perder casi todo da un Calmar cercano a -1, y una pérdida moderada da un valor peor. Por eso el "mejor" Calmar del θ único (-1.0) corresponde a una configuración que pierde 99.9% con 3,155 operaciones. El filtro de calidad la descarta, pero el valor no sirve para comparar configuraciones perdedoras.

## Limitaciones conocidas del procedimiento
- Test y validación duran 30 y 31 días: solo 19 y 18 operaciones. Sus métricas anualizadas (Sharpe, Calmar) no son informativas.
- Con ventanas de un mes, el optimizador encuentra una configuración con Calmar positivo en solo 7 de 17 ventanas (θ único) y la crisis tiene datos suficientes en 16 ventanas (12 pasan el filtro de calidad).
- La validación empieza con indicadores y régimen sin historia (hay 4 meses sin datos antes), así que la primera semana no tiene régimen etiquetado.
