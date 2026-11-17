# 🧭 Perú Travel Health — Dengue y Malaria (2000–2025)

Aplicación experimental orientada a **viajeros en Perú**, con Streamlit y estimaciones basadas en registros epidemiológicos del CDC-MINSA y clima de Open-Meteo.

> **Advertencia sanitaria:** no proporciona una probabilidad de infección personal, diagnóstico, incidencia real actual, ni alerta oficial de brote. La categoría `Bajo/Moderado/Alto` se refiere a un **nivel de actividad semanal departamental definido estadísticamente**; los porcentajes son salidas del modelo aún **no calibradas**. Los registros epidemiológicos llegan solo hasta 2025. Comprobar avisos del MINSA y consultar medicina del viajero.

## Requisitos

- Python **3.10 o superior** (recomendado 3.11/3.12)
- Internet para descargar el histórico de Open-Meteo y visualizar el clima en vivo
- Memoria libre suficiente para leer los archivos (~1 GB aconsejable)

## Estructura

```text
travel_health_peru/
├── app.py                        # Interfaz Streamlit para viajeros
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                 # 25 capitales departamentales
│   ├── epi.py                    # Normalización y semana (provisional)
│   ├── etl.py                    # Limpieza dengue y malaria
│   ├── weather.py                # Descarga Open-Meteo histórica/en vivo
│   ├── features.py               # Integración y 4 lags
│   ├── train.py                  # 4 pares de modelos y métricas
│   └── pipeline.py               # Ejecuta la secuencia completa
├── data/
│   ├── raw/
│   │   ├── dengue/
│   │   │   └── datos_abiertos_vigilancia_dengue_2000_2025_0.csv
│   │   └── malaria/
│   │       └── malaria_2000-2025_*.csv (22 archivos)
│   ├── interim/                 # Caché de descargas por departamento
│   └── processed/               # ETL, clima semanal, dataset modelado
├── models/                       # *.joblib al entrenar
├── reports/                      # Auditorías, métricas JSON
└── tests/
    └── test_pipeline.py
```

El **ZIP completo** incluye los 23 CSV fuente suministrados; el ZIP de **solo código** exige que coloques esos archivos manualmente en las carpetas indicadas. No hay necesidad de convertir los archivos originales.

## Ejecutar desde cero (Windows / VS Code)

Abre la carpeta `travel_health_peru` en VS Code y abre una terminal dentro de ella.

**PowerShell**:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.pipeline
streamlit run app.py
```

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
streamlit run app.py
```

Abre la URL local que muestra Streamlit (normalmente `http://localhost:8501`). La primera descarga es **larga**, porque consulta aproximadamente 25 capitales × 26 años, divididos en ventanas de 5 años; después, la caché local hace que las siguientes ejecuciones reutilicen lo ya descargado. Si falla una región, reejecuta `python -m src.pipeline`: se retoman los CSV meteorológicos ya guardados.

### Ejecutar por etapas

```bash
python -m src.etl       # Produce data/processed/epidemiologia_semanal.csv
python -m src.weather   # Descarga diario y agrupa clima semanal
python -m src.features  # Produce data/processed/dataset_modelado.csv
python -m src.train     # Entrena modelos y genera reportes de desempeño
streamlit run app.py    # Abre la aplicación
```

Para comprobar sin internet algunas reglas de tratamiento:

```bash
python -m pytest -q
```

## Qué hace la limpieza

- **Dengue:** las 1 065 274 filas de origen representan notificaciones; se filtran diagnósticos `A97.0/A97.1/A97.2` y estados `C` (confirmados), `P` (probables). Se cuentan por departamento, año y semana. `casos` = C+P, porque el objetivo demo es actividad notificada, **no solo incidencia confirmada**. Se retienen C y P por separado para revisión.
- **Malaria:** los 22 archivos contienen `ano,semana,provincia,malaria_total`; se suman los conteos provinciales por departamento (determinado por el nombre del archivo). No se inventa confirmación/probabilidad porque esos atributos no figuran en los CSV.
- Se quitan tildes, espacios sobrantes y variantes de nombres regionales. Los nombres de provincia con caracteres dañados no se utilizan como clave de integración, porque los archivos están etiquetados por departamento.
- **Semanas ausentes de dengue** quedan ausentes: **no se convierten en cero**, pues no está acreditado si son ceros epidemiológicos o falta de vigilancia.
- **Semana epidemiológica provisional:** domingo-sábado; SE1 contiene el 4 de enero. Esto permite un procesamiento reproducible, pero **NO sustituye los calendarios oficiales CDC Perú de cada año**. Registros fuera de ese calendario quedan en `reports/semanas_epidemiologicas_pendientes.csv`, sin ser reasignados.
- Las semanas meteorológicas deben estar completas (7 días con todas las variables disponibles).
- Los registros de clima se toman de la **capital departamental**, como aproximación imperfecta de la exposición climática en toda la región. Para un proyecto científico más robusto, muestrear o agregar una grilla por departamento.

## Modelo y validación

Se entrenan **dos tareas por enfermedad**: clasificación `Bajo/Moderado/Alto` y regresión de número de casos semanales.

| Familia | Características | Uso |
|---|---|---|
| `*_travel.joblib` | Coordenada regional (codificada por departamento), estación anual, año, rezagos climáticos semanas 1–4 | **Dashboard**: predicción sin suponer casos recientes inexistentes |
| `*_full.joblib` | Todas las variables anteriores **más rezagos de casos** semanas 1–4 | Investigación retrospectiva; no se utiliza en viajes futuros sin datos recientes |

Modelos RandomForest (100 árboles), divididos **cronológicamente**: entrenamiento **2000–2017**, validación **2018–2021**, prueba fuera de muestra **2022–2025**. Los umbrales se calculan con **solo el conjunto de entrenamiento**, por departamento, con los percentiles 60 y 90 de semanas con casos positivos. Es una taxonomía **estadística experimental**, no una definición clínica de brote. Las métricas MAE y accuracy se guardan en `reports/metricas_*.json` y nunca se muestran como evidencia de calibración.

Si una enfermedad no tiene registros en una región o no hay suficientes semanas completas para inferir clima, el dashboard muestra **sin datos**, no riesgo bajo.

Las probabilidades de `RandomForestClassifier` son **puntuaciones no calibradas**. La UI muestra confianza en la categoría predicha y P(actividad alta); ninguno equivale a la probabilidad de que un turista se contagie. La regresión estima **conteos semanales departamentales**, no tasas por población ni casos a nivel de hotel.

## Clima en vivo y fechas futuras

Se utiliza `https://api.open-meteo.com/v1/forecast` con **hasta 16 días de horizonte** (día actual incluido) y datos de las **seis semanas previas** para calcular 4 rezagos climáticos completos. Las temperaturas medias para pronóstico se calculan promediando las **temperaturas horarias**, porque ese endpoint no devuelve directamente la media diaria como el histórico. La probabilidad de precipitación diaria es `precipitation_probability_max` de Open-Meteo. Se separa visualmente del estimador semanal de actividad.

Una consulta por una fecha más allá del horizonte de Open-Meteo **no se habilita**, porque no existen valores meteorológicos pronosticados válidos para fechas arbitrariamente lejanas.

Para el mapa de 25 destinos hay un **botón voluntario**: la primera carga consulta las regiones concurrentemente y se guarda temporalmente en caché. Los marcadores representan capitales (no coropletas exactas).

## Auditoría real hecha con los archivos recibidos

- 1 065 274 notificaciones de dengue (2000–2025).
- 22 archivos departamentales de malaria.
- 23 departamentos con dengue y 22 con malaria en estos archivos.
- **469 registros semanales** no se ajustan a la definición calendárica provisional; contienen en conjunto **248 casos**. Se excluyen del dataset modelado, pero están guardados íntegramente para revisión y no deben descartarse en un análisis definitivo.

**Advertencias metodológicas importantes:** investigar semanas 53, posible subregistro de 2025, períodos sin vigilancia, definiciones confirmados/probables, variaciones de vigilancia en el tiempo, validación espacial y la representatividad de las capitales. Usar clima histórico de Open-Meteo requiere conexión y aceptar las políticas de uso/citación de ese servicio.

## Verificación de resultados y mantenimiento

- `data/processed/epidemiologia_semanal.csv`: conteos epidemiológicos aceptados.
- `data/processed/clima_diario_openmeteo.csv`: meteorología histórica descargada.
- `data/processed/clima_semanal_openmeteo.csv`: clima semanal completo.
- `data/processed/dataset_modelado.csv`: clima + dengue o malaria, sin ceros inventados.
- `models/dengue_travel.joblib`, `models/malaria_travel.joblib`: modelos del dashboard.
- `models/dengue_full.joblib`, `models/malaria_full.joblib`: modelos retrospectivos.
- `reports/metricas_*.json`: evaluación con fechas de entrenamiento y test.
- `reports/semanas_epidemiologicas_pendientes.csv`: elementos excluidos por fecha.
- `reports/cobertura_epi.csv`: presencia histórica por enfermedad, departamento y año.

**Nota de seguridad:** los archivos `.joblib` solo deben abrirse si provienen de tu propio entrenamiento; `joblib.load()` ejecuta serialización de Python y no debe usarse con archivos externos desconocidos.

## Fuentes y especificaciones de API

- Documentación oficial Open-Meteo histórico: https://open-meteo.com/en/docs/historical-weather-api
- Pronóstico Open-Meteo: https://open-meteo.com/en/docs
- Archivo epidemiológico y CSV departamentales: archivos proporcionados por el usuario, procedentes de vigilancia CDC-MINSA según su descripción.
- En implementación de producción hay que confirmar acceso, licencia, definición oficial de casos, verificación epidemiológica y calibración.
