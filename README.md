# Perú Travel Health
### Inteligencia artificial para el análisis de dengue y malaria en viajeros del Perú

**Proyecto de Machine Learning, epidemiología y visualización interactiva**

**Repositorio:** https://github.com/verenisseql/PeruTH

**Tecnologías:** Python · Scikit-learn · Pandas · Streamlit · Plotly · Open-Meteo

---

## 1. Descripción del proyecto

**Perú Travel Health** es un prototipo de inteligencia artificial orientado a viajeros que desean conocer las condiciones climáticas y los patrones estimados de actividad epidemiológica de dengue y malaria en diferentes departamentos del Perú.

El proyecto combina registros históricos de vigilancia epidemiológica correspondientes al periodo **2000–2025** con información meteorológica obtenida mediante la API de Open-Meteo.

A partir de estos datos, se entrenaron modelos de Machine Learning para estimar casos semanales y clasificar la actividad epidemiológica en tres categorías:

- **Bajo:** actividad histórica relativamente baja.
- **Moderado:** actividad histórica intermedia.
- **Alto:** actividad histórica elevada.

Los resultados se presentan mediante un dashboard interactivo desarrollado con Streamlit.

> **Importante:** Perú Travel Health es un prototipo académico. Sus predicciones no representan probabilidades individuales de contagio, diagnósticos médicos ni alertas epidemiológicas oficiales del MINSA. Las puntuaciones del clasificador no están calibradas.

## 2. ¿Qué problema buscamos resolver?

Perú posee una gran diversidad geográfica y climática. Las condiciones ambientales pueden variar considerablemente entre la costa, los Andes y la Amazonía.

Algunas enfermedades transmitidas por mosquitos, como el dengue y la malaria, presentan patrones epidemiológicos relacionados con factores ambientales, sociales y geográficos.

Sin embargo, la información epidemiológica histórica no siempre resulta sencilla de consultar e interpretar para una persona que está planificando un viaje.

Por ello, este proyecto plantea una aplicación que permita explorar información epidemiológica y meteorológica mediante una interfaz visual, accesible y orientada a viajeros.

## 3. ¿Por qué dengue y malaria?

### Dengue

El dengue es una enfermedad viral transmitida principalmente por mosquitos del género *Aedes*.

Se seleccionó por su importancia epidemiológica en el Perú y porque factores como la temperatura y las precipitaciones pueden influir en las condiciones ambientales relacionadas con su transmisión.

### Malaria

La malaria es una enfermedad parasitaria transmitida principalmente por mosquitos *Anopheles* infectados.

En el Perú, presenta especial relevancia en determinadas zonas amazónicas.

Su incorporación permite estudiar otra enfermedad transmitida por mosquitos que posee características epidemiológicas y geográficas diferentes a las del dengue.

### ¿Por qué analizarlas por separado?

Aunque ambas enfermedades son transmitidas por mosquitos, tienen distintos agentes causales, vectores y patrones de transmisión.

Por ello, Perú Travel Health utiliza **modelos independientes para dengue y malaria**, evitando combinarlas en una única predicción.

## 4. Objetivos

### Objetivo general

Desarrollar una aplicación basada en Machine Learning que integre datos epidemiológicos históricos y variables meteorológicas para estimar patrones de actividad semanal de dengue y malaria y visualizarlos mediante una interfaz interactiva para viajeros.

### Objetivos específicos

1. Procesar datos epidemiológicos históricos del Perú.
2. Obtener temperatura y precipitación mediante Open-Meteo.
3. Integrar información epidemiológica y climática por departamento y semana.
4. Entrenar modelos independientes de clasificación y regresión.
5. Evaluar el desempeño de los modelos utilizando datos de periodos posteriores al entrenamiento.
6. Desarrollar un dashboard interactivo que facilite la interpretación de resultados.

## 5. Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python | Desarrollo del proyecto |
| Pandas | Procesamiento e integración de datos |
| NumPy | Operaciones numéricas |
| Scikit-learn | Machine Learning |
| Random Forest | Clasificación y regresión |
| Joblib | Almacenamiento de modelos |
| Open-Meteo | Información meteorológica |
| Streamlit | Interfaz web interactiva |
| Plotly | Gráficos, medidores y mapas |
| GitHub | Control de versiones |

## 6. Arquitectura del sistema

```text
Registros epidemiológicos (2000–2025)
                 |
                 v
       Limpieza y normalización
                 |
                 v
     Agrupación epidemiológica
                 |
                 +---------------------+
                                       |
Open-Meteo                             |
    |                                  |
    v                                  |
Temperatura y precipitación            |
    |                                  |
    v                                  |
Agregación semanal                     |
    |                                  |
    +----------------------------------+
                 |
                 v
      Dataset integrado
                 |
                 v
    Ingeniería de características
    Rezagos climáticos: 1–4 semanas
                 |
                 v
        Random Forest
          /        \
         v          v
   Clasificación  Regresión
         \          /
          v        v
       Modelos entrenados
                 |
                 v
      Dashboard Streamlit
                 |
                 v
      Consulta del viajero
```

## 7. Procesamiento de los datos

### 7.1. Datos epidemiológicos

Se utilizaron archivos históricos de dengue y malaria asociados con registros de vigilancia epidemiológica del Perú.

**Dengue:**

- 1 065 274 registros originales.
- Periodo de estudio: 2000–2025.
- Diagnósticos A97.0, A97.1 y A97.2.
- Diferenciación entre casos confirmados y probables.
- Agregación por departamento y semana epidemiológica.

Para el análisis se utiliza la suma de casos confirmados y probables como medida de actividad notificada.

**Malaria:**

- 22 archivos departamentales.
- Registros organizados por provincia, año y semana.
- Agrupación de casos por departamento y semana.

Durante el procesamiento se normalizaron nombres geográficos, se revisaron registros inconsistentes y se evitó convertir automáticamente las semanas sin notificaciones en cero casos.

### 7.2. Datos meteorológicos

Se utilizó la API histórica de Open-Meteo para obtener:

- Temperatura media diaria.
- Temperatura máxima diaria.
- Temperatura mínima diaria.
- Precipitación diaria acumulada.

El periodo histórico corresponde a 2000–2025.

Se consultaron coordenadas de las capitales de los 25 departamentos considerados. Estas ubicaciones son puntos de referencia y no representan necesariamente las condiciones climáticas de cada departamento completo.

Los registros se transformaron a semanas de domingo a sábado y se conservaron únicamente las semanas meteorológicas completas.

### 7.3. Integración temporal

Los datos epidemiológicos y meteorológicos se relacionaron mediante:

- Departamento.
- Año epidemiológico.
- Semana epidemiológica.
- Fecha de inicio de semana.

La implementación utiliza una definición provisional de semana epidemiológica que requiere contrastarse con los calendarios históricos oficiales.

Durante la auditoría se identificaron **469 registros semanales con inconsistencias respecto de esa definición**, que contienen 248 casos en total. Estos registros se separaron para revisión.

## 8. Ingeniería de características

Se incorporaron variables temporales para estudiar relaciones entre condiciones ambientales y actividad epidemiológica.

Entre las características empleadas se encuentran:

- Departamento codificado.
- Año y estacionalidad.
- Temperatura de semanas anteriores.
- Precipitación de semanas anteriores.
- Rezagos climáticos de 1 a 4 semanas.

También se experimentó con rezagos epidemiológicos para analizar la información de casos previos.

Se establecieron dos modalidades:

| Modalidad | Descripción |
|---|---|
| **Travel** | Utiliza variables climáticas rezagadas y estacionalidad, sin requerir casos recientes de una semana futura |
| **Full** | Incorpora además rezagos epidemiológicos; orientada al análisis retrospectivo |

El dashboard utiliza los modelos Travel.

## 9. Entrenamiento y evaluación

Se implementaron modelos Random Forest independientes para ambas enfermedades.

Para cada enfermedad se desarrollaron dos tareas:

**Clasificación:** identificar niveles de actividad baja, moderada o alta.

**Regresión:** estimar el número de casos semanales departamentales.

Se aplicó una separación temporal de los datos:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | 2000–2017 |
| Validación | 2018–2021 |
| Prueba | 2022–2025 |

Los umbrales de clasificación se calcularon usando exclusivamente registros del conjunto de entrenamiento.

La clasificación es una definición estadística experimental, no una declaración oficial de brote epidemiológico.

### Resultados iniciales

| Modelo | MAE (casos) | Accuracy |
|---|---:|---:|
| Dengue — Travel | 161.48 | 31.7 % |
| Dengue — Full | 117.69 | 69.1 % |
| Malaria — Travel | 10.67 | 93.1 % |
| Malaria — Full | 4.42 | 97.2 % |

**Interpretación de métricas:**

- **MAE:** error absoluto promedio al estimar la cantidad de casos.
- **Accuracy:** proporción de clasificaciones correctas.
- **Precision:** proporción de predicciones de una categoría que fueron correctas.
- **Recall:** proporción de observaciones de una categoría que el modelo logró identificar.
- **F1-score:** medida que combina precision y recall.
- **Matriz de confusión:** muestra los aciertos y errores entre las categorías Bajo, Moderado y Alto.

### Análisis de resultados

El modelo Dengue Travel obtuvo una accuracy de 31.7 %, lo que evidencia un desempeño limitado y la necesidad de mejorar su capacidad de clasificación.

El modelo Malaria Travel obtuvo una accuracy de 93.1 %. Sin embargo, esta cifra no demuestra por sí sola que identifique adecuadamente las semanas de actividad elevada.

Por esta razón, se incorporaron métricas adicionales de evaluación, incluyendo matriz de confusión, F1 macro y recall de la categoría alta.

Los resultados deben examinarse teniendo en cuenta posibles desequilibrios entre categorías, diferencias de cobertura y departamentos sin suficientes registros de entrenamiento.

Los reportes de evaluación se encuentran en `reports/metricas_*.json`, cuando han sido generados y publicados.

## 10. Dashboard interactivo para viajeros

La interfaz se desarrolló utilizando Streamlit y Plotly.

### Funcionalidades

**Selección de destino**

Permite elegir un departamento del Perú.

**Selección de fecha**

Permite consultar fechas dentro del horizonte disponible de Open-Meteo.

**Indicadores de dengue y malaria**

Presenta niveles experimentales de actividad mediante gráficos semicirculares y etiquetas visuales.

**Clima del destino**

Muestra temperaturas y condiciones de precipitación disponibles para la fecha seleccionada.

**Mapa interactivo**

Representa las capitales departamentales mediante marcadores geográficos.

**Recomendaciones preventivas**

Proporciona orientaciones generales relacionadas con protección frente a picaduras de mosquitos y preparación del viaje.

Las recomendaciones no sustituyen una evaluación médica ni las recomendaciones sanitarias oficiales.

### Capturas de la aplicación

<img width="1876" height="1085" alt="Screenshot 2026-11-19 113050" src="https://github.com/user-attachments/assets/152c24f0-8795-480d-bd12-e1cf8d910654" />

*Figura 1. Pantalla principal de Perú Travel Health.*

<img width="1917" height="1101" alt="Screenshot 2026-11-19 113135" src="https://github.com/user-attachments/assets/12a17d1d-9b25-4bb6-8578-29dfe17290fa" />

*Figura 2. Indicadores experimentales de dengue y malaria.*

<img width="1917" height="1075" alt="Screenshot 2026-11-19 113305" src="https://github.com/user-attachments/assets/c2b36007-8baa-4525-aa97-551f7bf86f94" />

*Figura 3. Mapa interactivo y recomendaciones para viajeros.*

## 11. Dificultades encontradas y soluciones

### Compatibilidad de Python

Durante la configuración se presentaron errores en la instalación de bibliotecas.

Se utilizó finalmente Python 3.12 en un entorno virtual independiente para garantizar la compatibilidad de las dependencias.

### Límite de solicitudes de Open-Meteo

La descarga inicial fue interrumpida por un error HTTP 429 (*Too Many Requests*).

Se mejoró el mecanismo de descarga mediante almacenamiento por bloques, reutilización de archivos existentes y reintentos controlados.

### Departamentos sin antecedentes de entrenamiento

Durante la evaluación se identificaron departamentos presentes en validación o prueba que no contaban con información suficiente en entrenamiento.

Se incorporaron controles para evitar evaluar esos registros con umbrales inexistentes.

Esta limitación se conserva como parte de las consideraciones metodológicas.

### Diferencias de cobertura epidemiológica

Los registros históricos no presentan una cobertura uniforme en todos los departamentos y años.

Por ello, no se interpreta automáticamente la ausencia de información como ausencia de casos.

### Predicciones futuras

Los datos epidemiológicos utilizados llegan hasta 2025.

La aplicación utiliza modelos Travel para realizar estimaciones sin requerir los casos de la fecha futura consultada. Sin embargo, no incorpora automáticamente vigilancia epidemiológica actualizada.

## 12. Instalación y ejecución

### Requisitos

- Python 3.12 recomendado.
- Acceso a internet para consultar Open-Meteo.
- Dependencias incluidas en `requirements.txt`.

### Ejecutar el dashboard

Clonar o descargar el repositorio y abrir la carpeta en VS Code.

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv312
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abrir la dirección local indicada por Streamlit, normalmente:

`http://localhost:8501`

Para utilizar el dashboard se requieren los modelos `dengue_travel.joblib` y `malaria_travel.joblib` en la carpeta `models/`.

### Reentrenamiento completo

Para reproducir el procesamiento y entrenamiento desde cero, colocar previamente los archivos epidemiológicos originales en:

```text
data/raw/dengue/
data/raw/malaria/
```

El repositorio público no incluye necesariamente los CSV originales.

Después ejecutar:

```powershell
python -m src.pipeline
```

El procesamiento realiza limpieza epidemiológica, descarga meteorológica, integración, ingeniería de características y entrenamiento.

La descarga histórica puede estar sujeta a límites de Open-Meteo.

### Ejecutar las pruebas

```powershell
python -m pytest -q
```

## 13. Estructura del proyecto

```text
PeruTH/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── epi.py
│   ├── etl.py
│   ├── weather.py
│   ├── features.py
│   ├── train.py
│   └── pipeline.py
├── models/
│   ├── dengue_travel.joblib
│   └── malaria_travel.joblib
├── reports/
│   └── metricas_*.json
└── tests/
    └── test_pipeline.py
```

*Los archivos de métricas y modelos se incluyen según los cambios publicados en el repositorio.*

## 14. Próximas etapas y oportunidades de investigación

Perú Travel Health es un prototipo de investigación y aprendizaje que todavía presenta oportunidades de mejora:

1. Mejorar el desempeño predictivo del modelo de dengue.
2. Evaluar el F1-score y recall de las clases de actividad elevada.
3. Comparar Random Forest con modelos de referencia y otros algoritmos.
4. Incorporar nuevas variables ambientales y espaciales.
5. Evaluar calibración de probabilidades y desbalance de clases.
6. Validar calendarios epidemiológicos históricos.
7. Mejorar la representatividad espacial de los datos climáticos.
8. Incorporar registros epidemiológicos más recientes y fuentes oficiales de orientación para viajeros.

## 15. Conclusiones

Perú Travel Health demuestra la integración de ciencia de datos, Machine Learning y desarrollo de aplicaciones web para abordar un problema vinculado con la salud pública y el turismo.

El proyecto permitió construir un flujo de procesamiento reproducible que combina registros epidemiológicos históricos con datos meteorológicos, entrena modelos predictivos y presenta sus resultados mediante una interfaz interactiva.

Las evaluaciones iniciales evidencian tanto las posibilidades como las limitaciones de los modelos desarrollados.

El siguiente objetivo es fortalecer su metodología, mejorar el desempeño predictivo y continuar evaluando el potencial de la inteligencia artificial como herramienta de apoyo para la interpretación de información epidemiológica.

## 16. Referencias

1. Organización Mundial de la Salud. *Dengue*.  
   https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue

2. Organización Mundial de la Salud. *Malaria*.  
   https://www.who.int/news-room/fact-sheets/detail/malaria

3. Ministerio de Salud del Perú. *Plataforma Nacional de Datos Abiertos*.  
   https://www.datosabiertos.gob.pe/

4. Open-Meteo. *Historical Weather API*.  
   https://open-meteo.com/en/docs/historical-weather-api

5. Open-Meteo. *Weather Forecast API*.  
   https://open-meteo.com/en/docs

6. Scikit-learn. *RandomForestClassifier*.  
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

7. Streamlit. *Documentation*.  
   https://docs.streamlit.io/

---

**Proyecto académico de inteligencia artificial.**

*La aplicación no debe utilizarse como fuente única para decisiones médicas o sanitarias. Para información sobre riesgos y prevención, consultar fuentes oficiales y profesionales de salud.*
