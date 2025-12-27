# Predicción de Enfermedad Cardíaca mediante Machine Learning

## Descripción del proyecto

Las enfermedades cardiovasculares son la principal causa de mortalidad a nivel mundial. 
La detección temprana de pacientes con alto riesgo es clave para reducir la incidencia de eventos graves como infartos o insuficiencia cardíaca.

El objetivo de este proyecto es **desarrollar un modelo predictivo de Machine Learning** capaz de estimar la probabilidad de que un paciente presente enfermedad cardíaca, utilizando variables clínicas y demográficas.

El proyecto se desarrolla siguiendo un **workflow completo de Data Science**, desde el análisis exploratorio de datos (EDA) hasta el entrenamiento, evaluación e interpretación de modelos predictivos.

---

## Objetivo

- Predecir la presencia de **enfermedad cardíaca** (`HeartDisease`)
- Comparar distintos modelos de clasificación
- Evaluar su rendimiento mediante métricas estándar
- Seleccionar un modelo final interpretable y robusto

---

## Dataset

- **Fuente**: Kaggle – Heart Failure Prediction Dataset  
- **Origen**: Combinación de varios datasets clínicos (UCI Machine Learning Repository)
- **Observaciones finales**: 918 pacientes
- **Variables**: 11 variables clínicas + variable objetivo

### Variable objetivo
- `HeartDisease`  
  - 1 → Presencia de enfermedad cardíaca  
  - 0 → Paciente sano

---

## Estructura del repositorio

```text
src/
│
├── data/
│   ├── heart.csv
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_modeling.ipynb
│
├── model/
│   ├── production/
│   │   └── final_model.pkl
│
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
│
├── memoria.ipynb
└── README.md
```

---

## Metodología

1. **Análisis Exploratorio de Datos (EDA)**
   - Estudio de distribuciones
   - Análisis de correlaciones
   - Detección de valores no fisiológicos
   - Análisis de importancia de variables (Chi², Mutual Information, VIF)

2. **Preprocesado**
   - One-Hot Encoding para variables categóricas
   - Tratamiento de valores anómalos
   - Escalado para modelos lineales

3. **Modelado**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - Comparación de resultados mediante validación cruzada

4. **Evaluación**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

5. **Selección del modelo final**
   - Basada en rendimiento, estabilidad e interpretabilidad clínica

---

## Resultados principales

- Variables con mayor impacto:
  - Edad
  - Tipo de dolor en el pecho
  - Oldpeak
  - Frecuencia cardíaca máxima
  - Angina inducida por ejercicio

- Los modelos basados en árboles y boosting mostraron mejor rendimiento que los lineales.
- Se seleccionó un modelo final equilibrando **capacidad predictiva** e **interpretabilidad clínica**.

---

## Cómo ejecutar el proyecto

1. Clonar el repositorio
```bash
git clone https://github.com/usuario/heart-disease-ml.git
```

2. Instalar dependencias
```bash
pip install -r requirements.txt
```

3. Ejecutar notebooks en orden
```text
01_eda.ipynb → 02_feature_selection.ipynb → 03_modeling.ipynb
```

---

## Conclusiones

Este proyecto demuestra cómo el Machine Learning puede apoyar la detección temprana de enfermedades cardiovasculares, ofreciendo una herramienta de apoyo a la toma de decisiones clínicas.

El enfoque adoptado prioriza la **robustez del modelo**, la **interpretabilidad** y la **correcta validación**, aspectos fundamentales en entornos sanitarios.

---

## Trabajo futuro

- Ajuste fino de hiperparámetros
- Validación con datasets externos
- Análisis de explicabilidad con SHAP
- Desarrollo de una API para despliegue

---

## Autor

Proyecto desarrollado de forma individual como parte del **Máster / Bootcamp en Data Science**.
