# 🏥 Delphi + Qready Integration

Sistema completo de **modelado de trayectorias de salud** con **análisis de bienestar laboral** usando inteligencia artificial.

## 🎯 Características Principales

### 🧠 Delphi - Modelado de Trayectorias de Salud
- **Arquitectura Transformer GPT-2 modificada** para análisis de secuencias médicas
- **Predicción de riesgos** de enfermedades con probabilidades calibradas  
- **Visualización interactiva** de trayectorias de pacientes
- **Análisis de rendimiento** con métricas AUC y calibración
- **Manejo de +65 enfermedades** con categorización ICD

### 🏢 Qready - Análisis de Bienestar Laboral
- **4 Dimensiones de Bienestar**: Positividad, Estrés, Compromiso Laboral, Salud Física
- **Predicción de Absentismo** usando algoritmos de IA 
- **Análisis Organizacional** por departamentos
- **Recomendaciones Personalizadas** basadas en perfiles de riesgo
- **Metodología de Quirón Prevención** integrada

## 🛠️ Tecnologías Utilizadas

- **Backend**: Python, PyTorch, Streamlit
- **ML/AI**: Transformers, Scikit-learn, UMAP, T-SNE
- **Visualización**: Plotly, Matplotlib, Seaborn
- **Datos**: Pandas, NumPy, OpenPyXL, Python-docx

## 🚀 Instalación y Uso

```bash
# Instalar dependencias
pip install streamlit torch pandas plotly scikit-learn umap-learn matplotlib seaborn statsmodels python-docx openpyxl

# Ejecutar la aplicación
streamlit run app.py --server.port 5000
```

## 📊 Metodología Qready

Basada en la metodología de **Quirón Prevención** para evaluación de bienestar laboral:

### 📋 4 Dimensiones Evaluadas
1. **Bienestar (Positividad)** - Estado anímico y satisfacción general
2. **Estrés** - Nivel de presión laboral y capacidad de relajación  
3. **Compromiso Laboral** - Implicación y valoración en el trabajo
4. **Salud Física** - Hábitos de sueño, alimentación y ejercicio

### 🔮 Sistema de Predicción de Absentismo
- **Algoritmo Ponderado**: Estrés (40%) + Bienestar (30%) + Salud (30%)
- **Categorización de Riesgo**: Alto (>70%), Medio (50-70%), Bajo (<50%)
- **Visualizaciones**: Gráficos radar, distribuciones por departamento
- **Recomendaciones**: Plan de acción personalizado por dimensión

## 🎨 Funcionalidades de la Interfaz

### 📈 Delphi - Análisis Médico
- **Subir Datos**: Carga de datos CSV o uso de datos sintéticos
- **Entrenamiento**: Configuración y entrenamiento de modelos transformer
- **Análisis de Trayectorias**: Visualización de líneas de tiempo de pacientes
- **Predicción de Riesgos**: Generación de predicciones multi-horizonte
- **Interpretabilidad**: Análisis de atención y embeddings UMAP
- **Métricas**: Evaluación de rendimiento con ROC-AUC y calibración

### 🧠 Qready - Bienestar Laboral  
- **📊 Evaluación Individual**: Formulario de bienestar con gráficos radar
- **📈 Análisis Organizacional**: Comparación por departamentos (10-500 empleados)
- **🔮 Predicción Absentismo**: Algoritmo de IA basado en 4 dimensiones
- **📋 Recomendaciones**: Plan de acción con intervenciones específicas

## 📁 Estructura del Proyecto

```
delphi-qready-integration/
├── app.py                    # Aplicación principal Streamlit
├── qready_integration.py     # Módulo de integración Qready
├── model.py                  # Arquitectura transformer Delphi
├── utils.py                  # Utilidades de procesamiento de datos
├── train.py                  # Sistema de entrenamiento
├── plotting.py               # Funciones de visualización
├── evaluate_auc.py          # Métricas de evaluación
├── data/
│   └── synthetic_data.csv   # Datos sintéticos de ejemplo
├── .streamlit/
│   └── config.toml          # Configuración Streamlit
└── delphi_labels_chapters_colours_icd.csv  # Etiquetas de enfermedades
```

## 🔬 Innovación y Metodología

### Transformer Generativo para Medicina
- **Atención Causal** para modelado secuencial
- **Tokenización de Enfermedades** con vocabulario especializado
- **Embeddings Temporales** para eventos médicos
- **Predicción Multi-horizonte** (1 año, 5 años, 10 años, vida completa)

### Análisis de Bienestar Organizacional
- **Escalas Validadas** tipo Likert de Quirón Prevención
- **Algoritmo Predictivo** basado en combinación ponderada de factores
- **Análisis Departamental** con distribución de riesgo
- **Sistema de Alerta Temprana** para prevención de absentismo

## 🎉 Integración Completa

✅ **Sistema Unificado**: Combina análisis médico avanzado con bienestar laboral  
✅ **Interfaz en Español**: Completamente localizada para usuarios hispanohablantes  
✅ **IA Predictiva**: Algoritmos de última generación con arquitecturas transformer  
✅ **Escalable**: Arquitectura modular y extensible para nuevas funcionalidades  
✅ **Validado**: Metodologías respaldadas por investigación médica y organizacional  

## 📜 Licencia

Este proyecto integra metodologías de análisis de salud para investigación y desarrollo en el ámbito sanitario y de bienestar organizacional.

---

**Desarrollado con ❤️ para la investigación en salud y bienestar organizacional**
