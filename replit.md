# Delphi: Health Trajectory Modeling System

## Overview

Delphi is a machine learning system designed to analyze health trajectories and predict future disease risks based on patient medical histories. The system uses a transformer-based neural network architecture (similar to GPT) to model sequential patterns in disease occurrences across patient populations. The application provides a complete pipeline from data ingestion to risk prediction, with specialized focus on workplace wellbeing analysis through Qready methodology integration.

The system handles over 65 different diseases across medical specialties and provides a fully Spanish-localized interface for healthcare professionals and researchers to analyze patient trajectories, train predictive models, and generate risk assessments.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Model Architecture
- **Transformer-based Neural Network**: Custom implementation of a GPT-style transformer model (`DelphiModel`) specifically adapted for medical sequence modeling
- **Causal Self-Attention**: Uses causal masking to ensure predictions only depend on past medical events
- **Disease Tokenization**: Centralized tokenizer system that converts disease names to numerical tokens with PAD token handling
- **Sequence Processing**: Handles variable-length patient trajectories with proper padding and attention masking

### Data Processing Pipeline
- **Multi-format Support**: Processes Excel (.xlsx, .xls) and Word (.docx) documents for medical data extraction
- **Disease Mapping**: Comprehensive disease categorization system using ICD codes and medical classifications
- **Sequence Generation**: Converts patient medical histories into tokenized sequences for model training
- **Data Validation**: Includes minimum/maximum sequence length constraints and data quality checks

### Training Infrastructure
- **Configurable Training**: Modular configuration system for model hyperparameters, training schedules, and data splits
- **Learning Rate Scheduling**: Implements warmup and cosine decay learning rate scheduling
- **Model Checkpointing**: Automatic model saving and resuming capabilities
- **Cross-validation**: Support for model evaluation across different data splits

### Evaluation and Analytics
- **Risk Prediction**: Generates probability distributions for future disease occurrences
- **Model Interpretability**: Tools for understanding model decision-making and attention patterns
- **Performance Metrics**: Comprehensive evaluation including AUC, precision-recall, and calibration metrics
- **Visualization Suite**: Advanced plotting capabilities for trajectory analysis and risk visualization

### Specialized Integrations
- **Qready Methodology**: Integrated workplace wellbeing analysis framework with predefined questionnaires
- **Spanish Localization**: Complete Spanish language support for all user interfaces and documentation
- **Multi-dimensional Analysis**: Supports analysis across wellbeing dimensions including stress, engagement, and physical health

## External Dependencies

### Core ML Framework
- **PyTorch**: Primary deep learning framework for model implementation and training
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Scikit-learn**: Model evaluation metrics, calibration, and preprocessing utilities

### Data Processing
- **python-docx**: Microsoft Word document processing for medical records
- **openpyxl**: Excel file handling for structured medical data
- **JSON**: Configuration and results serialization

### Visualization and Analytics
- **Matplotlib/Seaborn**: Statistical plotting and visualization
- **Plotly**: Interactive web-based plotting for trajectory visualization
- **UMAP/t-SNE**: Dimensionality reduction for data exploration

### Development Tools
- **GitHub Integration**: Automated repository management through Octokit REST API
- **Node.js Dependencies**: Package management for web-based components

### Optional Dependencies
- **CUDA/MPS**: GPU acceleration support for model training (auto-detected)
- **Flash Attention**: Optimized attention computation for larger sequences
- **Model Compilation**: PyTorch 2.0+ compilation support for performance optimization

The system is designed to be self-contained with minimal external service dependencies, focusing on local computation and data processing while maintaining the ability to scale to larger datasets and more complex model architectures.

## Recent Changes

### Integración Completa con Metodología Qready (Septiembre 21, 2025)
- ✅ **Análisis de Documentos**: Extracción exitosa de información de documentos Qready/Quirón Prevención subidos
  - Plantilla de análisis de bienestar (Excel): 10 preguntas, 10 dimensiones
  - Escalas de respuesta auténticas: "Nunca;Raramente;Algunas veces;A menudo;Siempre"
  - Tipos de preguntas: Likert_5, numeric, single_choice

- ✅ **Nueva Sección "Análisis Qready Bienestar"** integrada en la aplicación principal:
  - **📊 Evaluación Individual**: Gráficos radar, métricas por dimensión
  - **📈 Análisis Organizacional**: Comparación por departamentos, distribución de riesgo
  - **🔮 Predicción Absentismo**: Algoritmo de IA basado en 4 dimensiones de bienestar
  - **📋 Recomendaciones**: Plan de acción personalizado con intervenciones específicas

- ✅ **4 Dimensiones de Bienestar** implementadas:
  - Bienestar (positividad): 3 preguntas sobre ánimo y satisfacción
  - Estrés: 3 preguntas sobre demandas laborales y relajación  
  - Compromiso Laboral: 3 preguntas sobre compromiso y valoración
  - Salud Física: 3 preguntas sobre sueño, alimentación y ejercicio

- ✅ **Funcionalidades Técnicas**:
  - Módulo `qready_integration.py` con clase `QreadyWellbeingAnalyzer`
  - Visualizaciones radar con Plotly para perfiles de bienestar
  - Algoritmo de predicción de absentismo (combinación ponderada de factores)
  - Sistema de indicadores de riesgo con codificación por colores
  - Configuración Streamlit optimizada (`.streamlit/config.toml`)

- ✅ **Correcciones Técnicas**:
  - Solucionados 7 errores LSP en `app.py`
  - Corregidos imports UMAP, manejo de archivos subidos, mapeo de horizontes de predicción
  - Implementación robusta de manejo de errores y validación de datos
  - Compatibilidad completa con la arquitectura Delphi existente

### Estado Actual
- ✅ Aplicación funcionando correctamente en puerto 5000
- ✅ Todas las funcionalidades de Delphi + Qready operativas
- ✅ Interfaz completamente en español
- ✅ Integración sin errores entre ambas metodologías