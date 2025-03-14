# Notas de Implementación - Proyecto de Clustering Semántico

## Visión General

Este proyecto implementa un sistema de análisis y clustering semántico para términos de búsqueda publicitarios. La solución combina características numéricas (impresiones, costos) con análisis de texto mediante embedding vectorial para agrupar los términos de búsqueda en clusters coherentes.

## Flujo de Trabajo

1. **Extracción de Datos**: Obtención de términos de búsqueda desde una base de datos SQLite.
2. **Procesamiento de Datos**: 
   - Limpieza de términos de búsqueda
   - Cálculo de métricas de costo por impresión
   - Generación de embeddings mediante Sentence Transformers
   - Reducción de dimensionalidad con PCA
3. **Clustering**: 
   - Determinación del número óptimo de clusters usando el método del codo y el score de silueta
   - Aplicación de K-means para el agrupamiento
   - Asignación de nombres descriptivos a los clusters
4. **Análisis y Visualización**:
   - Visualización con t-SNE
   - Análisis estadístico de características por cluster
   - Evaluación de coherencia semántica

## Componentes Principales

### DataExtractor

Responsable de obtener los datos de la base de datos SQLite y/o consultar información de cuentas a través de una API externa.

### DataProcessor

Encargado de la limpieza y preparación de los datos para el clustering:
- Limpieza de texto en términos de búsqueda
- Cálculo de métricas derivadas como 'cost_per_impression' (costo/impresión)
- Tratamiento de valores atípicos en variables numéricas
- Generación de embeddings para términos de búsqueda mediante Sentence Transformers
- Análisis de componentes principales (PCA) para reducción de dimensionalidad

### ClusterAnalyzer

Implementa el algoritmo de clustering y los análisis posteriores:
- Determinación del número óptimo de clusters
- Aplicación de K-means
- Visualización mediante t-SNE
- Análisis de características por cluster
- Evaluación de coherencia semántica

### DBManager

Gestiona la interacción con la base de datos:
- Consulta de tablas y esquemas
- Extracción de datos
- Almacenamiento de resultados

## Análisis Exploratorio

El notebook `notebooks/exploratory_analysis.ipynb` contiene el trabajo exploratorio inicial donde se probaron y refinaron los diferentes enfoques analíticos antes de implementarlos en los scripts modulares. Este enfoque permitió iterar rápidamente sobre diferentes técnicas de procesamiento de texto, reducción de dimensionalidad y visualización antes de incorporarlas al código final.

## Sobre los Tests

He incorporado tests unitarios para cada uno de los componentes principales del sistema. Esta es mi primera experiencia implementando tests en un proyecto de ciencia de datos, y debo admitir que no comprendo completamente su funcionamiento. Los incluí para añadir robustez al código, pero reconozco que necesito estudiar más a fondo cómo funcionan los tests y las mejores prácticas para su implementación. Esta es definitivamente un área donde planeo mejorar mis conocimientos en el futuro cercano.

La estructura de los tests sigue la organización de los módulos principales:
- `test_data_extractor.py`: Valida la extracción correcta de datos
- `test_data_processor.py`: Verifica el procesamiento de características
- `test_clustering.py`: Comprueba la funcionalidad de clustering
- `test_db_manager.py`: Asegura la correcta interacción con la base de datos

## Decisiones Técnicas Destacadas

### Embeddings Multilingües

Se utilizó el modelo `paraphrase-multilingual-MiniLM-L12-v2` para generar embeddings de los términos de búsqueda, lo que permite manejar adecuadamente términos en diferentes idiomas.

### Reducción de Dimensionalidad Adaptativa

El código determina automáticamente el número óptimo de componentes principales para retener al menos el 80% de la varianza, lo que equilibra la preservación de información con la eficiencia computacional.

### Fallback a TF-IDF

Se implementó un mecanismo de respaldo que utiliza TF-IDF si la generación de embeddings con Sentence Transformers falla, asegurando robustez en diferentes entornos.

### Análisis de Coherencia Semántica

Se analiza la coherencia semántica de los clusters calculando la distancia promedio de los términos al centroide del cluster en el espacio de embeddings, proporcionando una medida objetiva de la calidad del clustering.

## Posibles Mejoras Futuras

1. Implementación de técnicas más avanzadas de clustering (DBSCAN, HDBSCAN)
2. Integración con herramientas de visualización interactiva
3. Automatización del proceso de asignación de nombres a clusters
4. Ampliación de la cobertura de tests unitarios
5. Optimización del rendimiento para conjuntos de datos más grandes