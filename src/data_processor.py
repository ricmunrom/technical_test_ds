import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
from sentence_transformers import SentenceTransformer
from typing import Tuple

class DataProcessor:
    def __init__(self, df: pd.DataFrame, results_dir: str = 'notebooks'):
        """
        Inicializa el procesador de datos.
        
        Args:
            df: DataFrame con los datos extraídos de la base de datos
            results_dir: Directorio donde guardar los resultados y visualizaciones
        """
        self.df = df.copy()
        self.results_dir = results_dir
        self.numeric_features = ['impressions', 'cost', 'cost_per_impression']
        self.search_term_embeddings = None
        self.term_to_embedding = None
        
        # Asegurar que el directorio de resultados exista
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Descargar stopwords si es necesario
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Realiza el preprocesamiento de los datos para el clustering.
        
        Returns:
            DataFrame con los datos preprocesados
        """
        print("Preprocesando datos...")
        if self.df.empty:
            print("No hay datos para procesar")
            return pd.DataFrame()
        
        # Limpiar términos de búsqueda
        self._clean_search_terms()
        
        # Calcular costo por impresión
        self._calculate_cost_metrics()
        
        # Procesar características numéricas
        X_numeric_scaled = self._process_numeric_features()
        
        # Procesar características textuales
        text_features_reduced = self._process_text_features()
        
        # Combinar características
        X_combined = np.hstack((X_numeric_scaled, text_features_reduced))
        
        return X_combined
    
    def _clean_search_terms(self) -> None:
        """Limpia los términos de búsqueda"""
        print("Limpiando términos de búsqueda...")
        # Quitar caracteres especiales y convertir a minúsculas
        self.df['search_term_clean'] = self.df['search_term'].apply(
            lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower())
        )
        
        # Mostrar número de términos de búsqueda únicos
        unique_terms = self.df['search_term'].nunique()
        print(f"Número de términos de búsqueda únicos: {unique_terms}")
    
    def _calculate_cost_metrics(self) -> None:
        """Calcula métricas relacionadas con los costos"""
        # Calcular costo por impresión
        self.df['cost_per_impression'] = self.df['cost'] / self.df['impressions']
        # Manejar divisiones por cero
        self.df.loc[self.df['impressions'] == 0, 'cost_per_impression'] = 0
    
    def _process_numeric_features(self) -> np.ndarray:
        """
        Procesa las características numéricas para el clustering.
        
        Returns:
            Array NumPy con características numéricas procesadas
        """
        print("Procesando características numéricas...")
        # Seleccionar variables numéricas
        X_numeric = self.df[self.numeric_features].copy()
        
        # Manejar valores extremos (outliers)
        for feature in self.numeric_features:
            Q1 = X_numeric[feature].quantile(0.25)
            Q3 = X_numeric[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X_numeric[feature] = X_numeric[feature].clip(lower_bound, upper_bound)
        
        # Escalar características numéricas
        scaler = StandardScaler()
        X_numeric_scaled = scaler.fit_transform(X_numeric)
        
        return X_numeric_scaled
    
    def _process_text_features(self) -> np.ndarray:
        """
        Procesa las características textuales para el clustering.
        
        Returns:
            Array NumPy con características textuales procesadas
        """
        print("Procesando características textuales...")
        # Crear lista única de términos de búsqueda para procesar
        unique_search_terms = self.df['search_term_clean'].unique()
        print(f"Procesando {len(unique_search_terms)} términos de búsqueda únicos...")
        
        try:
            # Cargar modelo de embeddings multilingüe
            print("Cargando modelo de embeddings...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Modelo de embeddings cargado correctamente")
            
            # Generar embeddings para términos únicos
            print("Generando embeddings con Sentence Transformers...")
            term_embeddings = model.encode(unique_search_terms, show_progress_bar=True)
            
            # Crear diccionario para mapear términos a embeddings
            self.term_to_embedding = dict(zip(unique_search_terms, term_embeddings))
            
            # Asignar embeddings a cada fila del dataframe
            self.search_term_embeddings = np.array([
                self.term_to_embedding[term] for term in self.df['search_term_clean']
            ])
            
            # PASO INTERMEDIO: Determinar dimensionalidad óptima para PCA
            print("\nDeterminación de dimensionalidad óptima para PCA")
            
            # Aplicar PCA sin limitar componentes para análisis
            pca_analysis = PCA(random_state=42)
            pca_analysis.fit(self.search_term_embeddings)
            
            # Calcular varianza explicada acumulativa
            cumulative_variance = np.cumsum(pca_analysis.explained_variance_ratio_)
            
            # Graficar curva de varianza explicada
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
            plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% de varianza')
            plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% de varianza')
            plt.title('Varianza Explicada Acumulada vs Número de Componentes')
            plt.xlabel('Número de Componentes')
            plt.ylabel('Varianza Explicada Acumulada')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Guardar gráfico en el directorio de resultados
            save_path = os.path.join(self.results_dir, 'pca_variance_analysis.png')
            plt.savefig(save_path)
            print(f"Gráfico de varianza guardado en: {save_path}")
            
            # Determinar número de componentes para diferentes umbrales de varianza
            threshold_80 = np.argmax(cumulative_variance >= 0.8) + 1
            threshold_90 = np.argmax(cumulative_variance >= 0.9) + 1
            threshold_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            print(f"Componentes necesarios para explicar el 80% de varianza: {threshold_80}")
            print(f"Componentes necesarios para explicar el 90% de varianza: {threshold_90}")
            print(f"Componentes necesarios para explicar el 95% de varianza: {threshold_95}")
            
            # Seleccionar dimensionalidad basada en un umbral de varianza explicada
            optimal_components = threshold_80
            print(f"\nDimensionalidad óptima seleccionada: {optimal_components} componentes")
            
            # Continuar con PCA usando la dimensionalidad óptima
            print("\nAplicación de PCA con dimensionalidad óptima")
            pca_text = PCA(n_components=optimal_components, random_state=42)
            text_features_reduced = pca_text.fit_transform(self.search_term_embeddings)
            print(f"Varianza explicada con {optimal_components} componentes: {np.sum(pca_text.explained_variance_ratio_):.4f}")
            
        except Exception as e:
            print(f"Error al usar Sentence Transformers: {e}")
            print("Usando enfoque alternativo con TF-IDF")
            
            # Usar TF-IDF como alternativa si falla Sentence Transformers
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words=stopwords.words('spanish') + stopwords.words('english')
            )
            tfidf_matrix = tfidf.fit_transform(self.df['search_term_clean'])
            
            # Reducir dimensionalidad con PCA para TF-IDF
            pca_text = PCA(n_components=optimal_components, random_state=42)
            text_features_reduced = pca_text.fit_transform(tfidf_matrix.toarray())
            print(f"Varianza explicada por los componentes de PCA con TF-IDF: {np.sum(pca_text.explained_variance_ratio_):.2f}")
        
        print(f"Dimensionalidad reducida a {text_features_reduced.shape[1]} características")
        return text_features_reduced
    
    def get_processed_df(self) -> pd.DataFrame:
        """
        Retorna el DataFrame con todas las columnas procesadas.
        
        Returns:
            DataFrame procesado
        """
        return self.df
    
    def get_search_term_embeddings(self) -> Tuple[dict, np.ndarray]:
        """
        Retorna los embeddings de los términos de búsqueda.
        
        Returns:
            Tupla con diccionario de términos a embeddings y matriz de embeddings
        """
        return self.term_to_embedding, self.search_term_embeddings