import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import joblib
from typing import Dict, List, Tuple, Optional
import os

class ClusterAnalyzer:
    def __init__(self, X_combined: np.ndarray, df: pd.DataFrame, results_dir: str = 'notebooks'):
        """
        Inicializa el analizador de clusters.
        
        Args:
            X_combined: Array NumPy con características combinadas para clustering
            df: DataFrame original con datos
            results_dir: Directorio para guardar resultados
        """
        self.X_combined = X_combined
        self.df = df.copy()
        self.results_dir = results_dir
        self.optimal_k = None
        self.kmeans = None
        self.descriptive_names = None
        
        # Crear directorio para resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuraciones para visualización
        plt.style.use('seaborn-v0_8')
        sns.set_palette('viridis')
    
    def find_optimal_clusters(self, min_k: int = 2, max_k: int = 11) -> int:
        """
        Encuentra el número óptimo de clusters usando el método del codo y silueta.
        
        Args:
            min_k: Número mínimo de clusters a evaluar
            max_k: Número máximo de clusters a evaluar
            
        Returns:
            Número óptimo de clusters
        """
        print("Determinando número óptimo de clusters...")
        inertia = []
        silhouette_scores = []
        k_range = range(min_k, max_k)
        
        for k in k_range:
            print(f"Evaluando k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_combined)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_combined, kmeans.labels_))
        
        # Graficar método del codo y silhouette score
        plt.figure(figsize=(14, 6))
        
        # Método del codo
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertia, 'o-')
        plt.title('Método del Codo')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inercia')
        plt.grid(True)
        
        # Silhouette score
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'o-')
        plt.title('Puntaje de Silueta')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntaje de Silueta')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/optimal_clusters.png')
        
        # Determinar el número óptimo de clusters
        best_k_index = np.argmax(silhouette_scores)
        self.optimal_k = k_range[best_k_index]
        print(f"Número óptimo de clusters basado en silueta: {self.optimal_k}")
        
        # Garantizar al menos 3 clusters según los requisitos
        if self.optimal_k < 3:
            self.optimal_k = 3
            print(f"Ajustando a un mínimo de 3 clusters según los requisitos")
        
        return self.optimal_k
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Realiza el clustering con el número especificado o determinado de clusters.
        
        Args:
            n_clusters: Número de clusters a usar (opcional)
            
        Returns:
            DataFrame con datos y etiquetas de cluster asignadas
        """
        # Usar número determinado de clusters o encontrar el óptimo
        if n_clusters is None:
            if self.optimal_k is None:
                n_clusters = self.find_optimal_clusters()
            else:
                n_clusters = self.optimal_k
        else:
            self.optimal_k = n_clusters
        
        print(f"Aplicando clustering con {n_clusters} clusters...")
        # Aplicar KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_combined)
        
        # Guardar modelo para uso futuro
        joblib.dump(self.kmeans, f'{self.results_dir}/kmeans_model.pkl')
        print(f"Modelo de clustering guardado en '{self.results_dir}/kmeans_model.pkl'")
        
        return self.df
    
    def visualize_clusters(self) -> None:
        """
        Visualiza los clusters usando t-SNE para reducción de dimensionalidad.
        """
        if 'cluster' not in self.df.columns:
            print("No se han asignado clusters. Ejecute perform_clustering primero.")
            return
        
        print("Generando visualización t-SNE para los clusters...")
        # Aplicar t-SNE para visualización
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(self.X_combined)
        
        # Graficar clusters sin nombres descriptivos
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                             c=self.df['cluster'], cmap='viridis', 
                             alpha=0.7, s=50)
        plt.title(f'Visualización de Clusters con t-SNE (K={self.optimal_k})')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/clusters_tsne.png')
        
        # Si existen nombres descriptivos, hacer un segundo gráfico con ellos
        if self.descriptive_names is not None:
            self._visualize_named_clusters(X_tsne)
    
    def _visualize_named_clusters(self, X_tsne: np.ndarray) -> None:
        """
        Visualiza los clusters con nombres descriptivos.
        
        Args:
            X_tsne: Coordenadas t-SNE para visualización
        """
        plt.figure(figsize=(14, 12))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=self.df['cluster'], cmap='viridis', 
                              alpha=0.7, s=50)
        
        # Añadir etiquetas con los nombres de los clusters
        for cluster_id, name in self.descriptive_names.items():
            # Encontrar el centro del cluster en el espacio t-SNE
            cluster_points = X_tsne[self.df['cluster'] == cluster_id]
            if len(cluster_points) > 0:  # Verificar que el cluster tenga puntos
                centroid = cluster_points.mean(axis=0)
                # Añadir etiqueta
                plt.annotate(name, xy=centroid, xytext=(centroid[0], centroid[1]),
                            fontsize=12, weight='bold', color='black',
                            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
        
        plt.title('Visualización de Clusters con Nombres Descriptivos')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/clusters_named_tsne.png')
    
    def name_clusters(self, custom_names: Optional[Dict[int, str]] = None) -> Dict[int, str]:
        """
        Asigna nombres descriptivos a los clusters.
        
        Args:
            custom_names: Diccionario opcional con nombres personalizados
            
        Returns:
            Diccionario con nombres asignados a cada cluster
        """
        if 'cluster' not in self.df.columns:
            print("No se han asignado clusters. Ejecute perform_clustering primero.")
            return {}
        
        if custom_names is None:
            # Estos nombres deberían ser generados automáticamente o asignados manualmente
            # según el análisis de los términos más frecuentes en cada cluster
            # Aquí usaremos los nombres predefinidos de tu análisis
            self.descriptive_names = {
                0: "Productos básicos y abarrotes",
                1: "Refrigeradores y sistemas de refrigeración",
                2: "Tecnología y computación",
                3: "Lavadoras y electrodomésticos de lavandería",
                4: "Tiendas minoristas y retailers",
                5: "Bebidas alcohólicas",
                6: "Productos de consumo general y videojuegos",
                7: "Televisores y pantallas"
            }
            
            # Ajustar si hay menos clusters que nombres
            if self.optimal_k < len(self.descriptive_names):
                self.descriptive_names = {k: v for k, v in self.descriptive_names.items() if k < self.optimal_k}
        else:
            self.descriptive_names = custom_names
        
        # Aplicar nombres al DataFrame
        self.df['cluster_name'] = self.df['cluster'].map(self.descriptive_names)
        
        # Guardar nombres para uso futuro
        joblib.dump(self.descriptive_names, f'{self.results_dir}/cluster_names.pkl')
        print(f"Nombres de clusters guardados en '{self.results_dir}/cluster_names.pkl'")
        
        return self.descriptive_names
    
    def analyze_clusters(self) -> pd.DataFrame:
        """
        Analiza las características de cada cluster.
        
        Returns:
            DataFrame con estadísticas de cada cluster
        """
        if 'cluster' not in self.df.columns or 'cluster_name' not in self.df.columns:
            print("No se han asignado clusters o nombres. Ejecute perform_clustering y name_clusters primero.")
            return pd.DataFrame()
        
        # Métricas numéricas para análisis
        numeric_features = ['impressions', 'cost', 'cost_per_impression']
        
        # Calcular promedios globales
        global_impressions_avg = self.df['impressions'].mean()
        global_cost_avg = self.df['cost'].mean()
        global_cost_per_impression_avg = self.df['cost_per_impression'].mean()
        
        print("\nPromedios globales:")
        print(f"Impresiones promedio global: {global_impressions_avg:.2f}")
        print(f"Costo promedio global: {global_cost_avg:.2f}")
        print(f"Costo por impresión promedio global: {global_cost_per_impression_avg:.2f}")
        
        # Estadísticas por cluster
        cluster_stats = []
        
        for cluster_id, cluster_name in self.descriptive_names.items():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            # Si no hay datos para este cluster, continuar
            if len(cluster_data) == 0:
                continue
                
            # Métricas numéricas con comparación porcentual
            imp_avg = cluster_data['impressions'].mean()
            cost_avg = cluster_data['cost'].mean()
            cpi_avg = cluster_data['cost_per_impression'].mean()
            
            imp_pct = (imp_avg / global_impressions_avg - 1) * 100
            cost_pct = (cost_avg / global_cost_avg - 1) * 100
            cpi_pct = (cpi_avg / global_cost_per_impression_avg - 1) * 100
            
            # Términos más frecuentes
            top_terms = cluster_data['search_term'].value_counts().head(5).to_dict()
            
            # Añadir estadísticas a la lista
            cluster_stats.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df) * 100,
                'impressions_avg': imp_avg,
                'impressions_vs_global': imp_pct,
                'cost_avg': cost_avg,
                'cost_vs_global': cost_pct,
                'cost_per_impression_avg': cpi_avg,
                'cost_per_impression_vs_global': cpi_pct,
                'top_terms': top_terms
            })
        
        # Crear DataFrame con estadísticas
        stats_df = pd.DataFrame(cluster_stats)
        
        # Visualizar características numéricas por cluster
        self._visualize_numeric_features()
        
        # Guardar estadísticas
        stats_df.to_csv(f'{self.results_dir}/semantic_clusters_results.csv', index=False)
        
        return stats_df
    
    def _visualize_numeric_features(self) -> None:
        """
        Visualiza las características numéricas por cluster.
        """
        numeric_features = ['impressions', 'cost', 'cost_per_impression']
        
        plt.figure(figsize=(18, 8))
        for i, feature in enumerate(numeric_features, 1):
            plt.subplot(1, 3, i)
            ax = sns.boxplot(x='cluster_name', y=feature, data=self.df)
            
            # Rotar las etiquetas del eje x y ajustar posición
            plt.xticks(rotation=45, ha='right')
            
            plt.title(f'Distribución de {feature} por Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/numeric_features_by_cluster.png')
    
    def get_clustered_data(self) -> pd.DataFrame:
        """
        Retorna el DataFrame con las etiquetas de cluster asignadas.
        
        Returns:
            DataFrame con datos y etiquetas de cluster
        """
        return self.df
    
    def analyze_cluster_coherence(self, term_to_embedding: dict = None) -> None:
        """
        Analiza la coherencia semántica de los clusters.
        
        Args:
            term_to_embedding: Diccionario que mapea términos a embeddings
        """
        if term_to_embedding is None or 'cluster' not in self.df.columns:
            print("No se puede analizar la coherencia sin embeddings o clusters.")
            return
        
        print("Calculando coherencia semántica de los clusters...")
        # Para cada cluster, calcular la distancia promedio de sus embeddings al centroide
        cluster_coherence = {}
        
        for cluster_id in range(self.optimal_k):
            # Obtener los términos de búsqueda únicos en este cluster
            cluster_terms = self.df[self.df['cluster'] == cluster_id]['search_term_clean'].unique()
            
            if len(cluster_terms) > 0:
                # Obtener embeddings de estos términos
                cluster_embeddings = np.array([term_to_embedding[term] for term in cluster_terms])
                
                # Calcular el centroide (vector promedio)
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calcular distancias euclidianas al centroide
                distances = np.sqrt(np.sum((cluster_embeddings - centroid)**2, axis=1))
                
                # Guardar distancia promedio como medida de coherencia
                cluster_coherence[cluster_id] = np.mean(distances)
        
        # Mostrar resultados de coherencia (valores más bajos indican mayor coherencia)
        print("\nCoherencia semántica por cluster (distancia promedio al centroide):")
        
        # Ordenar por coherencia (de mayor a menor)
        sorted_coherence = sorted(cluster_coherence.items(), key=lambda x: x[1])
        
        for cluster_id, coherence in sorted_coherence:
            cluster_name = self.descriptive_names[cluster_id]
            print(f"{cluster_name}: {coherence:.4f}")
        
        # Visualización gráfica de la coherencia
        plt.figure(figsize=(12, 6))
        
        cluster_names = [self.descriptive_names[cluster_id] for cluster_id in cluster_coherence.keys()]
        coherence_values = list(cluster_coherence.values())
        
        # Ordenar para mejor visualización
        sorted_indices = np.argsort(coherence_values)
        sorted_names = [cluster_names[i] for i in sorted_indices]
        sorted_values = [coherence_values[i] for i in sorted_indices]
        
        bars = plt.barh(sorted_names, sorted_values, color='skyblue')
        
        # Añadir valores numéricos al final de cada barra
        for i, v in enumerate(sorted_values):
            plt.text(v + 0.05, i, f"{v:.3f}", va='center')
        
        plt.xlabel('Distancia promedio al centroide (menor = más coherente)')
        plt.title('Coherencia semántica por cluster')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/cluster_coherence.png')