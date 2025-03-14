from src.data_extractor import DataExtractor
from src.data_processor import DataProcessor
from src.clustering import ClusterAnalyzer
from src.db_manager import DBManager
from config import DB_PATH, API_URL, TARGET_ACCOUNT_ID, RESULTS_DIR
import pandas as pd
import time
import os
import sys

def extract_data():
    """Extrae los datos de la cuenta objetivo"""
    print("\n=== EXTRACCIÓN DE DATOS ===")
    # Inicializar el extractor de datos
    extractor = DataExtractor(DB_PATH, API_URL)
    
    # Obtener información de la cuenta
    account = extractor.get_account_data(TARGET_ACCOUNT_ID)
    if account:
        print(f"Cuenta encontrada: {account.name} (ID: {account.account_id})")
    else:
        print(f"No se encontró la cuenta con ID {TARGET_ACCOUNT_ID}")
        return None
    
    # Extraer datos del cliente
    df = extractor.extract_client_data(TARGET_ACCOUNT_ID)
    
    # Mostrar resumen de los datos
    if not df.empty:
        print("\nResumen de los datos extraídos:")
        print(f"Dimensiones: {df.shape}")
        print("\nPrimeras 5 filas:")
        print(df.head())
        
        # Guardar los datos extraídos en el directorio de resultados
        os.makedirs(RESULTS_DIR, exist_ok=True)  # Asegurar que el directorio existe
        df.to_csv(f"{RESULTS_DIR}/data_{TARGET_ACCOUNT_ID}.csv", index=False)
        print(f"\nDatos guardados en '{RESULTS_DIR}/data_{TARGET_ACCOUNT_ID}.csv'")
        
        return df
    else:
        return None

def process_data(df):
    """Procesa los datos para el clustering"""
    print("\n=== PROCESAMIENTO DE DATOS ===")
    # Inicializar procesador de datos
    processor = DataProcessor(df, results_dir=RESULTS_DIR)
    
    # Preprocesar datos
    X_combined = processor.preprocess_data()
    
    # Obtener DataFrame procesado
    processed_df = processor.get_processed_df()
    
    # Obtener embeddings si están disponibles
    term_to_embedding, _ = processor.get_search_term_embeddings()
    
    return X_combined, processed_df, term_to_embedding

def perform_clustering(X_combined, df, term_to_embedding):
    """Realiza el análisis de clustering"""
    print("\n=== ANÁLISIS DE CLUSTERING ===")
    
    # Crear directorio para resultados si no existe
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Inicializar analizador de clusters
    analyzer = ClusterAnalyzer(X_combined, df, RESULTS_DIR)
    
    # Encontrar número óptimo de clusters
    n_clusters = analyzer.find_optimal_clusters()
    
    # Realizar clustering
    clustered_df = analyzer.perform_clustering(n_clusters)
    
    # Asignar nombres a los clusters
    cluster_names = analyzer.name_clusters()
    print("\nNombres asignados a los clusters:")
    for cluster_id, name in cluster_names.items():
        print(f"  Cluster {cluster_id}: {name}")
    
    # Visualizar clusters
    analyzer.visualize_clusters()
    
    # Analizar características de los clusters
    stats_df = analyzer.analyze_clusters()
    print("\nEstadísticas de los clusters:")
    print(stats_df[['cluster_id', 'cluster_name', 'size', 'percentage']])
    
    # Analizar coherencia semántica si hay embeddings disponibles
    if term_to_embedding:
        analyzer.analyze_cluster_coherence(term_to_embedding)
    
    return clustered_df

def save_results(df, stats_df=None):
    """Guarda los resultados en la base de datos"""
    print("\n=== GUARDANDO RESULTADOS ===")
    
    # Inicializar gestor de base de datos
    db_manager = DBManager(DB_PATH)
    
    # Guardar resultados en la tabla 'clusters'
    success = db_manager.save_clusters_to_db(df, TARGET_ACCOUNT_ID)
    
    if success:
        print("Resultados guardados exitosamente en la base de datos")
    else:
        print("Hubo un problema al guardar los resultados")
    
    # Guardar un resumen de resultados en CSV
    if stats_df is not None:
        stats_df.to_csv(f"{RESULTS_DIR}/cluster_statistics.csv", index=False)
        print(f"Estadísticas guardadas en '{RESULTS_DIR}/cluster_statistics.csv'")
    
    # También guardar el DataFrame completo
    df.to_csv(f"{RESULTS_DIR}/clustered_data.csv", index=False)
    print(f"Datos con clusters guardados en '{RESULTS_DIR}/clustered_data.csv'")

def main():
    """Función principal"""
    start_time = time.time()
    
    print("=== PRUEBA TÉCNICA - DATA SCIENCE (CLUSTERING) ===")
    print(f"Cuenta objetivo: {TARGET_ACCOUNT_ID}")
    
    df = extract_data()
    
    if df is not None and not df.empty:
        # Procesar datos
        X_combined, processed_df, term_to_embedding = process_data(df)
        
        # Realizar clustering
        clustered_df = perform_clustering(X_combined, processed_df, term_to_embedding)
        
        # Guardar resultados
        save_results(clustered_df)
        
        print("\n=== PROCESO COMPLETADO EXITOSAMENTE ===")
    else:
        print("\n=== ERROR: No se pudieron cargar los datos ===")
    
    # Mostrar tiempo total de ejecución
    elapsed_time = time.time() - start_time
    print(f"\nTiempo total de ejecución: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        import traceback
        traceback.print_exc()