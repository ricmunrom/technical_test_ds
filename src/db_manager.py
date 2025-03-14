import sqlite3
import pandas as pd
from typing import List, Tuple, Optional

class DBManager:
    def __init__(self, db_path: str):
        """
        Inicializa el administrador de base de datos.
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Establece una conexión con la base de datos.
        
        Returns:
            Objeto de conexión con la base de datos
        """
        return sqlite3.connect(self.db_path)
    
    def get_tables(self) -> List[str]:
        """
        Obtiene la lista de tablas en la base de datos.
        
        Returns:
            Lista de nombres de tablas
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Consulta para obtener las tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        conn.close()
        
        return [table[0] for table in tables]
    
    def get_table_schema(self, table_name: str) -> List[Tuple]:
        """
        Obtiene el esquema de una tabla específica.
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            Lista de tuplas con información de las columnas
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Consulta para obtener el esquema de la tabla
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        
        conn.close()
        
        return schema
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Obtiene una muestra de datos de una tabla.
        
        Args:
            table_name: Nombre de la tabla
            limit: Número máximo de filas a obtener
            
        Returns:
            DataFrame con los datos de muestra
        """
        conn = self.get_connection()
        
        # Consulta para obtener datos de muestra
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        return df
    
    def save_clusters_to_db(self, df: pd.DataFrame, account_id: str) -> bool:
        """
        Guarda los resultados del clustering en una nueva tabla 'clusters'.
        
        Args:
            df: DataFrame con los datos y etiquetas de cluster
            account_id: ID de la cuenta analizada
            
        Returns:
            True si se guardaron correctamente, False en caso contrario
        """
        if 'cluster' not in df.columns or df.empty:
            print("No hay datos de clustering para guardar")
            return False
            
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Verificar si la tabla ya existe y crearla si no
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT,
                search_term TEXT,
                cluster_id INTEGER,
                cluster_name TEXT
            )
            """)
            
            # Preparar datos para inserción
            data_to_insert = []
            
            # Extraer campos relevantes
            for _, row in df.iterrows():
                data_to_insert.append((
                    account_id,
                    row['search_term'],
                    int(row['cluster']),
                    row['cluster_name'] if 'cluster_name' in df.columns else f"Cluster {row['cluster']}"
                ))
            
            # Insertar datos
            cursor.executemany("""
            INSERT INTO clusters (account_id, search_term, cluster_id, cluster_name)
            VALUES (?, ?, ?, ?)
            """, data_to_insert)
            
            # Guardar cambios
            conn.commit()
            conn.close()
            
            print(f"Se guardaron {len(data_to_insert)} registros en la tabla 'clusters'")
            return True
            
        except Exception as e:
            print(f"Error al guardar clusters en la base de datos: {str(e)}")
            return False