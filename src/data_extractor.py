from dataclasses import dataclass
import requests
import pandas as pd
import sqlite3
from typing import List, Optional


@dataclass
class AdsConfigData:
    name: str
    account_id: str


class DataExtractor:
    def __init__(self, db_path: str, api_url: str):
        """
        Inicializa el extractor de datos.
        
        Args:
            db_path: Ruta a la base de datos SQLite
            api_url: URL del API de cuentas
        """
        self.db_path = db_path
        self.api_url = api_url
    
    def get_accounts_from_api(self) -> List[AdsConfigData]:
        """
        Obtiene las cuentas desde la API externa.
        
        Returns:
            Lista de objetos AdsConfigData
        """
        try:
            response = requests.get(self.api_url)
            response.raise_for_status()  # Lanza excepción si hay error HTTP
            
            accounts_data = response.json()
            accounts = []
            
            for account in accounts_data:
                # Crear instancia de AdsConfigData para cada cuenta
                account_obj = AdsConfigData(
                    name=account["name"],
                    account_id=account["account_id"]
                )
                accounts.append(account_obj)
                
            return accounts
        except Exception as e:
            print(f"Error al obtener datos de la API: {str(e)}")
            return []
    
    def get_account_data(self, account_id: str) -> Optional[AdsConfigData]:
        """
        Busca una cuenta específica en la API por su ID.
        
        Args:
            account_id: ID de la cuenta a buscar
            
        Returns:
            Objeto AdsConfigData si se encuentra, None en caso contrario
        """
        accounts = self.get_accounts_from_api()
        for account in accounts:
            if account.account_id == account_id:
                return account
        return None
    
    def extract_client_data(self, account_id: str) -> pd.DataFrame:
        """
        Extrae los datos de la base de datos SQLite para un account_id específico.
        
        Args:
            account_id: ID de la cuenta para filtrar los datos
            
        Returns:
            DataFrame con los datos del cliente
        """
        try:
            # Conectar a la base de datos
            conn = sqlite3.connect(self.db_path)
            
            # Consultar datos de la cuenta específica
            query = f"SELECT * FROM search_terms WHERE account_id = '{account_id}'"
            
            # Leer datos en un DataFrame
            df = pd.read_sql_query(query, conn)
            
            # Cerrar conexión
            conn.close()
            
            if df.empty:
                print(f"No se encontraron datos para la cuenta {account_id}")
            else:
                print(f"Se extrajeron {len(df)} registros para la cuenta {account_id}")
                
            return df
        except Exception as e:
            print(f"Error al extraer datos de SQLite: {str(e)}")
            return pd.DataFrame()