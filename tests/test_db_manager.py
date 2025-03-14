import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlite3
import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_manager import DBManager


class TestDBManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_db_path = "test.sqlite3"
        self.db_manager = DBManager(self.test_db_path)
        
        # Sample test data
        self.test_tables = [('search_terms',), ('accounts',), ('clusters',)]
        self.test_schema = [
            (0, 'id', 'INTEGER', 1, None, 1),
            (1, 'search_term', 'TEXT', 0, None, 0),
            (2, 'impressions', 'INTEGER', 0, None, 0),
            (3, 'cost', 'REAL', 0, None, 0),
            (4, 'account_id', 'TEXT', 0, None, 0)
        ]
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'search_term': ['laptop', 'refrigerator', 'phone'],
            'impressions': [100, 200, 300],
            'cost': [10, 20, 30],
            'account_id': ['5555555555', '5555555555', '5555555555']
        })
        
        # Clustered data for testing save_clusters_to_db
        self.clustered_data = pd.DataFrame({
            'search_term': ['laptop', 'refrigerator', 'phone'],
            'impressions': [100, 200, 300],
            'cost': [10, 20, 30],
            'cluster': [0, 1, 0],
            'cluster_name': ['Tech', 'Appliances', 'Tech']
        })
    
    @patch('sqlite3.connect')
    def test_get_connection(self, mock_connect):
        """Test establishing a connection to the database."""
        # Configure the mock to return a mock connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Call the method being tested
        result = self.db_manager.get_connection()
        
        # Assert that the connection was created with the correct path
        mock_connect.assert_called_once_with(self.test_db_path)
        
        # Assert that the returned connection is correct
        self.assertEqual(result, mock_conn)
    
    @patch('sqlite3.connect')
    def test_get_tables(self, mock_connect):
        """Test getting tables from the database."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Configure the mock cursor to return our test tables
        mock_cursor.fetchall.return_value = self.test_tables
        
        # Call the method being tested
        result = self.db_manager.get_tables()
        
        # Assert that the cursor executed the correct SQL
        mock_cursor.execute.assert_called_once_with("SELECT name FROM sqlite_master WHERE type='table';")
        
        # Assert that the returned tables match our test data
        self.assertEqual(result, ['search_terms', 'accounts', 'clusters'])
    
    @patch('sqlite3.connect')
    def test_get_table_schema(self, mock_connect):
        """Test getting a table schema from the database."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Configure the mock cursor to return our test schema
        mock_cursor.fetchall.return_value = self.test_schema
        
        # Call the method being tested
        result = self.db_manager.get_table_schema('search_terms')
        
        # Assert that the cursor executed the correct SQL
        mock_cursor.execute.assert_called_once_with("PRAGMA table_info(search_terms);")
        
        # Assert that the returned schema matches our test data
        self.assertEqual(result, self.test_schema)
    
    @patch('sqlite3.connect')
    def test_get_sample_data(self, mock_connect):
        """Test getting sample data from a table."""
        # Set up the mock connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock pandas.read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=self.test_data) as mock_read_sql:
            # Call the method being tested
            result = self.db_manager.get_sample_data('search_terms', limit=3)
            
            # Assert that read_sql_query was called with the correct arguments
            mock_read_sql.assert_called_once_with("SELECT * FROM search_terms LIMIT 3", mock_conn)
            
            # Assert that the returned data matches our test data
            pd.testing.assert_frame_equal(result, self.test_data)
    
    @patch('sqlite3.connect')
    def test_save_clusters_to_db(self, mock_connect):
        """Test saving clustering results to the database."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Call the method being tested
        result = self.db_manager.save_clusters_to_db(self.clustered_data, '5555555555')
        
        # Assert that the cursor executed the correct SQL to create the table
        mock_cursor.execute.assert_called_with("""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id TEXT,
                search_term TEXT,
                cluster_id INTEGER,
                cluster_name TEXT
            )
            """)
        
        # Assert that executemany was called with correct parameters
        self.assertEqual(mock_cursor.executemany.call_count, 1)
        
        # Check the first argument of executemany (the SQL)
        sql_arg = mock_cursor.executemany.call_args[0][0]
        self.assertIn("INSERT INTO clusters", sql_arg)
        
        # Check data to be inserted (should have 3 rows)
        data_arg = mock_cursor.executemany.call_args[0][1]
        self.assertEqual(len(data_arg), 3)
        
        # Assert that changes were committed
        mock_conn.commit.assert_called_once()
        
        # Assert that the connection was closed
        mock_conn.close.assert_called_once()
        
        # Assert that the method returned True for success
        self.assertTrue(result)
    
    @patch('sqlite3.connect')
    def test_save_clusters_to_db_empty_df(self, mock_connect):
        """Test saving empty clustering results."""
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Call the method with empty data
        result = self.db_manager.save_clusters_to_db(empty_df, '5555555555')
        
        # Assert that the method returned False for failure
        self.assertFalse(result)
        
        # Assert that connect was not called
        mock_connect.assert_not_called()
    
    @patch('sqlite3.connect')
    def test_save_clusters_to_db_no_cluster_column(self, mock_connect):
        """Test saving data without cluster column."""
        # Create a DataFrame without cluster column
        df_no_cluster = pd.DataFrame({
            'search_term': ['laptop', 'refrigerator'],
            'impressions': [100, 200]
        })
        
        # Call the method with data missing cluster column
        result = self.db_manager.save_clusters_to_db(df_no_cluster, '5555555555')
        
        # Assert that the method returned False for failure
        self.assertFalse(result)
        
        # Assert that connect was not called
        mock_connect.assert_not_called()
    
    @patch('sqlite3.connect')
    def test_save_clusters_to_db_exception(self, mock_connect):
        """Test handling of exceptions when saving."""
        # Configure the mock to raise an exception
        mock_connect.side_effect = Exception("Database error")
        
        # Call the method
        result = self.db_manager.save_clusters_to_db(self.clustered_data, '5555555555')
        
        # Assert that the method returned False for failure
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()