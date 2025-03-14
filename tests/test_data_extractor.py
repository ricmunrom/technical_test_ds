import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlite3
import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_extractor import DataExtractor, AdsConfigData


class TestDataExtractor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_path = "mock_db.sqlite3"
        self.mock_api_url = "https://mock-api.com/accounts"
        self.extractor = DataExtractor(self.mock_db_path, self.mock_api_url)
        
        # Sample test data
        self.mock_accounts = [
            {"name": "Test Account 1", "account_id": "1111111111"},
            {"name": "Test Account 2", "account_id": "2222222222"},
            {"name": "Target Account", "account_id": "5555555555"}
        ]
        
        self.mock_search_terms = pd.DataFrame({
            'search_term': ['laptop', 'refrigerator', 'phone'],
            'impressions': [100, 200, 300],
            'cost': [10, 20, 30],
            'account_id': ['5555555555', '5555555555', '5555555555']
        })
    
    @patch('requests.get')
    def test_get_accounts_from_api(self, mock_get):
        """Test retrieving accounts from the API."""
        # Configure the mock to return a successful response with mock data
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_accounts
        mock_get.return_value = mock_response
        
        # Call the method being tested
        result = self.extractor.get_accounts_from_api()
        
        # Assert that the API was called with the correct URL
        mock_get.assert_called_once_with(self.mock_api_url)
        
        # Assert that the result has the correct structure
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], AdsConfigData)
        self.assertEqual(result[0].name, "Test Account 1")
        self.assertEqual(result[0].account_id, "1111111111")
    
    @patch('requests.get')
    def test_get_account_data_found(self, mock_get):
        """Test retrieving a specific account that exists."""
        # Configure the mock to return a successful response with mock data
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_accounts
        mock_get.return_value = mock_response
        
        # Call the method being tested
        result = self.extractor.get_account_data("5555555555")
        
        # Assert the result is correct
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Target Account")
        self.assertEqual(result.account_id, "5555555555")
    
    @patch('requests.get')
    def test_get_account_data_not_found(self, mock_get):
        """Test retrieving a specific account that doesn't exist."""
        # Configure the mock to return a successful response with mock data
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_accounts
        mock_get.return_value = mock_response
        
        # Call the method being tested
        result = self.extractor.get_account_data("9999999999")
        
        # Assert the result is None when account is not found
        self.assertIsNone(result)
    
    @patch('sqlite3.connect')
    def test_extract_client_data(self, mock_connect):
        """Test extracting client data from the database."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Configure the mock to return our test DataFrame
        mock_conn.execute.return_value = None
        pd.read_sql_query = MagicMock(return_value=self.mock_search_terms)
        
        # Override the pandas read_sql_query function
        with patch('pandas.read_sql_query', return_value=self.mock_search_terms):
            # Call the method being tested
            result = self.extractor.extract_client_data("5555555555")
        
            # Assert that the connection was created with the correct path
            mock_connect.assert_called_once_with(self.mock_db_path)
            
            # Assert that the returned DataFrame has the expected data
            self.assertEqual(len(result), 3)
            self.assertEqual(list(result['search_term']), ['laptop', 'refrigerator', 'phone'])
    
    @patch('sqlite3.connect')
    def test_extract_client_data_empty(self, mock_connect):
        """Test extracting client data when no data is found."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Configure the mock to return an empty DataFrame
        with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
            # Call the method being tested
            result = self.extractor.extract_client_data("9999999999")
            
            # Assert that an empty DataFrame is returned
            self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()