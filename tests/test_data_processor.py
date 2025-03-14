import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame for testing
        self.test_df = pd.DataFrame({
            'search_term': ['laptop new', 'refrigerator samsung', 'smartphone android', 'tv smart 4k', 'washing machine'],
            'impressions': [100, 200, 50, 300, 150],
            'cost': [10, 20, 5, 30, 15]
        })
        
        # Create a temporary directory for test results
        self.test_results_dir = "test_results"
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Initialize the processor with the test data
        self.processor = DataProcessor(self.test_df, self.test_results_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files if they exist
        import shutil
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
    
    def test_init(self):
        """Test initialization of the DataProcessor."""
        self.assertEqual(self.processor.numeric_features, ['impressions', 'cost', 'cost_per_impression'])
        self.assertTrue(hasattr(self.processor, 'df'))
        self.assertEqual(len(self.processor.df), 5)
    
    def test_clean_search_terms(self):
        """Test cleaning search terms."""
        self.processor._clean_search_terms()
        
        # Check if search_term_clean column was created
        self.assertIn('search_term_clean', self.processor.df.columns)
        
        # Check if the terms were cleaned correctly
        cleaned_terms = self.processor.df['search_term_clean'].tolist()
        self.assertEqual(cleaned_terms[0], 'laptop new')  # No special chars to remove
        self.assertEqual(cleaned_terms[1], 'refrigerator samsung')
    
    def test_calculate_cost_metrics(self):
        """Test calculating cost metrics."""
        self.processor._calculate_cost_metrics()
        
        # Check if cost_per_impression column was created
        self.assertIn('cost_per_impression', self.processor.df.columns)
        
        # Check correct calculation
        expected_cpi = [0.1, 0.1, 0.1, 0.1, 0.1]  # 10/100, 20/200, etc.
        actual_cpi = self.processor.df['cost_per_impression'].tolist()
        
        for expected, actual in zip(expected_cpi, actual_cpi):
            self.assertAlmostEqual(expected, actual, places=6)
        
        # Test handling of zero impressions
        zero_impressions_df = pd.DataFrame({
            'search_term': ['test'],
            'impressions': [0],
            'cost': [10]
        })
        zero_processor = DataProcessor(zero_impressions_df, self.test_results_dir)
        zero_processor._calculate_cost_metrics()
        self.assertEqual(zero_processor.df.loc[0, 'cost_per_impression'], 0)
    
    def test_process_numeric_features(self):
        """Test processing numeric features."""
        # First calculate cost metrics which is required
        self.processor._calculate_cost_metrics()
        
        # Process numeric features
        result = self.processor._process_numeric_features()
        
        # Check if the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check if the array has the right dimensions
        # Should have 5 rows (one for each sample) and 3 columns (one for each feature)
        self.assertEqual(result.shape, (5, 3))
        
        # Check if the values are standardized - el recorte de outliers afecta la desviación estándar
        # por lo que comprobamos que la media esté cerca de 0 y que la desviación estándar
        # esté en un rango razonable (puede no ser exactamente 1)
        self.assertAlmostEqual(np.mean(result), 0, places=1)
        # Verificar que la desviación estándar esté en un rango razonable
        self.assertTrue(0.7 <= np.std(result) <= 1.3, 
                       f"La desviación estándar {np.std(result)} está fuera del rango esperado [0.7, 1.3]")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_process_text_features(self, mock_transformer):
        """Test processing text features."""
        # Clean search terms first
        self.processor._clean_search_terms()
        
        # Mock the embedding model
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Create mock embeddings (5 terms, 10 dimensions each)
        mock_embeddings = np.random.random((5, 10))
        mock_model.encode.return_value = mock_embeddings
        
        # Mock PCA behavior
        with patch('sklearn.decomposition.PCA') as mock_pca:
            # Configure the mock PCA
            mock_pca_instance = MagicMock()
            mock_pca.return_value = mock_pca_instance
            mock_pca_instance.fit_transform.return_value = np.random.random((5, 5))
            mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.2, 0.1, 0.05, 0.03])
            
            # Process text features
            result = self.processor._process_text_features()
            
            # Check if the method returns a numpy array
            self.assertIsInstance(result, np.ndarray)
            
            # Check if the term_to_embedding dictionary was created
            self.assertIsNotNone(self.processor.term_to_embedding)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_preprocess_data(self, mock_transformer):
        """Test the complete preprocessing pipeline."""
        # Mock the embedding model
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Create mock embeddings
        mock_embeddings = np.random.random((5, 10))
        mock_model.encode.return_value = mock_embeddings
        
        # Mock PCA behavior
        with patch('sklearn.decomposition.PCA') as mock_pca:
            # Configure the mock PCA
            mock_pca_instance = MagicMock()
            mock_pca.return_value = mock_pca_instance
            mock_pca_instance.fit_transform.return_value = np.random.random((5, 5))
            mock_pca_instance.explained_variance_ratio_ = np.array([0.5, 0.2, 0.1, 0.05, 0.03])
            
            # Call the complete preprocessing method
            result = self.processor.preprocess_data()
            
            # Check if the result is a numpy array
            self.assertIsInstance(result, np.ndarray)
            
            # Check if all the required columns are in the processed DataFrame
            processed_df = self.processor.get_processed_df()
            required_columns = ['search_term', 'impressions', 'cost', 'cost_per_impression', 'search_term_clean']
            for column in required_columns:
                self.assertIn(column, processed_df.columns)
    
    def test_get_processed_df(self):
        """Test getting the processed DataFrame."""
        df = self.processor.get_processed_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
    
    def test_get_search_term_embeddings(self):
        """Test getting search term embeddings."""
        # Test when embeddings haven't been generated yet
        term_to_embedding, embeddings = self.processor.get_search_term_embeddings()
        self.assertIsNone(term_to_embedding)
        self.assertIsNone(embeddings)


if __name__ == '__main__':
    unittest.main()