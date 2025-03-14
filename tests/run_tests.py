import unittest
import sys
import os

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_data_extractor import TestDataExtractor
from tests.test_data_processor import TestDataProcessor
from tests.test_clustering import TestClusterAnalyzer
from tests.test_db_manager import TestDBManager

def run_tests():
    """
    Run all tests and display results.
    """
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Add test cases using loadTestsFromTestCase (recommended approach) instead of makeSuite
    test_suite.addTests(loader.loadTestsFromTestCase(TestDataExtractor))
    test_suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    test_suite.addTests(loader.loadTestsFromTestCase(TestClusterAnalyzer))
    test_suite.addTests(loader.loadTestsFromTestCase(TestDBManager))
    
    # Create test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    print("=" * 70)
    print("RUNNING TESTS FOR CLUSTERING PROJECT")
    print("=" * 70)
    result = test_runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: Run {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Return exit code based on test results (0 if all passed, 1 otherwise)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())