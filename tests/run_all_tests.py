# /tests/run_all_tests.py
import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from test_initialization import TestInitialization
from test_embedding import TestEmbedding
from test_concept_extraction import TestConceptExtraction
from test_graph_building import TestGraphBuilding
from test_dimensionality_reduction import TestDimensionalityReduction
from test_visualization import TestVisualization
from test_caching import TestCaching
from test_end_to_end_pair import TestEndToEndPair
from test_end_to_end_chat import TestEndToEndChat
from test_mock_openai import TestMockOpenAI
from test_mock_sentence_transformer import TestMockSentenceTransformer
from test_integration import TestIntegration

if __name__ == '__main__':
    # Create test suites
    unit_suite = unittest.TestSuite()
    unit_suite.addTest(unittest.makeSuite(TestInitialization))
    unit_suite.addTest(unittest.makeSuite(TestEmbedding))
    unit_suite.addTest(unittest.makeSuite(TestConceptExtraction))
    unit_suite.addTest(unittest.makeSuite(TestGraphBuilding))
    unit_suite.addTest(unittest.makeSuite(TestDimensionalityReduction))
    unit_suite.addTest(unittest.makeSuite(TestVisualization))
    unit_suite.addTest(unittest.makeSuite(TestCaching))
    
    integration_suite = unittest.TestSuite()
    integration_suite.addTest(unittest.makeSuite(TestEndToEndPair))
    integration_suite.addTest(unittest.makeSuite(TestEndToEndChat))
    integration_suite.addTest(unittest.makeSuite(TestIntegration))
    
    mock_suite = unittest.TestSuite()
    mock_suite.addTest(unittest.makeSuite(TestMockOpenAI))
    mock_suite.addTest(unittest.makeSuite(TestMockSentenceTransformer))
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print("\n=== Running Unit Tests ===")
    runner.run(unit_suite)
    
    print("\n=== Running Integration Tests ===")
    runner.run(integration_suite)
    
    print("\n=== Running Mock Tests ===")
    runner.run(mock_suite)
    
    # Run all tests
    print("\n=== Running All Tests ===")
    unittest.main(verbosity=2)