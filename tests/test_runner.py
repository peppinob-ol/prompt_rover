"""
Test runner per eseguire tutti i test di Prompt Rover
"""

import sys
import os
import pytest

# Aggiungi il percorso del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Esegue tutti i test"""
    print("=" * 60)
    print("ESECUZIONE TEST PROMPT ROVER")
    print("=" * 60)
    
    # Esegui pytest con opzioni verbose
    exit_code = pytest.main([
        "tests/",
        "-v",  # Verbose
        "--tb=short",  # Traceback corto
        "-p", "no:warnings",  # Disabilita warnings
        "--no-header",  # No header pytest
    ])
    
    if exit_code == 0:
        print("\n✅ TUTTI I TEST SONO PASSATI!")
    else:
        print(f"\n❌ ALCUNI TEST SONO FALLITI (codice: {exit_code})")
    
    return exit_code

def run_specific_tests(test_file):
    """Esegue test specifici"""
    print(f"Esecuzione test: {test_file}")
    return pytest.main([test_file, "-v"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Se specificato un file, esegui solo quello
        test_file = sys.argv[1]
        exit_code = run_specific_tests(test_file)
    else:
        # Altrimenti esegui tutti i test
        exit_code = run_all_tests()
    
    sys.exit(exit_code) 