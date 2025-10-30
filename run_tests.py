#!/usr/bin/env python3
"""
Test runner script for Model Compass test suite.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the complete test suite."""
    print("Model Compass - Comprehensive Test Suite")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install -r requirements-test.txt")
        return 1
    
    # Run tests with coverage
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "tests/",  # Test directory
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        test_args.extend([
            "--cov=model_compass",  # Coverage for model_compass package
            "--cov-report=term-missing",  # Show missing lines
            "--cov-report=html:htmlcov",  # HTML coverage report
        ])
        print("📊 Running tests with coverage analysis...")
    except ImportError:
        print("📋 Running tests without coverage (install pytest-cov for coverage)")
    
    print()
    
    # Run the tests
    exit_code = pytest.main(test_args)
    
    print()
    if exit_code == 0:
        print("✅ All tests passed!")
        print()
        print("Test Coverage:")
        print("- Configuration parsing (YAML/JSON)")
        print("- All resolution types (intent, logical, physical, alias)")
        print("- Circular alias detection and error handling")
        print("- Profile context switching and nesting")
        print("- Core API functions and validation")
        print("- Model registry and configuration templates")
        print("- Integration workflows and error recovery")
        print("- Logging and error context")
        
        try:
            import pytest_cov
            print()
            print("📊 Coverage report generated in htmlcov/index.html")
        except ImportError:
            pass
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return exit_code


def run_specific_test_module(module_name):
    """Run tests for a specific module."""
    test_file = f"tests/test_{module_name}.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"Running tests for {module_name}...")
    
    test_args = [
        "-v",
        "--tb=short",
        test_file
    ]
    
    return pytest.main(test_args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        module = sys.argv[1]
        exit_code = run_specific_test_module(module)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)