"""
ConvoLens Setup Test - Windows Compatible
This script tests that your environment is set up correctly.
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test that all required packages are installed."""
    print("Testing package imports...")
    try:
        import pandas as pd
        import numpy as np
        import anthropic
        import requests
        import matplotlib.pyplot as plt
        import seaborn as sns
        import gradio as gr
        import sklearn
        print("[PASS] All packages imported successfully!")
        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False

def test_api_key():
    """Test that API key is loaded correctly."""
    print("\nTesting API key configuration...")
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("[FAIL] No API key found in .env file")
        print("Make sure you created .env file with ANTHROPIC_API_KEY=your_key")
        return False
    
    if api_key == "your_api_key_here":
        print("[FAIL] You need to replace 'your_api_key_here' with your actual API key")
        return False
    
    print("[PASS] API key loaded successfully!")
    return True

def test_api_connection():
    """Test actual connection to Anthropic API."""
    print("\nTesting Anthropic API connection...")
    load_dotenv()
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Simple test message
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'Setup successful!' if you can read this."}
            ]
        )
        
        response = message.content[0].text
        print(f"[PASS] API connection successful!")
        print(f"Claude says: {response}")
        return True
        
    except Exception as e:
        print(f"[FAIL] API connection failed: {e}")
        return False

def test_folder_structure():
    """Check that folder structure is correct."""
    print("\nTesting folder structure...")
    required_folders = [
        'data',
        'data/raw_conversations',
        'data/annotated',
        'notebooks',
        'src',
        'results',
        'results/figures'
    ]
    
    all_exist = True
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"[PASS] {folder}")
        else:
            print(f"[FAIL] {folder} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("="*60)
    print("ConvoLens Setup Test")
    print("="*60)
    
    results = []
    results.append(("Package Imports", test_imports()))
    results.append(("API Key Configuration", test_api_key()))
    results.append(("Folder Structure", test_folder_structure()))
    
    # Only test API connection if key is configured
    if results[1][1]:
        results.append(("API Connection", test_api_connection()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("SUCCESS! All tests passed! You're ready to start!")
        print("="*60)
    else:
        print("WARNING: Some tests failed. Please fix the issues above.")
        print("="*60)

if __name__ == "__main__":
    main()
