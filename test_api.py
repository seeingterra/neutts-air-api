"""
Simple test script for NeuTTS Air API
Tests the API endpoints without requiring the full model to be loaded.
"""

import sys
import time
import requests
from pathlib import Path

API_BASE = "http://127.0.0.1:8011"
TIMEOUT = 5

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Health check passed: {data}")
            return True
        elif response.status_code == 503:
            print(f"  ⚠️  Service not ready (model loading): {response.json()}")
            return True  # This is expected during startup
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\nTesting / endpoint...")
    try:
        response = requests.get(f"{API_BASE}/", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Root endpoint passed")
            print(f"     API: {data.get('name')} v{data.get('version')}")
            print(f"     Endpoints: {', '.join(data.get('endpoints', {}).keys())}")
            print(f"     Whisper available: {data.get('features', {}).get('whisper_available', False)}")
            return True
        else:
            print(f"  ❌ Root endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection error: {e}")
        return False

def test_voices():
    """Test voices list endpoint"""
    print("\nTesting /voices endpoint...")
    try:
        response = requests.get(f"{API_BASE}/voices", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            voices = data.get('voices', [])
            print(f"  ✅ Voices endpoint passed")
            print(f"     Found {len(voices)} voice(s)")
            for voice in voices:
                print(f"       - {voice.get('id')}")
            return True
        else:
            print(f"  ❌ Voices endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection error: {e}")
        return False

def test_gui():
    """Test GUI endpoint"""
    print("\nTesting /gui endpoint...")
    try:
        response = requests.get(f"{API_BASE}/gui", timeout=TIMEOUT)
        if response.status_code == 200:
            content = response.text
            if "NeuTTS Air" in content and "<html" in content.lower():
                print(f"  ✅ GUI endpoint passed (HTML content served)")
                return True
            else:
                print(f"  ❌ GUI endpoint returned unexpected content")
                return False
        else:
            print(f"  ❌ GUI endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection error: {e}")
        return False

def test_synthesis_validation():
    """Test synthesis endpoint validation (without actually synthesizing)"""
    print("\nTesting /synthesize endpoint validation...")
    try:
        # Test with empty text (should fail)
        response = requests.post(
            f"{API_BASE}/synthesize",
            json={"text": "", "voice_id": "dave"},
            timeout=TIMEOUT
        )
        if response.status_code == 400:
            print(f"  ✅ Empty text validation works")
        else:
            print(f"  ⚠️  Unexpected response for empty text: {response.status_code}")
        
        # Test with invalid voice (should fail)
        response = requests.post(
            f"{API_BASE}/synthesize",
            json={"text": "test", "voice_id": "nonexistent_voice"},
            timeout=TIMEOUT
        )
        if response.status_code in [404, 503]:
            print(f"  ✅ Invalid voice validation works")
        else:
            print(f"  ⚠️  Unexpected response for invalid voice: {response.status_code}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Connection error: {e}")
        return False

def wait_for_server(max_wait=30):
    """Wait for server to be ready"""
    print(f"Waiting for server at {API_BASE}...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE}/", timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is ready!\n")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"  Waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    print(f"❌ Server did not start within {max_wait} seconds\n")
    return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("NeuTTS Air API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    if not wait_for_server(max_wait=5):
        print("\n❌ Server is not running!")
        print("\nTo start the server, run:")
        print("  python api_server.py")
        print("  or")
        print("  ./start_server.sh  (Linux/Mac)")
        print("  start_server.bat   (Windows)")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Root Endpoint", test_root()))
    results.append(("Voices Endpoint", test_voices()))
    results.append(("GUI Endpoint", test_gui()))
    results.append(("Synthesis Validation", test_synthesis_validation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
