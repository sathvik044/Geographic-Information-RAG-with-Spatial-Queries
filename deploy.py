"""
Deployment script for the Geographic Information RAG System.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("streamlit", "streamlit"),
        ("pandas", "pandas"), 
        ("numpy", "numpy"), 
        ("geopandas", "geopandas"), 
        ("shapely", "shapely"), 
        ("pyproj", "pyproj"), 
        ("rtree", "rtree"), 
        ("chromadb", "chromadb"), 
        ("sentence_transformers", "sentence_transformers"), 
        ("scikit-learn", "sklearn"), 
        ("scipy", "scipy")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "data/sample_data",
        "data/geographic_data", 
        "data/satellite_images",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")
    
    return True

def run_tests():
    """Run system tests."""
    print("🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_application():
    """Start the Streamlit application."""
    print("🚀 Starting Geographic Information RAG System...")
    
    try:
        # Start Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main deployment function."""
    print("🌍 Geographic Information RAG System - Deployment")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Deployment failed: Missing dependencies")
        return False
    
    print()
    
    # Setup directories
    if not setup_directories():
        print("❌ Deployment failed: Could not setup directories")
        return False
    
    print()
    
    # Run tests
    if not run_tests():
        print("❌ Deployment failed: Tests failed")
        return False
    
    print()
    
    # Start application
    print("🎯 Deployment successful! Starting application...")
    print("📱 The application will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print()
    
    start_application()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 