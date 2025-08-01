"""
Test script for the Geographic Information RAG System.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import GeographicRAGEngine
from src.utils import create_sample_geographic_data, save_sample_data

def test_basic_functionality():
    """Test basic RAG engine functionality."""
    print("🧪 Testing Geographic Information RAG System...")
    
    # Initialize RAG engine
    print("📦 Initializing RAG engine...")
    rag_engine = GeographicRAGEngine()
    
    # Load sample data
    print("📊 Loading sample data...")
    rag_engine.load_sample_data()
    
    # Test spatial query
    print("🔍 Testing spatial query...")
    results = rag_engine.process_spatial_query(
        query_text="What cities are near New York?",
        query_coordinates=(40.7128, -74.006),  # New York coordinates
        n_results=3
    )
    
    print(f"✅ Query processed successfully!")
    print(f"📝 Generated response: {results['generated_response'][:200]}...")
    print(f"📊 Retrieved {len(results['retrieved_documents'])} documents")
    print(f"🗺️ Found {len(results['spatial_results'])} spatial results")
    
    # Test system statistics
    print("📈 Testing system statistics...")
    stats = rag_engine.get_system_statistics()
    print(f"✅ System statistics retrieved:")
    print(f"   - Datasets: {stats['geographic_data']['datasets']}")
    print(f"   - Total features: {stats['geographic_data']['total_features']}")
    print(f"   - Total queries: {stats['query_history']['total_queries']}")
    
    # Test multi-scale analysis
    print("📊 Testing multi-scale analysis...")
    multi_scale_results = rag_engine.perform_multi_scale_analysis(
        center_coordinates=(40.7128, -74.006),
        scales=[10, 50, 100]
    )
    
    print(f"✅ Multi-scale analysis completed:")
    print(f"   - Scales analyzed: {multi_scale_results['summary']['scales_analyzed']}")
    print(f"   - Total features: {multi_scale_results['summary']['total_features_analyzed']}")
    
    print("🎉 All tests passed successfully!")
    return True

def test_sample_data_creation():
    """Test sample data creation and saving."""
    print("📝 Testing sample data creation...")
    
    # Create sample data
    sample_data = create_sample_geographic_data()
    
    # Save sample data
    save_sample_data(sample_data)
    
    print("✅ Sample data created and saved successfully!")
    return True

def main():
    """Main test function."""
    print("🚀 Starting Geographic Information RAG System Tests")
    print("=" * 60)
    
    try:
        # Test sample data creation
        test_sample_data_creation()
        print()
        
        # Test basic functionality
        test_basic_functionality()
        print()
        
        print("🎯 All tests completed successfully!")
        print("✅ The Geographic Information RAG System is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 