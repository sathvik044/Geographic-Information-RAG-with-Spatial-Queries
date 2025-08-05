"""
Geographic Information RAG System - Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import traceback

# Add error handling for imports
try:
    # Import our RAG system
    from src.rag_engine import GeographicRAGEngine
    from src.utils import create_spatial_query_examples
    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing RAG system: {e}")
    st.code(traceback.format_exc())
    RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Geographic Information RAG System",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource(ttl=3600)
def initialize_rag_engine():
    """Initialize the RAG engine with caching."""
    if not RAG_AVAILABLE:
        return None
    
    try:
        # Use a smaller embedding model for cloud deployment
        engine = GeographicRAGEngine(embedding_model="paraphrase-MiniLM-L3-v2")
        engine.load_sample_data()
        return engine
    except Exception as e:
        st.error(f"Error initializing RAG engine: {e}")
        st.code(traceback.format_exc())
        return None

def main():
    """Main application function."""
    
    # Header
    st.title("🌍 Geographic Information RAG System")
    st.markdown("Advanced spatial query processing with satellite imagery analysis and geographic data fusion")
    
    # Check if RAG system is available
    if not RAG_AVAILABLE:
        st.error("❌ RAG system components are not available. Switching to fallback mode.")
        st.info("This is a simplified version with limited functionality.")
        show_fallback_mode()
        return
    
    # Initialize RAG engine with error handling
    try:
        with st.spinner("Initializing Geographic RAG Engine..."):
            rag_engine = initialize_rag_engine()
        
        if rag_engine is None:
            st.error("❌ Failed to initialize RAG engine. Switching to fallback mode.")
            show_fallback_mode()
            return
    except Exception as e:
        st.error(f"❌ Error during initialization: {e}")
        st.info("Switching to fallback mode with limited functionality.")
        show_fallback_mode()
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Spatial Queries", "📊 System Statistics"]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(rag_engine)
    elif page == "🔍 Spatial Queries":
        show_spatial_queries_page(rag_engine)
    elif page == "📊 System Statistics":
        show_statistics_page(rag_engine)

def show_home_page(rag_engine):
    """Display the home page with system overview."""
    
    st.header("System Overview")
    
    # System capabilities
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🌐 Geographic Processing\n\nAdvanced spatial data processing with support for multiple formats.")
    
    with col2:
        st.info("🔍 Spatial Indexing\n\nEfficient R-tree spatial indexing for fast geographic queries.")
    
    with col3:
        st.info("🛰️ Satellite Analysis\n\nSatellite imagery analysis with NDVI calculation and feature extraction.")
    
    # Quick start section
    st.header("Quick Start")
    
    # Example queries
    example_queries = create_spatial_query_examples()
    
    st.markdown("**Try these example queries:**")
    for i, example in enumerate(example_queries[:3]):
        with st.expander(f"Example {i+1}: {example['query']}"):
            st.write(f"**Type:** {example['type']}")
            st.write(f"**Description:** {example['description']}")
    
    # System statistics
    st.header("System Status")
    
    stats = rag_engine.get_system_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datasets", stats["geographic_data"]["datasets"])
    
    with col2:
        st.metric("Total Features", stats["geographic_data"]["total_features"])
    
    with col3:
        st.metric("Indexed Datasets", stats["spatial_indexing"]["indexed_datasets"])
    
    with col4:
        st.metric("Total Queries", stats["query_history"]["total_queries"])

def show_spatial_queries_page(rag_engine):
    """Display the spatial queries page."""
    
    st.header("🔍 Spatial Queries")
    
    # Query input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query_text = st.text_area(
            "Enter your spatial query:",
            placeholder="e.g., What cities are within 100km of New York?",
            height=100
        )
    
    with col2:
        st.markdown("**Query Options:**")
        use_coordinates = st.checkbox("Include coordinates", value=False)
        
        if use_coordinates:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.7128, step=0.0001)
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-74.006, step=0.0001)
            query_coordinates = (lat, lon)
        else:
            query_coordinates = None
        
        n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    # Process query
    if st.button("🔍 Process Query", type="primary"):
        if query_text.strip():
            with st.spinner("Processing spatial query..."):
                results = rag_engine.process_spatial_query(
                    query_text=query_text,
                    query_coordinates=query_coordinates,
                    n_results=n_results
                )
            
            # Display results
            st.header("Query Results")
            
            # Generated response
            st.subheader("Generated Response:")
            st.write(results["generated_response"])
            
            # Results details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Retrieved Documents:")
                if results["retrieved_documents"]:
                    for i, doc in enumerate(results["retrieved_documents"][:3]):
                        with st.expander(f"Document {i+1}"):
                            st.write(f"**Content:** {doc.get('document', 'No content')}")
                            if 'distance_km' in doc:
                                st.write(f"**Distance:** {doc['distance_km']:.2f} km")
                            if 'metadata' in doc:
                                st.write(f"**Coordinates:** {doc['metadata'].get('coordinates_formatted', 'Unknown')}")
                else:
                    st.warning("No documents retrieved.")
            
            with col2:
                st.subheader("Spatial Results:")
                if results["spatial_results"]:
                    for i, result in enumerate(results["spatial_results"][:3]):
                        with st.expander(f"Spatial Result {i+1}"):
                            st.write(f"**Name:** {result.get('name', 'Unknown')}")
                            st.write(f"**Category:** {result.get('category', 'Unknown')}")
                            st.write(f"**Description:** {result.get('description', 'No description')}")
                else:
                    st.info("No spatial results available.")
        
        else:
            st.warning("Please enter a query.")

def show_fallback_mode():
    """Display a simplified fallback mode when the full system can't be initialized."""
    
    # Create demo data for the fallback mode
    def create_demo_data():
        return pd.DataFrame({
            'name': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'latitude': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
            'longitude': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
            'population': [8336817, 3979576, 2693976, 2320268, 1680992],
            'description': [
                'The Big Apple - Financial and cultural center',
                'City of Angels - Entertainment and technology hub',
                'Windy City - Transportation and business center',
                'Space City - Energy and aerospace industry',
                'Valley of the Sun - Technology and tourism'
            ]
        })
    
    # Demo mode indicator
    st.info("🔄 Running in Demo Mode - Full RAG system will be available after deployment optimization")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Sample Queries", "📊 Demo Data"]
    )
    
    # Page routing for fallback mode
    if page == "🏠 Home":
        st.header("System Overview")
        st.write("This is a simplified demo version of the Geographic Information RAG System.")
        st.write("The full system with advanced spatial querying and satellite imagery analysis will be available after deployment optimization.")
        
    elif page == "🔍 Sample Queries":
        st.header("Sample Spatial Queries")
        st.write("Here are some examples of the types of queries the full system can handle:")
        
        queries = [
            "Find all cities within 100 miles of Chicago",
            "What is the population density of Los Angeles?",
            "Show satellite imagery of Houston from the last 3 months",
            "Compare the urban development of New York and Phoenix"
        ]
        
        for query in queries:
            st.markdown(f"- {query}")
            
    elif page == "📊 Demo Data":
        st.header("Sample Geographic Data")
        demo_data = create_demo_data()
        st.dataframe(demo_data)
        
        # Simple map visualization
        st.subheader("City Locations")
        st.map(demo_data)

def show_statistics_page(rag_engine):
    """Display the system statistics page."""
    
    st.header("📈 System Statistics")
    
    # Get system statistics
    stats = rag_engine.get_system_statistics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datasets", stats["geographic_data"]["datasets"])
    
    with col2:
        st.metric("Total Features", stats["geographic_data"]["total_features"])
    
    with col3:
        st.metric("Indexed Datasets", stats["spatial_indexing"]["indexed_datasets"])
    
    with col4:
        st.metric("Total Queries", stats["query_history"]["total_queries"])
    
    # Detailed statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Geographic Data:")
        for name, data in stats["geographic_data"].items():
            if name not in ["datasets", "total_features"]:
                st.write(f"• **{name}**: {data.get('features', 0)} features")
    
    with col2:
        st.subheader("Vector Database:")
        for name, data in stats["vector_database"]["collection_statistics"].items():
            st.write(f"• **{name}**: {data.get('total_documents', 0)} documents")
    
    # Query history visualization
    if rag_engine.query_history:
        st.subheader("Query History:")
        
        df_history = pd.DataFrame(rag_engine.query_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        # Query results over time   
        st.line_chart(df_history.set_index('timestamp')['results_count'])
        
        # Query type distribution
        spatial_queries = sum(1 for q in rag_engine.query_history if q.get("coordinates") is not None)
        semantic_queries = len(rag_engine.query_history) - spatial_queries
        
        chart_data = pd.DataFrame({
            'Query Type': ['Spatial Queries', 'Semantic Queries'],
            'Count': [spatial_queries, semantic_queries]
        })
        
        st.bar_chart(chart_data.set_index('Query Type'))

if __name__ == "__main__":
    main()