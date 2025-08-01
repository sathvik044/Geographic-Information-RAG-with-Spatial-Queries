"""
Geographic Information RAG System - Fallback Demo Application
This version works even if some geospatial dependencies fail
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Geographic Information RAG System - Demo",
    page_icon="🌍",
    layout="wide"
)

def create_demo_data():
    """Create demo data for the fallback version."""
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

def main():
    """Main application function."""
    
    # Header
    st.title("🌍 Geographic Information RAG System - Demo")
    st.markdown("Advanced spatial query processing with satellite imagery analysis and geographic data fusion")
    
    # Demo mode indicator
    st.info("🔄 Running in Demo Mode - Full RAG system will be available after deployment optimization")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Spatial Queries", "📊 System Statistics"]
    )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page_demo()
    elif page == "🔍 Spatial Queries":
        show_spatial_queries_page_demo()
    elif page == "📊 System Statistics":
        show_statistics_page_demo()

def show_home_page_demo():
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
    example_queries = [
        {"query": "What cities are within 100km of New York?", "type": "Spatial Range", "description": "Find cities within a specific distance"},
        {"query": "Show me the population density of major US cities", "type": "Demographic Analysis", "description": "Analyze population data with geographic context"},
        {"query": "What are the environmental features near Los Angeles?", "type": "Environmental Analysis", "description": "Identify natural features and environmental data"}
    ]
    
    st.markdown("**Try these example queries:**")
    for i, example in enumerate(example_queries):
        with st.expander(f"Example {i+1}: {example['query']}"):
            st.write(f"**Type:** {example['type']}")
            st.write(f"**Description:** {example['description']}")
    
    # System statistics
    st.header("System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datasets", 1)
    
    with col2:
        st.metric("Total Features", 5)
    
    with col3:
        st.metric("Indexed Datasets", 1)
    
    with col4:
        st.metric("Total Queries", 0)

def show_spatial_queries_page_demo():
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
                # Demo response
                demo_data = create_demo_data()
                
                st.header("Query Results")
                
                # Generated response
                st.subheader("Generated Response:")
                st.write("Based on your spatial query, here are the relevant geographic features and locations:")
                
                # Results details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Retrieved Documents:")
                    for i, row in demo_data.head(3).iterrows():
                        with st.expander(f"Document {i+1}"):
                            st.write(f"**Content:** {row['description']}")
                            st.write(f"**Coordinates:** {row['latitude']:.4f}, {row['longitude']:.4f}")
                
                with col2:
                    st.subheader("Spatial Results:")
                    for i, row in demo_data.head(3).iterrows():
                        with st.expander(f"Spatial Result {i+1}"):
                            st.write(f"**Name:** {row['name']}")
                            st.write(f"**Population:** {row['population']:,}")
                            st.write(f"**Description:** {row['description']}")
                
                # Show data table
                st.subheader("Geographic Data:")
                st.dataframe(demo_data)
        
        else:
            st.warning("Please enter a query.")

def show_statistics_page_demo():
    """Display the system statistics page."""
    
    st.header("📈 System Statistics")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datasets", 1)
    
    with col2:
        st.metric("Total Features", 5)
    
    with col3:
        st.metric("Indexed Datasets", 1)
    
    with col4:
        st.metric("Total Queries", 0)
    
    # Demo data visualization
    demo_data = create_demo_data()
    
    st.subheader("Geographic Data Overview:")
    st.dataframe(demo_data)
    
    # Population chart
    st.subheader("City Population Comparison:")
    st.bar_chart(demo_data.set_index('name')['population'])

if __name__ == "__main__":
    main() 