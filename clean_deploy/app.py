"""
Geographic Information RAG System - Gradio Application
"""

import gradio as gr
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
    print(f"Error importing RAG system: {e}")
    RAG_AVAILABLE = False

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

def initialize_rag_engine():
    """Initialize the RAG engine."""
    if not RAG_AVAILABLE:
        return None
    
    try:
        engine = GeographicRAGEngine()
        engine.load_sample_data()
        return engine
    except Exception as e:
        print(f"Error initializing RAG engine: {e}")
        return None

def process_spatial_query(query_text, use_coordinates, latitude, longitude, n_results):
    """Process a spatial query."""
    
    if not query_text.strip():
        return "Please enter a query.", "", ""
    
    # Initialize RAG engine
    rag_engine = initialize_rag_engine()
    
    if rag_engine is None:
        # Fallback to demo mode
        demo_data = create_demo_data()
        
        response = f"Based on your spatial query '{query_text}', here are the relevant geographic features and locations:"
        
        # Create document results
        doc_results = ""
        for i, row in demo_data.head(3).iterrows():
            doc_results += f"Document {i+1}:\n"
            doc_results += f"Content: {row['description']}\n"
            doc_results += f"Coordinates: {row['latitude']:.4f}, {row['longitude']:.4f}\n\n"
        
        # Create spatial results
        spatial_results = ""
        for i, row in demo_data.head(3).iterrows():
            spatial_results += f"Spatial Result {i+1}:\n"
            spatial_results += f"Name: {row['name']}\n"
            spatial_results += f"Population: {row['population']:,}\n"
            spatial_results += f"Description: {row['description']}\n\n"
        
        return response, doc_results, spatial_results
    
    # Full RAG processing
    try:
        query_coordinates = (latitude, longitude) if use_coordinates else None
        
        results = rag_engine.process_spatial_query(
            query_text=query_text,
            query_coordinates=query_coordinates,
            n_results=n_results
        )
        
        response = results["generated_response"]
        
        # Format retrieved documents
        doc_results = ""
        if results["retrieved_documents"]:
            for i, doc in enumerate(results["retrieved_documents"][:3]):
                doc_results += f"Document {i+1}:\n"
                doc_results += f"Content: {doc.get('document', 'No content')}\n"
                if 'distance_km' in doc:
                    doc_results += f"Distance: {doc['distance_km']:.2f} km\n"
                if 'metadata' in doc:
                    doc_results += f"Coordinates: {doc['metadata'].get('coordinates_formatted', 'Unknown')}\n"
                doc_results += "\n"
        
        # Format spatial results
        spatial_results = ""
        if results["spatial_results"]:
            for i, result in enumerate(results["spatial_results"][:3]):
                spatial_results += f"Spatial Result {i+1}:\n"
                spatial_results += f"Name: {result.get('name', 'Unknown')}\n"
                spatial_results += f"Category: {result.get('category', 'Unknown')}\n"
                spatial_results += f"Description: {result.get('description', 'No description')}\n\n"
        
        return response, doc_results, spatial_results
        
    except Exception as e:
        return f"Error processing query: {str(e)}", "", ""

def get_system_statistics():
    """Get system statistics."""
    rag_engine = initialize_rag_engine()
    
    if rag_engine is None:
        # Demo statistics
        stats = {
            "datasets": 1,
            "total_features": 5,
            "indexed_datasets": 1,
            "total_queries": 0
        }
    else:
        try:
            stats = rag_engine.get_system_statistics()
            stats = {
                "datasets": stats["geographic_data"]["datasets"],
                "total_features": stats["geographic_data"]["total_features"],
                "indexed_datasets": stats["spatial_indexing"]["indexed_datasets"],
                "total_queries": stats["query_history"]["total_queries"]
            }
        except:
            stats = {"datasets": 0, "total_features": 0, "indexed_datasets": 0, "total_queries": 0}
    
    return f"""
    **System Statistics:**
    
    📊 Datasets: {stats['datasets']}
    🏗️ Total Features: {stats['total_features']}
    🔍 Indexed Datasets: {stats['indexed_datasets']}
    📈 Total Queries: {stats['total_queries']}
    """

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Geographic Information RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌍 Geographic Information RAG System")
        gr.Markdown("Advanced spatial query processing with satellite imagery analysis and geographic data fusion")
        
        if not RAG_AVAILABLE:
            gr.Info("🔄 Running in Demo Mode - Full RAG system will be available after deployment optimization")
        
        with gr.Tabs():
            # Home Tab
            with gr.TabItem("🏠 Home"):
                gr.Markdown("## System Overview")
                
                with gr.Row():
                    with gr.Column():
                        gr.Info("🌐 Geographic Processing\n\nAdvanced spatial data processing with support for multiple formats.")
                    with gr.Column():
                        gr.Info("🔍 Spatial Indexing\n\nEfficient R-tree spatial indexing for fast geographic queries.")
                    with gr.Column():
                        gr.Info("🛰️ Satellite Analysis\n\nSatellite imagery analysis with NDVI calculation and feature extraction.")
                
                gr.Markdown("## Quick Start")
                gr.Markdown("Try these example queries:")
                
                example_queries = [
                    "What cities are within 100km of New York?",
                    "Show me the population density of major US cities",
                    "What are the environmental features near Los Angeles?"
                ]
                
                for i, query in enumerate(example_queries, 1):
                    gr.Markdown(f"**Example {i}:** {query}")
                
                gr.Markdown("## System Status")
                stats_output = gr.Markdown()
                stats_output.value = get_system_statistics()
            
            # Spatial Queries Tab
            with gr.TabItem("🔍 Spatial Queries"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Enter your spatial query:",
                            placeholder="e.g., What cities are within 100km of New York?",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        use_coords = gr.Checkbox(label="Include coordinates", value=False)
                        lat_input = gr.Number(label="Latitude", value=40.7128, minimum=-90, maximum=90)
                        lon_input = gr.Number(label="Longitude", value=-74.006, minimum=-180, maximum=180)
                        n_results = gr.Slider(label="Number of results", minimum=1, maximum=20, value=5, step=1)
                
                process_btn = gr.Button("🔍 Process Query", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        response_output = gr.Textbox(label="Generated Response", lines=5)
                    with gr.Column():
                        docs_output = gr.Textbox(label="Retrieved Documents", lines=10)
                    with gr.Column():
                        spatial_output = gr.Textbox(label="Spatial Results", lines=10)
                
                process_btn.click(
                    fn=process_spatial_query,
                    inputs=[query_input, use_coords, lat_input, lon_input, n_results],
                    outputs=[response_output, docs_output, spatial_output]
                )
            
            # Statistics Tab
            with gr.TabItem("📊 System Statistics"):
                gr.Markdown("## System Statistics")
                stats_display = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh Statistics")
                
                refresh_btn.click(
                    fn=get_system_statistics,
                    outputs=stats_display
                )
                
                # Initialize stats
                stats_display.value = get_system_statistics()
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860) 