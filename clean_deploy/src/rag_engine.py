"""
Main RAG engine for geographic information retrieval and generation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from .geographic_processor import GeographicProcessor
from .spatial_indexer import SpatialIndexer
from .satellite_analyzer import SatelliteAnalyzer
from .embeddings import GeographicEmbeddings
from .utils import (
    create_sample_geographic_data,
    coordinates_to_point,
    format_coordinates,
    calculate_distance
)

logger = logging.getLogger(__name__)

class GeographicRAGEngine:
    """
    Main RAG engine for geographic information retrieval and generation.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chroma_persist_dir: str = "./chroma_db"):
        """
        Initialize the geographic RAG engine.
        
        Args:
            embedding_model: Name of the sentence transformer model
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        # Initialize components
        self.geographic_processor = GeographicProcessor()
        self.spatial_indexer = SpatialIndexer()
        self.satellite_analyzer = SatelliteAnalyzer()
        self.embeddings = GeographicEmbeddings(embedding_model)
        
        # Initialize vector database
        self.embeddings.initialize_vector_database(chroma_persist_dir)
        
        # Data storage
        self.geographic_data = {}
        self.satellite_data = {}
        self.query_history = []
        
        logger.info("Geographic RAG Engine initialized")
    
    def load_sample_data(self) -> None:
        """Load sample geographic data for demonstration."""
        sample_data = create_sample_geographic_data()
        
        # Convert to GeoDataFrame
        gdf = self.geographic_processor.create_geodataframe_from_dict(sample_data)
        
        # Store data
        self.geographic_data["sample"] = gdf
        
        # Build spatial index
        self.spatial_indexer.build_spatial_index(gdf, "sample")
        
        # Add to vector database
        documents = []
        coordinates = []
        
        for _, row in gdf.iterrows():
            coords = (row.geometry.y, row.geometry.x)  # lat, lon
            description = f"{row.get('name', 'Unknown')}: {row.get('description', 'No description')}"
            
            documents.append(description)
            coordinates.append(coords)
        
        self.embeddings.add_geographic_documents(
            "sample_data",
            documents,
            coordinates
        )
        
        logger.info("Sample data loaded and indexed")
    
    def process_spatial_query(self, 
                            query_text: str,
                            query_coordinates: Optional[Tuple[float, float]] = None,
                            query_type: str = "semantic_spatial",
                            n_results: int = 5) -> Dict[str, Any]:
        """
        Process a spatial query using the RAG system.
        
        Args:
            query_text: Natural language query
            query_coordinates: Optional (lat, lon) coordinates
            query_type: Type of query processing
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results and generated response
        """
        results = {
            "query": query_text,
            "query_coordinates": query_coordinates,
            "query_type": query_type,
            "retrieved_documents": [],
            "spatial_results": [],
            "satellite_analysis": None,
            "generated_response": "",
            "metadata": {}
        }
        
        try:
            # Step 1: Semantic retrieval from vector database
            if query_coordinates:
                # Spatial-semantic search
                search_results = self.embeddings.semantic_spatial_search(
                    "sample_data",
                    query_text,
                    (query_coordinates[0], query_coordinates[1], 100),  # 100km radius
                    n_results
                )
                results["retrieved_documents"] = search_results.get("results", [])
            else:
                # Pure semantic search
                search_results = self.embeddings.spatial_query(
                    "sample_data",
                    query_text,
                    (0, 0),  # Dummy coordinates
                    n_results
                )
                results["retrieved_documents"] = [
                    {
                        "document": doc,
                        "metadata": meta,
                        "distance": dist
                    }
                    for doc, meta, dist in zip(
                        search_results["documents"],
                        search_results["metadatas"],
                        search_results["distances"]
                    )
                ]
            
            # Step 2: Spatial analysis if coordinates provided
            if query_coordinates:
                query_point = coordinates_to_point(query_coordinates[0], query_coordinates[1])
                
                # Find nearby features
                nearby_features = self.spatial_indexer.nearest_neighbor_query(
                    query_point,
                    n_neighbors=n_results,
                    index_name="sample"
                )
                
                results["spatial_results"] = nearby_features.to_dict('records')
                
                # Range query
                range_results = self.spatial_indexer.range_query(
                    query_point,
                    radius_km=100,
                    index_name="sample"
                )
                
                results["metadata"]["range_query_results"] = len(range_results)
            
            # Step 3: Generate response
            results["generated_response"] = self._generate_response(
                query_text, results["retrieved_documents"], results["spatial_results"]
            )
            
            # Step 4: Add to query history
            self.query_history.append({
                "timestamp": pd.Timestamp.now(),
                "query": query_text,
                "coordinates": query_coordinates,
                "results_count": len(results["retrieved_documents"])
            })
            
        except Exception as e:
            logger.error(f"Error processing spatial query: {e}")
            results["error"] = str(e)
        
        return results
    
    def analyze_satellite_imagery(self, 
                                image_path: str,
                                analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze satellite imagery for geographic insights.
        
        Args:
            image_path: Path to satellite image
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "image_path": image_path,
            "analysis_type": analysis_type,
            "vegetation_indices": {},
            "land_cover_classification": {},
            "feature_extraction": {},
            "change_detection": None,
            "visualization_path": None
        }
        
        try:
            # Load satellite image
            satellite_data = self.satellite_analyzer.load_satellite_image(image_path)
            image_data = satellite_data['data']
            
            # Perform analysis based on type
            if analysis_type == "comprehensive":
                # Calculate vegetation indices
                if len(image_data.shape) == 3 and image_data.shape[0] >= 4:
                    indices = self.satellite_analyzer.calculate_vegetation_indices(image_data)
                    results["vegetation_indices"] = {
                        name: {
                            "mean": float(np.mean(index)),
                            "std": float(np.std(index)),
                            "min": float(np.min(index)),
                            "max": float(np.max(index))
                        }
                        for name, index in indices.items()
                    }
                
                # Land cover classification
                try:
                    classification = self.satellite_analyzer.classify_land_cover(image_data)
                    unique_classes, counts = np.unique(classification, return_counts=True)
                    results["land_cover_classification"] = {
                        "class_distribution": dict(zip(unique_classes.tolist(), counts.tolist())),
                        "total_pixels": int(np.sum(counts))
                    }
                except Exception as e:
                    results["land_cover_classification"]["error"] = str(e)
                
                # Feature extraction
                features = self.satellite_analyzer.extract_features(image_data)
                results["feature_extraction"] = {
                    "n_keypoints": len(features.get('keypoints', [])),
                    "texture_stats": features.get('texture_stats', {}),
                    "color_stats": features.get('color_stats', {})
                }
            
            elif analysis_type == "vegetation":
                # Focus on vegetation analysis
                if len(image_data.shape) == 3 and image_data.shape[0] >= 4:
                    indices = self.satellite_analyzer.calculate_vegetation_indices(image_data)
                    results["vegetation_indices"] = {
                        name: {
                            "mean": float(np.mean(index)),
                            "std": float(np.std(index))
                        }
                        for name, index in indices.items()
                    }
            
            elif analysis_type == "land_cover":
                # Focus on land cover classification
                try:
                    classification = self.satellite_analyzer.classify_land_cover(image_data)
                    unique_classes, counts = np.unique(classification, return_counts=True)
                    results["land_cover_classification"] = {
                        "class_distribution": dict(zip(unique_classes.tolist(), counts.tolist())),
                        "total_pixels": int(np.sum(counts))
                    }
                except Exception as e:
                    results["land_cover_classification"]["error"] = str(e)
            
            # Store satellite data
            self.satellite_data[image_path] = satellite_data
            
        except Exception as e:
            logger.error(f"Error analyzing satellite imagery: {e}")
            results["error"] = str(e)
        
        return results
    
    def perform_multi_scale_analysis(self, 
                                   center_coordinates: Tuple[float, float],
                                   scales: List[float] = [10, 50, 100]) -> Dict[str, Any]:
        """
        Perform multi-scale geographic analysis.
        
        Args:
            center_coordinates: Center point for analysis
            scales: List of analysis scales in kilometers
            
        Returns:
            Dictionary with multi-scale analysis results
        """
        results = {
            "center_coordinates": center_coordinates,
            "scales": scales,
            "scale_analyses": {},
            "summary": {}
        }
        
        try:
            center_point = coordinates_to_point(center_coordinates[0], center_coordinates[1])
            
            for scale in scales:
                scale_results = {}
                
                # Range query at this scale
                range_results = self.spatial_indexer.range_query(
                    center_point,
                    radius_km=scale,
                    index_name="sample"
                )
                
                scale_results["features_in_range"] = len(range_results)
                
                # Spatial statistics
                if len(range_results) > 0:
                    stats = self.geographic_processor.calculate_spatial_statistics(range_results)
                    scale_results["spatial_statistics"] = stats
                
                # Nearest neighbors
                nearest = self.spatial_indexer.nearest_neighbor_query(
                    center_point,
                    n_neighbors=5,
                    index_name="sample"
                )
                
                scale_results["nearest_features"] = nearest.to_dict('records')
                
                results["scale_analyses"][f"{scale}km"] = scale_results
            
            # Generate summary
            total_features = sum(analysis["features_in_range"] for analysis in results["scale_analyses"].values())
            results["summary"] = {
                "total_features_analyzed": total_features,
                "scales_analyzed": len(scales),
                "center_point": format_coordinates(center_coordinates[0], center_coordinates[1])
            }
            
        except Exception as e:
            logger.error(f"Error performing multi-scale analysis: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_response(self, 
                          query: str, 
                          retrieved_docs: List[Dict], 
                          spatial_results: List[Dict]) -> str:
        """
        Generate a natural language response based on retrieved documents and spatial results.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents from vector search
            spatial_results: Spatial analysis results
            
        Returns:
            Generated response text
        """
        # Simple response generation - in practice, you might use a more sophisticated LLM
        response_parts = []
        
        # Add query context
        response_parts.append(f"Based on your query: '{query}'")
        
        # Add retrieved document information
        if retrieved_docs:
            response_parts.append("\n\nRelevant geographic information:")
            for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 results
                if "document" in doc:
                    response_parts.append(f"{i}. {doc['document']}")
                if "distance_km" in doc:
                    response_parts.append(f"   Distance: {doc['distance_km']:.1f} km")
        
        # Add spatial analysis results
        if spatial_results:
            response_parts.append("\n\nNearby geographic features:")
            for i, result in enumerate(spatial_results[:3], 1):  # Top 3 results
                name = result.get('name', 'Unknown feature')
                description = result.get('description', 'No description available')
                response_parts.append(f"{i}. {name}: {description}")
        
        # Add summary
        total_results = len(retrieved_docs) + len(spatial_results)
        response_parts.append(f"\n\nFound {total_results} relevant geographic features for your query.")
        
        return " ".join(response_parts)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "geographic_data": {
                "datasets": len(self.geographic_data),
                "total_features": sum(len(gdf) for gdf in self.geographic_data.values())
            },
            "spatial_indexing": {
                "indexed_datasets": len(self.spatial_indexer.geodataframes),
                "index_statistics": {}
            },
            "satellite_analysis": {
                "loaded_images": len(self.satellite_analyzer.satellite_data),
                "analysis_results": len(self.satellite_analyzer.analysis_results)
            },
            "vector_database": {
                "collections": len(self.embeddings.collections),
                "collection_statistics": {}
            },
            "query_history": {
                "total_queries": len(self.query_history),
                "recent_queries": len([q for q in self.query_history if 
                                     (pd.Timestamp.now() - q['timestamp']).days < 1])
            }
        }
        
        # Add detailed statistics
        for name, gdf in self.geographic_data.items():
            stats["geographic_data"][name] = {
                "features": len(gdf),
                "geometry_types": gdf.geometry.geom_type.value_counts().to_dict()
            }
        
        for name in self.spatial_indexer.geodataframes:
            stats["spatial_indexing"]["index_statistics"][name] = \
                self.spatial_indexer.get_index_statistics(name)
        
        for name in self.embeddings.collections:
            stats["vector_database"]["collection_statistics"][name] = \
                self.embeddings.get_collection_statistics(name)
        
        return stats
    
    def export_results(self, filepath: str, results: Dict[str, Any]) -> None:
        """
        Export analysis results to a file.
        
        Args:
            filepath: Path to save results
            results: Results to export
        """
        import json
        
        # Convert numpy arrays and other non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, Point):
                return {"type": "Point", "coordinates": [obj.x, obj.y]}
            elif isinstance(obj, Polygon):
                return {"type": "Polygon", "coordinates": [list(obj.exterior.coords)]}
            elif isinstance(obj, gpd.GeoDataFrame):
                return obj.to_dict('records')
            else:
                return obj
        
        # Recursively convert objects
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_for_json(obj)
        
        exportable_results = recursive_convert(results)
        
        with open(filepath, 'w') as f:
            json.dump(exportable_results, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")
    
    def create_visualization(self, 
                           results: Dict[str, Any],
                           save_path: Optional[str] = None) -> None:
        """
        Create visualizations for analysis results.
        
        Args:
            results: Analysis results to visualize
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        import folium
        
        # Create a map centered on query coordinates if available
        if results.get("query_coordinates"):
            center_lat, center_lon = results["query_coordinates"]
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add query point
            folium.Marker(
                [center_lat, center_lon],
                popup=f"Query: {results['query']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add retrieved features
            for i, doc in enumerate(results.get("retrieved_documents", [])):
                if "metadata" in doc and "latitude" in doc["metadata"]:
                    lat = doc["metadata"]["latitude"]
                    lon = doc["metadata"]["longitude"]
                    name = doc.get("document", f"Feature {i+1}")
                    
                    folium.Marker(
                        [lat, lon],
                        popup=name,
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(m)
            
            # Save map
            if save_path:
                m.save(save_path)
                logger.info(f"Visualization saved to {save_path}")
            else:
                return m
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Query history
        if self.query_history:
            df_history = pd.DataFrame(self.query_history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            
            axes[0, 0].plot(df_history['timestamp'], df_history['results_count'])
            axes[0, 0].set_title('Query Results Over Time')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Results distribution
        if results.get("retrieved_documents"):
            distances = [doc.get("distance_km", 0) for doc in results["retrieved_documents"]]
            axes[0, 1].hist(distances, bins=10)
            axes[0, 1].set_title('Distance Distribution of Results')
            axes[0, 1].set_xlabel('Distance (km)')
        
        # System statistics
        stats = self.get_system_statistics()
        if stats["geographic_data"]["datasets"] > 0:
            dataset_names = list(stats["geographic_data"].keys())[1:]  # Skip 'datasets' key
            feature_counts = [stats["geographic_data"][name]["features"] for name in dataset_names]
            
            axes[1, 0].bar(dataset_names, feature_counts)
            axes[1, 0].set_title('Features per Dataset')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Query type distribution
        if self.query_history:
            query_types = [q.get("coordinates") is not None for q in self.query_history]
            spatial_count = sum(query_types)
            semantic_count = len(query_types) - spatial_count
            
            axes[1, 1].pie([spatial_count, semantic_count], 
                          labels=['Spatial', 'Semantic'],
                          autopct='%1.1f%%')
            axes[1, 1].set_title('Query Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.html', '_summary.png'), dpi=300, bbox_inches='tight')
        
        plt.show() 