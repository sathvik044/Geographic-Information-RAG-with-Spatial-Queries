"""
Geographic-aware embeddings for spatial data and text.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .utils import format_coordinates, calculate_distance

logger = logging.getLogger(__name__)

class GeographicEmbeddings:
    """
    Generate and manage geographic-aware embeddings for spatial data and text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the geographic embeddings system.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chroma_client = None
        self.collections = {}
        
    def initialize_vector_database(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB vector database.
        
        Args:
            persist_directory: Directory to persist the database
        """
        try:
            # Use in-memory client for cloud deployment to avoid file permission issues
            self.chroma_client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=False,
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_directory
                )
            )
            logger.info(f"Initialized ChromaDB in memory with reference to {persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def create_geographic_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Create a new collection for geographic data.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        if self.chroma_client is None:
            raise ValueError("ChromaDB not initialized. Call initialize_vector_database first.")
        
        try:
            # Use get_or_create to handle existing collections
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Geographic data collection: {collection_name}"}
            )
            self.collections[collection_name] = collection
            logger.info(f"Using collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating/getting collection {collection_name}: {e}")
            raise
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text data.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def generate_spatial_embeddings(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate embeddings for spatial coordinates.
        
        Args:
            coordinates: List of (lat, lon) coordinates
            
        Returns:
            Array of spatial embeddings
        """
        # Convert coordinates to normalized embeddings
        # This is a simplified approach - in practice, you might use more sophisticated methods
        embeddings = []
        
        for lat, lon in coordinates:
            # Normalize coordinates to [-1, 1] range
            norm_lat = lat / 90.0  # Latitude range: -90 to 90
            norm_lon = lon / 180.0  # Longitude range: -180 to 180
            
            # Create a simple embedding vector
            # In practice, you might use more sophisticated spatial encoding
            embedding = np.array([norm_lat, norm_lon, 
                                np.sin(np.radians(lat)), np.cos(np.radians(lat)),
                                np.sin(np.radians(lon)), np.cos(np.radians(lon))])
            
            # Pad to match text embedding dimension
            target_dim = self.model.get_sentence_embedding_dimension()
            if len(embedding) < target_dim:
                padding = np.zeros(target_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:target_dim]
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def generate_geographic_text_embeddings(self, 
                                          texts: List[str],
                                          coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate embeddings that combine text and spatial information.
        
        Args:
            texts: List of text descriptions
            coordinates: List of corresponding (lat, lon) coordinates
            
        Returns:
            Array of combined embeddings
        """
        # Generate text embeddings
        text_embeddings = self.generate_text_embeddings(texts)
        
        # Generate spatial embeddings
        spatial_embeddings = self.generate_spatial_embeddings(coordinates)
        
        # Combine text and spatial embeddings
        # Simple concatenation - could be more sophisticated
        combined_embeddings = np.concatenate([text_embeddings, spatial_embeddings], axis=1)
        
        return combined_embeddings
    
    def add_geographic_documents(self, 
                               collection_name: str,
                               documents: List[str],
                               coordinates: List[Tuple[float, float]],
                               metadata: Optional[List[Dict[str, Any]]] = None,
                               ids: Optional[List[str]] = None) -> None:
        """
        Add geographic documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            coordinates: List of (lat, lon) coordinates
            metadata: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        if collection_name not in self.collections:
            self.create_geographic_collection(collection_name)
        
        collection = self.collections[collection_name]
        
        # Generate embeddings
        embeddings = self.generate_geographic_text_embeddings(documents, coordinates)
        
        # Prepare metadata
        if metadata is None:
            metadata = []
            for i, (lat, lon) in enumerate(coordinates):
                metadata.append({
                    "latitude": lat,
                    "longitude": lon,
                    "coordinates_formatted": format_coordinates(lat, lon),
                    "document_index": i
                })
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection {collection_name}")
    
    def spatial_query(self, 
                     collection_name: str,
                     query_text: str,
                     query_coordinates: Tuple[float, float],
                     n_results: int = 5) -> Dict[str, Any]:
        """
        Perform a spatial query combining text and location.
        
        Args:
            collection_name: Name of the collection to query
            query_text: Query text
            query_coordinates: (lat, lon) coordinates for the query
            n_results: Number of results to return
            
        Returns:
            Dictionary with query results
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Generate query embedding
        query_embedding = self.generate_geographic_text_embeddings(
            [query_text], [query_coordinates]
        )[0]
        
        # Perform query
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }
    
    def range_query(self,
                   collection_name: str,
                   center_coordinates: Tuple[float, float],
                   radius_km: float,
                   query_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a range query within a geographic radius.
        
        Args:
            collection_name: Name of the collection to query
            center_coordinates: Center point (lat, lon)
            radius_km: Search radius in kilometers
            query_text: Optional text query
            
        Returns:
            Dictionary with query results
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Get all documents in collection
        all_results = collection.get(include=["documents", "metadatas"])
        
        # Filter by geographic distance
        filtered_results = []
        center_lat, center_lon = center_coordinates
        
        for i, metadata in enumerate(all_results["metadatas"]):
            doc_lat = metadata.get("latitude")
            doc_lon = metadata.get("longitude")
            
            if doc_lat is not None and doc_lon is not None:
                distance = calculate_distance(
                    (center_lat, center_lon),
                    (doc_lat, doc_lon)
                )
                
                if distance <= radius_km:
                    filtered_results.append({
                        "document": all_results["documents"][i],
                        "metadata": metadata,
                        "distance_km": distance,
                        "id": all_results["ids"][i]
                    })
        
        # Sort by distance
        filtered_results.sort(key=lambda x: x["distance_km"])
        
        # Apply text filtering if query provided
        if query_text:
            query_embedding = self.generate_text_embeddings([query_text])[0]
            
            for result in filtered_results:
                doc_embedding = self.generate_text_embeddings([result["document"]])[0]
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                result["text_similarity"] = similarity
            
            # Sort by text similarity
            filtered_results.sort(key=lambda x: x["text_similarity"], reverse=True)
        
        return {
            "results": filtered_results,
            "total_found": len(filtered_results),
            "query_center": center_coordinates,
            "query_radius_km": radius_km
        }
    
    def semantic_spatial_search(self,
                              collection_name: str,
                              query_text: str,
                              location_constraint: Optional[Tuple[float, float, float]] = None,
                              n_results: int = 5) -> Dict[str, Any]:
        """
        Perform semantic search with optional spatial constraints.
        
        Args:
            collection_name: Name of the collection to query
            query_text: Query text
            location_constraint: Optional (lat, lon, radius_km) constraint
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        if location_constraint:
            # Use range query with text filtering
            center_lat, center_lon, radius_km = location_constraint
            return self.range_query(
                collection_name,
                (center_lat, center_lon),
                radius_km,
                query_text
            )
        else:
            # Use standard semantic search
            return self.spatial_query(
                collection_name,
                query_text,
                (0, 0),  # Dummy coordinates
                n_results
            )
    
    def get_collection_statistics(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        if collection_name not in self.collections:
            return {}
        
        collection = self.collections[collection_name]
        all_results = collection.get(include=["metadatas"])
        
        # Calculate geographic statistics
        latitudes = []
        longitudes = []
        
        for metadata in all_results["metadatas"]:
            lat = metadata.get("latitude")
            lon = metadata.get("longitude")
            if lat is not None and lon is not None:
                latitudes.append(lat)
                longitudes.append(lon)
        
        stats = {
            "total_documents": len(all_results["ids"]),
            "documents_with_coordinates": len(latitudes),
            "geographic_bounds": {
                "min_lat": min(latitudes) if latitudes else None,
                "max_lat": max(latitudes) if latitudes else None,
                "min_lon": min(longitudes) if longitudes else None,
                "max_lon": max(longitudes) if longitudes else None
            },
            "coordinate_distribution": {
                "lat_mean": np.mean(latitudes) if latitudes else None,
                "lat_std": np.std(latitudes) if latitudes else None,
                "lon_mean": np.mean(longitudes) if longitudes else None,
                "lon_std": np.std(longitudes) if longitudes else None
            }
        }
        
        return stats
    
    def export_embeddings(self, collection_name: str, 
                         filepath: str) -> None:
        """
        Export embeddings from a collection to a file.
        
        Args:
            collection_name: Name of the collection
            filepath: Path to save the embeddings
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        all_results = collection.get(include=["embeddings", "documents", "metadatas"])
        
        # Save as numpy array
        embeddings = np.array(all_results["embeddings"])
        np.save(filepath, embeddings)
        
        logger.info(f"Exported {len(embeddings)} embeddings to {filepath}")
    
    def import_embeddings(self, collection_name: str,
                         embeddings_file: str,
                         documents: List[str],
                         coordinates: List[Tuple[float, float]],
                         metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Import pre-computed embeddings into a collection.
        
        Args:
            collection_name: Name of the collection
            embeddings_file: Path to the embeddings file
            documents: List of document texts
            coordinates: List of (lat, lon) coordinates
            metadata: Optional list of metadata dictionaries
        """
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        # Create collection if it doesn't exist
        if collection_name not in self.collections:
            self.create_geographic_collection(collection_name)
        
        collection = self.collections[collection_name]
        
        # Prepare metadata
        if metadata is None:
            metadata = []
            for i, (lat, lon) in enumerate(coordinates):
                metadata.append({
                    "latitude": lat,
                    "longitude": lon,
                    "coordinates_formatted": format_coordinates(lat, lon),
                    "document_index": i
                })
        
        # Generate IDs
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Imported {len(embeddings)} embeddings to collection {collection_name}")