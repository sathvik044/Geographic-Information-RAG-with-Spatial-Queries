"""
Utility functions for the Geographic Information RAG system.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_sample_geographic_data() -> Dict[str, Any]:
    """Create sample geographic data for testing."""
    sample_data = {
        "cities": [
            {
                "name": "New York",
                "coordinates": [-74.006, 40.7128],
                "population": 8336817,
                "area_km2": 778.2,
                "description": "Largest city in the United States, major financial center"
            },
            {
                "name": "London",
                "coordinates": [-0.1276, 51.5074],
                "population": 8982000,
                "area_km2": 1572,
                "description": "Capital of England, major global city"
            },
            {
                "name": "Tokyo",
                "coordinates": [139.6917, 35.6895],
                "population": 13929286,
                "area_km2": 2194,
                "description": "Capital of Japan, largest metropolitan area in the world"
            }
        ],
        "landmarks": [
            {
                "name": "Eiffel Tower",
                "coordinates": [2.2945, 48.8584],
                "type": "monument",
                "description": "Iconic iron lattice tower in Paris, France"
            },
            {
                "name": "Statue of Liberty",
                "coordinates": [-74.0445, 40.6892],
                "type": "monument",
                "description": "Famous statue in New York Harbor"
            }
        ],
        "natural_features": [
            {
                "name": "Mount Everest",
                "coordinates": [86.9250, 27.9881],
                "type": "mountain",
                "elevation_m": 8848,
                "description": "Highest peak in the world"
            },
            {
                "name": "Amazon River",
                "coordinates": [-58.3816, -3.4653],
                "type": "river",
                "length_km": 6575,
                "description": "Largest river by discharge volume"
            }
        ]
    }
    return sample_data

def coordinates_to_point(lat: float, lon: float) -> Point:
    """Convert latitude and longitude to a Shapely Point."""
    return Point(lon, lat)

def create_bounding_box(center_lat: float, center_lon: float, radius_km: float = 10) -> Polygon:
    """Create a bounding box around a center point."""
    # Approximate conversion: 1 degree ≈ 111 km
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    min_lat, max_lat = center_lat - lat_offset, center_lat + lat_offset
    min_lon, max_lon = center_lon - lon_offset, center_lon + lon_offset
    
    return Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat)
    ])

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate distance between two points using Haversine formula."""
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_lat / 2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def format_coordinates(lat: float, lon: float) -> str:
    """Format coordinates in a readable format."""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate that coordinates are within valid ranges."""
    return -90 <= lat <= 90 and -180 <= lon <= 180

def create_geodataframe_from_dict(data: Dict[str, List[Dict]]) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from a dictionary of geographic data."""
    all_features = []
    
    for category, features in data.items():
        for feature in features:
            coords = feature.get('coordinates', [0, 0])
            point = Point(coords[0], coords[1])  # lon, lat
            
            feature_dict = {
                'geometry': point,
                'category': category,
                **{k: v for k, v in feature.items() if k != 'coordinates'}
            }
            all_features.append(feature_dict)
    
    return gpd.GeoDataFrame(all_features, crs="EPSG:4326")

def save_sample_data(data: Dict[str, Any], filename: str = "sample_geographic_data.json") -> None:
    """Save sample data to a JSON file."""
    data_dir = "data/sample_data"
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Sample data saved to {filepath}")

def load_sample_data(filename: str = "sample_geographic_data.json") -> Dict[str, Any]:
    """Load sample data from a JSON file."""
    filepath = os.path.join("data/sample_data", filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"Sample data file not found: {filepath}")
        return create_sample_geographic_data()
    
    with open(filepath, 'r') as f:
        return json.load(f)

def create_spatial_query_examples() -> List[Dict[str, str]]:
    """Create example spatial queries for the system."""
    return [
        {
            "query": "What cities are within 100km of New York?",
            "type": "range_query",
            "description": "Find cities within a specific radius"
        },
        {
            "query": "What landmarks are near the Eiffel Tower?",
            "type": "proximity_query",
            "description": "Find nearby landmarks"
        },
        {
            "query": "What is the highest mountain in Asia?",
            "type": "attribute_query",
            "description": "Query based on geographic attributes"
        },
        {
            "query": "Show me all rivers longer than 5000km",
            "type": "filter_query",
            "description": "Filter by geographic properties"
        },
        {
            "query": "What's the population density around Tokyo?",
            "type": "analysis_query",
            "description": "Spatial analysis query"
        }
    ]

def get_satellite_image_info() -> Dict[str, Any]:
    """Get information about available satellite imagery."""
    return {
        "sources": [
            {
                "name": "Landsat 8",
                "resolution": "30m",
                "bands": ["Red", "Green", "Blue", "NIR", "SWIR"],
                "description": "Multispectral satellite imagery"
            },
            {
                "name": "Sentinel-2",
                "resolution": "10m",
                "bands": ["Red", "Green", "Blue", "NIR", "SWIR"],
                "description": "High-resolution multispectral imagery"
            },
            {
                "name": "Planet Labs",
                "resolution": "3-5m",
                "bands": ["Red", "Green", "Blue", "NIR"],
                "description": "High-resolution commercial imagery"
            }
        ],
        "analysis_types": [
            "NDVI (Vegetation Index)",
            "Land Cover Classification",
            "Change Detection",
            "Feature Extraction",
            "Urban Development Analysis"
        ]
    } 