"""
Geographic data processing and spatial operations.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
import pyproj
from pyproj import CRS, Transformer

from .utils import (
    coordinates_to_point, 
    create_bounding_box, 
    calculate_distance,
    validate_coordinates,
    format_coordinates
)

logger = logging.getLogger(__name__)

class GeographicProcessor:
    """
    Handles geographic data processing, spatial operations, and coordinate transformations.
    """
    
    def __init__(self, default_crs: str = "EPSG:4326"):
        """
        Initialize the geographic processor.
        
        Args:
            default_crs: Default coordinate reference system (WGS84)
        """
        self.default_crs = CRS(default_crs)
        self.data_sources = {}
        self.spatial_index = None
        
    def create_geodataframe_from_dict(self, data_dict: Dict[str, Any]) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame from a dictionary containing geographic data.
        
        Args:
            data_dict: Dictionary containing geographic data with categories and features
            
        Returns:
            GeoDataFrame with the geographic data
        """
        features = []
        
        for category, feature_list in data_dict.items():
            if not isinstance(feature_list, list):
                logger.warning(f"Category {category} does not contain a list of features")
                continue
                
            for feature_data in feature_list:
                # Extract coordinates
                coords = feature_data.get('coordinates', [])
                if not coords or len(coords) != 2:
                    logger.warning(f"Invalid coordinates for feature: {coords}")
                    continue
                    
                lon, lat = coords
                
                # Validate coordinates
                if not validate_coordinates(lat, lon):
                    logger.warning(f"Invalid coordinates: {lon}, {lat}")
                    continue
                
                # Create geometry
                geometry = Point(lon, lat)
                
                # Prepare attributes
                attributes = {
                    'name': feature_data.get('name', 'Unknown'),
                    'description': feature_data.get('description', ''),
                    'type': feature_data.get('type', category),
                    'category': category,
                    'latitude': lat,
                    'longitude': lon
                }
                
                # Add any additional attributes
                for key, value in feature_data.items():
                    if key not in ['coordinates', 'name', 'description', 'type']:
                        attributes[key] = value
                
                features.append({
                    'geometry': geometry,
                    **attributes
                })
        
        if not features:
            logger.warning("No valid features found in data dictionary")
            return gpd.GeoDataFrame(crs=self.default_crs)
        
        gdf = gpd.GeoDataFrame(features, crs=self.default_crs)
        logger.info(f"Created GeoDataFrame with {len(gdf)} features")
        return gdf

    def load_geographic_data(self, filepath: str, layer_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load geographic data from various file formats.
        
        Args:
            filepath: Path to the geographic data file
            layer_name: Layer name for multi-layer files (e.g., GeoPackage)
            
        Returns:
            GeoDataFrame containing the geographic data
        """
        try:
            if filepath.endswith('.geojson'):
                gdf = gpd.read_file(filepath)
            elif filepath.endswith('.shp'):
                gdf = gpd.read_file(filepath)
            elif filepath.endswith('.gpkg'):
                if layer_name:
                    gdf = gpd.read_file(filepath, layer=layer_name)
                else:
                    gdf = gpd.read_file(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf.set_crs(self.default_crs, inplace=True)
            
            logger.info(f"Loaded {len(gdf)} features from {filepath}")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading geographic data from {filepath}: {e}")
            raise
    
    def create_point_data(self, coordinates: List[Tuple[float, float]], 
                         attributes: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame from point coordinates and attributes.
        
        Args:
            coordinates: List of (lon, lat) coordinates
            attributes: List of attribute dictionaries
            
        Returns:
            GeoDataFrame with point geometries
        """
        geometries = [Point(lon, lat) for lon, lat in coordinates]
        
        # Validate coordinates
        for i, (lon, lat) in enumerate(coordinates):
            if not validate_coordinates(lat, lon):
                logger.warning(f"Invalid coordinates at index {i}: {lon}, {lat}")
        
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=self.default_crs)
        return gdf
    
    def spatial_query(self, gdf: gpd.GeoDataFrame, 
                     query_geometry: Union[Point, Polygon], 
                     query_type: str = "intersects") -> gpd.GeoDataFrame:
        """
        Perform spatial queries on a GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            query_geometry: Geometry to query against
            query_type: Type of spatial relationship ('intersects', 'contains', 'within', 'near')
            
        Returns:
            Filtered GeoDataFrame
        """
        if query_type == "intersects":
            result = gdf[gdf.geometry.intersects(query_geometry)]
        elif query_type == "contains":
            result = gdf[gdf.geometry.contains(query_geometry)]
        elif query_type == "within":
            result = gdf[gdf.geometry.within(query_geometry)]
        elif query_type == "near":
            # Find features within a buffer distance
            buffer_distance = 0.01  # ~1km in degrees
            buffered_geometry = query_geometry.buffer(buffer_distance)
            result = gdf[gdf.geometry.intersects(buffered_geometry)]
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        return result
    
    def buffer_analysis(self, gdf: gpd.GeoDataFrame, 
                       distance: float, 
                       unit: str = "degrees") -> gpd.GeoDataFrame:
        """
        Create buffer zones around geometries.
        
        Args:
            gdf: Input GeoDataFrame
            distance: Buffer distance
            unit: Unit of distance ('degrees', 'meters', 'kilometers')
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        if unit == "kilometers":
            # Convert km to degrees (approximate)
            distance = distance / 111.0
        elif unit == "meters":
            # Convert meters to degrees (approximate)
            distance = distance / 111000.0
        
        buffered_gdf = gdf.copy()
        buffered_gdf.geometry = gdf.geometry.buffer(distance)
        
        return buffered_gdf
    
    def calculate_spatial_statistics(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Calculate spatial statistics for a GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            Dictionary of spatial statistics
        """
        stats = {
            "total_features": len(gdf),
            "geometry_types": gdf.geometry.geom_type.value_counts().to_dict(),
            "bounds": gdf.total_bounds.tolist(),
            "area_km2": None,
            "perimeter_km": None
        }
        
        # Calculate area and perimeter if geometries are polygons
        if len(gdf) > 0 and gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            # Convert to a projected CRS for accurate area calculation
            projected_gdf = gdf.to_crs("EPSG:3857")  # Web Mercator
            stats["area_km2"] = projected_gdf.geometry.area.sum() / 1e6  # Convert to km²
            stats["perimeter_km"] = projected_gdf.geometry.length.sum() / 1000  # Convert to km
        
        return stats
    
    def find_nearest_features(self, gdf: gpd.GeoDataFrame, 
                            query_point: Point, 
                            n_neighbors: int = 5) -> gpd.GeoDataFrame:
        """
        Find the nearest features to a query point.
        
        Args:
            gdf: Input GeoDataFrame
            query_point: Point to find nearest features to
            n_neighbors: Number of nearest neighbors to return
            
        Returns:
            GeoDataFrame with nearest features
        """
        # Calculate distances to query point
        distances = gdf.geometry.distance(query_point)
        
        # Sort by distance and get top n_neighbors
        nearest_indices = distances.nsmallest(n_neighbors).index
        nearest_features = gdf.loc[nearest_indices].copy()
        
        # Add distance column
        nearest_features['distance_to_query'] = distances[nearest_indices]
        
        return nearest_features
    
    def spatial_join(self, left_gdf: gpd.GeoDataFrame, 
                    right_gdf: gpd.GeoDataFrame, 
                    how: str = "left") -> gpd.GeoDataFrame:
        """
        Perform spatial join between two GeoDataFrames.
        
        Args:
            left_gdf: Left GeoDataFrame
            right_gdf: Right GeoDataFrame
            how: Join type ('left', 'right', 'inner', 'outer')
            
        Returns:
            Joined GeoDataFrame
        """
        return gpd.sjoin(left_gdf, right_gdf, how=how, predicate='intersects')
    
    def reproject_data(self, gdf: gpd.GeoDataFrame, 
                      target_crs: str) -> gpd.GeoDataFrame:
        """
        Reproject data to a different coordinate reference system.
        
        Args:
            gdf: Input GeoDataFrame
            target_crs: Target coordinate reference system
            
        Returns:
            Reprojected GeoDataFrame
        """
        return gdf.to_crs(target_crs)
    
    def create_grid(self, bounds: Tuple[float, float, float, float], 
                   cell_size: float) -> gpd.GeoDataFrame:
        """
        Create a regular grid within specified bounds.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y)
            cell_size: Size of grid cells in degrees
            
        Returns:
            GeoDataFrame with grid polygons
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Create grid coordinates
        x_coords = np.arange(min_x, max_x + cell_size, cell_size)
        y_coords = np.arange(min_y, max_y + cell_size, cell_size)
        
        grid_polygons = []
        for i in range(len(x_coords) - 1):
            for j in range(len(y_coords) - 1):
                polygon = Polygon([
                    (x_coords[i], y_coords[j]),
                    (x_coords[i + 1], y_coords[j]),
                    (x_coords[i + 1], y_coords[j + 1]),
                    (x_coords[i], y_coords[j + 1]),
                    (x_coords[i], y_coords[j])
                ])
                grid_polygons.append(polygon)
        
        grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=self.default_crs)
        return grid_gdf
    
    def calculate_centroids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate centroids for all geometries.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with centroid points
        """
        centroids = gdf.geometry.centroid
        centroid_gdf = gpd.GeoDataFrame(
            gdf.drop(columns=['geometry']),
            geometry=centroids,
            crs=gdf.crs
        )
        return centroid_gdf
    
    def dissolve_features(self, gdf: gpd.GeoDataFrame, 
                        by: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Dissolve features based on an attribute or merge all features.
        
        Args:
            gdf: Input GeoDataFrame
            by: Column name to dissolve by (None for all features)
            
        Returns:
            Dissolved GeoDataFrame
        """
        if by:
            dissolved = gdf.dissolve(by=by)
        else:
            dissolved = gdf.dissolve()
        
        return dissolved
    
    def clip_data(self, gdf: gpd.GeoDataFrame, 
                  clip_geometry: Union[Polygon, MultiPolygon]) -> gpd.GeoDataFrame:
        """
        Clip data to a specific geometry.
        
        Args:
            gdf: Input GeoDataFrame
            clip_geometry: Geometry to clip to
            
        Returns:
            Clipped GeoDataFrame
        """
        clipped = gpd.clip(gdf, clip_geometry)
        return clipped
    
    def validate_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and fix geometries in a GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with valid geometries
        """
        # Remove invalid geometries
        valid_mask = gdf.geometry.is_valid
        valid_gdf = gdf[valid_mask].copy()
        
        # Fix simple geometry issues
        valid_gdf.geometry = valid_gdf.geometry.buffer(0)
        
        logger.info(f"Removed {len(gdf) - len(valid_gdf)} invalid geometries")
        return valid_gdf
    
    def export_data(self, gdf: gpd.GeoDataFrame, 
                   filepath: str, 
                   driver: str = "GeoJSON") -> None:
        """
        Export GeoDataFrame to various formats.
        
        Args:
            gdf: GeoDataFrame to export
            filepath: Output file path
            driver: Output format driver
        """
        try:
            gdf.to_file(filepath, driver=driver)
            logger.info(f"Exported data to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting data to {filepath}: {e}")
            raise 