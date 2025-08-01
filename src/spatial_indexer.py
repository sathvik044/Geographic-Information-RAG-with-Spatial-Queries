"""
Spatial indexing and query processing for geographic data.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.prepared import prep
import rtree
from rtree import index

from .utils import calculate_distance, format_coordinates

logger = logging.getLogger(__name__)

class SpatialIndexer:
    """
    Efficient spatial indexing and query processing using R-tree indexing.
    """
    
    def __init__(self):
        """Initialize the spatial indexer."""
        self.spatial_index = None
        self.geodataframes = {}
        self.indexed_features = {}
        
    def build_spatial_index(self, gdf: gpd.GeoDataFrame, 
                           index_name: str = "default") -> None:
        """
        Build a spatial index for efficient querying.
        
        Args:
            gdf: GeoDataFrame to index
            index_name: Name for the spatial index
        """
        try:
            # Create R-tree spatial index
            idx = index.Index()
            
            # Add geometries to the index
            for i, geometry in enumerate(gdf.geometry):
                if geometry is not None and geometry.is_valid:
                    bounds = geometry.bounds
                    idx.insert(i, bounds)
            
            self.spatial_index = idx
            self.geodataframes[index_name] = gdf
            self.indexed_features[index_name] = list(range(len(gdf)))
            
            logger.info(f"Built spatial index for {len(gdf)} features")
            
        except Exception as e:
            logger.error(f"Error building spatial index: {e}")
            raise
    
    def spatial_query(self, query_geometry: Union[Point, Polygon], 
                     index_name: str = "default",
                     query_type: str = "intersects") -> gpd.GeoDataFrame:
        """
        Perform spatial queries using the spatial index.
        
        Args:
            query_geometry: Geometry to query against
            index_name: Name of the spatial index to use
            query_type: Type of spatial relationship
            
        Returns:
            GeoDataFrame with matching features
        """
        if self.spatial_index is None:
            raise ValueError("Spatial index not built. Call build_spatial_index first.")
        
        gdf = self.geodataframes[index_name]
        query_bounds = query_geometry.bounds
        
        # Get candidate features from spatial index
        candidate_indices = list(self.spatial_index.intersection(query_bounds))
        
        # Filter candidates based on actual spatial relationship
        matching_indices = []
        for idx in candidate_indices:
            geometry = gdf.geometry.iloc[idx]
            
            if query_type == "intersects":
                if geometry.intersects(query_geometry):
                    matching_indices.append(idx)
            elif query_type == "contains":
                if geometry.contains(query_geometry):
                    matching_indices.append(idx)
            elif query_type == "within":
                if geometry.within(query_geometry):
                    matching_indices.append(idx)
            elif query_type == "near":
                # Find features within a buffer distance
                buffer_distance = 0.01  # ~1km in degrees
                buffered_query = query_geometry.buffer(buffer_distance)
                if geometry.intersects(buffered_query):
                    matching_indices.append(idx)
        
        return gdf.iloc[matching_indices]
    
    def range_query(self, center_point: Point, 
                   radius_km: float,
                   index_name: str = "default") -> gpd.GeoDataFrame:
        """
        Find features within a specified radius of a point.
        
        Args:
            center_point: Center point for the range query
            radius_km: Search radius in kilometers
            index_name: Name of the spatial index to use
            
        Returns:
            GeoDataFrame with features within the range
        """
        # Convert radius to degrees (approximate)
        radius_degrees = radius_km / 111.0
        
        # Create buffer around center point
        buffer_geometry = center_point.buffer(radius_degrees)
        
        return self.spatial_query(buffer_geometry, index_name, "intersects")
    
    def nearest_neighbor_query(self, query_point: Point,
                             n_neighbors: int = 5,
                             index_name: str = "default") -> gpd.GeoDataFrame:
        """
        Find the nearest neighbors to a query point.
        
        Args:
            query_point: Point to find neighbors for
            n_neighbors: Number of nearest neighbors to return
            index_name: Name of the spatial index to use
            
        Returns:
            GeoDataFrame with nearest neighbors
        """
        gdf = self.geodataframes[index_name]
        
        # Calculate distances to all features
        distances = gdf.geometry.distance(query_point)
        
        # Get indices of nearest neighbors
        nearest_indices = distances.nsmallest(n_neighbors).index
        
        # Create result GeoDataFrame
        result = gdf.loc[nearest_indices].copy()
        result['distance_to_query'] = distances[nearest_indices]
        
        return result
    
    def bounding_box_query(self, bounds: Tuple[float, float, float, float],
                          index_name: str = "default") -> gpd.GeoDataFrame:
        """
        Find features within a bounding box.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) bounding box
            index_name: Name of the spatial index to use
            
        Returns:
            GeoDataFrame with features in the bounding box
        """
        min_x, min_y, max_x, max_y = bounds
        bbox = Polygon([
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y)
        ])
        
        return self.spatial_query(bbox, index_name, "intersects")
    
    def spatial_join_indexed(self, left_index_name: str,
                           right_index_name: str,
                           how: str = "left") -> gpd.GeoDataFrame:
        """
        Perform spatial join between two indexed datasets.
        
        Args:
            left_index_name: Name of left spatial index
            right_index_name: Name of right spatial index
            how: Join type ('left', 'right', 'inner', 'outer')
            
        Returns:
            Joined GeoDataFrame
        """
        left_gdf = self.geodataframes[left_index_name]
        right_gdf = self.geodataframes[right_index_name]
        
        return gpd.sjoin(left_gdf, right_gdf, how=how, predicate='intersects')
    
    def create_spatial_grid(self, bounds: Tuple[float, float, float, float],
                          cell_size: float) -> gpd.GeoDataFrame:
        """
        Create a spatial grid for analysis.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) bounds
            cell_size: Size of grid cells in degrees
            
        Returns:
            GeoDataFrame with grid cells
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
        
        grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs="EPSG:4326")
        return grid_gdf
    
    def point_in_polygon_query(self, points_gdf: gpd.GeoDataFrame,
                             polygons_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Find which polygon each point falls within.
        
        Args:
            points_gdf: GeoDataFrame with point geometries
            polygons_gdf: GeoDataFrame with polygon geometries
            
        Returns:
            GeoDataFrame with points and their containing polygons
        """
        return gpd.sjoin(points_gdf, polygons_gdf, how="left", predicate="within")
    
    def calculate_spatial_density(self, points_gdf: gpd.GeoDataFrame,
                                cell_size: float = 0.01) -> gpd.GeoDataFrame:
        """
        Calculate spatial density of points using a grid.
        
        Args:
            points_gdf: GeoDataFrame with point geometries
            cell_size: Size of grid cells in degrees
            
        Returns:
            GeoDataFrame with density values per grid cell
        """
        bounds = points_gdf.total_bounds
        grid_gdf = self.create_spatial_grid(bounds, cell_size)
        
        # Count points in each grid cell
        point_counts = gpd.sjoin(grid_gdf, points_gdf, how="left", predicate="contains")
        density_gdf = point_counts.groupby(point_counts.index).size().reset_index(name='point_count')
        
        # Merge back with grid geometry
        result = grid_gdf.merge(density_gdf, left_index=True, right_on='index')
        result['density'] = result['point_count'] / (cell_size ** 2)
        
        return result
    
    def find_spatial_clusters(self, points_gdf: gpd.GeoDataFrame,
                            distance_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Find spatial clusters in point data.
        
        Args:
            points_gdf: GeoDataFrame with point geometries
            distance_threshold: Distance threshold for clustering in degrees
            
        Returns:
            Dictionary with cluster information
        """
        from sklearn.cluster import DBSCAN
        
        # Extract coordinates
        coords = np.array([[point.x, point.y] for point in points_gdf.geometry])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=distance_threshold, min_samples=2).fit(coords)
        
        # Add cluster labels to GeoDataFrame
        clustered_gdf = points_gdf.copy()
        clustered_gdf['cluster_id'] = clustering.labels_
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Skip noise points
                cluster_points = clustered_gdf[clustered_gdf['cluster_id'] == cluster_id]
                cluster_stats[cluster_id] = {
                    'size': len(cluster_points),
                    'centroid': cluster_points.geometry.unary_union.centroid,
                    'bounds': cluster_points.total_bounds.tolist()
                }
        
        return {
            'clustered_data': clustered_gdf,
            'cluster_statistics': cluster_stats,
            'n_clusters': len([c for c in clustering.labels_ if c != -1]),
            'n_noise_points': list(clustering.labels_).count(-1)
        }
    
    def calculate_spatial_autocorrelation(self, gdf: gpd.GeoDataFrame,
                                       value_column: str) -> Dict[str, float]:
        """
        Calculate spatial autocorrelation (Moran's I) for a numeric column.
        
        Args:
            gdf: GeoDataFrame with data
            value_column: Name of the numeric column to analyze
            
        Returns:
            Dictionary with autocorrelation statistics
        """
        from libpysal.weights import W
        from libpysal.explore.esda.moran import Moran
        
        # Create spatial weights matrix
        weights = W.from_geodataframe(gdf, k=5)  # k-nearest neighbors
        
        # Calculate Moran's I
        moran = Moran(gdf[value_column], weights)
        
        return {
            'morans_i': moran.I,
            'expected_i': moran.EI,
            'variance': moran.VI_norm,
            'z_score': moran.z_norm,
            'p_value': moran.p_norm
        }
    
    def get_index_statistics(self, index_name: str = "default") -> Dict[str, Any]:
        """
        Get statistics about the spatial index.
        
        Args:
            index_name: Name of the spatial index
            
        Returns:
            Dictionary with index statistics
        """
        if index_name not in self.geodataframes:
            return {}
        
        gdf = self.geodataframes[index_name]
        
        stats = {
            'total_features': len(gdf),
            'geometry_types': gdf.geometry.geom_type.value_counts().to_dict(),
            'bounds': gdf.total_bounds.tolist(),
            'crs': str(gdf.crs),
            'indexed': index_name in self.indexed_features
        }
        
        return stats 