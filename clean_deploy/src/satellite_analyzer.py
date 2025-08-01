"""
Satellite imagery analysis and processing.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import format_coordinates

logger = logging.getLogger(__name__)

class SatelliteAnalyzer:
    """
    Analyze satellite imagery for feature extraction and land cover classification.
    """
    
    def __init__(self):
        """Initialize the satellite analyzer."""
        self.satellite_data = {}
        self.analysis_results = {}
        
    def load_satellite_image(self, filepath: str, 
                           image_name: str = "default") -> Dict[str, Any]:
        """
        Load satellite imagery from various formats.
        
        Args:
            filepath: Path to the satellite image file
            image_name: Name for the loaded image
            
        Returns:
            Dictionary with image data and metadata
        """
        try:
            if filepath.endswith(('.tif', '.tiff')):
                # Load GeoTIFF
                with rasterio.open(filepath) as src:
                    image_data = src.read()
                    metadata = src.meta
                    bounds = src.bounds
                    crs = src.crs
                    
                    satellite_data = {
                        'data': image_data,
                        'metadata': metadata,
                        'bounds': bounds,
                        'crs': crs,
                        'filepath': filepath,
                        'name': image_name
                    }
                    
            elif filepath.endswith(('.jpg', '.jpeg', '.png')):
                # Load regular image
                image = Image.open(filepath)
                image_array = np.array(image)
                
                satellite_data = {
                    'data': image_array,
                    'metadata': {
                        'width': image.width,
                        'height': image.height,
                        'mode': image.mode
                    },
                    'filepath': filepath,
                    'name': image_name
                }
                
            else:
                raise ValueError(f"Unsupported image format: {filepath}")
            
            self.satellite_data[image_name] = satellite_data
            logger.info(f"Loaded satellite image: {image_name}")
            
            return satellite_data
            
        except Exception as e:
            logger.error(f"Error loading satellite image from {filepath}: {e}")
            raise
    
    def calculate_ndvi(self, red_band: np.ndarray, 
                      nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Args:
            red_band: Red band data
            nir_band: Near-infrared band data
            
        Returns:
            NDVI array
        """
        # Ensure data types are float
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Avoid division by zero
        denominator = red + nir
        denominator[denominator == 0] = 1
        
        ndvi = (nir - red) / denominator
        return ndvi
    
    def calculate_vegetation_indices(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate various vegetation indices from multispectral data.
        
        Args:
            image_data: Multispectral image data (bands, height, width)
            
        Returns:
            Dictionary with various vegetation indices
        """
        indices = {}
        
        if image_data.shape[0] >= 4:  # At least 4 bands (RGB + NIR)
            red = image_data[2]  # Red band (typically band 3)
            nir = image_data[3]  # NIR band (typically band 4)
            
            # NDVI
            indices['ndvi'] = self.calculate_ndvi(red, nir)
            
            # EVI (Enhanced Vegetation Index)
            blue = image_data[0]  # Blue band
            denominator = red + 6 * blue - 7.5
            denominator[denominator == 0] = 1
            indices['evi'] = 2.5 * (nir - red) / denominator
            
            # SAVI (Soil-Adjusted Vegetation Index)
            L = 0.5  # Soil brightness correction factor
            denominator = red + nir + L
            denominator[denominator == 0] = 1
            indices['savi'] = 1.5 * (nir - red) / denominator
        
        return indices
    
    def extract_features(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from satellite imagery.
        
        Args:
            image_data: Satellite image data
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        # Convert to grayscale if needed
        if len(image_data.shape) == 3 and image_data.shape[0] > 1:
            # Multispectral image - use first band for basic features
            gray = image_data[0]
        else:
            gray = image_data
        
        # Edge detection
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        features['edges'] = edges
        
        # Feature detection using SIFT
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray.astype(np.uint8), None)
        features['keypoints'] = keypoints
        features['descriptors'] = descriptors
        
        # Texture analysis using GLCM (simplified)
        features['texture_stats'] = self._calculate_texture_statistics(gray)
        
        # Color statistics
        if len(image_data.shape) == 3:
            features['color_stats'] = self._calculate_color_statistics(image_data)
        
        return features
    
    def _calculate_texture_statistics(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Calculate texture statistics from grayscale image."""
        # Simple texture measures
        stats = {
            'mean': np.mean(gray_image),
            'std': np.std(gray_image),
            'variance': np.var(gray_image),
            'skewness': self._calculate_skewness(gray_image),
            'kurtosis': self._calculate_kurtosis(gray_image)
        }
        return stats
    
    def _calculate_color_statistics(self, image_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate color statistics for each band."""
        color_stats = {}
        
        for i in range(min(image_data.shape[0], 4)):  # Limit to first 4 bands
            band = image_data[i]
            color_stats[f'band_{i}'] = {
                'mean': np.mean(band),
                'std': np.std(band),
                'min': np.min(band),
                'max': np.max(band),
                'median': np.median(band)
            }
        
        return color_stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def classify_land_cover(self, image_data: np.ndarray, 
                          method: str = "ndvi_threshold") -> np.ndarray:
        """
        Classify land cover from satellite imagery.
        
        Args:
            image_data: Satellite image data
            method: Classification method
            
        Returns:
            Classification map
        """
        if method == "ndvi_threshold":
            return self._ndvi_threshold_classification(image_data)
        elif method == "kmeans":
            return self._kmeans_classification(image_data)
        else:
            raise ValueError(f"Unsupported classification method: {method}")
    
    def _ndvi_threshold_classification(self, image_data: np.ndarray) -> np.ndarray:
        """Classify land cover using NDVI thresholds."""
        if image_data.shape[0] < 4:
            raise ValueError("Need at least 4 bands for NDVI classification")
        
        # Calculate NDVI
        red = image_data[2]
        nir = image_data[3]
        ndvi = self.calculate_ndvi(red, nir)
        
        # Simple threshold-based classification
        classification = np.zeros_like(ndvi, dtype=np.uint8)
        
        # Water
        classification[ndvi < -0.1] = 1
        # Built-up areas
        classification[(ndvi >= -0.1) & (ndvi < 0.1)] = 2
        # Vegetation
        classification[ndvi >= 0.1] = 3
        
        return classification
    
    def _kmeans_classification(self, image_data: np.ndarray, 
                             n_clusters: int = 5) -> np.ndarray:
        """Classify land cover using K-means clustering."""
        from sklearn.cluster import KMeans
        
        # Reshape data for clustering
        if len(image_data.shape) == 3:
            # Multispectral image
            height, width, bands = image_data.shape
            reshaped_data = image_data.reshape(height * width, bands)
        else:
            # Single band image
            height, width = image_data.shape
            reshaped_data = image_data.reshape(height * width, 1)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped_data)
        
        # Reshape back to original dimensions
        classification = labels.reshape(height, width)
        
        return classification
    
    def detect_changes(self, image1: np.ndarray, 
                      image2: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect changes between two satellite images.
        
        Args:
            image1: First satellite image
            image2: Second satellite image
            
        Returns:
            Dictionary with change detection results
        """
        # Ensure images have same dimensions
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Calculate difference
        difference = np.abs(image2.astype(np.float32) - image1.astype(np.float32))
        
        # Calculate change magnitude
        change_magnitude = np.mean(difference, axis=0) if len(difference.shape) == 3 else difference
        
        # Threshold for significant changes
        threshold = np.percentile(change_magnitude, 95)
        change_mask = change_magnitude > threshold
        
        return {
            'difference': difference,
            'change_magnitude': change_magnitude,
            'change_mask': change_mask,
            'threshold': threshold
        }
    
    def segment_image(self, image_data: np.ndarray, 
                     method: str = "watershed") -> np.ndarray:
        """
        Segment satellite image into regions.
        
        Args:
            image_data: Satellite image data
            method: Segmentation method
            
        Returns:
            Segmentation map
        """
        if method == "watershed":
            return self._watershed_segmentation(image_data)
        elif method == "slic":
            return self._slic_segmentation(image_data)
        else:
            raise ValueError(f"Unsupported segmentation method: {method}")
    
    def _watershed_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Perform watershed segmentation."""
        # Convert to grayscale if needed
        if len(image_data.shape) == 3:
            gray = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray = image_data
        
        # Apply watershed segmentation
        # This is a simplified version
        _, markers = cv2.connectedComponents(gray.astype(np.uint8))
        
        return markers
    
    def _slic_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Perform SLIC superpixel segmentation."""
        from skimage.segmentation import slic
        
        # Ensure image is in correct format
        if len(image_data.shape) == 3:
            # Multispectral image
            image_rgb = image_data.transpose(1, 2, 0)
        else:
            # Single band image
            image_rgb = np.stack([image_data] * 3, axis=-1)
        
        # Normalize to 0-1 range
        image_normalized = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())
        
        # Apply SLIC segmentation
        segments = slic(image_normalized, n_segments=100, compactness=10)
        
        return segments
    
    def generate_analysis_report(self, image_name: str) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report for a satellite image.
        
        Args:
            image_name: Name of the satellite image to analyze
            
        Returns:
            Dictionary with analysis report
        """
        if image_name not in self.satellite_data:
            raise ValueError(f"Image {image_name} not found")
        
        satellite_data = self.satellite_data[image_name]
        image_data = satellite_data['data']
        
        report = {
            'image_name': image_name,
            'metadata': satellite_data.get('metadata', {}),
            'dimensions': image_data.shape,
            'data_type': str(image_data.dtype),
            'statistics': {}
        }
        
        # Basic statistics
        report['statistics']['basic'] = {
            'min': np.min(image_data),
            'max': np.max(image_data),
            'mean': np.mean(image_data),
            'std': np.std(image_data)
        }
        
        # Vegetation indices (if multispectral)
        if len(image_data.shape) == 3 and image_data.shape[0] >= 4:
            indices = self.calculate_vegetation_indices(image_data)
            report['vegetation_indices'] = {
                name: {
                    'mean': np.mean(index),
                    'std': np.std(index),
                    'min': np.min(index),
                    'max': np.max(index)
                }
                for name, index in indices.items()
            }
        
        # Feature extraction
        features = self.extract_features(image_data)
        report['features'] = {
            'n_keypoints': len(features.get('keypoints', [])),
            'texture_stats': features.get('texture_stats', {}),
            'color_stats': features.get('color_stats', {})
        }
        
        # Land cover classification
        try:
            classification = self.classify_land_cover(image_data)
            unique_classes, counts = np.unique(classification, return_counts=True)
            report['land_cover'] = {
                'classification_map_shape': classification.shape,
                'class_distribution': dict(zip(unique_classes, counts))
            }
        except Exception as e:
            report['land_cover'] = {'error': str(e)}
        
        return report
    
    def visualize_results(self, image_name: str, 
                        results: Dict[str, Any],
                        save_path: Optional[str] = None) -> None:
        """
        Visualize satellite image analysis results.
        
        Args:
            image_name: Name of the satellite image
            results: Analysis results to visualize
            save_path: Path to save visualization (optional)
        """
        if image_name not in self.satellite_data:
            raise ValueError(f"Image {image_name} not found")
        
        satellite_data = self.satellite_data[image_name]
        image_data = satellite_data['data']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        if len(image_data.shape) == 3:
            # Multispectral image - show RGB composite
            rgb_image = image_data[:3].transpose(1, 2, 0)
            rgb_normalized = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            axes[0, 0].imshow(rgb_normalized)
            axes[0, 0].set_title('Original Image (RGB Composite)')
        else:
            axes[0, 0].imshow(image_data, cmap='gray')
            axes[0, 0].set_title('Original Image')
        
        # NDVI (if available)
        if 'vegetation_indices' in results:
            ndvi = self.calculate_vegetation_indices(image_data)['ndvi']
            im = axes[0, 1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0, 1].set_title('NDVI')
            plt.colorbar(im, ax=axes[0, 1])
        
        # Land cover classification
        if 'land_cover' in results and 'classification_map_shape' in results['land_cover']:
            classification = self.classify_land_cover(image_data)
            axes[1, 0].imshow(classification, cmap='tab10')
            axes[1, 0].set_title('Land Cover Classification')
        
        # Feature extraction results
        if 'features' in results:
            # Show edge detection
            features = self.extract_features(image_data)
            if 'edges' in features:
                axes[1, 1].imshow(features['edges'], cmap='gray')
                axes[1, 1].set_title('Edge Detection')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show() 