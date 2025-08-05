# Geographic Information RAG with Spatial Queries

A comprehensive RAG (Retrieval-Augmented Generation) system that combines geographic data, satellite imagery, and location-based information to answer spatial queries and provide location-specific insights.

## 🌍 Features

- **Geographic Data Processing**: Handle various geographic data formats (GeoJSON, Shapefiles, etc.)
- **Spatial Indexing**: Efficient spatial queries using R-tree indexing
- **Satellite Imagery Analysis**: Process and analyze satellite images for feature extraction
- **Location-based Information Retrieval**: Retrieve relevant geographic information
- **Spatial Query Processing**: Advanced spatial relationship modeling and querying
- **Multi-scale Analysis**: Support for different geographic scales and resolutions

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9** (specified in runtime.txt)
2. **Git** for cloning the repository

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd geograph
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

## 🌐 Deployment to Streamlit Cloud

### Option 1: Direct Deployment (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy

### Troubleshooting Deployment Issues

If you encounter issues with the deployment:

1. **Check the logs** in the Streamlit Cloud dashboard
2. **Verify system dependencies** in `packages.txt` are correct
3. **Ensure Python version** is set to 3.9 in `runtime.txt`
4. **Try the fallback app** by changing the main file to `app-fallback.py`

### Memory Optimization

The application has been optimized for cloud deployment with:

- Memory-efficient spatial indexing
- In-memory database configuration
- Smaller embedding models
- Fallback mode for limited environments

**Option 3: Using Python module**
```bash
python -m streamlit run app.py --server.port 8501
```

The application will be available at `http://localhost:8501`

### Testing the System

Run the test suite to verify everything is working:
```bash
python test_system.py
```

## 📁 Project Structure

```
geograph/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── data/                          # Data directory
│   ├── geographic_data/           # Geographic datasets
│   ├── satellite_images/          # Satellite imagery
│   └── sample_data/              # Sample data for testing
├── src/                           # Source code
│   ├── __init__.py
│   ├── geographic_processor.py    # Geographic data processing
│   ├── spatial_indexer.py        # Spatial indexing and queries
│   ├── satellite_analyzer.py     # Satellite imagery analysis
│   ├── rag_engine.py             # Main RAG engine
│   ├── embeddings.py             # Embedding models
│   └── utils.py                  # Utility functions
├── notebooks/                     # Jupyter notebooks for exploration
└── tests/                        # Unit tests
```

## 🔧 Technical Architecture

### Core Components

1. **Geographic Processor**: Handles geographic data loading, preprocessing, and spatial operations
2. **Spatial Indexer**: Implements R-tree spatial indexing for efficient queries
3. **Satellite Analyzer**: Processes satellite imagery using computer vision
4. **RAG Engine**: Combines retrieval and generation for spatial queries
5. **Embeddings**: Geographic-aware embedding models

### Key Technologies

- **Geospatial**: GeoPandas, Shapely, PyProj
- **Spatial Indexing**: Rtree, SpatialIndex
- **Satellite Imagery**: Rasterio, EarthPy, OpenCV
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers, OpenAI
- **UI**: Streamlit
- **Data Processing**: Pandas, NumPy

## 📊 Capabilities

### Spatial Queries Supported

- **Point Queries**: Find information about specific coordinates
- **Range Queries**: Search within geographic boundaries
- **Spatial Relationships**: Find features based on spatial relationships
- **Multi-scale Analysis**: Analyze data at different geographic scales
- **Temporal Analysis**: Time-series geographic data analysis

### Satellite Imagery Features

- **Feature Extraction**: Extract buildings, roads, water bodies
- **Land Cover Classification**: Classify different land types
- **Change Detection**: Detect changes over time
- **NDVI Analysis**: Vegetation index analysis

## 🎯 Use Cases

1. **Urban Planning**: Analyze urban development patterns
2. **Environmental Monitoring**: Track environmental changes
3. **Disaster Response**: Assess damage and plan responses
4. **Infrastructure Planning**: Plan infrastructure development
5. **Agricultural Analysis**: Monitor crop health and land use

## 📈 Performance Metrics

- **Retrieval Accuracy**: Precision and recall for spatial queries
- **Query Latency**: Response time for spatial operations
- **Spatial Index Performance**: R-tree query efficiency
- **Image Processing Speed**: Satellite imagery analysis time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🎯 Demo & Deployment

### Live Demo
The Geographic Information RAG System is designed to be deployed on various platforms:

- **Streamlit Cloud**: Deploy directly from GitHub
- **Hugging Face Spaces**: For easy sharing and collaboration
- **Local Development**: Full development environment

### Deployment Options

#### 1. **Streamlit Cloud Deployment (Recommended - Free):**
   1. Push this repository to GitHub
   2. Go to [share.streamlit.io](https://share.streamlit.io)
   3. Sign in with your GitHub account
   4. Click "New app"
   5. Select your repository and set main file path to `app.py`
   6. Click "Deploy"

#### 2. **Hugging Face Spaces:**
   1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   2. Create a new Space
   3. Choose "Streamlit" as the SDK
   4. Upload your application files
   5. Configure the Space for Python

#### 3. **Local Development:**
   - Clone the repository
   - Install dependencies
   - Run with `python deploy.py`

### Performance Metrics

- **Query Response Time**: < 2 seconds for spatial queries
- **Spatial Index Performance**: R-tree query efficiency
- **Memory Usage**: Optimized for large geographic datasets
- **Scalability**: Supports multiple concurrent users

## 🔗 Links

- **GitHub Repository**: [Link to be added]
- **Deployed Application**: [Link to be added]
- **Documentation**: [Link to be added]

## 👥 Team

- Geographic Data Processing: [Your Name]
- Satellite Imagery Analysis: [Your Name]
- RAG Engine Development: [Your Name]
- UI/UX Design: [Your Name]

---

*Built with ❤️ for Geographic Information Systems and Spatial Analysis*