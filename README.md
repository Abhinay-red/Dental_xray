# Dental AI - Radiograph Analysis System

An AI-powered dental radiograph analysis system that uses EfficientNet deep learning model to analyze dental X-rays.

## Features

-  Dental radiograph upload and analysis
-  EfficientNet-based deep learning model
-  Automated report generation
-  Modern web interface

## Technology Stack

### Backend

- **Framework:** FastAPI
- **AI/ML:** PyTorch, TorchVision, EfficientNet
- **Image Processing:** OpenCV, Pillow
- **Report Generation:** ReportLab

### Frontend

- HTML/CSS/JavaScript (Single Page Application)

## Setup Instructions

### Prerequisites

- Python 3.8+

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Abhinay-red/Dental_xray
   cd dental-ai
   ```

2. **Set up the backend**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the backend server**

   ```bash
   python main.py
   ```

4. **Open the frontend**
   - Open `frontend/index.html` in your browser
   - Or serve it using a local web server

## Project Structure

```
dental-ai/
├── backend/
│   ├── main.py              # FastAPI server entry point
│   ├── local_analyzer.py    # EfficientNet model implementation
│   ├── processor.py         # Image processing utilities
│   ├── reporter.py          # Report generation
│   ├── train.py             # Model training script
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # Trained models directory
│   ├── uploads/             # Uploaded images (runtime)
│   └── results/             # Analysis results (runtime)
└── frontend/
    └── index.html           # Web interface
```

## Author

Abhinay Reddy
