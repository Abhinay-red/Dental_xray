# Dental AI - Radiograph Analysis System

An AI-powered dental radiograph analysis system that uses Google Gemini AI and local deep learning models to analyze dental X-rays.

## Features

- Dental radiograph upload and analysis
- Dual analysis modes:
  - Google Gemini AI API integration
  - Local EfficientNet-based model
- Automated report generation
- Modern web interface

## Technology Stack

### Backend

- **Framework:** FastAPI
- **AI/ML:** Google Gemini API, PyTorch, TorchVision
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
   git clone <repository-url>
   cd dental-ai
   ```

2. **Set up the backend**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Run the backend server**

   ```bash
   python main.py
   ```

5. **Open the frontend**
   - Open `frontend/index.html` in your browser
   - Or serve it using a local web server

## Project Structure

```
dental-ai/
├── backend/
│   ├── main.py              # FastAPI server entry point
│   ├── analyzer.py          # Gemini AI integration
│   ├── local_analyzer.py    # Local model implementation
│   ├── processor.py         # Image processing utilities
│   ├── reporter.py          # Report generation
│   ├── train.py             # Model training script
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Environment variables template
│   ├── models/              # Trained models directory
│   ├── uploads/             # Uploaded images
│   └── results/             # Analysis results
└── frontend/
    └── index.html           # Web interface

```
## Author

Abhinay Reddy
