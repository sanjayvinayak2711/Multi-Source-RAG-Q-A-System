# Scripts Directory

This directory contains utility scripts for the Multi-Source RAG System.

## Available Scripts

### `run.sh` (Linux/macOS)
Startup script for Unix-like systems.
- Creates and activates virtual environment
- Installs dependencies
- Sets up necessary directories
- Starts the RAG system

**Usage:**
```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

### `run.bat` (Windows)
Startup script for Windows systems.
- Creates and activates virtual environment
- Installs dependencies
- Sets up necessary directories
- Starts the RAG system

**Usage:**
```cmd
scripts\run.bat
```

### `setup.py` (Cross-platform)
Setup script for initial environment configuration.
- Checks Python version compatibility
- Creates virtual environment
- Installs dependencies (base and dev)
- Sets up environment file
- Creates necessary directories
- Runs tests (optional)
- Sets up Git hooks (optional)

**Usage:**
```bash
# Basic setup
python scripts/setup.py

# Setup with development dependencies
python scripts/setup.py --dev

# Setup with Git hooks
python scripts/setup.py --git-hooks

# Run tests after setup
python scripts/setup.py --test

# Custom virtual environment path
python scripts/setup.py --venv-path myenv

# Skip virtual environment creation
python scripts/setup.py --no-venv
```

## Environment Variables

The scripts respect the following environment variables:

- `ENV`: Environment mode (`development` or `production`)
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `PYTHONPATH`: Python path (automatically set)

## Directory Structure

After running the setup scripts, the following directories will be created:

```
Multi-Source-RAG-Q-A-System/
├── data/
│   ├── source_documents/    # Your PDF, TXT, DOCX files
│   └── vector_store/       # ChromaDB data
├── logs/                   # Application logs
├── uploads/                # Temporary uploads
├── temp/                   # Temporary files
├── models/
│   └── configs/            # Model configurations
├── prompts/
│   └── cache/              # Prompt cache
└── venv/                   # Virtual environment
```

## Configuration

### Environment File (.env)
The scripts will create `.env` from `.env.example` if it doesn't exist. Edit this file with:

```env
# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# Database
CHROMA_PERSIST_DIRECTORY=data/vector_store

# Application
ENV=development
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
```

### Configuration Files
YAML configuration files in the `config/` directory:

- `app_config.yaml`: Main application configuration
- `models_config.yaml`: Model definitions and settings
- `prompts_config.yaml`: Prompt templates and settings

## Development Workflow

### Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd Multi-Source-RAG-Q-A-System

# Run setup with development dependencies
python scripts/setup.py --dev --git-hooks

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Daily Development
```bash
# Start the application
./scripts/run.sh  # Linux/macOS
# or
scripts\run.bat   # Windows

# Run tests
python -m pytest tests/ -v

# Install new dependencies
pip install new-package
pip freeze > requirements.txt
```

### Production Deployment
```bash
# Set production environment
export ENV=production
export LOG_LEVEL=INFO

# Run with production settings
./scripts/run.sh
```

## Troubleshooting

### Common Issues

1. **Python Version Error**
   ```
   Error: Python 3.8 or higher is required.
   ```
   **Solution:** Install Python 3.8+ from python.org or use pyenv/conda

2. **Virtual Environment Issues**
   ```
   Error: Virtual environment creation failed.
   ```
   **Solution:** 
   ```bash
   # Remove existing venv
   rm -rf venv
   # Recreate
   python scripts/setup.py
   ```

3. **Permission Issues (Linux/macOS)**
   ```
   Error: Permission denied
   ```
   **Solution:**
   ```bash
   chmod +x scripts/run.sh
   chmod +x scripts/setup.py
   ```

4. **Dependencies Installation Failures**
   ```
   Error: Failed to install requirements
   ```
   **Solution:**
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   # Try installing again
   pip install -r requirements.txt
   ```

5. **Port Already in Use**
   ```
   Error: Port 8000 is already in use
   ```
   **Solution:**
   ```bash
   # Kill existing process
   lsof -ti:8000 | xargs kill -9  # Linux/macOS
   # or use different port
   export PORT=8001
   ./scripts/run.sh
   ```

### Debug Mode

For debugging, run with debug settings:
```bash
export ENV=development
export LOG_LEVEL=DEBUG
./scripts/run.sh
```

### Logs

Check application logs in the `logs/` directory:
```bash
tail -f logs/app.log
```

## Contributing

When adding new scripts:

1. Make scripts executable (`chmod +x`)
2. Add documentation to this README
3. Include error handling
4. Support both Windows and Unix-like systems when possible
5. Use environment variables for configuration
6. Add appropriate logging

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Review `.env.example` before sharing
- Use HTTPS for API calls in production
- Set appropriate file permissions for scripts
