#!/bin/bash

# Multi-Source RAG System - Linux/macOS Startup Script
# This script sets up the environment and starts the RAG system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

print_status "Starting Multi-Source RAG System..."
print_status "Project root: $PROJECT_ROOT"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. Found version: $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION detected"

# Check if virtual environment exists
VENV_DIR="$PROJECT_ROOT/venv"

if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    print_status "Installing requirements..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    print_success "Requirements installed"
fi

# Install dev requirements if requirements-dev.txt exists
if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
    print_status "Installing development requirements..."
    pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    print_success "Development requirements installed"
fi

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        print_warning "Please edit .env file with your API keys and configuration"
    else
        print_warning ".env file not found. Please create it with your configuration."
    fi
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/data/source_documents"
mkdir -p "$PROJECT_ROOT/data/vector_store"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/uploads"
mkdir -p "$PROJECT_ROOT/temp"
print_success "Directories created"

# Check configuration files
if [ -f "$PROJECT_ROOT/config/app_config.yaml" ]; then
    print_success "Configuration files found"
else
    print_warning "Configuration files not found in config/ directory"
fi

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export ENV="${ENV:-development}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start the application
print_status "Starting the RAG system..."
print_status "Environment: $ENV"
print_status "Log Level: $LOG_LEVEL"

# Check if main.py exists
if [ ! -f "$PROJECT_ROOT/main.py" ]; then
    print_error "main.py not found in project root"
    exit 1
fi

# Run the main application
if [ "$ENV" = "development" ]; then
    print_status "Running in development mode with auto-reload..."
    python3 main.py --reload --debug
else
    print_status "Running in production mode..."
    python3 main.py
fi

print_success "RAG system started successfully!"
print_status "Access the application at: http://localhost:8000"
print_status "API documentation at: http://localhost:8000/docs"
print_status "Press Ctrl+C to stop the server"

# Wait for the process
wait
