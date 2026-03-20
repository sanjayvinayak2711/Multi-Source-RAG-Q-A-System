#!/usr/bin/env python3
"""
Multi-Source RAG System Setup Script
This script helps set up the development environment and dependencies.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        if capture_output:
            return result.stdout.strip(), result.stderr.strip()
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e}")
        if capture_output:
            return e.stdout.strip(), e.stderr.strip()
        return None, None

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} is compatible.")
    return True

def create_virtual_environment(venv_path):
    """Create a virtual environment."""
    print(f"Creating virtual environment at {venv_path}...")
    stdout, stderr = run_command(f"python -m venv {venv_path}", capture_output=True)
    if stderr:
        print(f"Warning: {stderr}")
    print("Virtual environment created successfully.")

def activate_virtual_environment(venv_path):
    """Get activation command for virtual environment."""
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
        return f'call "{activate_script}" && '
    else:  # Unix-like
        activate_script = os.path.join(venv_path, 'bin', 'activate')
        return f'source "{activate_script}" && '

def install_requirements(venv_path, dev=False):
    """Install requirements from requirements.txt files."""
    venv_activate = activate_virtual_environment(venv_path)
    
    # Install base requirements
    if os.path.exists('requirements.txt'):
        print("Installing base requirements...")
        command = f'{venv_activate}pip install -r requirements.txt'
        stdout, stderr = run_command(command, capture_output=True)
        if stderr and "ERROR" in stderr.upper():
            print(f"Error installing requirements: {stderr}")
            return False
        print("Base requirements installed successfully.")
    else:
        print("Warning: requirements.txt not found.")
    
    # Install dev requirements if requested
    if dev and os.path.exists('requirements-dev.txt'):
        print("Installing development requirements...")
        command = f'{venv_activate}pip install -r requirements-dev.txt'
        stdout, stderr = run_command(command, capture_output=True)
        if stderr and "ERROR" in stderr.upper():
            print(f"Error installing dev requirements: {stderr}")
            return False
        print("Development requirements installed successfully.")
    elif dev:
        print("Warning: requirements-dev.txt not found.")
    
    return True

def setup_environment_file():
    """Set up .env file from .env.example."""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        print("Creating .env file from .env.example...")
        import shutil
        shutil.copy('.env.example', '.env')
        print("Please edit .env file with your API keys and configuration.")
        return True
    elif os.path.exists('.env'):
        print(".env file already exists.")
        return True
    else:
        print("Warning: Neither .env nor .env.example found.")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        'data/source_documents',
        'data/vector_store',
        'logs',
        'uploads',
        'temp',
        'models/configs',
        'prompts/cache'
    ]
    
    print("Creating necessary directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def run_tests(venv_path):
    """Run the test suite."""
    print("Running tests...")
    venv_activate = activate_virtual_environment(venv_path)
    command = f'{venv_activate}python -m pytest tests/ -v'
    stdout, stderr = run_command(command, capture_output=True)
    if stderr and "ERROR" in stderr.upper():
        print(f"Error running tests: {stderr}")
        return False
    print("Tests completed successfully.")
    return True

def setup_git_hooks():
    """Set up Git hooks for development."""
    if os.path.exists('.git'):
        print("Setting up Git hooks...")
        hooks_dir = Path('.git/hooks')
        
        # Pre-commit hook
        pre_commit_hook = """#!/bin/sh
# Pre-commit hook for RAG system
echo "Running pre-commit checks..."

# Run tests
python -m pytest tests/ --tb=short -q
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Run linting (if available)
if command -v flake8 &> /dev/null; then
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    if [ $? -ne 0 ]; then
        echo "Linting failed. Commit aborted."
        exit 1
    fi
fi

echo "Pre-commit checks passed."
"""
        
        hook_file = hooks_dir / 'pre-commit'
        with open(hook_file, 'w') as f:
            f.write(pre_commit_hook)
        
        # Make hook executable
        os.chmod(hook_file, 0o755)
        print("Git hooks set up successfully.")
    else:
        print("Not a Git repository. Skipping Git hooks setup.")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up Multi-Source RAG System')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    parser.add_argument('--no-venv', action='store_true', help='Skip virtual environment creation')
    parser.add_argument('--test', action='store_true', help='Run tests after setup')
    parser.add_argument('--git-hooks', action='store_true', help='Set up Git hooks')
    parser.add_argument('--venv-path', default='venv', help='Virtual environment path')
    
    args = parser.parse_args()
    
    print("=== Multi-Source RAG System Setup ===")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not args.no_venv:
        venv_path = Path(args.venv_path)
        if not venv_path.exists():
            create_virtual_environment(venv_path)
        else:
            print(f"Virtual environment already exists at {venv_path}")
    else:
        venv_path = None
        print("Skipping virtual environment creation.")
    
    # Install requirements
    if venv_path:
        if not install_requirements(venv_path, dev=args.dev):
            print("Failed to install requirements.")
            sys.exit(1)
    else:
        print("Skipping requirements installation (no virtual environment).")
    
    # Set up environment file
    setup_environment_file()
    
    # Create directories
    create_directories()
    
    # Set up Git hooks if requested
    if args.git_hooks:
        setup_git_hooks()
    
    # Run tests if requested
    if args.test and venv_path:
        if not run_tests(venv_path):
            print("Tests failed. Setup completed but tests have issues.")
            sys.exit(1)
    
    print()
    print("=== Setup Complete ===")
    print()
    print("Next steps:")
    print("1. Edit .env file with your API keys and configuration")
    print("2. Add your documents to data/source_documents/")
    print("3. Run the application:")
    
    if os.name == 'nt':  # Windows
        print("   scripts\\run.bat")
    else:  # Unix-like
        print("   chmod +x scripts/run.sh")
        print("   ./scripts/run.sh")
    
    print()
    print("For development mode:")
    print("1. Install dev dependencies: python scripts/setup.py --dev")
    print("2. Set up Git hooks: python scripts/setup.py --git-hooks")
    print("3. Run tests: python scripts/setup.py --test")

if __name__ == "__main__":
    main()
