#!/usr/bin/env python3
"""
Setup script for AI vs Human Writing Classification project.
This script helps set up the development environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Please upgrade your Python version.")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")

def create_virtual_environment():
    """Create a virtual environment for the project."""
    venv_path = Path.cwd() / "venv"
    if venv_path.exists():
        print("ℹ️  Virtual environment already exists.")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_requirements():
    """Install Python requirements."""
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
    
    commands = [
        (f"{pip_path} install --upgrade pip", "Upgrading pip"),
        (f"{pip_path} install -r requirements.txt", "Installing core requirements"),
    ]
    
    # Install development requirements if file exists
    if Path("requirements-dev.txt").exists():
        commands.append((f"{pip_path} install -r requirements-dev.txt", "Installing development requirements"))
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def download_nltk_data():
    """Download required NLTK data."""
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    nltk_command = f'''{python_path} -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
print('NLTK data downloaded successfully!')
"'''
    
    return run_command(nltk_command, "Downloading NLTK data")

def verify_installation():
    """Verify that the installation was successful."""
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    test_command = f'''{python_path} -c "
import pandas, numpy, sklearn, torch, transformers, nltk, textstat
print('✅ All core libraries imported successfully!')
print(f'PyTorch version: {{torch.__version__}}')
print(f'Transformers version: {{transformers.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
"'''
    
    return run_command(test_command, "Verifying installation")

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 ENVIRONMENT SETUP COMPLETE!")
    print("="*60)
    
    activation_cmd = "venv\\Scripts\\activate" if os.name == 'nt' else "source venv/bin/activate"
    
    print(f"""
📝 Next Steps:

1. Activate the virtual environment:
   {activation_cmd}

2. Open the setup notebook:
   jupyter notebook notebooks/01_environment_setup.ipynb

3. Prepare your dataset:
   - Place CSV files in data/raw/
   - Ensure columns: 'text' and 'label'
   - Labels: 0 = human, 1 = AI-generated

4. Start training:
   python train.py --data data/raw/your_dataset.csv

5. Explore the notebooks:
   - notebooks/exploratory/ for data analysis
   - notebooks/experiments/ for model experiments

📁 Project Structure:
   data/          - Your datasets
   models/        - Trained models
   notebooks/     - Jupyter notebooks
   src/           - Source code
   results/       - Experiment results

🚀 Happy coding!
    """)

def main():
    """Main setup function."""
    print("🚀 AI vs Human Writing Classification - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed at virtual environment creation.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation.")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed, but continuing...")
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Installation verification failed, but continuing...")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
