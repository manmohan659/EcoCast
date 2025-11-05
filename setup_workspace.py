#!/usr/bin/env python3
"""
🚀 EcoCast Workspace Setup Script
Automatically installs and configures all dependencies for teammates
"""

import sys
import subprocess
import os
from pathlib import Path

# ANSI colors for pretty output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}{text.center(60)}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{RESET}")

def run_command(cmd, check=True, capture=True):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture,
            text=True
        )
        return result.returncode == 0, result.stdout.strip() if capture else ""
    except subprocess.CalledProcessError as e:
        return False, e.stderr if capture else ""

def check_python_version():
    """Check if Python version is adequate"""
    print_header("🐍 Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible (requires >= 3.8)")
        return True
    else:
        print_error(f"Python {version_str} is too old. Need Python >= 3.8")
        print_info("Please install Python 3.8+ from https://www.python.org/downloads/")
        return False

def check_pip():
    """Check if pip is available"""
    print_header("📦 Checking pip")
    
    success, output = run_command("pip --version")
    if success:
        print_success(f"pip is installed: {output}")
        return True
    
    success, output = run_command("pip3 --version")
    if success:
        print_success(f"pip3 is installed: {output}")
        return True
    
    print_error("pip/pip3 not found")
    print_info("Attempting to install pip...")
    
    success, _ = run_command("python -m ensurepip --upgrade")
    if success:
        print_success("pip installed successfully")
        return True
    
    return False

def install_python_dependencies():
    """Install Python dependencies from requirements.txt"""
    print_header("📚 Installing Python Dependencies")
    
    req_file = Path(__file__).parent / "requirements.txt"
    
    if not req_file.exists():
        print_error(f"requirements.txt not found at {req_file}")
        return False
    
    print_info("Installing packages from requirements.txt...")
    print_info("This may take a few minutes...")
    
    # Try with pip first, then pip3
    success, output = run_command(
        f"pip install -r {req_file} --upgrade",
        check=False,
        capture=False
    )
    
    if not success:
        print_warning("pip failed, trying pip3...")
        success, output = run_command(
            f"pip3 install -r {req_file} --upgrade",
            check=False,
            capture=False
        )
    
    if success:
        print_success("Python dependencies installed successfully")
        return True
    else:
        print_error("Failed to install Python dependencies")
        return False

def verify_python_imports():
    """Test that critical Python packages can be imported"""
    print_header("🧪 Verifying Python Imports")
    
    critical_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'prophet',
        'statsmodels',
        'joblib'
    ]
    
    all_success = True
    
    for package in critical_packages:
        try:
            __import__(package)
            print_success(f"{package} ✓")
        except ImportError as e:
            print_error(f"{package} failed: {e}")
            all_success = False
    
    return all_success

def check_node():
    """Check if Node.js is installed"""
    print_header("🟢 Checking Node.js")
    
    success, output = run_command("node --version")
    
    if success:
        version = output.replace('v', '')
        major_version = int(version.split('.')[0])
        
        print_info(f"Node.js version: {output}")
        
        if major_version >= 16:
            print_success(f"Node.js {output} is compatible (requires >= 16)")
            return True
        else:
            print_warning(f"Node.js {output} is old. Recommend >= 18")
            return True
    else:
        print_error("Node.js not found")
        print_info("Please install Node.js from https://nodejs.org/ (LTS version)")
        return False

def check_npm():
    """Check if npm is installed"""
    print_header("📦 Checking npm")
    
    success, output = run_command("npm --version")
    
    if success:
        print_success(f"npm version: {output}")
        return True
    else:
        print_error("npm not found (should come with Node.js)")
        return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print_header("🎨 Installing Frontend Dependencies")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print_error(f"frontend directory not found at {frontend_dir}")
        return False
    
    print_info("Installing frontend packages...")
    print_info("This may take a few minutes...")
    
    os.chdir(frontend_dir)
    
    success, _ = run_command("npm install", check=False, capture=False)
    
    if success:
        print_success("Frontend dependencies installed successfully")
        return True
    else:
        print_error("Failed to install frontend dependencies")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    print_header("⚙️  Checking Configuration")
    
    env_file = Path(__file__).parent / "backend" / ".env"
    
    if env_file.exists():
        print_success(".env file already exists")
        return True
    
    print_info("Creating .env file...")
    
    env_content = """# Backend Configuration
PORT=8000
HOST=0.0.0.0

# CORS Configuration
FRONTEND_URL=http://localhost:5173

# Data paths
DATA_PATH=../data_work
MODELS_PATH=../models
ARTEFACTS_PATH=../artefacts
"""
    
    try:
        env_file.parent.mkdir(exist_ok=True)
        env_file.write_text(env_content)
        print_success(".env file created")
        return True
    except Exception as e:
        print_warning(f"Could not create .env: {e}")
        return True  # Not critical

def print_final_instructions():
    """Print instructions to boot the app"""
    print_header("🎉 Setup Complete!")
    
    print(f"\n{GREEN}{BOLD}To start the application:{RESET}\n")
    print(f"{BLUE}1. Start Backend (Terminal 1):{RESET}")
    print(f"   cd backend")
    print(f"   python -m uvicorn app.main:app --reload --port 8000\n")
    
    print(f"{BLUE}2. Start Frontend (Terminal 2):{RESET}")
    print(f"   cd frontend")
    print(f"   npm run dev\n")
    
    print(f"{BLUE}3. Open Browser:{RESET}")
    print(f"   http://localhost:5173\n")
    
    print(f"{GREEN}You should now see the EcoCast dashboard with RF insights! 🚀{RESET}\n")

def main():
    """Main setup flow"""
    print_header("🚀 EcoCast Workspace Setup")
    print_info("This script will install and configure all dependencies")
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print_error("Failed to install Python dependencies")
        sys.exit(1)
    
    # Verify imports
    if not verify_python_imports():
        print_error("Some Python packages failed to import")
        print_info("Try running: pip install -r requirements.txt --upgrade")
        sys.exit(1)
    
    # Check Node/npm
    if not check_node():
        print_warning("Node.js not found - frontend won't work")
        print_info("Install Node.js and run this script again")
    else:
        if not check_npm():
            print_warning("npm not found - frontend won't work")
        else:
            # Install frontend dependencies
            install_frontend_dependencies()
    
    # Create config files
    create_env_file()
    
    # Final instructions
    print_final_instructions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_error("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

