"""
Startup script for the AI-Based Athlete Injury Prediction Web Application

This script checks dependencies and starts the Flask web application.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'sklearn', 
        'joblib', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("All required packages are installed!")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'models/injury_days_reg.joblib',
        'models/injury_flag_clf.joblib',
        'models/metadata.json',
        'data/processed_injury_dataset.csv',
        'app.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure you have:")
        print("   1. Trained the model: python -u scripts/train_and_evaluate.py")
        print("   2. Converted the data: python convert_excel_to_csv.py")
        return False
    
    print("All required files are present!")
    return True

def main():
    """Main startup function"""
    print("AI-Based Athlete Injury Prediction System")
    print("=" * 50)
    print("Starting web application...")
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print()
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    print()
    print("Starting Flask web application...")
    print("Open your browser and go to: http://localhost:5000")
    print("⏹Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
