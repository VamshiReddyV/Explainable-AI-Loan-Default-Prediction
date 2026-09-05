"""
Convenience launcher for Explainable AI Loan Default Prediction platform.
Automatically detects and runs the application using virtual environment or active Python.
"""
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for .venv
venv_python = os.path.join(BASE_DIR, '.venv', 'Scripts', 'python.exe')

if os.path.exists(venv_python):
    python_exec = venv_python
else:
    python_exec = sys.executable

app_py = os.path.join(BASE_DIR, 'app.py')

print("=" * 60)
print("🚀 Starting LoanPredict AI - Explainable AI Platform")
print("=" * 60)
print(f"Using Python: {python_exec}")
print(f"Server URL:   http://127.0.0.1:5000")
print("Press CTRL+C to stop the server.")
print("=" * 60)

try:
    subprocess.run([python_exec, app_py])
except KeyboardInterrupt:
    print("\n👋 Server stopped.")
