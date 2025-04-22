import subprocess
import sys
import nltk

def install_packages():
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        # Install packages from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during package installation: {e}")
        sys.exit(1)

def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

if __name__ == "__main__":
    install_packages()
    download_nltk_data()
    print("All dependencies installed and NLTK data downloaded successfully.")
