import os
import argparse
import requests
import zipfile
import shutil
import sys
import json
from tqdm import tqdm

def download_file(url, destination, headers=None):
    """Download a file from a URL with a progress bar
    
    Args:
        url: URL to download
        destination: Path to save the file
        headers: Optional headers for the request
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        # Make request with optional headers
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Setup progress bar
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True,
            desc=f"Downloading {os.path.basename(destination)}"
        )
        
        # Download with progress
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        
        # Check if download is complete
        if total_size > 0 and os.path.getsize(destination) != total_size:
            print("Downloaded file size does not match expected size")
            return False
            
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def extract_zip(zip_path, extract_path):
    """Extract a zip file
    
    Args:
        zip_path: Path to the zip file
        extract_path: Path to extract to
        
    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        # Verify that the file is a valid zip file
        if not zipfile.is_zipfile(zip_path):
            print(f"Error: {zip_path} is not a valid zip file")
            return False
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files
            total_files = len(zip_ref.namelist())
            
            # Setup progress bar
            with tqdm(total=total_files, desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
                # Extract one file at a time
                for file in zip_ref.namelist():
                    zip_ref.extract(file, extract_path)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def create_sample_data(output_dir):
    """Create a small sample dataset for testing when downloads fail
    
    Args:
        output_dir: Directory to save the sample data
    """
    print("Creating sample data for testing...")
    
    # Create database directories
    db_dir = os.path.join(output_dir, "dev_databases")
    os.makedirs(db_dir, exist_ok=True)
    
    # Create a sample database
    sample_db_dir = os.path.join(db_dir, "card_games")
    os.makedirs(sample_db_dir, exist_ok=True)
    
    # Create an empty SQLite database
    import sqlite3
    db_path = os.path.join(sample_db_dir, "card_games.sqlite")
    conn = sqlite3.connect(db_path)
    
    # Create a sample table
    conn.execute('''
    CREATE TABLE cards (
        id INTEGER PRIMARY KEY,
        name TEXT,
        attack INTEGER,
        defense INTEGER,
        type TEXT
    )
    ''')
    
    # Insert some sample data
    conn.execute("INSERT INTO cards VALUES (1, 'Dragon', 8, 6, 'Monster')")
    conn.execute("INSERT INTO cards VALUES (2, 'Wizard', 5, 3, 'Monster')")
    conn.execute("INSERT INTO cards VALUES (3, 'Sword', 3, 0, 'Weapon')")
    conn.execute("INSERT INTO cards VALUES (4, 'Shield', 0, 4, 'Defense')")
    conn.execute("INSERT INTO cards VALUES (5, 'Heal', 0, 0, 'Spell')")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    # Create a sample dataset JSON
    sample_data = [
        {
            "db_id": "card_games",
            "question": "How many cards have more than 5 attack points?",
            "evidence": "The attack points are stored in the attack column.",
            "SQL": "SELECT COUNT(*) FROM cards WHERE attack > 5"
        },
        {
            "db_id": "card_games",
            "question": "List all cards of type Monster.",
            "evidence": "The type of card is stored in the type column.",
            "SQL": "SELECT * FROM cards WHERE type = 'Monster'"
        }
    ]
    
    # Save the sample dataset
    dataset_path = os.path.join(output_dir, "mini_dev_sqlite.json")
    with open(dataset_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Create a gold SQL file
    gold_path = os.path.join(output_dir, "mini_dev_sqlite_gold.sql")
    with open(gold_path, 'w') as f:
        for item in sample_data:
            f.write(f"{item['SQL']}\t----- bird -----\t{item['db_id']}\n")
    
    print(f"Sample data created in {output_dir}")

def download_from_huggingface(output_dir):
    """Alternative method to download from Hugging Face
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        # Download the dataset files
        print("Downloading dataset from Hugging Face...")
        dataset_file = hf_hub_download(
            repo_id="bird-bench/mini_dev",
            filename="mini_dev_sqlite.json",
            repo_type="dataset"
        )
        
        # Copy to output directory
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(dataset_file, os.path.join(output_dir, "mini_dev_sqlite.json"))
        
        # Download the database files
        print("Downloading databases from Hugging Face...")
        # This might be a large download, so we'll show it's in progress
        print("This could take some time...")
        
        # Try to download the databases directly
        try:
            db_file = hf_hub_download(
                repo_id="bird-bench/mini_dev",
                filename="dev_databases.zip",
                repo_type="dataset"
            )
            
            # Extract databases
            zip_path = os.path.join(output_dir, "dev_databases.zip")
            shutil.copy(db_file, zip_path)
            
            if extract_zip(zip_path, output_dir):
                os.remove(zip_path)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error downloading databases: {e}")
            return False
            
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        print("Please install the huggingface_hub package: pip install huggingface_hub")
        return False

def download_bird_mini_dev(output_dir="./mini_dev_data"):
    """Download and extract the BIRD Mini-Dev dataset
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try GitHub first
    print("Attempting to download BIRD-SQL Mini-Dev dataset from GitHub...")
    github_success = False
    
    # GitHub URL for the dataset files
    dataset_url = "https://github.com/bird-bench/mini_dev/archive/refs/heads/main.zip"
    
    # Temporary zip file path
    zip_path = os.path.join(output_dir, "mini_dev.zip")
    
    # Try to download from GitHub
    if download_file(dataset_url, zip_path):
        # Extract dataset
        print("Extracting dataset...")
        if extract_zip(zip_path, output_dir):
            # Move files from extraction directory to output directory
            extracted_dir = os.path.join(output_dir, "mini_dev-main")
            if os.path.exists(extracted_dir):
                # Move all files from extracted directory to output directory
                for item in os.listdir(extracted_dir):
                    source = os.path.join(extracted_dir, item)
                    dest = os.path.join(output_dir, item)
                    
                    if os.path.exists(dest):
                        if os.path.isdir(dest):
                            shutil.rmtree(dest)
                        else:
                            os.remove(dest)
                    
                    shutil.move(source, dest)
                
                # Remove extracted directory
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)
                
                github_success = True
    
    # Clean up zip file if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # If GitHub download failed, try Hugging Face
    if not github_success:
        print("GitHub download failed. Trying Hugging Face...")
        huggingface_success = download_from_huggingface(output_dir)
        
        if not huggingface_success:
            print("Both download methods failed.")
            print("Creating sample data for testing instead.")
            create_sample_data(output_dir)
            return
    
    print(f"Dataset downloaded and extracted to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download BIRD-SQL Mini-Dev dataset")
    parser.add_argument("--output-dir", type=str, default="./mini_dev_data", help="Directory to save the dataset")
    
    args = parser.parse_args()
    
    download_bird_mini_dev(args.output_dir)

if __name__ == "__main__":
    main()