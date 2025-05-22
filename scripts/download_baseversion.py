import os
import subprocess

URL = "https://drive.google.com/drive/folders/15QDnPR9t3FO18fVzCyqUu4h-7COl9Utd?usp=sharing"
DEST = os.path.join("data", "base_version")


def main():
    os.makedirs(DEST, exist_ok=True)
    cmd = ["gdown", "--folder", URL, "-O", DEST]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Failed to download dataset. Please check your network connection and gdown installation.")
        raise e


if __name__ == "__main__":
    main()
