import os
import subprocess

def clean_metadata(file_path):
    # Usa ExifTool per cancellare ogni traccia di GPS o ID
    print(f"Pulizia metadati per: {file_path}")
    subprocess.run(["exiftool", "-all=", "-overwrite_original", file_path])
    print("Video pulito. Pronto per il caricamento anonimo.")

if __name__ == "__main__":
    video = input("Trascina qui il file video da pulire: ")
    clean_metadata(video)
