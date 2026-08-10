from pathlib import Path
import shutil

# Parent folder
def check_folders(PARENT_FOLDER):
    PARENT_FOLDER= Path(PARENT_FOLDER)
    for folder in PARENT_FOLDER.iterdir():

        # Only process directories
        if not folder.is_dir():
            continue

        images_folder = folder / "images"
        labels_folder = folder / "labels"

        # Check that both folders exist
        if not images_folder.is_dir() or not labels_folder.is_dir():
            print(f"Deleting: {folder}")
            shutil.rmtree(folder)

        else:
            print(f"OK: {folder}")

    print("Done.")