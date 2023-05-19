import os
import shutil

# Set your source and destination folders
source_dir = '../../../naver-webtoon-data/faces/'
destination_dir = '../dataset/cartoon_faces/'

# Create the destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Initialize a counter for file renaming
counter = 0

# Iterate through all folders in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)

    # Check if the current path is a folder
    if os.path.isdir(folder_path):

        # Iterate through all files in the current folder
        number_of_files = len(os.listdir(folder_path))
        print("Processing folder:", folder_name, "with", number_of_files)
        for idx, file_name in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)

            # Check if the current path is a file
            if os.path.isfile(file_path):

                # Create a new file name with the counter value
                new_file_name = f"{counter:06d}"
                new_file_path = os.path.join(destination_dir, new_file_name)

                # Copy the file to the destination folder with the new name
                shutil.copy(file_path, new_file_path)

                # Increment the counter
                counter += 1

print("Done")