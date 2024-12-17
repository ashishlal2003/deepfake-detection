import os
import shutil

def reorganize_folders(base_dir):
    splits = ['train', 'test']
    categories = ['real', 'fake']

    for split in splits:
        for category in categories:
            source_dir = os.path.join(base_dir, category, split)
            target_dir = os.path.join(base_dir, split, category)
            os.makedirs(target_dir, exist_ok=True)

            for file_name in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# Run the reorganization
reorganize_folders('data/split')
