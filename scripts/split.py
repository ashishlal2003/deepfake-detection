from sklearn.model_selection import train_test_split
import shutil
import os

def split_data(input_dir, output_dir, test_size=0.2):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    train_files, test_files = train_test_split(files, test_size=test_size)

    for split, split_files in [('train', train_files), ('test', test_files)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for file_name in split_files:
            shutil.copy(os.path.join(input_dir, file_name), os.path.join(split_dir, file_name))

split_data('data/preprocessed/real', 'data/split/real')
split_data('data/preprocessed/fake', 'data/split/fake')