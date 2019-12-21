import os


def ensure_dir(file_path):
    """
    Ensure target dir exists and if it doesnt, creates it

    :param file_path:
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_file(file_name, original_path, target_path):
    """
    Move file to train and valid dir

    :param file_name:
    :param original_path:
    :param target_path:

    """
    try:
        os.rename(original_path + file_name, target_path + file_name)
    except (FileNotFoundError, FileNotFoundError) as e:
        pass
