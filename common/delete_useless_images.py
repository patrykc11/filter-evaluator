import os

def find_and_remove_files(root_directory, target_word, target_file_substrings):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if target_word in os.path.basename(dirpath):
            print(f"Found folder: {dirpath}")
            
            for filename in filenames:
                if any(substring in filename for substring in target_file_substrings):
                    file_to_delete = os.path.join(dirpath, filename)
                    print(f"Deleting file: {file_to_delete}")

                    try:
                        os.remove(file_to_delete)
                        print(f"Successfully deleted {file_to_delete}")
                    except OSError as e:
                        print(f"Error deleting {file_to_delete}: {e}")

if __name__ == "__main__":
    root_directory = "./images"
    target_word = "iseo"
    target_file_substrings = ["0013", "0014", "0016", "0018", "0017"]

    find_and_remove_files(root_directory, target_word, target_file_substrings)
