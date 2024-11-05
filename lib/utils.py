
import os

def get_model(voice_model):
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            model_filename = file
        elif ext == ".index":
            index_filename = file

    if model_filename is None:
        print(f"No model file exists in {model_dir}.")
        return None, None

    return os.path.join(model_dir, model_filename), os.path.join(model_dir, index_filename or "")
