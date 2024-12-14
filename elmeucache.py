import os

def ensure_directory_exists(path):
    # Normalize the path for cross-platform compatibility
    normalized_path = os.path.normpath(path)
    
    try:
        # Create the directory and all intermediate-level directories if necessary
        os.makedirs(normalized_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory '{normalized_path}': {e}")
        raise
    return normalized_path

# Redirect cache to OneDrive path
cache_path = 'C:/Users/merit/OneDrive/.cache/torch'

try:
    fixed_path = ensure_directory_exists(cache_path)
    os.environ['TORCH_HOME'] = fixed_path  # Redirect PyTorch to use the new cache location
    print(f"Cache directory set to: {fixed_path}")
except Exception as e:
    print(f"Failed to ensure directory: {e}")
