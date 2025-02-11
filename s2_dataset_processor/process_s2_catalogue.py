import numpy as np
import os
import pandas as pd
import json
import shutil


TEMP_METADATA_FILE = "s2_catalogue/temp_metadata.json"

def save_metadata(dict_1, dict_2):
    """ Save metadata to a temporary JSON file """
    # Merge both dictionaries
    combined_metadata = {**dict_1, **dict_2}  
    with open(TEMP_METADATA_FILE, "w") as file:
        json.dump(combined_metadata, file)

def load_metadata():
    """ Load metadata from the temporary JSON file """
    if os.path.exists(TEMP_METADATA_FILE):
        with open(TEMP_METADATA_FILE, "r") as file:
            return json.load(file)
    return {}  # Return empty dictionary if file doesn't exist


def preprocess_images_npy(subscene_path, num_bands):
    """
    Process subscenes stored as .npy files by:
      - Memory-mapping the file to avoid loading all data into RAM.
      - Loading only the selected bands.
      - Padding a 1022x1022 image to 1024x1024.
      - Splitting into four 512x512 tiles.
      - Converting from float32 to uint16 with scaling (multiplying by 10000).
      - Saving each tile as a separate .npy file.
      
    Parameters:
      subscene_path (str): Directory containing the .npy subscene files.
      num_bands (int): Number of bands to load (e.g., 3 for RGB).
    """
    # Creates and rewrites directories
    output_dir = "s2_catalogue\images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)           # Removes all the subdirectories
        os.makedirs(output_dir)

    # List all .npy files
    files = []
    for f in os.listdir(subscene_path):
        if f.endswith(".npy"):
            files.append(f)

    # Metadata bins
    tile_id = []
    image_filename = []
    sentinel_ID = []
    subscene_coordinate = []
    cloud_percentage = []

    # Read cloud percentage information as a list
    
    data = pd.read_csv("dataset/classification_tags.csv")
    df_cloud = data['cloud_percent'].tolist()

    subscene_num = 1
    # Goes through each file in "files"
    for file in files:
        file_path = os.path.join(subscene_path, file)
        
        # Memory-mapping the .npy file, it won't load the whole file into RAM immediately.
        subscene_memmap = np.load(file_path, mmap_mode='r')
        
        # Select only the desired bands. This slicing should only load the required part.
        subscene = subscene_memmap[:, :, 0:num_bands]
        
        # Pad the image from 1022x1022 to 1024x1024
        padded_data = np.pad(subscene, ((1, 1), (1, 1), (0, 0)), mode='constant')
        
        # Split into 4 tiles (each 512x512)
        top_half, bottom_half = np.vsplit(padded_data, 2)
        tiles = np.hsplit(top_half, 2) + np.hsplit(bottom_half, 2)
        
        current_cloud_percent = df_cloud[subscene_num - 1]

        tile_coords = []

        # Process each tile individually
        for i, tile in enumerate(tiles):
            # Multiply by 10000, clip to [0, 65535], and convert to uint16.
            converted_tile = np.clip(tile * 10000, 0, 65535).astype(np.uint16)
            output_filename = f"tile_{i}_{file}"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, converted_tile)
            
            
            # Handling metadata
            tile_id.append(f"subscene{subscene_num}_tile_{i}")
            image_filename.append(output_filename)
            sentinel_ID.append(file)

            if i == 0:
                coord = "rows 0-511, cols 0-511"
            elif i == 1:
                coord = "rows 0-511, cols 512-1023"
            elif i == 2:
                coord = "rows 512-1023, cols 0-511"
            elif i == 3:
                coord = "rows 512-1023, cols 512-1023"
            else:
                coord = "unknown"
            
            tile_coords.append(coord)
            cloud_percentage.append(current_cloud_percent)
            

        subscene_coordinate.extend(tile_coords)
        subscene_num += 1

        # Writing medatada to dictionary
        dict_1 = {
            'Tile ID': tile_id,
            'Image Filename': image_filename,
            'Sentinel-2 product ID': sentinel_ID,
            'Subscene coordinate': subscene_coordinate,
            'Cloud coverige percentage': cloud_percentage
          }

    save_metadata(dict_1, {})  # Save only image metadata first
    return dict_1



def convert_mask(mask_path):
    """
    Convert a one-hot encoded mask (shape: H x W x 3) to a single-channel mask (H x W)
    where:
      0 = CLEAR, 1 = CLOUD, 2 = CLOUD_SHADOW.

    Parameters:
      mask_path (str): Directory containing the .npy subscene files.
    """
    
    # Create and rewrite directories
    output_dir = "s2_catalogue\masks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)           # Removes all the subdirectories
        os.makedirs(output_dir)

    # List all .npy files
    files = []
    for f in os.listdir(mask_path):
        if f.endswith(".npy"):
            files.append(f)

    mask_filename = []

    # Goes through each file in "files"
    for file in files:
        
        file_path = os.path.join(mask_path, file)
        mask = np.load(file_path)

        # Pad the image from 1022x1022 to 1024x1024
        padded_mask = np.pad(mask, ((1, 1), (1, 1), (0, 0)), mode='constant')

        # Remove one-hot encoding
        decoded_mask = np.argmax(padded_mask, axis=-1)

        # Split into 4 tiles (each 512x512)
        top_half, bottom_half = np.vsplit(decoded_mask, 2)
        tiles = np.hsplit(top_half, 2) + np.hsplit(bottom_half, 2)
        
        for i, tile in enumerate(tiles):

          # Convert to uint8.
          converted_tile = tile.astype(np.uint8)
          output_filename = f"tile_{i}_{file}"
          output_path = os.path.join(output_dir, output_filename)
          np.save(output_path, converted_tile)
          mask_filename.append(output_filename)

    # Handling metadata
    dict_2 = {'Mask filename': mask_filename}

    # Merge with existing metadata
    existing_metadata = load_metadata()
    save_metadata(existing_metadata, dict_2)

    return dict_2


def tile_metadata():

    """ Load metadata and save it as CSV """
    metadata = load_metadata()
    df = pd.DataFrame(metadata)
    df.to_csv("s2_catalogue/tile_metadata.csv", index=False)
    print("Tile metadata saved successfully.")
    os.remove(TEMP_METADATA_FILE)


def dataset_metadata():
    dataset_dictionary = {
        "class_mapping": {
            "0": "Clear",
            "1": "Cloud",
            "2": "Cloud Shadow"
        },
        "band_info": [
            {
            "band_id": 1,
            "band_name": "Blue",
            "center_wavelength": 442.7,
            "bandwidth": 20,
            "gsd": 60
            },
            {
            "band_id": 2,
            "band_name": "Green",
            "center_wavelength": 492.7,
            "bandwidth": 65,
            "gsd": 10
            },
            {
            "band_id": 3,
            "band_name": "Green_2",
            "center_wavelength": 559.8,
            "bandwidth": 35,
            "gsd": 10
            },
            {
            "band_id": 4,
            "band_name": "Red",
            "center_wavelength": 664.6,
            "bandwidth": 30,
            "gsd": 10
            },
            {
            "band_id": 5,
            "band_name": "Red_2",
            "center_wavelength": 704.1,
            "bandwidth": 14,
            "gsd": 20
            },
            {
            "band_id": 6,
            "band_name": "Red_3",
            "center_wavelength": 740.5,
            "bandwidth": 14,
            "gsd": 20
            },
            {
            "band_id": 7,
            "band_name": "Red_4",
            "center_wavelength": 782.8,
            "bandwidth": 19,
            "gsd": 20
            },
            {
            "band_id": 8,
            "band_name": "Infra_red",
            "center_wavelength": 832.8,
            "bandwidth": 105,
            "gsd": 10
            },
            {
            "band_id": 8.5,
            "band_name": "Infra_red_2",
            "center_wavelength": 864.7,
            "bandwidth": 21,
            "gsd": 20
            },
            {
            "band_id": 9,
            "band_name": "Infra_red_3",
            "center_wavelength": 945.1,
            "bandwidth": 19,
            "gsd": 60
            },
            {
            "band_id": 10,
            "band_name": "Infra_red_4",
            "center_wavelength": 1373.5,
            "bandwidth": 29,
            "gsd": 60
            },
            {
            "band_id": 11,
            "band_name": "Infra_red_5",
            "center_wavelength": 1613.7,
            "bandwidth": 90,
            "gsd": 20
            },
            {
            "band_id": 12,
            "band_name": "Infra_red_6",
            "center_wavelength": 2202.4,
            "bandwidth": 174,
            "gsd": 20
            }
            ]
            }

    with open("s2_catalogue/dataset_metadata.json", "w") as outfile: 
        json.dump(dataset_dictionary, outfile)
    

# Handling subscenes
preprocess_images_npy("dataset/subscenes", num_bands=3)

# Handling masks
convert_mask("dataset/masks")

# Handling metadata in tile level
tile_metadata()

# Handling metadata in dataset level
dataset_metadata()





