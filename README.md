## Purpose
This package preprocesses the **Sentinel-2 Cloud Mask Catalogue** dataset into a machine learning (ML) ready format.  

The preprocessing of `.npy` files involves 3 key steps:
- Loading only the selected bands.
- Padding and tiling images.
- Converting images from float32 to uint16 data type.

### ** Mask processing**
- Implementing same matching and tiling strategy as in the images.
- Each mask shares the same filename to its corresponding image.
- Stores the mask as a 2D array where each value corresponds to a class ID:

    | Class ID | Label         |
    |----------|---------------|
    | 0        | Clear         |
    | 1        | Cloud         |
    | 2        | Cloud Shadow  |


### **Metadata Generation**  
The package also **produces metadata** in the following formats:  
- **Tile-level metadata** → stored in `.csv` format.
- **Dataset-level metadata** → stored in `.json` format.


## Package architecture
- **`process_s2_catalogue.py`** Contains the core functions for preprocessing images, generating masks, and saving metadata.
- **`__init__.py`**.py: Defines the package and makes functions importable.

## Usage instructions:
- Use a .py file where you import those 4 functions: `from s2_dataset_processor.process_s2_catalogue import preprocess_images_npy, convert_mask, tile_metadata, dataset_metadata`
- Call each functions as follows:
    `preprocess_images_npy("Your dataset subscene path", num_bands="from 1 to 13")`  # Process images and save metadata
    `convert_mask("Your dataset masks path")`  # Process masks and save metadata
    `tile_metadata()`  # Load metadata and write CSV
    `dataset_metadata()`  # Generate dataset-level metadata JSON
- IMPORTANT! Place the dataset next to the package.



## Installation:
- Go inside s2_dataset_processor and use command `pip install .`



