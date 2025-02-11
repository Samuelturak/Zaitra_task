from s2_dataset_processor.process_s2_catalogue import preprocess_images_npy, convert_mask, tile_metadata, dataset_metadata

preprocess_images_npy("dataset\subscenes", num_bands=3)  # Process images and save metadata
convert_mask("dataset\masks")  # Process masks and save metadata
tile_metadata()  # Load metadata and write CSV
dataset_metadata()  # Generate dataset-level metadata JSON

print("Processing completed! Check 's2_catalogue/' for output files.")