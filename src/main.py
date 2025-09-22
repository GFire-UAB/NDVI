from tools import process_indices, generate_cropland_mask, add_ids_to_croplands, create_df_crops
from update_fuel_map import split_shapefile_by_planted, replace_pixels_within_shapefile
from model import predict_with_catboost

# ---------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------
region = "Nalec"
date = "2019-06-17"
processed = "../data/processed"
model = "../data/model"
raw = "../data/raw"
output = "../output"
# ---------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------
# Compute indexes ---
process_indices(base_dir=raw, date_str=date, output_dir=processed, region=region)

# Generate pixel mask for crop fields ---
generate_cropland_mask(region=region, input_dir=raw, output_dir=processed, date=date)

# Assign IDs to Nalec crop fields ---
add_ids_to_croplands(input_dir=processed, region=region)

# Create dataframe with statistics of the indexes for each crop to use them for AI predictions ---
create_df_crops(region=region, input_dir=processed, date=date)

# Predict if crops are planted or not using the created df ---

predict_with_catboost(
    model_path = f"{model}/catboost_model_trained.cbm",
    scaler_path = f"{model}/scaler.pkl",
    input_data_path = f"{processed}/output_croplands_{region}_{date}_stats.txt",
    output_data_path = f"{processed}/output_croplands_{region}_{date}_prediction.txt")

# Separate planted crops vs non-planted crops ---
split_shapefile_by_planted(processed, region, date)

# Update fuel map with planted crops ---
replace_pixels_within_shapefile(
    tif_path=f"{raw}/{region}_fuel_map.tif",
    shp_path=f"{processed}/croplands_{region}_{date}_1.shp",
    output_path=f"{processed}/{region}_fuel_map_planted.tif",
    new_value=104
)

# Update fuel map with non-planted crops ---
replace_pixels_within_shapefile(
    tif_path=f"{processed}/{region}_fuel_map_planted.tif",
    shp_path=f"{processed}/croplands_{region}_{date}_0.shp",
    output_path=f"{output}/{region}_fuel_map_output_nada.tif",
    new_value=93
)
