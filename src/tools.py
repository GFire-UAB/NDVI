import re
from rasterio.features import geometry_mask
from shapely.affinity import scale
from shapely.geometry import shape, box
import fiona
import os
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask


# ---------------------------------------------------------------------
# Compute NDVI, GNDVI, MSAVI, and EVI
# ---------------------------------------------------------------------
def process_indices(base_dir, date_str, output_dir, region):
    """
    Processes satellite bands for a specific date and saves NDVI, GNDVI, MSAVI, and EVI.

    Parameters:
    - base_dir (str): folder where the .tiff files are located
    - date_str (str): date in format "YYYY-MM-DD"
    - output_dir (str): folder where results will be saved
    - region (str): region name
    """

    # Compile regex pattern for filenames matching the given date
    pattern = re.compile(r"(B0[2348])_" + re.escape(date_str) + "_" + region + r"\.tiff")

    # Dictionary to store band file paths
    bands = {}

    # Search for files that match the date
    for filename in os.listdir(base_dir):
        match = pattern.match(filename)
        if match:
            band = match.group(1)
            bands[band] = os.path.join(base_dir, filename)

    # Check if all required bands are present
    if not all(b in bands for b in ['B02', 'B03', 'B04', 'B08']):
        print(f"❌ Missing bands for date {date_str}")
        return

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open each band file
        with rasterio.open(bands['B02']) as src_b2, \
             rasterio.open(bands['B03']) as src_b3, \
             rasterio.open(bands['B04']) as src_b4, \
             rasterio.open(bands['B08']) as src_b8:

            # Read band data as float32
            b2 = src_b2.read(1).astype("float32")
            b3 = src_b3.read(1).astype("float32")
            b4 = src_b4.read(1).astype("float32")
            b8 = src_b8.read(1).astype("float32")

            # Compute vegetation indices
            ndvi = (b8 - b4) / (b8 + b4 + 1e-6)  # add small constant to avoid division by zero
            gndvi = (b8 - b3) / (b8 + b3 + 1e-6)
            msavi = (2 * b8 + 1 - ((2 * b8 + 1) ** 2 - 8 * (b8 - b4)) ** 0.5) / 2
            evi = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)

            # Update raster profile to save results
            profile = src_b4.profile
            profile.update(dtype=rasterio.float32, count=1)

            # Save indices as GeoTIFFs
            indices = {'NDVI': ndvi, 'GNDVI': gndvi, 'MSAVI': msavi, 'EVI': evi}
            for index_name, index_data in indices.items():
                output_path = os.path.join(output_dir, f"{index_name}_{date_str}_{region}.tiff")
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(index_data.astype(rasterio.float32), 1)

            print(f"✅ Indices computed and saved for date {date_str}")

    except rasterio.errors.RasterioIOError as e:
        print(f"⚠️ Error opening a file for date {date_str}: {e}")

# ---------------------------------------------------------------------
# Create pixels cropland mask
# ---------------------------------------------------------------------
def generate_cropland_mask(region: str, input_dir: str, output_dir: str, date: str, tolerance: float = 1.7):
    """
    Generate a cropland mask (TIFF) from a raster and a shapefile of crop fields.

    Args:
        region (str): Region name
        input_dir (str): Base directory containing the input data.
        output_dir (str): Directory where the results will be saved.
        date (str): Date of the NDVI file in format 'YYYY-MM-DD'.

    """

    # Input and output paths
    ndvi_filename = f"NDVI_{date}_{region}.tiff"
    ndvi_path = os.path.join(output_dir,ndvi_filename)
    shp_path = os.path.join(input_dir, "croplands_"+ region + ".dbf")

    # Output files
    shp_path_region = os.path.join(output_dir, f"croplands_{region}.shp")
    mask_path = os.path.join(output_dir, f"mask_croplands_{region}.tif")

    os.makedirs(os.path.dirname(shp_path_region), exist_ok=True)

    # ------------------------------------------------------------------------------------------
    # Filter croplands that are fully inside the raster bounds
    shp_data = gpd.read_file(shp_path)
    with rasterio.open(ndvi_path) as src:
        raster_bounds = src.bounds
        raster_polygon = gpd.GeoSeries([box(*raster_bounds)], crs=src.crs)

    filtered_polygons = shp_data[shp_data.geometry.within(raster_polygon.iloc[0])]
    filtered_polygons.to_file(shp_path_region)

    # ------------------------------------------------------------------------------------------
    # Read raster metadata to build the mask
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        ndvi_shape = src.shape
        raster_bounds = box(*src.bounds)

    mask = np.zeros(ndvi_shape, dtype=np.uint8)

    # Loop through filtered polygons and refine mask with tolerance
    with fiona.open(shp_path_region, "r") as shapefile:
        for feature in shapefile:
            polygon = shape(feature["geometry"])

            if raster_bounds.contains(polygon):
                initial_mask = geometry_mask(
                    [polygon], transform=transform, invert=True, out_shape=ndvi_shape
                )

                coords = np.argwhere(initial_mask)
                for r, c in coords:
                    minx, miny = rasterio.transform.xy(transform, r, c)
                    maxx, maxy = rasterio.transform.xy(transform, r + 1, c + 1)

                    pixel_geom = box(minx, miny, maxx, maxy)
                    scaled_pixel_geom = scale(pixel_geom, xfact=tolerance, yfact=tolerance, origin="center")

                    if polygon.contains(scaled_pixel_geom):
                        mask[r, c] = 1
            else:
                print(f"Polygon {feature['id']} discarded (outside raster bounds).")

    # ------------------------------------------------------------------------------------------
    # Save mask as GeoTIFF
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    print(f"Refined mask saved at {mask_path}")
    return mask_path

# ---------------------------------------------------------------------
# Index croplands
# ---------------------------------------------------------------------

def add_ids_to_croplands(input_dir: str, region: str):
    """
    Add a unique ID column to a croplands shapefile and save it as a new file.

    Parameters
    ----------
    input_dir : str
        Directory containing the croplands shapefile.
    region : str
        Name of the region.

    Returns
    -------
    str
        Path to the newly saved shapefile with IDs.
    """

    # Define input shapefile path
    input_path = os.path.join(input_dir, f"croplands_{region}.shp")

    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(input_path)

    # Add a new 'id' column with unique IDs for each polygon
    gdf['id'] = range(1, len(gdf) + 1)

    # Define the output shapefile path
    output_path = os.path.join(input_dir, f"croplands_{region}_id.shp")

    # Save the new shapefile with IDs
    gdf.to_file(output_path)

    return output_path


# ---------------------------------------------------------------------
# Create df
# ---------------------------------------------------------------------

def create_df_crops(region: str, input_dir: str, date: str):
    """
    Calculate crop-level statistics (median) of remote sensing indices
    for a single date using predefined masks for each field.

    Parameters
    ----------
    region : str
        Name of the region
    input_dir : str
        Input directory
    date : str
        Date of the index images to process
    """

    # Paths
    mask_path = os.path.join(input_dir, f"mask_croplands_{region}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")
    output_txt = os.path.join(input_dir, f"output_croplands_{region}_{date}_stats.txt")

    # Read mask
    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        mask_meta = src_mask.meta

    # Read shapefile
    gdf = gpd.read_file(shp_path)

    # Create dictionary with pixel positions for each field
    pixel_positions = {}
    for _, geom in gdf.iterrows():
        field_id = geom['id']
        geom_shape = [geom['geometry']]

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**mask_meta) as temp_ds:
                temp_ds.write(mask_data, 1)

                try:
                    out_image, _ = mask(temp_ds, geom_shape, crop=False, filled=False)
                    rows, cols = np.where(out_image[0] == 1)
                    pixel_positions[field_id] = list(zip(rows, cols))
                except ValueError:
                    pixel_positions[field_id] = []

    # Filter index files for the given date
    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]

    required_indexes = ['NDVI', 'EVI', 'GNDVI', 'MSAVI']
    date_files = {f.split('_')[0]: os.path.join(input_dir, f) for f in index_files}

    if not all(idx in date_files for idx in required_indexes):
        raise ValueError(f"Missing indices for date {date}.")

    # Read index rasters
    index_data = {}
    for idx, path in date_files.items():
        try:
            with rasterio.open(path) as src:
                index_data[idx] = src.read(1)
        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")

    # Calculate stats for each field
    results = []
    for _, geom in gdf.iterrows():
        field_id = geom['id']
        stats = {'date': date, 'id': field_id, 'Crop': geom['Cultiu']}

        if field_id not in pixel_positions or not pixel_positions[field_id]:
            continue

        rows, cols = zip(*pixel_positions[field_id])

        for idx in required_indexes:
            valid_values = index_data[idx][rows, cols]
            valid_values = valid_values[~np.isnan(valid_values)]

            stats[idx] = np.median(valid_values) if valid_values.size > 0 else np.nan

        results.append(stats)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_txt, sep='\t', index=False)

    return df_results


