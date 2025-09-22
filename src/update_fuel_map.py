import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape, box
from shapely.affinity import scale
import geopandas as gpd
import fiona

# ---------------------------------------------------------------------
# Split shapefile into planted vs not planted
# ---------------------------------------------------------------------
def split_shapefile_by_planted(base_path, region, date):
    """
    Splits a shapefile into two separate files based on the 'planted' column
    from a prediction text file.
    """
    txt_file = os.path.join(base_path, f"output_croplands_{region}_{date}_prediction.txt")
    shp_file = os.path.join(base_path, f"croplands_{region}_id.shp")

    df = pd.read_csv(txt_file, sep="\t")
    gdf = gpd.read_file(shp_file)

    gdf = gdf.merge(df, on="id")

    gdf_planted = gdf[gdf["planted"] == 1]
    gdf_not_planted = gdf[gdf["planted"] == 0]

    out_planted = os.path.join(base_path, f"croplands_{region}_{date}_1.shp")
    out_not_planted = os.path.join(base_path, f"croplands_{region}_{date}_0.shp")

    gdf_planted.to_file(out_planted)
    gdf_not_planted.to_file(out_not_planted)

    print(f"✅ Saved: {out_planted} and {out_not_planted}")


# ---------------------------------------------------------------------
# Replace raster values inside shapefile polygons
# ---------------------------------------------------------------------
def replace_pixels_within_shapefile(tif_path, shp_path, output_path, new_value):
    """
    Replaces all pixels in a raster (TIFF) that fall inside a shapefile with a specified value.
    """
    # Load shapefile
    shapefile = gpd.read_file(shp_path)

    # Open the raster
    with rasterio.open(tif_path) as src:
        raster_data = src.read(1)
        out_meta = src.meta.copy()

        # Convert geometries to a list of mappings
        shapes = [feature["geometry"] for feature in shapefile.__geo_interface__["features"]]

        # Create a mask: True = inside the shapes, False = outside
        mask_inside = geometry_mask(shapes, transform=src.transform, invert=True,
                                    out_shape=(src.height, src.width))

        # Replace pixels **inside** the shapefile
        raster_data[mask_inside] = new_value

        # Save the modified raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)


