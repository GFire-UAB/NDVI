import openeo
from datetime import datetime, timedelta
import os


def download_sentinel2_bands_for_date(date: datetime, spatial_extent: dict, output_dir):
    """
    Downloads Sentinel-2 L2A bands B02, B03, B04, B08 for a single date and applies SCL masking.

    Parameters:
    - date: datetime object representing the date to download
    - spatial_extent: dict with keys 'west', 'south', 'east', 'north'
    - output_dir: directory to save files
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bands = ["B02", "B03", "B04", "B08"]

    # Connect to Copernicus backend
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    con.authenticate_oidc()

    end_date = date + timedelta(days=1)

    # Load Sentinel-2 collection for the given date
    datacube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
        bands=bands + ["SCL"],
        max_cloud_cover=100,
    )

    # Mask invalid pixels based on SCL band
    scl_band = datacube.band("SCL")
    mask_valid = ~((scl_band == 4) | (scl_band == 5) | (scl_band == 6) | (scl_band == 7))

    # Apply mask and download each band
    for band in bands:
        masked_band = datacube.band(band).mask(mask_valid)
        output_path = os.path.join(output_dir, f"{band}_{date.strftime('%Y-%m-%d')}_east.tiff")
        masked_band.download(output_path)

    return [os.path.join(output_dir, f"{band}_{date.strftime('%Y-%m-%d')}_east.tiff") for band in bands]


# Example call
spatial_extent_example = {
    "west": 1.41,
    "south": 41.10,
    "east": 3.36,
    "north": 42.68
}

downloaded_files = download_sentinel2_bands_for_date(datetime(2023, 11, 8), spatial_extent_example, output_dir:"../data/processed")

