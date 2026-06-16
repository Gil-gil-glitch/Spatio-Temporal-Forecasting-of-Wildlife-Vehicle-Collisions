#
## gridcode.py
#
#  Script to examine the GRIDCODE attribute in the corridor shapefiles to understand the distribution of corridor classifications
#


import geopandas as gpd

def examine_gridcodes(file_path, name):
    """
    Examines the GRIDCODE attribute in a shapefile and prints the value counts
    """
    gdf = gpd.read_file(file_path)
    print(f"\n=== {name} GRIDCODE Breakdown ===")
    print(gdf['GRIDCODE'].value_counts().sort_index())
    print(f"Total Rows: {len(gdf)}")


examine_gridcodes("xxx.shp", "Pronghorn") #<- Adjust path and name as needed

examine_gridcodes("xxx.shp", "Elk") #<- Adjust path and name as needed