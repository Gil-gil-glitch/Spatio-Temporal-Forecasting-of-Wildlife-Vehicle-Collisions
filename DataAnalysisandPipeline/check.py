#
## check.py
#
#  Functions for auditing migration corridor data
#

import geopandas as gpd

def audit_corridor_layer(file_path, layer_name):
    """"
    Audits a migration corridor layer for:
    - CRS (should be projected for distance calculations)
    - Geometry types (should be LineString or Polygon)
    - Attribute columns (should include use classifications)    
    """
    print(f"=== Auditing Layer: {layer_name} ===")
    gdf = gpd.read_file(file_path)
    
    print(f"[*] Native CRS: {gdf.crs}")
    if gdf.crs.is_geographic:
        print("[!] WARNING: Data is in a geographic CRS (lat/lon). Reprojecting to UTM Zone 12N (EPSG:26912)...")
        gdf = gdf.to_crs(epsg=26912)
    
    geom_types = gdf.geometry.type.unique()
    print(f"[*] Geometry Types Found: {geom_types}")
    
    print("[*] Column Names:")
    print(gdf.columns.tolist())
    
    for col in ['Layer', 'Use_Class', 'type', 'status']:
        if col in gdf.columns:
            print(f"[*] Unique categories in '{col}':")
            print(gdf[col].value_counts())
            
    print(f"[*] Total Vector Features: {len(gdf)}")
    print("=====================================\n")
    return gdf

# usage:

pronghorn_corridors = audit_corridor_layer("xxx.shp", "Pronghorn Corridors") # <- REPLACE WITH ACTUAL PATH TO PRONGHORN CORRIDORS SHAPEFILE

print("==========================================\n")

elk_corridors = audit_corridor_layer("xxx.shp", "Elk Corridors") # <- REPLACE WITH ACTUAL PATH TO ELK CORRIDORS SHAPEFILE