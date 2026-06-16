#
## spatial_feature_extraction_pipeline.py
#
#  Comprehensive script to execute the spatial feature extraction pipeline for the I-40 corridor project, integrating all previous data analysis steps into a cohesive workflow.
#  
#  This script performs the following key functions:
#  1. Loads and reprojects the HPMS, pronghorn corridors, and
#     elk corridors shapefiles to a common projected CRS (UTM Zone 12N).
#  2. Dynamically identifies the correct column in the HPMS data that contains route
#     identifiers and filters for segments corresponding to I-40 and SR-260.
#  3. Segregates the pronghorn and elk corridor data into high-use
#     migration corridors (GRIDCODE 2.0) and localized foraging stopovers (GRIDCODE 20.0).
#  4. Iteratively computes both binary intersection features and proximity
#     distance features for each filtered highway segment relative to the pronghorn and elk corridor categories
#  5. Compiles the extracted features into a clean tabular format and saves it as a CSV for downstream machine learning applications.
# Note: Ensure that the file paths for the input shapefiles are correctly set to your local directory structure before running this script.
#

import geopandas as gpd
import pandas as pd
import numpy as np
import os

def create_spatial_feature_matrix(hpms_path, pronghorn_path, elk_path, output_csv_path):
    print("[*] Phase 1: Loading and Aligning Spatial Data Horizons...")
    
    target_crs = "EPSG:26912" # Arizona UTM Zone 12N, ideal for distance calculations in this region
    
    if not os.path.exists(hpms_path):
        raise FileNotFoundError(f"Base HPMS file not found at {hpms_path}")

    # Read shapefiles and reproject them to the Arizona UTM grid
    hpms_gdf = gpd.read_file(hpms_path).to_crs(target_crs)
    pronghorn_gdf = gpd.read_file(pronghorn_path).to_crs(target_crs)
    elk_gdf = gpd.read_file(elk_path).to_crs(target_crs)
    
    print(f"[+] Spatial synchronization complete. Target projection: {target_crs}")

    print("[*] Phase 2: Filtering Target Transportation Corridors...")
    # Dynamically locate the column containing route names
    route_col = [col for col in hpms_gdf.columns if 'ROUTE' in col.upper()][0]
    print(f"[*] Identified route logging column: '{route_col}'")
    
    # EXACT STRING MATCH PATTERN:
    # This regex directly targets ADOT's specific internal naming schema 
    # 'I 040' (Mainline I-40), 'S 260' (Mainline SR-260), and 'SB040' (I-40 Business loops)
    matching_pattern = r'I 040|S 260|SB040'
    
    filtered_roads = hpms_gdf[hpms_gdf[route_col].str.contains(matching_pattern, case=False, na=False)].copy()
    print(f"[+] Successfully isolated {len(filtered_roads)} micro-segments along the target corridors.")

    if len(filtered_roads) == 0:
        print("[!] WARNING: Still isolated 0 roads. Double-check your column matching logic.")
        return None

    print("[*] Phase 3: Segmenting Species Vectors by Biological Urgency (GRIDCODE)...")
    # Separate core high-use migration corridors (2.0) from localized foraging stopovers (20.0)
    p_high_use = pronghorn_gdf[pronghorn_gdf['GRIDCODE'] == 2.0]
    p_stopover = pronghorn_gdf[pronghorn_gdf['GRIDCODE'] == 20.0]
    
    e_high_use = elk_gdf[elk_gdf['GRIDCODE'] == 2.0]
    e_stopover = elk_gdf[elk_gdf['GRIDCODE'] == 20.0]

    print("[*] Phase 4: Executing Geometric Matrix Intersection Loop...")
    tabular_features = []

    # Loop through every single 0.1-mile highway segment found
    for idx, row in filtered_roads.iterrows():
        segment_geom = row.geometry
        segment_centroid = segment_geom.centroid
        
        # A. Binary Intersection Extraction (Direct Spatial Overlap)
        has_p_high = 1 if p_high_use.intersects(segment_geom).any() else 0
        has_p_stop = 1 if p_stopover.intersects(segment_geom).any() else 0
        has_e_high = 1 if e_high_use.intersects(segment_geom).any() else 0
        has_e_stop = 1 if e_stopover.intersects(segment_geom).any() else 0
        
        # B. Proximity Vector Mapping (Shortest distance in meters from segment center to polygon boundary)
        # Note: If a segment intersects a polygon natively, its distance is computed as 0.0
        dist_p_high = p_high_use.distance(segment_centroid).min() if not p_high_use.empty else np.nan
        dist_p_stop = p_stopover.distance(segment_centroid).min() if not p_stopover.empty else np.nan
        dist_e_high = e_high_use.distance(segment_centroid).min() if not e_high_use.empty else np.nan
        dist_e_stop = e_stopover.distance(segment_centroid).min() if not e_stopover.empty else np.nan

        # Package the outputs into a row dictionary
        feature_row = {
            'segment_id': row.get('OBJECTID', idx),
            'highway_name': row[route_col].strip(), # strip out hanging whitespace
            'segment_length_m': round(segment_geom.length, 2),
            'pronghorn_overlap_high_use': has_p_high,
            'pronghorn_overlap_stopover': has_p_stop,
            'elk_overlap_high_use': has_e_high,
            'elk_overlap_stopover': has_e_stop,
            'meters_to_pronghorn_highway': round(dist_p_high, 2) if not np.isnan(dist_p_high) else -1,
            'meters_to_pronghorn_stopover': round(dist_p_stop, 2) if not np.isnan(dist_p_stop) else -1,
            'meters_to_elk_highway': round(dist_e_high, 2) if not np.isnan(dist_e_high) else -1,
            'meters_to_elk_stopover': round(dist_e_stop, 2) if not np.isnan(dist_e_stop) else -1
        }
        tabular_features.append(feature_row)

    # Compile the array of dictionaries into a clean Pandas DataFrame and write to disk
    output_df = pd.DataFrame(tabular_features)
    output_df.to_csv(output_csv_path, index=False)
    print(f"[++] SUCCESS: Engineered matrix saved with shape {output_df.shape} to {output_csv_path}")
    
    return output_df

# Execute the pipeline using your local directory setup
if __name__ == "__main__":
    HPMS_FILE = r"C:\Users\Lanie\Documents\Classes\2026\Spring Semester\Applied Informatics 1\Final Project\Data\ADOT_2024_Pavement_Conditions\ADOT_2024_Pavement_Conditions.shp"
    PRONGHORN_FILE = r"C:\Users\Lanie\Documents\Classes\2026\Spring Semester\Applied Informatics 1\Final Project\Data\PR_AZ_NorthI40_Corridors_Ver1_2021\PR_AZ_NorthI40_Corridors_Ver1_2021.shp"  # Ensure these filenames match your directory
    ELK_FILE = r"C:\Users\Lanie\Documents\Classes\2026\Spring Semester\Applied Informatics 1\Final Project\Data\ELK_AZ_NorthI40_Corridors_Ver1_2021\ELK_AZ_NorthI40_Corridors_Ver1_2021.shp"
    OUTPUT_FILE = "spatial_features_base.csv"
    
    create_spatial_feature_matrix(HPMS_FILE, PRONGHORN_FILE, ELK_FILE, OUTPUT_FILE)