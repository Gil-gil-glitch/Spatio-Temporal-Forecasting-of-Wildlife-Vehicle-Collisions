import geopandas as gpd

# Absolute path to your local 2024 ADOT HPMS shapefile
hpms_path = r"C:\Users\Lanie\Documents\Classes\2026\Spring Semester\Applied Informatics 1\Final Project\Data\ADOT_2024_Pavement_Conditions\ADOT_2024_Pavement_Conditions.shp"

print("[*] Ingesting shapefile headers to inspect RouteID formats...")
gdf = gpd.read_file(hpms_path)

# Extract unique values, drop missing entries, and cast to a standard sorted Python list
unique_routes = sorted(list(gdf['RouteID'].dropna().unique()))

print(f"[+] Found {len(unique_routes)} unique route identifiers.")
print("\n=== Sample of First 50 Route IDs ===")
for route in unique_routes[:50]:
    print(f" - {route}")

print("\n=== Keyword Search Sample ===")
# Scan the sorted array specifically for strings containing '40' or '260'
matches = [r for r in unique_routes if '40' in str(r) or '260' in str(r)]
print(f"Found {len(matches)} potential matches containing '40' or '260':")
for match in matches[:20]:
    print(f" -> {match}")