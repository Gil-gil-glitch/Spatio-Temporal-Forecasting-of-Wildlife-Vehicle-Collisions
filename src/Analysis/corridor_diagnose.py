#
## corridor_diagnose.py
#
# Diagnostic script to investigate issues with reading the animal corridor shapefile
#
#

import os
import geopandas as gpd
import pyogrio

path = r"xxxx.shp" # <- REPLACE WITH ACTUAL PATH TO ELK CORRIDORS SHAPEFILE

print("=" * 60)
print("BASIC INFORMATION")
print("=" * 60)

print("File exists:", os.path.exists(path))

print("\nGeoPandas version:", gpd.__version__)
print("Pyogrio version:", pyogrio.__version__)

print("\nFile sizes:")

for ext in [".shp", ".shx", ".dbf", ".prj"]:
    f = path.replace(".shp", ext)
    if os.path.exists(f):
        print(f"{ext}: {os.path.getsize(f):,} bytes")
    else:
        print(f"{ext}: MISSING")

print("\nDirectory contents:")
folder = os.path.dirname(path)

for f in sorted(os.listdir(folder)):
    print("  ", f)

print("\n" + "=" * 60)
print("TEST 1: pyogrio.list_layers")
print("=" * 60)

try:
    layers = pyogrio.list_layers(path)
    print("SUCCESS")
    print(layers)
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(e)

print("\n" + "=" * 60)
print("TEST 2: Read first feature")
print("=" * 60)

try:
    gdf = pyogrio.read_dataframe(path, max_features=1)
    print("SUCCESS")
    print(gdf.head())
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(e)

print("\n" + "=" * 60)
print("TEST 3: Read entire file with GeoPandas")
print("=" * 60)

try:
    gdf = gpd.read_file(path)
    print("SUCCESS")
    print("Features:", len(gdf))
    print("CRS:", gdf.crs)
    print("Geometry Types:", gdf.geometry.type.unique())
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(e)

print("\n" + "=" * 60)
print("TEST 4: Open raw SHP header")
print("=" * 60)

try:
    with open(path, "rb") as f:
        header = f.read(100)

    print("SUCCESS")
    print("Header length:", len(header))
    print("First 32 bytes:", header[:32])
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(e)