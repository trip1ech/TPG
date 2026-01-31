import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

# --- CONFIG ---
DATA_DIR = Path("../data") # Adjust to your path
STOP_FILE = DATA_DIR / "stop_df.csv"
SHAPE_FILE = DATA_DIR / "TPG_LIGNES.shp"
OUTPUT_DIR = Path("app_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_geometry_data(seg_df):
    """
    Args:
        seg_df: Your main dataframe (seg) containing 'from_stop', 'to_stop', 'SegmentKey'
    """
    print("📍 Processing Stop Coordinates...")
    
    # 1. Load Stop Data
    stops = pd.read_csv(STOP_FILE)
    
    # 2. Convert from Swiss Grid (LV03) to Lat/Lon (WGS84)
    #    Create a GeoDataFrame
    gdf_stops = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops.X, stops.Y),
        crs="EPSG:21781" # Old Swiss Grid
    )
    
    #    Reproject to WGS84 (Lat/Lon)
    gdf_stops = gdf_stops.to_crs("EPSG:4326")
    
    #    Extract Lon/Lat
    gdf_stops["lon"] = gdf_stops.geometry.x
    gdf_stops["lat"] = gdf_stops.geometry.y
    
    # 3. Create a Lookup Dictionary (CodeLong -> [Lon, Lat])
    #    MNLP matches your CodeLong
    stop_lookup = gdf_stops.set_index("MNLP")[["lon", "lat"]].T.to_dict("list")
    
    print(f"   -> Loaded {len(stop_lookup)} stops.")

    # 4. Generate Map Segments (The "Canonical" Routes)
    #    We only want unique segments to keep the file small
    print("🗺️ Building Segment Geometries...")
    
    #    Identify Canonical Segments (Most frequent pattern per line/dir)
    pattern_counts = seg_df.groupby(["line", "dir"])["SegmentKey"].value_counts().reset_index(name="count")
    #    Filter: Keep segments appearing in >50 trips (removes rare deviations)
    canonical_segments = pattern_counts[pattern_counts["count"] > 50]["SegmentKey"].unique()
    
    #    Get unique segment definitions
    unique_segs = seg_df[seg_df["SegmentKey"].isin(canonical_segments)].drop_duplicates("SegmentKey")
    
    map_features = []
    missing_coords = 0
    
    for _, row in unique_segs.iterrows():
        origin = row["from_stop"]
        dest = row["to_stop"]
        
        # Check if we have coords for both stops
        if origin in stop_lookup and dest in stop_lookup:
            coord_start = stop_lookup[origin] # [Lon, Lat]
            coord_end = stop_lookup[dest]     # [Lon, Lat]
            
            map_features.append({
                "segmentId": row["SegmentKey"],
                "line": str(row["line"]),
                "dir": str(row["dir"]),
                # Deck.gl expects: [[lon1, lat1], [lon2, lat2]]
                "path": [coord_start, coord_end] 
            })
        else:
            missing_coords += 1

    print(f"   -> Created {len(map_features)} mappable segments.")
    if missing_coords > 0:
        print(f"   ⚠️ Warning: {missing_coords} segments skipped due to missing stop coordinates.")

    # 5. Export to JSON
    out_path = OUTPUT_DIR / "map_geometry.json"
    with open(out_path, "w") as f:
        json.dump(map_features, f)
    
    print(f"✅ Saved Map Geometry to {out_path}")

    # --- OPTIONAL: Convert Shapefile for Background Layer ---
    if SHAPE_FILE.exists():
        print("📐 Processing TPG Network Shapes (Background)...")
        try:
            gdf_shapes = gpd.read_file(SHAPE_FILE)
            
            # Check current CRS, usually shapefiles have it embedded
            if gdf_shapes.crs is None:
                # Assuming LV03 if undefined, based on stops
                gdf_shapes.set_crs("EPSG:21781", inplace=True) 
            
            # Convert to WGS84
            gdf_shapes = gdf_shapes.to_crs("EPSG:4326")
            
            # Export to GeoJSON
            gdf_shapes.to_file(OUTPUT_DIR / "tpg_network.geojson", driver="GeoJSON")
            print(f"✅ Saved Background Network to {OUTPUT_DIR / 'tpg_network.geojson'}")
        except Exception as e:
            print(f"   ❌ Failed to process shapefile: {e}")

# --- USAGE ---
# Ensure you have your 'seg' dataframe loaded in memory from previous steps
if 'seg' in locals():
    process_geometry_data(seg)
else:
    print("❌ Error: 'seg' dataframe not found. Run your data ingestion code first.")