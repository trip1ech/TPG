import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from pyproj import Transformer

# Set page config
st.set_page_config(layout="wide", page_title="TPG Segment Performance Dashboard")

# Constants & Config
# Resolve paths relative to the script to ensure they work regardless of CWD
current_file = Path(__file__).resolve()
APP_DIR = current_file.parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "app_data"

RAW_DATA_PATH = DATA_DIR / "test_preds.parquet"
STOPS_PATH = DATA_DIR / "stops_df.csv"
WEATHER_PATH = DATA_DIR / "weather-info.csv"

# fallback if app_data doesn't exist (e.g. running from root with different structure)
if not DATA_DIR.exists():
    # Try current directory or simple relative path
    DATA_DIR = Path("app_data")
    STOPS_PATH = col_path = DATA_DIR / "stops_df.csv"
    WEATHER_PATH = DATA_DIR / "weather-info.csv"

# Coordinate Transformer (EPSG:21781 -> EPSG:4326)
# Geneva uses CH1903 / LV03 (21781) usually, sometimes LV95 (2056). 
# Assuming 21781 based on prompt.
transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)

@st.cache_data
def load_stops():
    """Load and process stop coordinates (convert to WGS84)."""
    if not STOPS_PATH.exists():
        st.error(f"Stops file not found at {STOPS_PATH}")
        return pd.DataFrame()
    
    stops = pd.read_csv(STOPS_PATH)
    stops.columns = stops.columns.str.strip()
    
    # Ensure columns exist
    if not {"MNLP", "X", "Y"}.issubset(stops.columns):
        st.error(f"Stops file missing required columns. Found: {stops.columns}")
        return pd.DataFrame()
        
    stops["MNLP"] = stops["MNLP"].astype(str).str.strip()
    stops = stops.dropna(subset=["X", "Y"]).drop_duplicates("MNLP")
    
    # Project coords
    lons, lats = transformer.transform(stops["X"].values, stops["Y"].values)
    stops["lon"] = lons
    stops["lat"] = lats
    
    return stops.set_index("MNLP")[["lon", "lat"]]

@st.cache_data
def load_weather():
    """Load hourly weather data."""
    if not WEATHER_PATH.exists():
        return pd.DataFrame()
        
    w = pd.read_csv(WEATHER_PATH)
    # Flexible column mapping
    col_map = {
        "time": "time",
        "temperature_2m (°C)": "temp",
        "temperature_2m": "temp",
        "rain (mm)": "rain",
        "rain": "rain",
        "snowfall (cm)": "snow",
        "snowfall": "snow"
    }
    w = w.rename(columns=col_map)
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time").sort_index()

@st.cache_data
def load_predictions():
    """Load prediction data from parquet."""
    # Look for separate stage files in app_data
    files = list(DATA_DIR.glob("pred_stage*_test.parquet"))
    
    if files:
        dfs = []
        for f in files:
            d = pd.read_parquet(f)
            # Ensure model name if missing (extract from filename)
            if "model" not in d.columns:
                name = f.stem.replace("pred_", "").replace("_test", "")
                d["model"] = name
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
    
    # Fallback to single file
    elif RAW_DATA_PATH.exists():
        df = pd.read_parquet(RAW_DATA_PATH)
        
    else:
        st.error(f"Prediction data not found in {DATA_DIR}")
        return pd.DataFrame()
        
    # Minimal required processing
    df["link_start_time"] = pd.to_datetime(df["link_start_time"])
    df["line"] = df["line"].astype(str)
    df["dir"] = df["dir"].astype(str)
    
    # Helper cols
    df["hour"] = df["link_start_time"].dt.hour
    df["dow"] = df["link_start_time"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6])
    
    # Error metrics
    if "residual" not in df.columns and "y_true" in df.columns and "y_pred" in df.columns:
        df["residual"] = df["y_pred"] - df["y_true"]
    if "abs_err" not in df.columns and "y_true" in df.columns and "y_pred" in df.columns:
        df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    if "pct_err" not in df.columns and "y_true" in df.columns and "y_pred" in df.columns:
        # Avoid division by zero
        df["pct_err"] = df["abs_err"] / df["y_true"].replace(0, np.nan) 

    # Parse segments
    if "from_stop" not in df.columns:
        splits = df["SegmentKey"].astype(str).str.split("→", expand=True, n=1)
        df["from_stop"] = splits[0].str.strip()
        df["to_stop"] = splits[1].str.strip() if splits.shape[1] > 1 else ""

    # Merge Weather if available
    w_df = load_weather()
    if not w_df.empty:
        # Round to nearest hour for join
        df["time_h"] = df["link_start_time"].dt.round("H")
        df = df.join(w_df, on="time_h", how="left")

    return df

def apply_filters(df, filters):
    mask = pd.Series(True, index=df.index)
    
    if filters.get("models"):
        mask &= df["model"].isin(filters["models"])
        
    if filters.get("lines"):
        mask &= df["line"].isin(filters["lines"])
        
    if filters.get("dirs"):
        mask &= df["dir"].isin(filters["dirs"])
        
    # Date range
    if filters.get("date_range"):
        d_mask = (df["link_start_time"].dt.date >= filters["date_range"][0]) & \
                 (df["link_start_time"].dt.date <= filters["date_range"][1])
        mask &= d_mask
    
    # Hour range
    if filters.get("hour_range"):
        h_mask = (df["hour"] >= filters["hour_range"][0]) & \
                 (df["hour"] <= filters["hour_range"][1])
        mask &= h_mask
    
    # Day type
    if filters.get("day_type"):
        if filters["day_type"] == "Weekday":
            mask &= (~df["is_weekend"])
        elif filters["day_type"] == "Weekend":
            mask &= (df["is_weekend"])
        
    # Weather Filters
    if filters.get("rain_only"):
        mask &= (df["rain"] > 0)
    if filters.get("snow_only"):
        mask &= (df["snow"] > 0)
        
    return df.loc[mask]

def agg_segments(filtered_df, stops_xy, universe_df=None):
    """Aggregate stats by SegmentKey for the map."""
    
    # 1. Define the Universe of Segments to show
    if universe_df is not None and not universe_df.empty:
        # Use simple aggregation to get metadata for ALL segments in the selected lines/dirs
        # We grab the FIRST occurrence of static attributes
        agg_base = universe_df.groupby("SegmentKey").agg({
            "from_stop": "first",
            "to_stop": "first",
            "line": lambda x: list(x.unique()),
            # If distance exists, average it
            **({"distance_m": "mean"} if "distance_m" in universe_df.columns else {})
        })
        if "distance_m" in agg_base.columns:
            agg_base = agg_base.rename(columns={"distance_m": "distance"})
            
    elif not filtered_df.empty:
        agg_base = pd.DataFrame(index=filtered_df["SegmentKey"].unique())
        # We'll fill metadata below if universe wasn't passed, but this path is fallback
        agg_base["from_stop"] = filtered_df.groupby("SegmentKey")["from_stop"].first()
        agg_base["to_stop"] = filtered_df.groupby("SegmentKey")["to_stop"].first()
        agg_base["line"] = filtered_df.groupby("SegmentKey")["line"].apply(list)
    else:
        return pd.DataFrame() 

    # 2. Calculate Metrics on the Filtered Subset
    if not filtered_df.empty:
        metrics = filtered_df.groupby("SegmentKey").agg(
            n=("y_true", "size"),
            mean_tt=("y_true", "mean"),
            pred_tt=("y_pred", "mean"),
            base_tt=("y_base", "mean"),
            mae=("abs_err", "mean"),
            mape=("pct_err", "mean"),
            std_tt=("y_true", "std"),
        )
        
        # Join metrics to base (Left Join preserves topology)
        agg = agg_base.join(metrics, how="left")
    else:
        agg = agg_base
        # Init metric columns with NaNs
        for c in ["n", "mean_tt", "pred_tt", "mae", "mape", "std_tt"]:
            agg[c] = np.nan
            
    # Fill N with 0 for missing
    agg["n"] = agg["n"].fillna(0)
    agg = agg.reset_index()

    # Derived Metrics
    agg["mape"] = agg["mape"] * 100
    agg["cv"] = agg["std_tt"] / agg["mean_tt"]
    
    # Geometry
    agg = agg.join(stops_xy.add_prefix("from_"), on="from_stop")
    agg = agg.join(stops_xy.add_prefix("to_"), on="to_stop")
    
    # Filter valid coords (absolute requirement for mapping, but warn if many dropped?)
    len_pre = len(agg)
    agg = agg.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"])
    # Simple warning if we lost segments
    if len(agg) < len_pre:
        # st.toast(f"Warning: {len_pre - len(agg)} segments dropped due to missing coordinates.")
        pass
        
    # Improve metric: improvement over baseline
    if not filtered_df.empty:
        agg_base_mae = filtered_df.groupby("SegmentKey").apply(lambda x: (x["y_true"] - x["y_base"]).abs().mean())
        # Map to agg
        agg["base_mae"] = agg["SegmentKey"].map(agg_base_mae)
    else:
        agg["base_mae"] = np.nan

    agg["imp_sec"] = agg["base_mae"] - agg["mae"]
    agg["imp_pct"] = (agg["imp_sec"] / agg["base_mae"]) * 100
    
    # Speed
    if "distance" in agg.columns:
        agg["speed_kmh"] = (agg["distance"] / agg["mean_tt"]) * 3.6
    else:
        agg["speed_kmh"] = np.nan
        
    return agg


# --- APP START ---
stops_xy = load_stops()
weather_df = load_weather()
preds_df = load_predictions()

if preds_df.empty:
    st.error("No prediction data loaded. Check paths.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Filters")
    
    all_models = sorted(preds_df["model"].unique())
    sel_models = st.multiselect("Models", all_models, default=all_models[:1])
    
    all_lines = sorted(preds_df["line"].unique())
    sel_lines = st.multiselect("Lines", all_lines, default=all_lines[:5])
    
    all_dirs = sorted(preds_df["dir"].unique())
    sel_dirs = st.multiselect("Direction", all_dirs, default=all_dirs)
    
    min_date = preds_df["link_start_time"].min().date()
    max_date = preds_df["link_start_time"].max().date()
    sel_dates = st.slider("Date Range", min_date, max_date, (min_date, max_date))
    
    sel_hours = st.slider("Hour of Day", 0, 23, (6, 22))
    
    sel_daytype = st.radio("Day Type", ["All", "Weekday", "Weekend"])
    
    st.markdown("### Weather Conditions")
    rain_o = st.checkbox("Rainy Intervals Only")
    snow_o = st.checkbox("Snowy Intervals Only")

    # Build filter dict
    filters = {
        "models": sel_models,
        "lines": sel_lines,
        "dirs": sel_dirs,
        "date_range": sel_dates,
        "hour_range": sel_hours,
        "day_type": sel_daytype,
        "rain_only": rain_o,
        "snow_only": snow_o
    }

# --- GLOBAL DATA ---
# Split filters into Structural (Line/Dir) and Temporal (Date/Weather)
# This allows us to show the full "Network Topology" (Structural) even if data is missing for specific times (Temporal).
struct_filters = {k: v for k, v in filters.items() if k in ["models", "lines", "dirs"]}
temporal_filters = {k: v for k, v in filters.items() if k not in ["models", "lines", "dirs"]}

df_struct = apply_filters(preds_df, struct_filters)
df_filtered = apply_filters(df_struct, temporal_filters)

st.sidebar.markdown(f"**{len(df_filtered):,}** records selected ({len(df_filtered)/len(preds_df):.1%})")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🗺️ Map Overview", "📊 Segment Detail", "📈 Line Analysis"])

# --- TAB 1: MAP ---
with tab1:
    st.header("Segment Performance Map")
    
    if df_struct.empty:
        st.write("No data found for selected Line/Model.")
    else:
        col_m1, col_m2 = st.columns([1, 3])
        
        # Aggregate with topology preservation
        seg_agg = agg_segments(df_filtered, stops_xy, universe_df=df_struct)

        with col_m1:
            metric = st.selectbox("Color By", 
                ["mape", "mae", "mean_tt", "cv", "imp_pct", "speed_kmh", "n"], 
                index=0,
                format_func=lambda x: x.upper().replace("_", " ")
            )
            
            # Dynamic Quantiles for defaults
            if not seg_agg.empty and metric in seg_agg.columns:
                q_min = float(seg_agg[metric].quantile(0.05))
                q_max = float(seg_agg[metric].quantile(0.95))
                # Ensure rational defaults
                if pd.isna(q_min): q_min = 0.0
                if pd.isna(q_max): q_max = 100.0
                if q_max <= q_min: q_max = q_min + 1.0
            else:
                q_min, q_max = 0.0, 100.0

            # Map settings
            scale_min = st.number_input("Scale Min", value=q_min, format="%.2f")
            scale_max = st.number_input("Scale Max", value=q_max, format="%.2f")
        
        # Color Scale Logic
        def get_color(val, vmin, vmax, cmap="r"):
            if pd.isna(val):
                return [128, 128, 128, 100] # Grey for missing data
            
            # Simple linear interp 0-255
            norm = np.clip((val - vmin) / (vmax - vmin + 1e-9), 0, 1)
            
            if pd.isna(norm):
                 return [128, 128, 128, 100]

            # Red to Green (Improvement/Speed) or Green to Red (Error)
            if metric in ["imp_pct", "speed_kmh"]: # Higher is better (Red -> Green)
                r = 255 * (1 - norm)
                g = 255 * norm
            else: # Lower is better (Green -> Red)
                r = 255 * norm
                g = 255 * (1 - norm)
            
            # Ensure valid int conversion
            try:
                return [int(r), int(g), 0, 180]
            except ValueError:
                return [128, 128, 128, 100]

        if not seg_agg.empty:
            seg_agg["color"] = seg_agg[metric].apply(lambda x: get_color(x, scale_min, scale_max))
            # Tweak width: Thinner segments [2, 12] width range
            seg_agg["width"] = np.clip(seg_agg["n"] / seg_agg["n"].max() * 20, 2, 12)
            
            # Format metric for tooltip
            seg_agg["metric_formatted"] = seg_agg[metric].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

            # Create Path Layer
            layer = pdk.Layer(
                "LineLayer",
                seg_agg,
                id="segments_layer",         # <--- IMPORTANT: ID for selection
                get_source_position="[from_lon, from_lat]",
                get_target_position="[to_lon, to_lat]",
                get_color="color",
                get_width="width",
                pickable=True,
                auto_highlight=True,
            )
            
            # Stops Layer
            # Filter stops involved in these segments
            unique_mnlps = set(seg_agg["from_stop"]).union(set(seg_agg["to_stop"]))
            stops_filtered = stops_xy.loc[stops_xy.index.isin(unique_mnlps)].reset_index()
            
            # Simple styling for stops: small white/black circles
            stops_layer = pdk.Layer(
                "ScatterplotLayer",
                stops_filtered,
                id="stops_layer",           # <--- Added ID here too
                get_position="[lon, lat]",
                get_radius=30,
                get_fill_color=[255, 255, 255, 200],
                get_line_color=[0, 0, 0, 150],
                get_line_width=2,
                pickable=True,
                auto_highlight=True,
            )
            
            view_state = pdk.ViewState(
                latitude=seg_agg["from_lat"].mean(),
                longitude=seg_agg["from_lon"].mean(),
                zoom=12,
                pitch=0
            )
            
            tooltip = {
                "html": "<b>{SegmentKey}</b><br/>"
                        f"Line: {{line}}<br/>"
                        f"{metric.upper()}: {{metric_formatted}}<br/>"
                        "N: {n}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
            # Interactive Selection
            deck = pdk.Deck(
                layers=[layer, stops_layer], 
                initial_view_state=view_state, 
                tooltip=tooltip, 
                map_style="light"
            )
            
            # Using current streamlit API for selection
            selection = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object")
            
            selected_seg_key = None
            
            # Extract selection
            if selection.selection:
                 # selection.selection["objects"] -> maps layer_id to list of indices
                 objects = selection.selection.get("objects", {})
                 
                 # Check if 'segments_layer' was clicked
                 indices = objects.get("segments_layer", [])
                 if indices:
                      first_item = indices[0]
                      if isinstance(first_item, int):
                           # Index lookup
                           selected_seg_key = seg_agg.iloc[first_item]["SegmentKey"]
                      elif isinstance(first_item, dict):
                           # Object data returned directly
                           selected_seg_key = first_item.get("SegmentKey")
                      else:
                           st.warning(f"Unknown selection format: {type(first_item)}")
            
            # --- RENDER DETAILS IF SELECTED ---
            if selected_seg_key:
                st.divider()
                st.markdown(f"### Details for Segment: `{selected_seg_key}`")
                
                # Filter raw data to segment (from df_filtered, which has temporal filters)
                seg_data = df_filtered[df_filtered["SegmentKey"] == selected_seg_key].copy()
                
                if seg_data.empty:
                    st.warning("No data for this segment in the current time range (it exists in topology but not in filtered data).")
                else:
                    # Metrics row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Count (n)", len(seg_data))
                    c2.metric("Avg TT (s)", f"{seg_data['y_true'].mean():.1f}")
                    mae_val = seg_data['abs_err'].mean()
                    mape_val = seg_data['pct_err'].mean() * 100
                    c3.metric("MAE (s)", f"{mae_val:.1f}")
                    c4.metric("MAPE (%)", f"{mape_val:.1f}%")
                    
                    # Charts
                    col_charts_1, col_charts_2 = st.columns(2)
                    
                    with col_charts_1:
                        st.subheader("Distribution")
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=seg_data["y_true"], name="True TT", opacity=0.6))
                        fig.add_trace(go.Histogram(x=seg_data["y_pred"], name="Pred TT", opacity=0.6))
                        fig.update_layout(barmode="overlay", margin=dict(l=20,r=20,t=30,b=20), height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_charts_2:
                        st.subheader("Time Series (Hourly)")
                        
                        # Aggregate required columns
                        agg_dict = {"y_true": "mean", "y_pred": "mean"}
                        if "temp" in seg_data.columns:
                            agg_dict["temp"] = "mean"
                        if "rain" in seg_data.columns:
                            agg_dict["rain"] = "max"
                        if "snow" in seg_data.columns:
                            agg_dict["snow"] = "max"
                            
                        ts = seg_data.set_index("link_start_time").resample("H").agg(agg_dict).dropna()
                        
                        # Create Dual Axis Chart
                        fig_ts = go.Figure()
                        
                        # Travel Times (Left Axis)
                        fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["y_true"], name="True TT", line=dict(color='blue')))
                        fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["y_pred"], name="Pred TT", line=dict(color='red', dash='dot')))
                        
                        # Weather (Right Axis)
                        if "rain" in ts.columns and ts["rain"].sum() > 0:
                            fig_ts.add_trace(go.Bar(x=ts.index, y=ts["rain"], name="Rain (mm)", yaxis="y2", opacity=0.3, marker_color="lightblue"))
                        elif "snow" in ts.columns and ts["snow"].sum() > 0:
                            fig_ts.add_trace(go.Bar(x=ts.index, y=ts["snow"], name="Snow (cm)", yaxis="y2", opacity=0.3, marker_color="white"))
                        elif "temp" in ts.columns:
                            fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["temp"], name="Temp (°C)", yaxis="y2", line=dict(color='orange', width=1), mode='lines'))

                        fig_ts.update_layout(
                            title="Mean Travel Time & Weather",
                            yaxis=dict(title="Travel Time (s)"),
                            yaxis2=dict(title="Weather", overlaying="y", side="right", showgrid=False),
                            margin=dict(l=20,r=20,t=30,b=20), 
                            height=300,
                            legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)

                    # --- ADDED: Error Profiles & Alignment ---
                    st.divider()
                    st.subheader("Error Profiles")
                    c_prof1, c_prof2 = st.columns(2)
                    
                    with c_prof1:
                        # Hourly bias
                        if "hour" in seg_data.columns:
                            err_h = seg_data.groupby("hour").agg(
                                mae=("abs_err","mean"), 
                                bias=("residual","mean")
                            ).reset_index()
                            fig_h = px.bar(err_h, x="hour", y=["mae", "bias"], barmode="group", title="Error by Hour of Day")
                            st.plotly_chart(fig_h, use_container_width=True)
                        
                    with c_prof2:
                        # Dow bias
                        if "dow" in seg_data.columns:
                            err_d = seg_data.groupby("dow").agg(
                                mae=("abs_err","mean"), 
                                bias=("residual","mean")
                            ).reset_index()
                            # Map dow to name if numeric
                            if pd.api.types.is_numeric_dtype(err_d["dow"]):
                                days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                                err_d["day"] = err_d["dow"].apply(lambda x: days[x] if 0 <= x < 7 else x)
                            else:
                                err_d["day"] = err_d["dow"]
                                
                            fig_d = px.bar(err_d, x="day", y=["mae", "bias"], barmode="group", title="Error by Day of Week")
                            st.plotly_chart(fig_d, use_container_width=True)
                        
                    # Scatter True vs Pred
                    st.subheader("Prediction Alignment")
                    fig_scat = px.scatter(seg_data, x="y_true", y="y_pred", color="hour", hover_data=["abs_err"], title="True vs Predicted (Color=Hour)")
                    # Add 1:1 line
                    max_val = max(seg_data["y_true"].max(), seg_data["y_pred"].max()) if not seg_data.empty else 100
                    fig_scat.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_scat, use_container_width=True)

        else:
            st.warning("No segments valid for mapping (check coordinates).")


# --- TAB 2: SEGMENT DETAIL ---
with tab2:
    st.header("Drill Down")
    
    # Segment Selector (sorted by error usually interesting)
    # Re-use seg_agg to fill the selectbox, sorted by current metric
    if not df_filtered.empty and not seg_agg.empty:
        sort_metric = st.selectbox("Sort Segments By", ["n", "mae", "mape", "mean_tt"], index=0)
        sorted_segs = seg_agg.sort_values(sort_metric, ascending=False)
        
        # Format: "Key (Metric=Val)"
        options = sorted_segs["SegmentKey"].tolist()
        captions = [f"{k} ({sort_metric}={v:.1f})" for k,v in zip(options, sorted_segs[sort_metric])]
        
        selected_seg_key = st.selectbox("Select Segment", options, format_func=lambda x: f"{x} - {sorted_segs.loc[sorted_segs['SegmentKey']==x, sort_metric].values[0]:.1f}")
        
        # Filter raw data to segment
        seg_data = df_filtered[df_filtered["SegmentKey"] == selected_seg_key].copy()
        
        # Calculate Baseline Metrics
        base_mae_val = (seg_data["y_true"] - seg_data["y_base"]).abs().mean()
        base_mape_val = ((seg_data["y_true"] - seg_data["y_base"]).abs() / seg_data["y_true"].replace(0, np.nan)).mean() * 100
        
        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Count (n)", len(seg_data))
        c2.metric("Avg TT (s)", f"{seg_data['y_true'].mean():.1f}")
        
        # Improve Metrics: Show Model vs Baseline
        mae_val = seg_data['abs_err'].mean()
        delta_mae = base_mae_val - mae_val # Positive = Improvement
        c3.metric("MAE (s)", f"{mae_val:.1f}", delta=f"{delta_mae:.1f} vs Base", delta_color="normal")
        
        mape_val = seg_data['pct_err'].mean() * 100
        delta_mape = base_mape_val - mape_val
        c4.metric("MAPE (%)", f"{mape_val:.1f}%", delta=f"{delta_mape:.1f}% vs Base", delta_color="normal")
        
        # Charts
        col_charts_1, col_charts_2 = st.columns(2)
        
        with col_charts_1:
            st.subheader("Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=seg_data["y_true"], name="True TT", opacity=0.6))
            fig.add_trace(go.Histogram(x=seg_data["y_base"], name="Base TT", opacity=0.6, marker_color="gray"))
            fig.add_trace(go.Histogram(x=seg_data["y_pred"], name="Pred TT", opacity=0.6))
            fig.update_layout(barmode="overlay", margin=dict(l=20,r=20,t=30,b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_charts_2:
            st.subheader("Time Series (Hourly)")
            
            # Aggregate required columns
            agg_dict = {"y_true": "mean", "y_pred": "mean", "y_base": "mean"}
            if "temp" in seg_data.columns:
                agg_dict["temp"] = "mean"
            if "rain" in seg_data.columns:
                agg_dict["rain"] = "max"
            if "snow" in seg_data.columns:
                agg_dict["snow"] = "max"
                
            ts = seg_data.set_index("link_start_time").resample("H").agg(agg_dict).dropna()
            
            # Create Dual Axis Chart
            fig_ts = go.Figure()
            
            # Travel Times (Left Axis)
            fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["y_true"], name="True TT", line=dict(color='blue')))
            fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["y_base"], name="Base TT", line=dict(color='gray', dash='longdash')))
            fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["y_pred"], name="Pred TT", line=dict(color='red', dash='dot')))
            
            # Weather (Right Axis)
            if "rain" in ts.columns and ts["rain"].sum() > 0:
                fig_ts.add_trace(go.Bar(x=ts.index, y=ts["rain"], name="Rain (mm)", yaxis="y2", opacity=0.3, marker_color="lightblue"))
            elif "snow" in ts.columns and ts["snow"].sum() > 0:
                fig_ts.add_trace(go.Bar(x=ts.index, y=ts["snow"], name="Snow (cm)", yaxis="y2", opacity=0.3, marker_color="white"))
            elif "temp" in ts.columns:
                fig_ts.add_trace(go.Scatter(x=ts.index, y=ts["temp"], name="Temp (°C)", yaxis="y2", line=dict(color='orange', width=1), mode='lines'))

            fig_ts.update_layout(
                title="Mean Travel Time & Weather",
                yaxis=dict(title="Travel Time (s)"),
                yaxis2=dict(title="Weather", overlaying="y", side="right", showgrid=False),
                margin=dict(l=20,r=20,t=30,b=20), 
                height=300,
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
        # Error Profiling
        st.subheader("Error Profiles")
        c_prof1, c_prof2 = st.columns(2)
        
        with c_prof1:
            # Hourly bias
            err_h = seg_data.groupby("hour").agg(
                mae=("abs_err","mean"), 
                bias=("residual","mean")
            ).reset_index()
            fig_h = px.bar(err_h, x="hour", y=["mae", "bias"], barmode="group", title="Error by Hour of Day")
            st.plotly_chart(fig_h, use_container_width=True)
            
        with c_prof2:
            # Dow bias
            err_d = seg_data.groupby("dow").agg(
                mae=("abs_err","mean"), 
                bias=("residual","mean")
            ).reset_index()
            # Map dow to name
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            err_d["day"] = err_d["dow"].map(lambda x: days[x])
            fig_d = px.bar(err_d, x="day", y=["mae", "bias"], barmode="group", title="Error by Day of Week")
            st.plotly_chart(fig_d, use_container_width=True)
            
        # Scatter True vs Pred
        st.subheader("Prediction Alignment")
        fig_scat = px.scatter(seg_data, x="y_true", y="y_pred", color="hour", hover_data=["abs_err"], title="True vs Predicted")
        # Add 1:1 line
        max_val = max(seg_data["y_true"].max(), seg_data["y_pred"].max())
        fig_scat.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig_scat, use_container_width=True)
        
    else:
        st.info("Select data in the filters to analyze segments.")

# --- TAB 3: LINE ANALYSIS ---
with tab3:
    st.header("Line Level Ranking")
    
    if not df_filtered.empty:
        # Group by Line (and maybe model)
        line_grp = df_filtered.groupby(["line", "model"]).agg(
            n=("y_true", "size"),
            mae=("abs_err", "mean"),
            std_ae=("abs_err", "std"),
            mape=("pct_err", "mean")
        ).reset_index()
        
        line_grp["mape"] = line_grp["mape"] * 100
        
        # Uncertainty (sem)
        line_grp["mae_sem"] = line_grp["std_ae"] / np.sqrt(line_grp["n"])
        line_grp["ci_95"] = 1.96 * line_grp["mae_sem"]
        
        # Filters
        min_n = st.number_input("Minimum Samples n", value=50, step=50)
        line_grp = line_grp[line_grp["n"] >= min_n]
        
        st.subheader("Performance Table")
        st.dataframe(line_grp.sort_values("mae", ascending=True), use_container_width=True)
        
        # Ranking Plot
        st.subheader("MAE Ranking (with 95% CI)")
        fig_rank = px.bar(
            line_grp.sort_values("mae"), 
            x="line", y="mae", color="model",
            error_y="ci_95",
            hover_data=["n", "mape"]
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
        st.subheader("MAPE Ranking")
        fig_rank_mape = px.bar(
            line_grp.sort_values("mape"), 
            x="line", y="mape", color="model",
            hover_data=["n", "mae"]
        )
        st.plotly_chart(fig_rank_mape, use_container_width=True)
