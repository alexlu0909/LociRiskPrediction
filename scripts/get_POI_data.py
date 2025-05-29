import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from tqdm import tqdm
import os


PLACE_NAME = "New York City, New York, USA"
INPUT_CSV_PATH = "DNN_DATA_WITH_LAW_LABEL.csv"
OUTPUT_CSV_WITH_POI = "DNN_DATA_with_LAW_LABEL_and_OSM_POI.csv"
TARGET_COLUMN = 'law_level'
SEARCH_RADIUS_METERS = 200


POI_CATEGORIES_OSM = {
    'nightlife': {'amenity': ['bar', 'nightclub', 'pub']},
    'food_drink': {'amenity': ['restaurant', 'cafe', 'fast_food']},
    'retail_convenience': {'shop': ['convenience', 'supermarket', 'mall']},
    'retail_risky': {'shop': ['alcohol', 'tobacco', 'pawnbroker', 'erotic_shop']},
    'transport_hub': {'railway': ['station', 'subway_entrance'], 'amenity': ['bus_station'], 'highway': ['bus_stop']},
    'education': {'amenity': ['school', 'university', 'college', 'library', 'kindergarten']},
    'finance': {'amenity': ['bank', 'atm']},
    'public_safety': {'amenity': ['police', 'fire_station']},
    'healthcare': {'amenity': ['hospital', 'clinic', 'doctors', 'pharmacy']},
    'recreation_open': {'leisure': ['park', 'playground', 'dog_park', 'garden']},
    'recreation_indoor': {'leisure': ['fitness_centre', 'sports_centre', 'stadium'], 'amenity': ['theatre', 'cinema']}
}




def create_crime_geodataframe(df_locations, lat_col='Latitude', lon_col='Longitude', verbose=True):

    if verbose:
        print("\n--- 3. Creating GeoDataFrame for locations ---")

    if lat_col not in df_locations.columns or lon_col not in df_locations.columns:
        if verbose:
            print(f"Error: Latitude ('{lat_col}') or Longitude ('{lon_col}') columns not found in input DataFrame.")
        return None

    # 保留原始索引以便後續操作，或者依賴傳入的 df_locations 已經有唯一ID欄
    df_locations_clean = df_locations.dropna(subset=[lat_col, lon_col]).copy()  # 使用 .copy() 避免 SettingWithCopyWarning

    if len(df_locations_clean) < len(df_locations) and verbose:
        print(f"Dropped {len(df_locations) - len(df_locations_clean)} rows with missing coordinates.")

    if df_locations_clean.empty:
        if verbose:
            print("No valid locations to process after dropping NaNs for coordinates.")
        return None

    try:
        geometry = [Point(xy) for xy in zip(df_locations_clean[lon_col], df_locations_clean[lat_col])]
        # 傳遞 df_locations_clean 以保留所有欄位，包括 'unique_id_for_poi_merge' 和原始索引（如果需要）
        crime_gdf = gpd.GeoDataFrame(df_locations_clean, geometry=geometry, crs="EPSG:4326")
        if verbose:
            print(f"Created GeoDataFrame with {len(crime_gdf)} locations. CRS set to EPSG:4326.")
        return crime_gdf
    except Exception as e_geom:
        if verbose:
            print(f"Error creating GeoDataFrame geometry: {e_geom}")
        return None


def count_pois_in_radius(points_gdf, pois_gdf, major_categories_osm, radius_meters,
                         enable_category_progress=True):

    if enable_category_progress:
        print(
            f"\n--- 4. Counting POIs within {radius_meters}m radius (Batch Mode for {len(points_gdf) if points_gdf is not None else 0} points) ---")

    if points_gdf is None or points_gdf.empty:
        if enable_category_progress:
            print("Skipping POI counting: input points_gdf is None or empty.")
        return pd.DataFrame()

    if pois_gdf is None or pois_gdf.empty:
        if enable_category_progress:
            print("Skipping POI counting: input pois_gdf is None or empty.")
        poi_counts_all_points_empty = pd.DataFrame(index=points_gdf.index)
        for cat_name_empty in major_categories_osm.keys():
            poi_counts_all_points_empty[f'poi_count_{cat_name_empty}'] = 0
        return poi_counts_all_points_empty

    target_crs = "EPSG:32618"
    projected_points_gdf = None
    projected_pois_gdf = None

    try:
        if enable_category_progress:
            print(f"Projecting location coordinates to {target_crs} for accurate buffering...")
        projected_points_gdf = points_gdf.to_crs(target_crs)

        if enable_category_progress:
            print(f"Filtering and projecting POI coordinates to {target_crs}...")
        pois_gdf_valid_geom = pois_gdf[pois_gdf.geometry.is_valid & (~pois_gdf.geometry.is_empty)]
        if pois_gdf_valid_geom.empty:
            if enable_category_progress:
                print("No valid POI geometries after filtering. Cannot count POIs.")

            poi_counts_all_points_empty = pd.DataFrame(index=points_gdf.index)
            for cat_name_empty in major_categories_osm.keys():
                poi_counts_all_points_empty[f'poi_count_{cat_name_empty}'] = 0
            return poi_counts_all_points_empty
        projected_pois_gdf = pois_gdf_valid_geom.to_crs(target_crs)

    except Exception as e_crs:
        if enable_category_progress:
            print(f"Error during CRS transformation: {e_crs}")
            print("Ensure your GeoDataFrames have valid geometries.")

        poi_counts_all_points_error = pd.DataFrame(index=points_gdf.index)
        for cat_name_error in major_categories_osm.keys():
            poi_counts_all_points_error[f'poi_count_{cat_name_error}'] = 0
        return poi_counts_all_points_error

    if projected_pois_gdf.empty:
        if enable_category_progress:
            print("No valid POIs to count after CRS transformation or filtering. Returning zero counts.")
        poi_counts_all_points_empty = pd.DataFrame(index=points_gdf.index)
        for cat_name_empty in major_categories_osm.keys():
            poi_counts_all_points_empty[f'poi_count_{cat_name_empty}'] = 0
        return poi_counts_all_points_empty

    try:

        projected_points_gdf = projected_points_gdf.copy()
        projected_points_gdf['buffer'] = projected_points_gdf.geometry.buffer(radius_meters)
    except Exception as e_buffer:
        if enable_category_progress:
            print(f"Error creating buffers for points: {e_buffer}")
        poi_counts_all_points_error = pd.DataFrame(index=points_gdf.index)
        for cat_name_error in major_categories_osm.keys():
            poi_counts_all_points_error[f'poi_count_{cat_name_error}'] = 0
        return poi_counts_all_points_error


    poi_counts_all_points = pd.DataFrame(index=projected_points_gdf.index)

    iterable_categories = major_categories_osm.items()
    if enable_category_progress:
        desc_text = f"Counting POIs per category for {len(projected_points_gdf)} points"
        iterable_categories = tqdm(major_categories_osm.items(), desc=desc_text)

    for major_cat_name, osm_tags_dict in iterable_categories:
        current_major_cat_pois_list = []
        for osm_key, osm_values in osm_tags_dict.items():
            if osm_key in projected_pois_gdf.columns:  # 先檢查欄位是否存在
                if isinstance(osm_values, list):
                    current_major_cat_pois_list.append(projected_pois_gdf[projected_pois_gdf[osm_key].isin(osm_values)])
                elif isinstance(osm_values, str) and osm_values == '*':
                    current_major_cat_pois_list.append(projected_pois_gdf[projected_pois_gdf[osm_key].notna()])
                elif isinstance(osm_values, str):
                    current_major_cat_pois_list.append(projected_pois_gdf[projected_pois_gdf[osm_key] == osm_values])
            elif enable_category_progress:
                (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   Warning: OSM key '{osm_key}' not found in projected_pois_gdf columns for category '{major_cat_name}'."
                )

        if not current_major_cat_pois_list:
            if enable_category_progress:
                (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   No POIs found matching tags for category '{major_cat_name}'."
                )
            poi_counts_all_points[f'poi_count_{major_cat_name}'] = 0
            continue

        temp_concat_gdf = pd.concat(current_major_cat_pois_list) if current_major_cat_pois_list else gpd.GeoDataFrame()

        pois_for_major_cat_gdf = gpd.GeoDataFrame()
        if not temp_concat_gdf.empty:

            if 'osmid' in temp_concat_gdf.columns:
                pois_for_major_cat_gdf = temp_concat_gdf.drop_duplicates(subset=['osmid'])
            else:
                pois_for_major_cat_gdf = temp_concat_gdf

        if pois_for_major_cat_gdf.empty:
            if enable_category_progress:
                (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   No POIs for major category '{major_cat_name}' after attempting to concat/deduplicate."
                )
            poi_counts_all_points[f'poi_count_{major_cat_name}'] = 0
            continue


        points_buffer_as_geom = projected_points_gdf.set_geometry('buffer')

        try:

            if not pois_for_major_cat_gdf.geometry.is_valid.all():
                if enable_category_progress: (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   Warning: Invalid geometries found in pois_for_major_cat_gdf for category {major_cat_name}.")
                pois_for_major_cat_gdf = pois_for_major_cat_gdf[pois_for_major_cat_gdf.geometry.is_valid]
            if not points_buffer_as_geom.geometry.is_valid.all():  # 'buffer' 欄現在是 geometry
                if enable_category_progress: (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   Warning: Invalid buffer geometries found for category {major_cat_name}.")


            if pois_for_major_cat_gdf.empty or points_buffer_as_geom.empty:
                poi_counts_all_points[f'poi_count_{major_cat_name}'] = 0
                continue

            joined = gpd.sjoin(pois_for_major_cat_gdf, points_buffer_as_geom, how='inner', predicate='intersects')
        except Exception as e_sjoin:
            if enable_category_progress:
                (tqdm.write if 'tqdm' in str(type(iterable_categories)) else print)(
                    f"   Error during sjoin for category {major_cat_name}: {e_sjoin}"
                )
            poi_counts_all_points[f'poi_count_{major_cat_name}'] = 0
            continue

        if not joined.empty:
            counts = joined.groupby(
                'index_right').size()
            poi_counts_all_points[f'poi_count_{major_cat_name}'] = counts
        else:
            poi_counts_all_points[f'poi_count_{major_cat_name}'] = 0

        poi_counts_all_points[f'poi_count_{major_cat_name}'] = poi_counts_all_points[
            f'poi_count_{major_cat_name}'].fillna(0).astype(int)

        if enable_category_progress:
            max_count_val = poi_counts_all_points[f'poi_count_{major_cat_name}'].max()

            write_func = tqdm.write if 'tqdm' in str(type(iterable_categories)) else print
            msg_prefix = "   Finished counting for"

            if max_count_val > 0:
                write_func(f"{msg_prefix} {major_cat_name}. Max count: {max_count_val}")
            elif not current_major_cat_pois_list or pois_for_major_cat_gdf.empty:
                pass
            else:
                write_func(f"{msg_prefix} {major_cat_name}. Max count: 0")

    return poi_counts_all_points



if __name__ == "__main__":

    print("--- Starting OSM POI Feature Extraction Script ---")
    script_start_time = pd.Timestamp.now()
    print(f"Script execution started at: {script_start_time}")

    print("\n--- 1. Loading and Initial Cleaning of Main Data ---")
    try:
        main_df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded {INPUT_CSV_PATH}, shape: {main_df.shape}")
        if 'Unnamed: 0' in main_df.columns:
            main_df = main_df.drop(columns=['Unnamed: 0'])
            print("Dropped 'Unnamed: 0' column.")

        cols_for_initial_dropna = [
            'Latitude', 'Longitude', 'median_income', 'unemployment_rate', 'no_high_school_ratio',
            'bachelor_or_higher_ratio', 'poverty_rate', 'population_density',
            'apparent_temperature', 'precipitation', TARGET_COLUMN
        ]
        actual_cols_for_initial_dropna = [col for col in cols_for_initial_dropna if col in main_df.columns]

        print(f"Columns used for initial NaN drop: {actual_cols_for_initial_dropna}")
        rows_before_drop = len(main_df)
        main_df.dropna(subset=actual_cols_for_initial_dropna, inplace=True)
        print(
            f"Dropped {rows_before_drop - len(main_df)} rows from main_df due to NaNs in critical columns. New shape: {main_df.shape}")

        if main_df.empty:
            raise ValueError("Main DataFrame is empty after initial NaN drop. Cannot proceed.")


        main_df.reset_index(drop=True, inplace=True)

        main_df.reset_index(inplace=True)

        main_df.rename(columns={'index': 'unique_id_for_poi_merge'}, inplace=True)
        print(f"Added 'unique_id_for_poi_merge' column. Current main_df index type: {type(main_df.index)}")

    except FileNotFoundError:
        print(f"Error: Input file {INPUT_CSV_PATH} not found. Please check the path.")
        exit()
    except ValueError as ve:
        print(f"ValueError: {ve}")
        exit()
    except Exception as e:
        print(f"Error loading or initially cleaning {INPUT_CSV_PATH}: {e}")
        exit()


    df_locations_to_process = main_df.copy()

    #df_locations_to_process = main_df.head(5).copy()
    # if len(df_locations_to_process) < len(main_df):
    #     print(f"\n--- DEBUG MODE: Processing only the first {len(df_locations_to_process)} locations ---")
    print(f"\n--- Preparing to process {len(df_locations_to_process)} locations ---")

    print("\n--- 2. Fetching POI Data (from cache or OSM) ---")
    all_osm_tags_to_fetch = {}
    for cat_name, tags_for_cat in POI_CATEGORIES_OSM.items():
        for key, values in tags_for_cat.items():
            if key not in all_osm_tags_to_fetch:
                all_osm_tags_to_fetch[key] = []  # 初始化為列表

            current_tags_for_key = all_osm_tags_to_fetch[key]
            if isinstance(current_tags_for_key, bool):  # 如果已經是 True (萬用字元)，則不再添加
                continue

            if isinstance(values, list):
                for val in values:
                    if val not in current_tags_for_key:
                        current_tags_for_key.append(val)
            elif values not in current_tags_for_key:
                current_tags_for_key.append(values)

    for key in list(all_osm_tags_to_fetch.keys()):
        if isinstance(all_osm_tags_to_fetch[key], list) and not all_osm_tags_to_fetch[key]:
            all_osm_tags_to_fetch[key] = True
    print(f"Aggregated OSM tags to fetch: {all_osm_tags_to_fetch}")

    poi_cache_path = f"pois_{PLACE_NAME.replace(' ', '_').replace(',', '')}_{SEARCH_RADIUS_METERS}m_cache.pkl"
    pois_gdf_area = None

    if os.path.exists(poi_cache_path):
        try:
            print(f"Loading POIs from cache: {poi_cache_path}")
            pois_gdf_area = pd.read_pickle(poi_cache_path)
            if not isinstance(pois_gdf_area, gpd.GeoDataFrame):
                print("Cached data is not a GeoDataFrame. Refetching.")
                pois_gdf_area = None
            elif pois_gdf_area.empty:
                print("Cached POI GeoDataFrame is empty. Will attempt to refetch.")
            else:
                print(f"Successfully loaded {len(pois_gdf_area)} POIs from cache.")
        except Exception as e_cache:
            print(f"Error loading POIs from cache: {e_cache}. Will attempt to refetch.")
            pois_gdf_area = None

    if pois_gdf_area is None or (isinstance(pois_gdf_area, gpd.GeoDataFrame) and pois_gdf_area.empty):
        if pois_gdf_area is None:
            print(f"Cache not found. Fetching POIs for {PLACE_NAME} using osmnx...")
        else:
            print(f"Cached POI data was empty. Fetching POIs for {PLACE_NAME} using osmnx...")
        try:
            pois_gdf_area = ox.features_from_place(PLACE_NAME, tags=all_osm_tags_to_fetch)
            if pois_gdf_area is not None and not pois_gdf_area.empty:
                if 'geometry' in pois_gdf_area.columns:
                    pois_gdf_area = pois_gdf_area[pois_gdf_area['geometry'].notna()]
                    pois_gdf_area = pois_gdf_area[pois_gdf_area.is_valid]
                else:
                    print("Fetched data from osmnx does not contain a 'geometry' column.")
                    pois_gdf_area = gpd.GeoDataFrame()

                if not pois_gdf_area.empty:
                    pois_gdf_area.to_pickle(poi_cache_path)
                    print(f"POIs fetched ({len(pois_gdf_area)}) and cached to {poi_cache_path}")
                else:
                    print(f"No valid POIs fetched for {PLACE_NAME} after geometry validation.")
                    if not isinstance(pois_gdf_area, gpd.GeoDataFrame): pois_gdf_area = gpd.GeoDataFrame()
            else:
                print(f"Could not fetch POIs for {PLACE_NAME} or result was empty.")
                pois_gdf_area = gpd.GeoDataFrame()
        except Exception as e_fetch:
            print(f"Error fetching POIs using osmnx: {e_fetch}")
            pois_gdf_area = gpd.GeoDataFrame()


    if df_locations_to_process.empty:
        print("\nNo locations to process. Exiting.")
    elif pois_gdf_area.empty:
        print("\nNo POI data available (from cache or fetch). Cannot add POI features. Exiting.")
    else:

        crime_locations_gdf = create_crime_geodataframe(df_locations_to_process, verbose=True)

        if crime_locations_gdf is not None and not crime_locations_gdf.empty:

            poi_features_df = count_pois_in_radius(
                crime_locations_gdf,
                pois_gdf_area,
                POI_CATEGORIES_OSM,
                SEARCH_RADIUS_METERS,
                enable_category_progress=True
            )

            if poi_features_df is not None and not poi_features_df.empty:
                print("\n--- 5. Merging POI features back to main DataFrame ---")

                if not df_locations_to_process.index.equals(poi_features_df.index):
                    print("Warning: Indices of df_locations_to_process and poi_features_df do not match directly.")
                    print(
                        f"df_locations_to_process index: {df_locations_to_process.index[:5]}... (type: {type(df_locations_to_process.index)})")
                    print(
                        f"poi_features_df index: {poi_features_df.index[:5]}... (type: {type(poi_features_df.index)})")
                    print("Attempting join. If errors occur or data is misaligned, investigate index handling.")

                df_with_poi = df_locations_to_process.join(poi_features_df, how='left')

                poi_count_cols = [col for col in df_with_poi.columns if col.startswith('poi_count_')]
                df_with_poi[poi_count_cols] = df_with_poi[poi_count_cols].fillna(0).astype(int)

                print(f"Successfully merged POI features. New DataFrame shape: {df_with_poi.shape}")

                if 'unique_id_for_poi_merge' in df_with_poi.columns:
                    print(f"Number of unique IDs in merged df: {df_with_poi['unique_id_for_poi_merge'].nunique()}")
                print("\n--- DataFrame with new OSM POI features (first 5 rows) ---")
                print(df_with_poi.head())


                print("\n--- 6. Saving results to CSV ---")
                try:
                    df_with_poi.to_csv(OUTPUT_CSV_WITH_POI, index=False)
                    print(f"Successfully saved data with POI features to {OUTPUT_CSV_WITH_POI}")
                except Exception as e_save:
                    print(f"Error saving data with POI features: {e_save}")
            else:
                print("No POI features were generated by count_pois_in_radius. Nothing to merge or save.")
        else:
            print("Could not create GeoDataFrame for input locations. POI features cannot be calculated.")

    script_end_time = pd.Timestamp.now()
    print(f"\n--- OSM POI Script Finished at: {script_end_time} ---")
    print(f"Total execution time: {script_end_time - script_start_time}")