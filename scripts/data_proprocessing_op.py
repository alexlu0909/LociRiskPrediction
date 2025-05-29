from statistics import median
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
import requests
from retry_requests import retry
from tqdm import tqdm
import time
from pathlib import Path
import os
import json

poi_key = "fsq3SDAjX9GmFUEOAlZMR+DnH0r/s4S+t+0+DaZAfWet8/w="

tqdm.pandas()
# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)


def process_raw_data(data): # get the needed features from the raw data
    selected_columns = data[['CMPLNT_FR_DT', 'CMPLNT_FR_TM', 'PD_DESC', 'Latitude', 'Longitude']]
    return selected_columns



def get_day_of_week(data):
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], format='%m/%d/%Y')
    data['weekday'] = data['CMPLNT_FR_DT'].dt.weekday  # 0 = Monday, 6 = Sunday

    data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)

    return data

def get_time_cyclical_encode(data):
    time = data['CMPLNT_FR_TM']
    time = pd.to_datetime(time, format='%H:%M:%S')

    hour = time.dt.hour
    minute = time.dt.minute
    data['hour'] = hour
    data['minute'] = minute

    hour = (hour + (minute >= 30)).mod(24)

    data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return data
#-----------------weather data------------------------------------------------------------------------------------------
CACHE_DIR = Path("weather_cache")
DAILY_CACHE_DIR = Path("daily_weather_cache")
CACHE_DIR.mkdir(exist_ok=True)
DAILY_CACHE_DIR.mkdir(exist_ok=True)

def build_weather_cache_key(lat, lon, date_str, hour):
    return f"{lat:.4f}_{lon:.4f}_{date_str}_{hour}"

def load_weather_from_cache(key):
    file = CACHE_DIR / f"{key}.json"
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_weather_to_cache(key, result_dict):
    file = CACHE_DIR / f"{key}.json"
    with open(file, 'w') as f:
        json.dump(result_dict, f)

def query_weather_api(lat, lon, date_str, hour):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["apparent_temperature", "precipitation", "weather_code"],
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York"
    }

    for attempt in range(10):
        try:
            responses = openmeteo.weather_api(url, params=params, timeout=10)
            response = responses[0]

            hourly = response.Hourly()
            try:
                return {
                    "apparent_temperature": float(hourly.Variables(0).ValuesAsNumpy()[hour]),
                    "precipitation": float(hourly.Variables(1).ValuesAsNumpy()[hour]),
                    "weather_code": int(hourly.Variables(2).ValuesAsNumpy()[hour])
                }
            except Exception as parse_err:
                print(f" Parsing error in hourly weather data: {parse_err}")
                return {
                    "apparent_temperature": None,
                    "precipitation": None,
                    "weather_code": None
                }

        except Exception as e:
            error_str = str(e)
            if "API request limit exceeded" in error_str:
                print(error_str)
                exit(1)
            else:
                print(f"Attempt {attempt + 1} failed for Open-Meteo API: {e}")

                if attempt < 9:
                    time.sleep(1)
                else:
                    print(f"[FAIL] lat:{lat}, lon:{lon}, date:{date_str}, hour:{hour}")
                    return {
                        "apparent_temperature": None,
                        "precipitation": None,
                        "weather_code": None
                    }

def get_weather_data_row_optimized(row):
    lat = row['Latitude']
    lon = row['Longitude']
    dt = pd.to_datetime(row['CMPLNT_FR_DT'])
    hour = int(row['hour'])
    minute = int(row['minute'])

    if minute >= 30:
        hour = (hour + 1) % 24
        if hour == 0:
            dt += pd.Timedelta(days=1)

    date_str = dt.strftime('%Y-%m-%d')
    key = build_weather_cache_key(lat, lon, date_str, hour)

    cached = load_weather_from_cache(key)
    if cached is not None:
        return pd.Series(cached)

    result = query_weather_api(lat, lon, date_str, hour)
    save_weather_to_cache(key, result)
    return pd.Series(result)

def get_weather_data(df):
    tqdm.pandas(desc="Get Weather Data (cached)")
    weather = df.progress_apply(get_weather_data_row_optimized, axis=1)
    df['apparent_temperature'] = weather['apparent_temperature']
    df['precipitation'] = weather['precipitation']
    df['weather_code'] = weather['weather_code']
    return df
#-------------------------------
def get_daily_weather_data(df):
    tqdm.pandas(desc="Get Daily Weather Data (cached)")
    weather = df.progress_apply(get_daily_weather_data_row, axis=1)
    df['apparent_temperature'] = weather['apparent_temperature']
    df['precipitation'] = weather['precipitation']
    df['weather_code'] = weather['weather_code']
    return df

def load_daily_weather_from_cache(key):
    file = DAILY_CACHE_DIR / f"{key}.json"
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_daily_weather_to_cache(key, result_dict):
    file = DAILY_CACHE_DIR / f"{key}.json"
    with open(file, 'w') as f:
        json.dump(result_dict, f)

def get_daily_weather_data_row(row):
    lat = row['Latitude']
    lon = row['Longitude']
    dt = pd.to_datetime(row['date'])
    date_str = dt.strftime('%Y-%m-%d')
    key = f"{lat:.4f}_{lon:.4f}_{date_str}_daily"

    cached = load_daily_weather_from_cache(key)
    if cached is not None:
        return pd.Series(cached)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["apparent_temperature", "precipitation", "weather_code"],
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York"
    }

    try:
        responses = openmeteo.weather_api(url, params=params, timeout=10)
        response = responses[0]
        hourly = response.Hourly()

        temp = hourly.Variables(0).ValuesAsNumpy()
        rain = hourly.Variables(1).ValuesAsNumpy()
        weather = hourly.Variables(2).ValuesAsNumpy()

        result = {
            "apparent_temperature": float(np.nanmean(temp)),
            "precipitation": float(np.nansum(rain)),  # ☔ 累積降雨量
            "weather_code": int(pd.Series(weather).mode().iloc[0]) if len(weather) > 0 else None
        }

    except Exception as e:
        print(f" Weather API failed on {date_str} ({lat},{lon}): {e}")
        result = {
            "apparent_temperature": None,
            "precipitation": None,
            "weather_code": None
        }

    save_daily_weather_to_cache(key, result)
    return pd.Series(result)
#------------------------get fcc info-------------------------------
CENSUS_CACHE_DIR = Path("census_cache")
CENSUS_CACHE_DIR.mkdir(exist_ok=True)

# ------- cache tools -------
def get_cache_file(subdir, key):
    path = CENSUS_CACHE_DIR / subdir
    path.mkdir(exist_ok=True)
    return path / f"{key}.json"

def load_from_cache(subdir, key):
    file = get_cache_file(subdir, key)
    if file.exists():
        with open(file, 'r') as f:
            return json.load(f)
    return None

def save_to_cache(subdir, key, value):
    file = get_cache_file(subdir, key)
    with open(file, 'w') as f:
        json.dump(value, f)

def build_key(*args):
    return "_".join(str(x) for x in args)

# ------- API  -------
def call_api_with_retry(url, params, retries=10, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            try:
                return response.json()
            except Exception as e:
                print(f"[Parse Error] JSON parse failed: {e}")
                return None
        except Exception as e:
            print(f"[Attempt {attempt+1}] API error: {e}")
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None

#------------------get fcc info-----------------------------------------------------------------------------------------
def get_fcc_info_row(row):
    lat = round(float(row['Latitude']), 4)
    lon = round(float(row['Longitude']), 4)
    key = build_key(lat, lon)
    cached = load_from_cache("fcc", key)
    if cached: return pd.Series(cached)

    url = "https://geo.fcc.gov/api/census/block/find"
    params = {"latitude": lat, "longitude": lon, "format": "json"}
    result = call_api_with_retry(url, params)

    if result and 'County' in result and 'FIPS' in result['County'] and 'Block' in result and 'FIPS' in result['Block']:
        try:
            output = {
                'state_fips': result['County']['FIPS'][:2],
                'county_fips': result['County']['FIPS'][2:5],
                'tract_code': result['Block']['FIPS'][5:11]
            }
        except Exception as parse_err:
            print(f" Parsing FCC result failed: {parse_err}")
            output = {'state_fips': None, 'county_fips': None, 'tract_code': None}
    else:
        print(f" FCC API returned incomplete or invalid response for ({lat}, {lon})")
        output = {'state_fips': None, 'county_fips': None, 'tract_code': None}

    save_to_cache("fcc", key, output)
    return pd.Series(output)

def get_fcc_info(df):
    tqdm.pandas(desc="FCC info")
    result = df.progress_apply(get_fcc_info_row, axis=1)
    return pd.concat([df, result], axis=1)

# ------- Median Income -------
def get_median_income_row(row):
    key = build_key(row['state_fips'], row['county_fips'], row['tract_code'])
    cached = load_from_cache("median_income", key)
    if cached is not None:
        return cached

    url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "B19013_001E",
        "for": f"tract:{row['tract_code']}",
        "in": f"state:{row['state_fips']}+county:{row['county_fips']}",
        "key": "239089ca5d3aead839725ca5a9b9c0c46a52cfb8"
    }
    result = call_api_with_retry(url, params)

    if not result or len(result) < 2 or not isinstance(result[1], list) or len(result[1]) < 1:
        print(f" API result invalid or missing for: {key}")
        value = None
    else:
        try:
            raw = result[1][0]
            value = None if raw in (None, "null") else float(raw)
        except Exception as e:
            print(f" Error parsing median income for {key}: {e}")
            value = None

    save_to_cache("median_income", key, value)
    return value

def get_median_incomes(df):
    tqdm.pandas(desc="median income")
    df['median_income'] = df.progress_apply(get_median_income_row, axis=1)
    return df
#---------------get education data--------------------------------------------------------------------------------------
def get_education_row(row):
    key = build_key(row['state_fips'], row['county_fips'], row['tract_code'])
    cached = load_from_cache("education", key)
    if cached:
        return pd.Series(cached)

    url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "B15003_001E,B15003_017E,B15003_022E,B15003_023E,B15003_025E",
        "for": f"tract:{row['tract_code']}",
        "in": f"state:{row['state_fips']}+county:{row['county_fips']}",
        "key": "239089ca5d3aead839725ca5a9b9c0c46a52cfb8"
    }

    result = call_api_with_retry(url, params)
    if not result or len(result) < 2 or len(result[1]) < 5:
        print(f" Education API returned invalid data for: {key}")
        fallback = {"no_high_school_ratio": None, "bachelor_or_higher_ratio": None}
        save_to_cache("education", key, fallback)
        return pd.Series(fallback)

    try:
        total = int(result[1][0])
        hs = int(result[1][1])
        ba = int(result[1][2])
        ma = int(result[1][3])
        ph = int(result[1][4])

        if total == 0:
            values = {"no_high_school_ratio": None, "bachelor_or_higher_ratio": None}
        else:
            hs_or_higher = hs + ba + ma + ph
            ba_or_higher = ba + ma + ph
            values = {
                "no_high_school_ratio": 1 - (hs_or_higher / total),
                "bachelor_or_higher_ratio": ba_or_higher / total
            }
    except Exception as e:
        print(f" Error parsing education data for {key}: {e}")
        values = {"no_high_school_ratio": None, "bachelor_or_higher_ratio": None}

    save_to_cache("education", key, values)
    return pd.Series(values)

def get_education(df):
    tqdm.pandas(desc="Get education (cached)")
    result = df.progress_apply(get_education_row, axis=1)
    df["no_high_school_ratio"] = result["no_high_school_ratio"]
    df["bachelor_or_higher_ratio"] = result["bachelor_or_higher_ratio"]
    return df

#---------------------get unemployment data-----------------------------------------------------------------------------
def get_unemployment_rate_row(row):
    key = build_key(row['state_fips'], row['county_fips'], row['tract_code'])
    cached = load_from_cache("unemployment", key)
    if cached is not None:
        return cached

    url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "B23025_003E,B23025_005E",
        "for": f"tract:{row['tract_code']}",
        "in": f"state:{row['state_fips']}+county:{row['county_fips']}",
        "key": "239089ca5d3aead839725ca5a9b9c0c46a52cfb8"
    }

    result = call_api_with_retry(url, params)
    if not result or len(result) < 2 or len(result[1]) < 2:
        print(f"[Unemployment] Invalid result for key {key}: {result}")
        save_to_cache("unemployment", key, None)
        return None

    try:
        labor_force = int(result[1][0])
        unemployed = int(result[1][1])

        if labor_force == 0:
            save_to_cache("unemployment", key, None)
            return None

        unemployment_rate = (unemployed / labor_force) * 100
    except Exception as e:
        print(f"[Unemployment] Error parsing result for key {key}: {e}")
        unemployment_rate = None

    save_to_cache("unemployment", key, unemployment_rate)
    return unemployment_rate

def get_unemployment_rate(df):
    tqdm.pandas(desc="Get unemployment rate (cached)")
    df["unemployment_rate"] = df.progress_apply(get_unemployment_rate_row, axis=1)
    return df

#-------------------get poverty rate data-------------------------------------------------------------------------------
def get_poverty_rate_row(row):
    key = build_key(row['state_fips'], row['county_fips'], row['tract_code'])
    cached = load_from_cache("poverty", key)
    if cached is not None:
        return cached

    url = "https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "B17001_001E,B17001_002E",
        "for": f"tract:{row['tract_code']}",
        "in": f"state:{row['state_fips']}+county:{row['county_fips']}",
        "key": "239089ca5d3aead839725ca5a9b9c0c46a52cfb8"
    }

    result = call_api_with_retry(url, params)
    try:
        if not result or len(result) < 2 or len(result[1]) < 2:
            raise ValueError("Result missing or incomplete")

        total_population = int(result[1][0])
        below_poverty = int(result[1][1])

        if total_population == 0:
            raise ZeroDivisionError("Population is zero")

        poverty_rate = (below_poverty / total_population) * 100
        save_to_cache("poverty", key, poverty_rate)
        return poverty_rate

    except Exception as e:
        print(f"[Poverty API Error] {e} for key: {key}")
        save_to_cache("poverty", key, None)
        return None

def get_poverty_rate(df):
    tqdm.pandas(desc="Get poverty rate (cached)")
    df["poverty_rate"] = df.progress_apply(get_poverty_rate_row, axis=1)
    return df

#----------------get population density data----------------------------------------------------------------------------
def get_population_density_row(row):
    #start = time.time()
    key = build_key(row['state_fips'], row['county_fips'], row['tract_code'])
    cached = load_from_cache("population_density", key)
    if cached is not None:
        return cached

    try:
        # Step 1: 查總人口數
        census_url = "https://api.census.gov/data/2023/acs/acs5"
        params_census = {
            "get": "B01003_001E",
            "for": f"tract:{row['tract_code']}",
            "in": f"state:{row['state_fips']}+county:{row['county_fips']}",
            "key": "239089ca5d3aead839725ca5a9b9c0c46a52cfb8"
        }

        result = call_api_with_retry(census_url, params_census)
        if not result or len(result) < 2 or result[1][0] in (None, "null"):
            raise ValueError("Missing or invalid total population")

        total_population = int(result[1][0])

        # Step 2: area
        tiger_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query"
        params_tiger = {
            "where": f"STATE='{row['state_fips']}' AND COUNTY='{row['county_fips']}' AND TRACT='{row['tract_code']}'",
            "outFields": "AREALAND",
            "f": "json"
        }

        tiger_result = call_api_with_retry(tiger_url, params_tiger)
        if not tiger_result or "features" not in tiger_result or not tiger_result["features"]:
            raise ValueError("Missing land area information")

        aland = int(tiger_result["features"][0]["attributes"]["AREALAND"])
        if aland == 0:
            raise ValueError("Land area is zero")

        density = total_population / aland
        save_to_cache("population_density", key, density)
        #print(f"[{key}] finished in {time.time() - start:.2f}s")
        return density

    except Exception as e:
        print(f"[Error] Failed to get population density for {key}: {e}")
        save_to_cache("population_density", key, None)
        return None

def get_population_density(df):
    tqdm.pandas(desc="Get population density (cached)")
    df["population_density"] = df.progress_apply(get_population_density_row, axis=1)
    return df
#-----------------------------------------------------------------------------------------------------------------------
def build_row_key(row):
    return (
        pd.to_datetime(row['CMPLNT_FR_DT']).strftime('%Y-%m-%d'),
        pd.to_datetime(row['CMPLNT_FR_TM']).strftime('%H:%M:%S'),
        round(float(row['Latitude']), 6),
        round(float(row['Longitude']), 6)
    )
#---------------------record missing data-------------------------------------------------------------------------------------------
def record_miss_data(df):
    fields = [
        'apparent_temperature', 'precipitation', 'weather_code', 'state_fips',
        'county_fips', 'tract_code', 'median_income', 'unemployment_rate',
        'no_high_school_ratio', 'bachelor_or_higher_ratio', 'poverty_rate',
        'population_density'
    ]

    result = []
    for idx, row in df.iterrows():
        missing = [col for col in fields if pd.isna(row[col])]
        if missing:
            result.append({'miss_row_index': idx})
    return pd.DataFrame(result)

def merge_law_back(full_df, raw_df, date_col: str = 'CMPLNT_FR_DT', time_col: str = 'CMPLNT_FR_TM', lat_col: str = 'Latitude', lon_col: str = 'Longitude', law_col: str = 'LAW_CAT_CD'):
    if not pd.api.types.is_datetime64_any_dtype(full_df[date_col]):
        full_df[date_col] = pd.to_datetime(full_df[date_col])
    if not pd.api.types.is_datetime64_any_dtype(raw_df[date_col]):
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])

        # Perform the merge
    merged_df = full_df.merge(
        raw_df[[date_col, time_col, lat_col, lon_col, law_col]],
        on=[date_col, time_col, lat_col, lon_col],
        how='left'
    )
    return merged_df


RAW_CSV = "raw_crime_data.csv"
OUTPUT_CSV = "full_data.csv"
MAX_PER_RUN = 4
BATCH_SIZE = 4


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    raw_data = pd.read_csv(RAW_CSV)
    data_no_miss = raw_data.dropna(how="any")
    #find_api_miss_data = pd.read_csv("full_data.csv")
    #print(record_miss_data(find_api_miss_data))
    full_data = pd.read_csv('full_data.csv')
    full_data_new_label = merge_law_back(full_data, raw_data)
    print(full_data_new_label.head(5))
    full_data_new_label.to_csv('full_data_with_law', index=False)


    start_index = 137395
    to_process = data_no_miss.iloc[start_index:start_index + MAX_PER_RUN]
    print(f"start processing {start_index + 1} to {start_index + len(to_process)} data...")

    # store every 500
    for i in range(0, len(to_process), BATCH_SIZE):
        batch = to_process.iloc[i:i + BATCH_SIZE]
        print(f"process {i + 1} to {min(i + BATCH_SIZE, len(to_process))} data...")

        ready = process_raw_data(batch)
        ready = get_day_of_week(ready)
        ready = get_time_cyclical_encode(ready)
        ready = get_weather_data(ready)
        ready = get_fcc_info(ready)
        ready = get_median_incomes(ready)
        ready = get_unemployment_rate(ready)
        ready = get_education(ready)
        ready = get_poverty_rate(ready)
        ready = get_population_density(ready)

        # store results
        if os.path.exists(OUTPUT_CSV):
            ready.to_csv(OUTPUT_CSV, mode='a', index=False, header=False)
        else:
            ready.to_csv(OUTPUT_CSV, index=False)

        print(f" store number {(i // BATCH_SIZE) + 1} 批，共 {len(ready)} 筆")

    print("finished all batch process and storage")

    last_row = to_process.iloc[-1]
    last_index = raw_data[
        (raw_data['CMPLNT_FR_DT'] == last_row['CMPLNT_FR_DT']) &
        (raw_data['CMPLNT_FR_TM'] == last_row['CMPLNT_FR_TM']) &
        (raw_data['Latitude'] == last_row['Latitude']) &
        (raw_data['Longitude'] == last_row['Longitude'])
        ].index[0]

    print(f"round end index：{last_index}")

