import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from datetime import datetime
from datetime import timedelta
import os
from tqdm import tqdm


def crime_type(crime_df):
    return crime_df['PD_DESC'].unique()
def build_law_keywords():
    law_keywords={
        3: ['FELONY'],
        2:['MISDEMEANOR'],
        1:['VIOLATION']
    }
    return law_keywords
def match_law_level(law, law_keywords):
    desc = str(law).upper()
    for level, keywords in law_keywords.items():
        for keyword in keywords:
            if keyword in desc:
                return level
    print(f"{desc} can't match any danger level")
    return None
def assign_law_level(crime_df):
    law_keywords = build_law_keywords()
    df = crime_df.copy()
    tqdm.pandas(desc="Assigning law levels")
    df['law_level'] = df['LAW_CAT_CD'].progress_apply(lambda x: match_danger_level(x, law_keywords))
    return df


def build_danger_keywords():
    danger_keywords = {
        3: [
            'RAPE 3', 'RAPE 1', 'ASSAULT 2,1,UNCLASSIFIED', 'BURGLARY,RESIDENCE,NIGHT', 'STRANGULATION 1ST',
            'IMPRISONMENT 1,UNLAWFUL', 'RECKLESS ENDANGERMENT 1',
            'LARCENY,GRAND BY EXTORTION', 'ROBBERY,DWELLING', 'ROBBERY,RESIDENTIAL COMMON AREA',
            'ROBBERY,COMMERCIAL UNCLASSIFIED', 'CONTROLLED SUBSTANCE,SALE 1',
            'SODOMY 1', 'OBSTR BREATH/CIRCUL', 'SEXUAL MISCONDUCT,INTERCOURSE', 'ROBBERY,BODEGA/CONVENIENCE STORE',
            'LEAVING SCENE-ACCIDENT-PERSONA',
            'SEX TRAFFICKING', 'BURGLARY,RESIDENCE,UNKNOWN TIM', 'ROBBERY,BEGIN AS SHOPLIFTING', 'SODOMY 3', 'RAPE 2',
            'ROBBERY,HOME INVASION', 'SEXUAL ABUSE',
            'SEXUAL MISCONDUCT,DEVIATE', 'AGGRAVATED HARASSMENT 1', 'ARSON 2,3,4', 'ARSON, MOTOR VEHICLE 1 2 3 & 4',
            'RAPE 1,ATTEMPT', 'ROBBERY,OPEN AREA UNCLASSIFIED',
            'AGGRAVATED SEXUAL ASBUSE', 'BURGLARY,RESIDENCE,DAY', 'ROBBERY,ATM LOCATION', 'BURGLARY,UNCLASSIFIED,DAY',
            'ROBBERY,NECKCHAIN/JEWELRY',
            'BURGLARY,UNCLASSIFIED,UNKNOWN', 'IMPRISONMENT 2,UNLAWFUL', 'ROBBERY,BAR/RESTAURANT',
            'MAKING TERRORISTIC THREAT', 'ROBBERY,LICENSED FOR HIRE VEHICLE',
            'ROBBERY,LICENSED MEDALLION CAB', 'ROBBERY,DELIVERY PERSON', 'MENACING 1ST DEGREE (VICT PEAC',
            'ROBBERY,PUBLIC PLACE INSIDE', 'ROBBERY,CLOTHING',
            'TRESPASS 1,CRIMINAL', 'ROBBERY,GAS STATION', 'ROBBERY,HIJACKING', 'ROBBERY,BICYCLE', 'ROBBERY,CAR JACKING',
            'ROBBERY, CHAIN STORE', 'COERCION 1',
            'INCOMPETENT PERSON,KNOWINGLY ENDANGERING', 'ROBBERY,ON BUS/ OR BUS DRIVER', 'ROBBERY,BANK',
            'VEHICULAR ASSAULT (INTOX DRIVE', 'ROBBERY,LIQUOR STORE',
            'ROBBERY,UNLICENSED FOR HIRE VEHICLE', 'ROBBERY,PHARMACY', 'HOMICIDE,NEGLIGENT,UNCLASSIFIE',
            'SUPP. ACT TERR 2ND', 'LURING A CHILD', 'KIDNAPPING 2',
            'ROBBERY,DOCTOR/DENTIST OFFICE', 'ROBBERY, PAYROLL', 'KIDNAPPING 1', 'STALKING COMMIT SEX OFFENSE',

        ],
        2: [
            'SEXUAL ABUSE 3,2', 'AGGRAVATED HARASSMENT 2', 'ASSAULT 3', 'CRIMINAL CONTEMPT 1',
            'ROBBERY,PERSONAL ELECTRONIC DEVICE', 'HARASSMENT,SUBD 3,4,5',
            'CONTROLLED SUBSTANCE, POSSESSI', 'CONTROLLED SUBSTANCE,SALE 2', 'CONTROLLED SUBSTANCE,SALE 3',
            'WEAPONS, POSSESSION, ETC', 'CONTROLLED SUBSTANCE,INTENT TO',
            'NY STATE LAWS,UNCLASSIFIED FEL', 'TRESPASS 2, CRIMINAL', 'VIOLATION OF ORDER OF PROTECTI',
            'HARASSMENT,SUBD 1,CIVILIAN', 'IMPERSONATION 2, PUBLIC SERVAN',
            'CRIMINAL POSSESSION WEAPON', 'MENACING,UNCLASSIFIED', 'ROBBERY,POCKETBOOK/CARRIED BAG',
            'MISCHIEF,CRIMINAL,    UNCL 2ND', 'RECKLESS ENDANGERMENT 2',
            'CHILD, ENDANGERING WELFARE', 'LARCENY,GRAND FROM RESIDENCE/BUILDING,UNATTENDED, PACKAGE THEFT INSIDE',
            'LARCENY,GRAND FROM PERSON,PICK', 'LARCENY,GRAND FROM PERSON, BAG OPEN/DIP',
            'COERCION 2', 'LARCENY,GRAND BY THEFT OF CREDIT CARD', 'CONTROLLED SUBSTANCE, SALE 5',
            'TAMPERING 1,CRIMINAL', 'INTOXICATED DRIVING,ALCOHOL', 'BURGLARY,COMMERCIAL,DAY',
            'ASSAULT POLICE/PEACE OFFICER', 'UNAUTH. SALE OF TRANS. SERVICE',
            'LARCENY,GRAND FROM PERSON,PERSONAL ELECTRONIC DEVICE(SNATCH)', 'OBSCENE MATERIAL - UNDER 17 YE',
            'OBSCENITY, PERFORMANCE 3', 'CUSTODIAL INTERFERENCE 1', 'WEAPONS POSSESSION 3',
            'CONTROLLED SUBSTANCE,POSSESS.', 'CONSPIRACY 2, 1', 'CAUSE SPI/KILL ANIMAL',
            'CRIMINAL DISPOSAL FIREARM 1', 'LARCENY,GRAND FROM PERSON,UNCL', 'BURGLARY,UNCLASSIFIED,NIGHT',
            'MISCHIEF, CRIMINAL 3&2, BY FIR', 'ESCAPE 2,1',
            'ASSAULT OTHER PUBLIC SERVICE EMPLOYEE', 'LARCENY,GRAND FROM PERSON,LUSH WORKER(SLEEPING/UNCON VICTIM)',
            'LARCENY,GRAND FROM PERSON,PURS',
            'RESISTING ARREST', 'LARCENY,GRAND PERSON,NECK CHAI', 'CONTROLLED SUBSTANCE, INTENT T',
            'IMPERSONATION 1, POLICE OFFICE', 'ACCOSTING,FRAUDULENT',
            'RIOT 2/INCITING', 'MENACING,PEACE OFFICER', 'CHILD ABANDONMENT', 'MENACING 1ST DEGREE (VICT NOT',
            'LEAVING THE SCENE OF AN ACCIDENT (SPI)',
            'IMPAIRED DRIVING,DRUG', 'SALE SCHOOL GROUNDS 4', 'ESCAPE 3', 'POSSES OR CARRY A KNIFE',
            'INCOMPETENT PERSON,RECKLESSY ENDANGERING', 'ASSAULT SCHOOL SAFETY AGENT',
            'CRIM POS WEAP 4', 'UNFINSH FRAME 2', 'ROBBERY,OF TRUCK DRIVER', 'UNLAWFUL POSS. WEAPON UPON SCH',
            'CHILD,OFFENSES AGAINST,UNCLASS', 'IMITATION PISTOL/AIR RIFLE',
            'FALSE REPORT BOMB', 'INCEST 3', 'ASSAULT TRAFFIC AGENT', 'POSS METH MANUFACT MATERIAL', 'INCEST 1, 2',

        ],
        1: [
            'LARCENY,PETIT FROM AUTO', 'UNAUTHORIZED USE VEHICLE 3', 'PETIT LARCENY-CHECK FROM MAILB',
            'FRAUD,UNCLASSIFIED-FELONY', 'LARCENY, GRAND OF MOPED',
            'MISCHIEF, CRIMINAL 4, OF MOTOR', 'LARCENY,PETIT OF LICENSE PLATE', 'LARCENY,PETIT OF VEHICLE ACCES',
            'LARCENY,GRAND BY FALSE PROMISE-IN PERSON CONTACT',
            'FRAUD,UNCLASSIFIED-MISDEMEANOR', 'LARCENY,PETIT FROM BUILDING,UN', 'LARCENY,PETIT FROM OPEN AREAS,',
            'LARCENY,GRAND FROM OPEN AREAS, UNATTENDED',
            'LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED', 'MISCHIEF, CRIMINAL 3 & 2, OF M',
            'LARCENY,GRAND FROM RESIDENCE, UNATTENDED',
            'LARCENY,PETIT FROM BUILDING,UNATTENDED, PACKAGE THEFT INSIDE', 'LARCENY,GRAND OF AUTO',
            'LARCENY,PETIT BY CREDIT CARD U', 'LARCENY,GRAND BY FALSE PROMISE-NOT IN PERSON CONTACT',
            'LARCENY,PETIT FROM BUILDING,UNATTENDED, PACKAGE THEFT OUTSIDE',
            'LARCENY,GRAND BY CREDIT CARD ACCT COMPROMISE-EXISTING ACCT', 'BURGLARY,TRUCK DAY',
            'LARCENY,GRAND BY OPEN CREDIT CARD (NEW ACCT)',
            'BURGLARY,COMMERCIAL,NIGHT', 'LARCENY,PETIT FROM STORE-SHOPL', 'LARCENY,GRAND FROM STORE-SHOPL',
            'CRIMINAL MISCHIEF,UNCLASSIFIED 4', 'LARCENY,PETIT BY ACQUIRING LOS',
            'LARCENY,GRAND BY OPEN/COMPROMISE CELL PHONE ACCT', 'TRAFFIC,UNCLASSIFIED MISDEMEAN',
            'LARCENY,GRAND BY BANK ACCT COMPROMISE-UNCLASSIFIED',
            'LARCENY,GRAND FROM EATERY, UNATTENDED', 'LARCENY,GRAND BY IDENTITY THEFT-UNCLASSIFIED',
            'FORGERY,ETC.,UNCLASSIFIED-FELO', 'LARCENY,GRAND BY BANK ACCT COMPROMISE-REPRODUCED CHECK',
            'LARCENY,GRAND BY ACQUIRING LOST CREDIT CARD', 'LARCENY,GRAND OF VEHICULAR/MOTORCYCLE ACCESSORIES',
            'LARCENY,GRAND BY BANK ACCT COMPROMISE-TELLER',
            'BAIL JUMPING 1 & 2', 'LARCENY,PETIT OF BICYCLE', 'LARCENY,GRAND OF BICYCLE',
            'LARCENY,GRAND BY BANK ACCT COMPROMISE-ATM TRANSACTION', 'MATERIAL              OFFENSIV',
            'NUISANCE, CRIMINAL', 'LARCENY,GRAND BY BANK ACCT COMPROMISE-UNAUTHORIZED PURCHASE', 'CONTEMPT,CRIMINAL',
            'LARCENY,GRAND FROM RESIDENCE/BUILDING,UNATTENDED, PACKAGE THEFT OUTSIDE',
            'LARCENY,GRAND OF TRUCK', 'TAMPERING 3,2, CRIMINAL', 'LARCENY,GRAND FROM NIGHT CLUB, UNATTENDED',
            'FALSE REPORT UNCLASSIFIED', 'LARCENY,PETIT BY FALSE PROMISE',
            'STOLEN PROPERTY 2,1,POSSESSION', 'LARCENY,GRAND OF MOTORCYCLE', 'LARCENY,GRAND BY ACQUIRING LOS',
            'LARCENY,PETIT OF AUTO', 'LARCENY,GRAND BY OPEN BANK ACCT',
            'COMPUTER UNAUTH. USE/TAMPER', 'LARCENY,PETIT FROM TRUCK', 'FALSE REPORT 1,FIRE',
            'RECKLESS ENDANGERMENT OF PROPE', 'LARCENY,GRAND FROM VEHICLE/MOTORCYCLE',
            'UNAUTHORIZED USE VEHICLE 2', 'LARCENY,PETIT BY CHECK USE', 'TRESPASS 3, CRIMINAL', 'CRIMINAL MIS 2 & 3',
            'FORGERY,DRIVERS LICENSE', 'LARCENY,GRAND FROM RETAIL STORE, UNATTENDED',
            'BURGLARY,COMMERCIAL,UNKNOWN TI', 'LARCENY, PETIT OF MOPED', 'CHECK,BAD', 'STOLEN PROPERTY 3,POSSESSION',
            'THEFT,RELATED OFFENSES,UNCLASS', 'LARCENY,PETIT FROM PIER',
            'BURGLARY, TRUCK UNKNOWN TIME', 'MISCHIEF, CRIMINAL 4, BY FIRE', 'BURGLARY,TRUCK NIGHT',
            'PETIT LARCENY OF ANIMAL', 'STOLEN PROPERTY-MOTOR VEH 2ND,',
            'AGGRAVATED CRIMINAL CONTEMPT', 'RECKLESS DRIVING', 'PUBLIC HEALTH LAW,GLUE,UNLAWFU',
            'BURGLARS TOOLS,UNCLASSIFIED', 'FORGERY-ILLEGAL POSSESSION,VEH',
            'DISORDERLY CONDUCT', 'LEWDNESS,PUBLIC', 'TORTURE/INJURE ANIMAL CRUELTY', 'DRUG PARAPHERNALIA,   POSSESSE',
            'ABANDON ANIMAL', 'LARCENY,PETIT OF MOTORCYCLE',
            'CANNABIS SALE, 3', 'FACILITATION 4, CRIMINAL', 'LARCENY,GRAND FROM PIER, UNATTENDED',
            'PROSTITUTION, PATRONIZING 4, 3', 'PROSTITUTION, PATRONIZING 2, 1',
            'UNLAWFUL DISCLOSURE OF AN INTIMATE IMAGE',
            'PROSTITUTION 3,PROMOTING BUSIN', 'PROSTITUTION', 'JOSTLING', 'MARIJUANA, SALE 4 & 5', 'CANNABIS SALE',
            'RECORDS,FALSIFY-TAMPER', 'LARCENY, GRAND OF AUTO - ATTEM',
            'CONTROLLED SUBSTANCE, SALE 4', 'OBSCENITY 1', 'PROMOTING A SEXUAL PERFORMANCE', 'LARCENY,GRAND OF BOAT',
            'CHILD,ALCOHOL SALE TO', 'FIREWORKS, POSSESS/USE',
            'LARCENY,PETIT FROM COIN MACHIN', 'FIREWORKS, SALE', 'LARCENY,GRAND FROM TRUCK, UNATTENDED',
            'EXPOSURE OF A PERSON', 'EAVESDROPPING', 'CANNABIS SALE, 2&1',
            'PROSTITUTION,PERMITTING', 'PRIVACY,OFFENSES AGAINST,UNCLA', 'NUISANCE,CRIMINAL,UNCLASSIFIED',
            'SALES OF PRESCRIPTION', 'FORGERY,PRESCRIPTION',
            'DISSEMINATING A FALSE SEX OFFEND', 'COMPUTER TAMPER/TRESSPASS', 'STOLEN PROPERTY 2,POSSESSION B',
            'STOLEN PROP-MOTOR VEHICLE 3RD,', 'CONTROLLED SUBSTANCE,POSSESS.-',
            'NEGLECT/POISON ANIMAL', 'BRIBERY, POLICE OFFICER', 'UNLAWFUL SALE SYNTHETIC MARIJUANA', 'CONSPIRACY 6, 5',
            'FIREARMS LICENSING LAWS', 'FACILITATION 3,2,1, CRIMINAL',
            'PROSTITUTION 4,PROMOTING&SECUR', 'CONSPIRACY 4, 3', 'ASSEMBLY,UNLAWFUL', 'INAPPROPIATE SHELTER DOG LEFT'
        ],
        0: [
            'LARCENY,GRAND BY DISHONEST EMP', 'THEFT OF SERVICES, UNCLASSIFIE', 'FORGERY,ETC.-MISD.',
            'ADM.CODE,UNCLASSIFIED VIOLATIO', 'FORGERY,M.V. REGISTRATION',
            'PUBLIC ADMINISTATION,UNCLASS M', 'CRIMINAL MISCHIEF 4TH, GRAFFIT', 'N.Y.C. TRANSIT AUTH. R&R',
            'TRESPASS 4,CRIMINAL SUB 2',
            'LARCENY,PETIT BY DISHONEST EMP', 'ADM.CODE,UNCLASSIFIED MISDEMEA', 'BAIL JUMPING 3',
            'PUBLIC ADMINISTRATION,UNCLASSI', 'CUSTODIAL INTERFERENCE 2',
            'NY STATE LAWS,UNCLASSIFIED MIS', 'AGRICULTURE & MARKETS LAW,UNCL', 'PUBLIC HEALTH LAW,UNCLASSIFIED',
            'ALCOHOLIC BEVERAGE CONTROL LAW',
            'PUBLIC SAFETY,UNCLASSIFIED MIS', 'BRIBERY,PUBLIC ADMINISTRATION', 'GAMBLING, DEVICE, POSSESSION',
            'CANNABIS POSSESSION, 3', 'HEALTH CODE,UNCLASSIFIED MISDE',
            'NY STATE LAWS,UNCLASSIFIED VIO', 'CANNABIS POSSESSION, 2&1', 'AIRPOLLUTION',
            'GAMBLING 2,PROMOTING,UNCLASSIF', 'CANNABIS POSSESSION', 'ALCOHOLIC BEVERAGES,PUBLIC CON',
            'TAX LAW', 'GAMBLING 1,PROMOTING,POLICY', 'GAMBLING 1,PROMOTING,BOOKMAKIN', 'PERJURY 2,1,ETC',
            'GAMBLING 2, PROMOTING, BOOKMAK', 'CONFINING ANIMAL IN VEHICLE/SHELTER',
            'GAMBLING 2, PROMOTING, POLICY-', 'GENERAL BUSINESS LAW,TICKET SP', 'RADIO DEVICES,UNLAWFUL POSSESS',
            'EDUCATION LAW,STREET TRADE', 'HEALTH CODE,VIOLATION',
            'THEFT OF SERVICES- CABLE TV SE', 'POSTING ADVERTISEMENTS'
        ]
    }
    return danger_keywords


def match_danger_level(pd_desc, danger_keywords):
    desc = str(pd_desc).upper()
    for level, keywords in danger_keywords.items():
        for keyword in keywords:
            if keyword in desc:
                return level
    print(f"{desc} can't match any danger level")
    return None


def assign_danger_level(crime_df):
    danger_keywords = build_danger_keywords()
    df = crime_df.copy()
    tqdm.pandas(desc="Assigning danger levels")
    df['danger_level'] = df['PD_DESC'].progress_apply(lambda x: match_danger_level(x, danger_keywords))
    return df


def get_lng_lat_only_data(df):
    return df[['Longitude', 'Latitude']]


def how_many_data(df):
    return df.shape[0]


def plot_crime_grid(df, lon_col='Longitude', lat_col='Latitude', crs_in='EPSG:4326',
                    crs_out='EPSG:32618', grid_size=0, boundary=None, show_debug=False):

    # GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs_in
    ).to_crs(crs_out)


    minx, miny, maxx, maxy = boundary.total_bounds
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
            y += grid_size
        x += grid_size

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf.crs)


    if boundary is not None:
        grid = grid[grid.intersects(boundary.unary_union)]
        print(f"Number of square: {len(grid.index)}")


    joined = gpd.sjoin(gdf, grid, predicate='within', how='left')
    if show_debug:
        unmatched = joined[joined['index_right'].isna()]
        print(f"Number of crime points not matched to any grid cell: {len(unmatched)}")

    crime_counts = joined.groupby('index_right').size()
    grid['crime_count'] = crime_counts.reindex(grid.index, fill_value=0)
    return grid


def create_map(grid, boundary=None, grid_size=0, cmap='Reds'):  # 繪圖
    print("Start creating the map")
    fig, ax = plt.subplots(figsize=(12, 12))
    grid[grid['crime_count'] > 0].plot(
        column='crime_count',
        cmap=cmap,
        ax=ax,
        edgecolor='white',
        linewidth=0.1,
        legend=True,
        legend_kwds={'label': 'Crime Count', 'shrink': 0.6}
    )
    grid.boundary.plot(ax=ax, edgecolor='lightgray', linewidth=0.1)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color='black', linewidth=1)

    plt.title(f"{grid_size}m x {grid_size}m Crime Density Grid")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def add_tensor_coordinations(grid):


    grid['centroid_x'] = grid.geometry.centroid.x
    grid['centroid_y'] = grid.geometry.centroid.y


    grid_sorted = grid.sort_values(by=['centroid_y', 'centroid_x'], ascending=[False, True]).reset_index(drop=False)


    unique_y = np.sort(grid_sorted['centroid_y'].unique())[::-1]
    unique_x = np.sort(grid_sorted['centroid_x'].unique())
    H = len(unique_y)
    W = len(unique_x)


    grid_sorted['tensor_row'] = [i // W for i in range(len(grid_sorted))]
    grid_sorted['tensor_col'] = [i % W for i in range(len(grid_sorted))]

    return grid_sorted


def map_crime_to_tensor_coords(df, grid_with_coords, show_unmatched=False):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs='EPSG:4326'
    ).to_crs('EPSG:32618')

    grid_with_coords_gdf = grid_with_coords.copy()

    grid_with_coords_gdf = gpd.GeoDataFrame(grid_with_coords_gdf, geometry='geometry', crs='EPSG:4326')
    grid_with_coords_gdf = grid_with_coords_gdf.to_crs('EPSG:32618')

    joined = gpd.sjoin(gdf, grid_with_coords_gdf[['geometry', 'tensor_row', 'tensor_col']], predicate='within',
                       how='left')
    if show_unmatched:
        unmatched = joined[joined['index_right'].isna()]
        print(f" Number of unmatched crime points: {len(unmatched)}")
        if len(unmatched) > 0:
            print(unmatched[['Longitude', 'Latitude', 'geometry']].head(114))

    df_with_mapcoords = df.copy()
    df_with_mapcoords['tensor_row'] = joined['tensor_row'].values
    df_with_mapcoords['tensor_col'] = joined['tensor_col'].values
    df_with_mapcoords['grid_index'] = joined['index_right'].values
    return df_with_mapcoords


# find the grids without crime happens
def get_empty_grid(grid_with_coords, full_data_with_map_coords):
    used_indices = full_data_with_map_coords['grid_index'].dropna().unique()
    empty_grid = grid_with_coords[~grid_with_coords['index'].isin(used_indices)].copy()
    return empty_grid['index'].reset_index(drop=True)



def track_missing_external_features(df):
    fields = [
        'median_income',
        'unemployment_rate',
        'no_high_school_ratio',
        'bachelor_or_higher_ratio',
        'poverty_rate',
        'population_density'
    ]

    missing_report = []

    for idx, row in df.iterrows():
        missing_fields = [field for field in fields if pd.isna(row[field])]
        if missing_fields:
            missing_report.append({
                'index': idx,
                'missing_fields': missing_fields
            })

    return pd.DataFrame(missing_report)


def knn_impute_missing_feature(df, feature_col, n_neighbors=5):
    cols = ['Latitude', 'Longitude', feature_col]
    df_subset = df[cols].copy()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(cols)

    df[feature_cols] = imputed[:, 2]
    return df



def get_empty_grid_lat_lon(grid_with_coords, full_data_with_map_coords):
    used_indices = full_data_with_map_coords['grid_index'].dropna().unique()
    empty_grid = grid_with_coords[~grid_with_coords['index'].isin(used_indices)].copy()
    empty_grids_gdf = gpd.GeoDataFrame(empty_grid, geometry='geometry', crs='EPSG:32618')
    empty_grids_gdf = empty_grids_gdf.to_crs('EPSG:4326')
    empty_grids_gdf['Latitude'] = empty_grids_gdf.geometry.centroid.y
    empty_grids_gdf['Longitude'] = empty_grids_gdf.geometry.centroid.x
    return empty_grids_gdf


def add_lat_lon_for_each_grid(grid_with_coords):
    grid_with_lat_lon = gpd.GeoDataFrame(grid_with_coords, geometry='geometry', crs='EPSG:32618')
    grid_with_lat_lon = grid_with_lat_lon.to_crs('EPSG:4326')
    grid_with_lat_lon['Latitude'] = grid_with_lat_lon.geometry.centroid.y
    grid_with_lat_lon['Longitude'] = grid_with_lat_lon.geometry.centroid.x
    return grid_with_lat_lon




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

crime_df = pd.read_csv('/Users/peilin/PycharmProjects/ESE561 AI_Final_project/full_data_with_law')
crime_df = crime_df[(crime_df['Longitude'] != 0) & (crime_df['Latitude'] != 0)]
#print(crime_df.head(5))
crime_df = assign_law_level(crime_df)
#print(f"after assigned danger level: {crime_df.head(5)}")
print(f"None law level:{crime_df['law_level'].isna().sum()}")
crime_df = crime_df.dropna(subset=['law_level'])
print(f"after drop:{crime_df['law_level'].isna().sum()}")
print(crime_df.head(5))
crime_df = crime_df.drop('PD_DESC', axis=1)
crime_df.to_csv('full_data_with_lawlabel',index=False)


cols = [
    "Latitude", "Longitude",
    "weekday_sin", "weekday_cos",
    "hour_sin", "hour_cos",
    "apparent_temperature", "precipitation",
    "median_income", "unemployment_rate",
    "no_high_school_ratio", "bachelor_or_higher_ratio",
    "poverty_rate", "population_density",
    "law_level"
]


DNN_training_data_lawlabel = crime_df[cols]


print(f"Number of DNN data: {len(DNN_training_data_lawlabel)}")

DNN_data_without_miss = DNN_training_data_lawlabel.dropna(subset=cols)


DNN_data_without_miss.to_csv("DNN_DATA_WITH_LAW_LABEL.csv", index=False)



number = how_many_data(crime_df) #count the number of data that already get external features
print(f"Number of data that finished api calling:{number}")

#raw_data = pd.read_csv('raw_crime_data.csv')
#data_no_miss = raw_data.dropna(how="any")
#print(f"Number of raw data:{how_many_data(data_no_miss)}") #count the number of raw data without any feature missing

print(f"number of different crime type: {len(crime_type(crime_df))}")

nyc = gpd.read_file("nybb.shp").to_crs(epsg=32618)
manhattan = nyc[nyc["BoroName"] == "Manhattan"]

grid_size = 100
grid = plot_crime_grid(
    crime_df,
    grid_size = grid_size,
    boundary=manhattan,
    show_debug=True,
)

create_map(grid, boundary=manhattan, grid_size=grid_size, cmap='YlOrRd')

grid_with_coords = add_tensor_coordinations(grid)

grid_with_coords = add_lat_lon_for_each_grid(grid_with_coords)
print(grid_with_coords.head(5))

full_data_with_map_coords = map_crime_to_tensor_coords(crime_df, grid_with_coords, show_unmatched=True)
H = full_data_with_map_coords['tensor_row'].max()+1
W = full_data_with_map_coords['tensor_col'].max()+1
print(f"Tensor shape: H ={H}, W ={W}")

print(full_data_with_map_coords.head(5))
