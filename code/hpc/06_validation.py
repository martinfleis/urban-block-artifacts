import os
os.environ["USE_PYGEOS"] = "0"
import geopandas
import warnings
import osmnx

# sample meta data
sample = geopandas.read_parquet("sample.parquet")

# only the cities we haven't included yet
sample = sample[sample["eFUA_name"].isin(
        [
            "London", 
            "Dortmund"
        ]
    )
]

# Filter warnings about GeoParquet implementation.
warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

for ix, row in sample.iterrows():
    city = row.eFUA_name
    print(f"dowloading {city} building data")
    # Download OSM buildings
    buildings = osmnx.features_from_polygon(
        sample[sample["eFUA_name"] == city].geometry.values[0],
        tags={"building": True},
    )
    print(f"{city} building data downloaded")
    # drop tags (not needed for analysis)
    buildings = buildings[["geometry"]]

    # drop points and linestrings, assert we only have polygons in the gdf
    buildings = buildings.drop(
        buildings[buildings.geometry.type == "Point"].index, axis=0
    ).reset_index(drop=True)
    buildings = buildings.drop(
        buildings[buildings.geometry.type == "LineString"].index, axis=0
    ).reset_index(drop=True)

    # explode multipolygons
    buildings = buildings.explode(index_parts=False)

    # check that we now only have polygons in the data set
    assert all(buildings.geometry.type == "Polygon")

    # save "cleaned" data set
    buildings.to_file(f"buildings_{city}_clean.gpkg", index=0)
    
    del(buildings)

    print(f"{city} building data saved")
