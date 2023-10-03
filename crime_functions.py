import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import folium
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import os
from scipy import stats
from folium.plugins import HeatMap
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import ttest_1samp
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report


def check_columns(DataFrame, reports=False, graphs=False, dates=False):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe.
    """

    dataframeinfo = []

    # Check information about the index
    index_dtype = DataFrame.index.dtype
    index_unique_vals = DataFrame.index.unique()
    index_num_unique = DataFrame.index.nunique()
    index_num_null = DataFrame.index.isna().sum()
    index_pct_null = index_num_null / len(DataFrame.index)

    if pd.api.types.is_numeric_dtype(index_dtype) and not isinstance(
        DataFrame.index, pd.RangeIndex
    ):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (index_min_val, index_max_val)
    elif pd.api.types.is_datetime64_any_dtype(index_dtype):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (
            index_min_val.strftime("%Y-%m-%d"),
            index_max_val.strftime("%Y-%m-%d"),
        )

        # Check for missing dates in the index if dates kwarg is True
        if dates:
            full_date_range = pd.date_range(
                start=index_min_val, end=index_max_val, freq="D"
            )
            missing_dates = full_date_range.difference(DataFrame.index)
            if not missing_dates.empty:
                print(
                    f"Missing dates in index: ({len(missing_dates)} Total) {missing_dates.tolist()}"
                )
    else:
        index_range_vals = None

    dataframeinfo.append(
        [
            "index",
            index_dtype,
            index_num_unique,
            index_num_null,
            index_pct_null,
            index_unique_vals,
            index_range_vals,
        ]
    )

    print(f"Total rows: {DataFrame.shape[0]}")
    print(f"Total columns: {DataFrame.shape[1]}")

    if reports:
        describe = DataFrame.describe().round(2)
        print(describe)

    if graphs:
        DataFrame.hist(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    for column in DataFrame.columns:
        dtype = DataFrame[column].dtype
        unique_vals = DataFrame[column].unique()
        num_unique = DataFrame[column].nunique()
        num_null = DataFrame[column].isna().sum()
        pct_null = DataFrame[column].isna().mean().round(5)

        if pd.api.types.is_numeric_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            mean_val = DataFrame[column].mean().round(2)
            range_vals = (min_val, max_val, mean_val)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            range_vals = (min_val.strftime("%Y-%m-%d"), max_val.strftime("%Y-%m-%d"))

            if dates:
                full_date_range_col = pd.date_range(
                    start=min_val, end=max_val, freq="D"
                )
                missing_dates_col = full_date_range_col.difference(DataFrame[column])
                if not missing_dates_col.empty:
                    print(
                        f"Missing dates in column '{column}': ({len(missing_dates_col)} Total) {missing_dates_col.tolist()}"
                    )
                else:
                    print(f"No missing dates in column '{column}'")

        else:
            range_vals = None

        dataframeinfo.append(
            [column, dtype, num_unique, num_null, pct_null, unique_vals, range_vals]
        )

    return pd.DataFrame(
        dataframeinfo,
        columns=[
            "col_name",
            "dtype",
            "num_unique",
            "num_null",
            "pct_null",
            "unique_values",
            "range (min, max, mean)",
        ],
    )


def load_crime_data():
    # Check if the cached CSV file exists
    if os.path.exists("crime_data.csv"):
        # Load the cached CSV file
        data = pd.read_csv("crime_data.csv")
        print(
            f"File found. Loaded crime_data.csv with {data.shape[0]} rows and {data.shape[1]} columns."
        )
    else:
        # Load the two CSV files
        data1 = pd.read_csv("crime_data_2010_2019.csv")
        data2 = pd.read_csv("crime_data_2020_2023.csv")
        # Concatenate the two CSV files
        data = pd.concat([data1, data2])
        # Cache the concatenated DataFrame as a CSV file
        data.to_csv("crime_data.csv", index=False)
        print(
            f"Created/loaded crime_data.csv with {data.shape[0]} rows and {data.shape[1]} columns."
        )

    return data


def split_data(df, random_state=123):
    """Split into train, validate, test with a 60% train, 20% validate, 20% test"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    print(f"train: {len(train)} ({round(len(train)/len(df)*100)}% of {len(df)})")
    print(
        f"validate: {len(validate)} ({round(len(validate)/len(df)*100)}% of {len(df)})"
    )
    print(f"test: {len(test)} ({round(len(test)/len(df)*100)}% of {len(df)})")
    return train, validate, test


import folium
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer


def plot_crime_clusters(train, k_optimal=5):
    """
    Plots the crime clusters on a map using the KMeans algorithm.

    Parameters:
    train (pandas.DataFrame): The training data with columns "lat" and "lon".
    k_optimal (int): The number of clusters to use. Default is 5.

    Returns:
    folium.Map: The map with the crime clusters and cluster centers plotted.
    """

    # Scale the coordinates
    scaler = StandardScaler()
    train_scaled_coordinates = scaler.fit_transform(train[["lat", "lon"]])

    # Find the optimal number of clusters using the elbow method
    visualizer = KElbowVisualizer(
        KMeans(), k=(2, 12), metric="distortion", timings=False
    )
    visualizer.fit(train_scaled_coordinates)
    visualizer.show()

    # Cluster the data using KMeans
    kmeans = KMeans(n_clusters=k_optimal)
    train["cluster"] = kmeans.fit_predict(train_scaled_coordinates)

    # Inverse transform the cluster centers to the original scale
    center_clusters = scaler.inverse_transform(kmeans.cluster_centers_)

    # Plot the clusters on a map
    map_center = [train["lat"].mean(), train["lon"].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=10, dragging=False)

    # Define a color map for the clusters
    cluster_colors = ["red", "blue", "green", "purple", "orange", "black"]

    # Add crime points to the map
    for idx, row in train.sample(
        frac=0.05
    ).iterrows():  # Sampling to avoid overcrowding the map
        color = cluster_colors[int(row["cluster"])]
        cluster_label = row["cluster"]
        folium.CircleMarker(
            (row["lat"], row["lon"]),
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            tooltip=f"Cluster: {cluster_label}",
        ).add_to(crime_map)

    # Add cluster centers
    for i, center in enumerate(center_clusters):
        folium.Marker(location=center, icon=folium.Icon(color="black")).add_to(
            crime_map
        )
        folium.Circle(
            location=center,
        ).add_to(crime_map)

    return crime_map


def plot_dbscan_clusters(train, eps=0.075, min_samples=25):
    """
    Plots the DBSCAN clusters on a map using the given parameters.

    Parameters:
    train (pandas.DataFrame): The training data with columns "lat" and "lon".
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.075.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is 25.

    Returns:
    folium.Map: The map with the DBSCAN clusters and noise points plotted.
    """
    scaler = StandardScaler()

    coordinates = train[["lat", "lon"]]
    train_scaled_coordinates = scaler.fit_transform(coordinates)

    # Fit a nearest neighbors model to identify distances between points
    neighbors_model = NearestNeighbors(n_neighbors=2)
    neighbors_model.fit(train_scaled_coordinates)
    distances, indices = neighbors_model.kneighbors(train_scaled_coordinates)

    # Draw a dashed line at the given eps and label it
    plt.hlines(
        y=eps,
        xmin=0,
        xmax=len(train_scaled_coordinates),
        linestyles="dashed",
        colors="red",
    )
    plt.text(x=0, y=0.25, s=f"eps = {eps}")

    # Sort the distances and plot them
    distances = distances[:, 1]
    plt.plot(distances)
    plt.ylabel("Distance")
    plt.xlabel("Points")
    plt.title("K-Distance Graph")
    plt.show()

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train["cluster"] = dbscan.fit_predict(train_scaled_coordinates)

    # Count the number of unique clusters (including noise)
    unique_clusters = np.unique(train["cluster"])

    # Define the colors for the clusters (including noise)
    cluster_colors = ["black", "yellow", "green", "blue", "purple", "orange", "red"]

    # Re-create the map with the clusters
    map_center = [train["lat"].mean(), train["lon"].mean()]
    cluster_map = folium.Map(location=map_center, zoom_start=10, tiles="Stamen Terrain")

    # Add cluster points to the map
    for idx, row in train.iterrows():
        if row["cluster"] == -1:
            cluster_color = cluster_colors[0]  # black for noise
        else:
            cluster_color = cluster_colors[
                row["cluster"] + 1
            ]  # +1 to adjust for 0 index of clusters
        folium.CircleMarker(
            (row["lat"], row["lon"]),
            radius=1,
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
        ).add_to(cluster_map)

    # Add a legend to the map
    legend_html = """
         <div style="position: fixed; 
                     top: 400px; right: 800px; width: 150px; height: 250px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color: white;
                     ">
             <p style="margin: 10px;"><b><u>Legend</u></b></p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:yellow"></i>&nbsp;Zone 1</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:green"></i>&nbsp;Zone 2</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:blue"></i>&nbsp;Zone 3</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:orange"></i>&nbsp;Zone 4</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:purple"></i>&nbsp;Zone 5</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:red"></i>&nbsp;Zone 6</p>
             <p style="margin: 10px;"><i class="fa fa-circle fa-1x" style="color:black"></i>&nbsp;Outliers (Noise)</p>
         </div>
    """

    cluster_map.get_root().html.add_child(folium.Element(legend_html))

    print(f"Unique Clusters: {unique_clusters}")

    return cluster_map


import matplotlib.pyplot as plt
import seaborn as sns


def plot_top_crimes_by_cluster(train):
    """
    Plots the top 5 most common crime descriptions for each cluster.

    Parameters:
    train (pandas.DataFrame): The training data with columns "cluster" and "crime_code_description".

    Returns:
    None.
    """

    # Extract the top 5 most common crime descriptions for each cluster
    top_crimes_per_cluster = (
        train.groupby("cluster")["crime_code_description"]
        .value_counts(normalize=True)
        .groupby(level=0)
        .nlargest(5)
        .reset_index(level=0, drop=True)
    )

    # Set up the figure and axes
    fig, axes = plt.subplots(
        len(top_crimes_per_cluster.index.levels[0]), 1, figsize=(10, 10), sharex=True
    )

    # Plot the top 5 crimes for each cluster
    for idx, cluster in enumerate(top_crimes_per_cluster.index.levels[0]):
        subset = top_crimes_per_cluster.loc[cluster]
        ax = axes[idx]
        sns.barplot(y=subset.index, x=subset.values, ax=ax, palette="viridis")
        ax.set_title(f"Cluster {cluster}")
        ax.set_ylabel("")
        ax.set_xlim(0, 0.5)

    fig.suptitle("Top 5 Crimes by Cluster", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_average_robbery_victim_age(train, cluster_num):
    """
    Plots the average robbery victim age for the given cluster and the other clusters where is_robbery is equal to 1.

    Parameters:
    train (pandas.DataFrame): The training data with columns "cluster", "is_robbery", and "victim_age".
    cluster_num (int): The cluster number to plot the average robbery victim age for.

    Returns:
    None.
    """

    # Get the average victim age for the given cluster where is_robbery is equal to 1
    cluster_robbery_victim_age = train[
        (train["cluster"] == cluster_num) & (train["is_robbery"] == 1)
    ]["victim_age"].mean()

    # Get the average victim age for the other clusters where is_robbery is equal to 1
    other_clusters_robbery_victim_age = train[
        (train["cluster"] != cluster_num) & (train["is_robbery"] == 1)
    ]["victim_age"].mean()

    # Plot the average victim age for the given cluster and the other clusters where is_robbery is equal to 1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        ["Cluster " + str(cluster_num), "Other Clusters"],
        [cluster_robbery_victim_age, other_clusters_robbery_victim_age],
    )
    ax.set_ylabel("Average Victim Age")
    ax.set_title(
        f"Average Robbery Victim Age for Cluster {cluster_num} vs Other Clusters"
    )
    plt.show()


import matplotlib.pyplot as plt


def plot_victim_sex_counts(train, cluster_num):
    """
    Plots the victim sex counts for the given cluster and the other clusters where is_robbery is equal to 1.

    Parameters:
    train (pandas.DataFrame): The training data with columns "cluster", "is_robbery", and "victim_sex".
    cluster_num (int): The cluster number to plot the victim sex counts for.

    Returns:
    None.
    """

    # Get unique victim sex categories to ensure consistent ordering
    categories = train["victim_sex"].unique()

    # Get the victim sex counts for the given cluster where is_robbery is equal to 1
    cluster_victim_sex_counts = train[
        (train["cluster"] == cluster_num) & (train["is_robbery"] == 1)
    ]["victim_sex"].value_counts(normalize=True)

    # Get the victim sex counts for the other clusters where is_robbery is equal to 1
    other_clusters_victim_sex_counts = train[
        (train["cluster"] != cluster_num) & (train["is_robbery"] == 1)
    ]["victim_sex"].value_counts(normalize=True)

    # Reindex to ensure both series have the same index order
    cluster_victim_sex_counts = cluster_victim_sex_counts.reindex(categories).fillna(0)
    other_clusters_victim_sex_counts = other_clusters_victim_sex_counts.reindex(
        categories
    ).fillna(0)

    # Get the x-coordinates for the bars
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bars for cluster_victim_sex_counts using the x-coordinates
    ax.bar(
        x,
        cluster_victim_sex_counts.values,
        width=0.4,
        label=f"Cluster {cluster_num} Robbery",
    )

    # Plot the bars for other_clusters_victim_sex_counts by shifting the x-coordinates by bar_width
    ax.bar(
        x + 0.4,
        other_clusters_victim_sex_counts.values,
        width=0.4,
        label="Other Clusters Robbery",
    )

    # Set the x-ticks to the middle of the grouped bars
    ax.set_xticks(x + 0.4 / 2)

    # Label the x-ticks with the categories
    ax.set_xticklabels(categories)

    ax.set_ylabel("Victim Sex Counts")
    ax.set_title(f"Victim Sex for Cluster {cluster_num} vs Other Clusters")
    ax.legend()
    plt.show()


def plot_victim_descent_distribution(cluster_df, rest_df):
    # Get the victim descent distribution for cluster 4
    cluster_victim_descent = cluster_df["victim_descent"].value_counts(normalize=True)

    # Get the victim descent distribution for the rest of the dataset
    rest_victim_descent = rest_df["victim_descent"].value_counts(normalize=True)

    # Define the order of the bars
    bar_order = ["Hispanic", "Black", "White", "Other"]

    # Plot the victim descent distribution for cluster 4 vs the rest of the dataset using a stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, descent in enumerate(bar_order):
        ax.bar(
            ["Cluster 4", "Rest of Dataset"],
            [
                cluster_victim_descent.get(descent, 0),
                rest_victim_descent.get(descent, 0),
            ],
            bottom=[
                sum(cluster_victim_descent.get(bar_order[j], 0) for j in range(i)),
                sum(rest_victim_descent.get(bar_order[j], 0) for j in range(i)),
            ],
            label=descent,
        )
    ax.set_ylabel("Proportion")
    ax.set_title("Victim Descent Distribution")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.show()


def calculate_robbery_proportions(train, cluster_num):
    """
    Calculates and prints the robbery and attempted robbery proportions for the given cluster and the other clusters.

    Parameters:
    train (pandas.DataFrame): The training data with columns "cluster" and "crime_code_description".
    cluster_num (int): The cluster number to calculate the robbery and attempted robbery proportions for.

    Returns:
    None.
    """

    # Calculate the robbery and attempted robbery proportions for the given cluster
    cluster_robbery_prop = train[
        (train["cluster"] == cluster_num)
        & (train["crime_code_description"] == "ROBBERY")
    ]["crime_code_description"].count() / len(train[train["cluster"] == cluster_num])
    cluster_attempted_robbery_prop = train[
        (train["cluster"] == cluster_num)
        & (train["crime_code_description"] == "ATTEMPTED ROBBERY")
    ]["crime_code_description"].count() / len(train[train["cluster"] == cluster_num])

    # Calculate the robbery and attempted robbery proportions for the other clusters
    other_clusters_robbery_prop = train[
        (train["cluster"] != cluster_num)
        & (train["crime_code_description"] == "ROBBERY")
    ]["crime_code_description"].count() / len(train[train["cluster"] != cluster_num])
    other_clusters_attempted_robbery_prop = train[
        (train["cluster"] != cluster_num)
        & (train["crime_code_description"] == "ATTEMPTED ROBBERY")
    ]["crime_code_description"].count() / len(train[train["cluster"] != cluster_num])

    # Print the robbery and attempted robbery proportions for the given cluster and the other clusters
    print(f"Cluster {cluster_num}:")
    print(f"Robbery Proportion: {cluster_robbery_prop:.2f}")
    print(f"Attempted Robbery Proportion: {cluster_attempted_robbery_prop:.2f}")
    print()
    print("Other Clusters:")
    print(f"Robbery Proportion: {other_clusters_robbery_prop:.2f}")
    print(f"Attempted Robbery Proportion: {other_clusters_attempted_robbery_prop:.2f}")


import matplotlib.pyplot as plt


def plot_weapon_category_proportions(train, cluster_num):
    """
    Plots the weapon category proportions for the given cluster and the other clusters.

    Parameters:
    train (pandas.DataFrame): The training data with columns "cluster" and "weapon_description".
    cluster_num (int): The cluster number to plot the weapon category proportions for.

    Returns:
    None.
    """

    # Get the weapon category counts for the given cluster
    cluster_weapon_category_counts = (
        train[train["cluster"] == cluster_num]["weapon_description"]
        .value_counts()
        .head()
    )

    # Get the weapon category counts for the other clusters
    other_clusters_weapon_category_counts = (
        train[train["cluster"] != cluster_num]["weapon_description"]
        .value_counts()
        .head()
    )

    # Normalize the counts to get proportions
    cluster_total = cluster_weapon_category_counts.sum()
    other_clusters_total = other_clusters_weapon_category_counts.sum()

    cluster_weapon_category_proportions = cluster_weapon_category_counts / cluster_total
    other_clusters_weapon_category_proportions = (
        other_clusters_weapon_category_counts / other_clusters_total
    )

    # Get the x-coordinates for the bars
    x = np.arange(len(cluster_weapon_category_proportions))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the bars for cluster_weapon_category_proportions using the x-coordinates
    ax.bar(
        x,
        cluster_weapon_category_proportions.values,
        width=0.4,
        label=f"Cluster {cluster_num}",
    )

    # Plot the bars for other_clusters_weapon_category_proportions by shifting the x-coordinates by 0.4
    ax.bar(
        x + 0.4,
        other_clusters_weapon_category_proportions.values,
        width=0.4,
        label="Other Clusters",
    )

    # Set the x-ticks to the middle of the grouped bars
    ax.set_xticks(x + 0.4 / 2)

    # Label the x-ticks with the categories
    ax.set_xticklabels(
        cluster_weapon_category_proportions.index, rotation=45, ha="right"
    )

    ax.set_ylabel("Weapon Category Proportions")
    ax.set_title(
        f"Weapon Category Proportions for Cluster {cluster_num} vs Other Clusters"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


import pandas as pd


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the given data according to specific transformations, renaming, and conditions.

    Parameters:
    - data (pd.DataFrame): The input data to be processed.

    Returns:
    - pd.DataFrame: The processed data.
    """

    # Rename columns for logical interpretation
    data = data.rename(
        columns={
            "DR_NO": "dragnet_number",
            "Date Rptd": "date_reported",
            "DATE OCC": "date_occurred",
            "TIME OCC": "time_occurred",
            "AREA": "area_id",
            "AREA NAME": "area_name",
            "Rpt Dist No": "report_district",
            "Part 1-2": "part_1_2",
            "Crm Cd": "crime_code",
            "Crm Cd Desc": "crime_code_description",
            "Mocodes": "modus_operandi_code",
            "Vict Age": "victim_age",
            "Vict Sex": "victim_sex",
            "Vict Descent": "victim_descent",
            "Premis Cd": "premise_code",
            "Premis Desc": "premise_description",
            "Weapon Used Cd": "weapon_used_code",
            "Weapon Desc": "weapon_description",
            "Status": "status",
            "Status Desc": "status_description",
            "Crm Cd 1": "crime_code_1",
            "Crm Cd 2": "crime_code_2",
            "Crm Cd 3": "crime_code_3",
            "Crm Cd 4": "crime_code_4",
            "LOCATION": "location",
            "Cross Street": "cross_street",
            "LAT": "lat",
            "LON": "lon",
            "AREA ": "area_id",
        }
    )

    # Select relevant columns
    data = data[
        [
            "date_reported",
            "date_occurred",
            "time_occurred",
            "area_name",
            "part_1_2",
            "crime_code_description",
            "victim_age",
            "victim_sex",
            "victim_descent",
            "premise_description",
            "weapon_description",
            "location",
            "lat",
            "lon",
        ]
    ]

    # Convert date columns to datetime format
    data["date_reported"] = pd.to_datetime(
        data["date_reported"], format="%m/%d/%Y %I:%M:%S %p"
    )
    data["date_occurred"] = pd.to_datetime(
        data["date_occurred"], format="%m/%d/%Y %I:%M:%S %p"
    )

    # Format the time_occurred column
    data["time_occurred"] = data["time_occurred"].astype(str).str.zfill(4)
    data["time_occurred"] = pd.to_datetime(data["time_occurred"], format="%H%M").dt.time

    # Adjust the date_occurred column by adding hours from time_occurred
    data["date_occurred"] = data["date_occurred"] + pd.to_timedelta(
        data["time_occurred"].astype(str)
    )

    # Remove rows with negative ages
    data = data[data["victim_age"] >= 0]

    # Map victim_descent codes to their full names
    descent_categories = {
        "H": "Hispanic",
        "W": "White",
        "O": "Unknown",
        "B": "Black",
        "A": "Asian",
        "X": "Unknown",
        "F": "Filipino",
        "K": "Korean",
        "C": "Chinese",
        "U": "Pacific Islander",
        "J": "Japanese",
        "V": "Vietnamese",
        "I": "American Indian/Alaskan Native",
        "G": "Guamanian",
        "P": "Asian Indian",
        "Z": "Asian Indian",
        "S": "Samoan",
        "D": "Cambodian",
        "L": "Laotian",
        "N": "Asian Indian",
        " ": "Unknown",
        "-": "Unknown",
    }
    data["victim_descent"] = data["victim_descent"].map(
        lambda desc: descent_categories.get(desc, "Unknown")
    )

    # Clean up the victim_sex column
    data.loc[~data["victim_sex"].isin(["M", "F", "X"]), "victim_sex"] = "Unknown"

    # Handle missing values in premise and weapon descriptions
    data["premise_description"] = data["premise_description"].fillna("Unknown")
    data["weapon_description"] = data["weapon_description"].fillna("No Weapon")

    # Categorize weapons
    weapon_categories = {
        "No Weapon": ["No Weapon"],
        "Firearm": [
            "M1-1 SEMIAUTOMATIC ASSAULT RIFLE",
            "SEMI-AUTOMATIC PISTOL",
            "HAND GUN",
            "SIMULATED GUN",
            "UNKNOWN FIREARM",
            "SHOTGUN",
            "AIR PISTOL/REVOLVER/RIFLE/BB GUN",
            "REVOLVER",
            "ASSAULT WEAPON/UZI/AK47/ETC",
            "ANTIQUE FIREARM",
            "SEMI-AUTOMATIC RIFLE",
            "RIFLE",
            "HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE",
            "MAC-11 SEMIAUTOMATIC ASSAULT WEAPON",
            "SAWED OFF RIFLE/SHOTGUN",
            "HECKLER & KOCH 91 SEMIAUTOMATIC ASSAULT RIFLE",
            "UZI SEMIAUTOMATIC ASSAULT RIFLE",
            "UNK TYPE SEMIAUTOMATIC ASSAULT RIFLE",
            "M-14 SEMIAUTOMATIC ASSAULT RIFLE",
            "AUTOMATIC WEAPON/SUB-MACHINE GUN",
            "STARTER PISTOL/REVOLVER",
            "MAC-10 SEMIAUTOMATIC ASSAULT WEAPON",
            "RELIC FIREARM",
            "OTHER FIREARM",
        ],
        "Melee Object": [
            "STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)",
            "ROCK/THROWN OBJECT",
            "BLUNT INSTRUMENT",
            "BOTTLE",
            "CLUB/BAT",
            "STICK",
            "PIPE/METAL PIPE",
            "HAMMER",
            "BELT FLAILING INSTRUMENT/CHAIN",
            "TIRE IRON",
            "CONCRETE BLOCK/BRICK",
            "BOARD",
            "BLACKJACK",
            "BRASS KNUCKLES",
            "MACHETE",
            "OTHER CUTTING INSTRUMENT",
            "FOLDING KNIFE",
            "OTHER KNIFE",
            "KNIFE WITH BLADE 6INCHES OR LESS",
            "ICE PICK",
            "KNIFE WITH BLADE OVER 6 INCHES IN LENGTH",
            "KITCHEN KNIFE",
            "SWITCH BLADE",
            "DIRK/DAGGER",
            "BOWIE KNIFE",
            "STRAIGHT RAZOR",
            "CLEAVER",
            "RAZOR BLADE",
            "SCISSORS",
            "AXE",
            "UNKNOWN TYPE CUTTING INSTRUMENT",
            "SWORD",
            "RAZOR",
            "SCREWDRIVER",
            "BOW AND ARROW",
            "SYRINGE",
        ],
        "Threats": [
            "UNKNOWN WEAPON/OTHER WEAPON",
            "VERBAL THREAT",
            "PHYSICAL PRESENCE",
            "DEMAND NOTE",
            "BOMB THREAT",
        ],
        "Other": [
            "GLASS",
            "MACE/PEPPER SPRAY",
            "STUN GUN",
            "EXPLOXIVE DEVICE",
            "DOG/ANIMAL (SIC ANIMAL ON)",
            "SCALDING LIQUID",
            "ROPE/LIGATURE",
            "TOY GUN",
            "CAUSTIC CHEMICAL/POISON",
            "MARTIAL ARTS WEAPONS",
            "LIQUOR/DRUGS",
            "FIRE",
            "FIXED OBJECT",
        ],
        "Vehicle": [
            "VEHICLE",
        ],
    }
    data["weapon_category"] = data["weapon_description"].map(
        lambda desc: next(
            (cat for cat, weapons in weapon_categories.items() if desc in weapons),
            "Unknown",
        )
    )

    # Calculate the time to report and round to the nearest day
    data["time_to_report"] = data["date_reported"] - data["date_occurred"]
    data["time_to_report"] = data["time_to_report"].dt.round("D")
    data.loc[
        data["time_to_report"] < pd.Timedelta(days=0), "time_to_report"
    ] = pd.Timedelta(days=0)
    data["time_to_report"] = data["time_to_report"].dt.days.astype(int)

    # Rename and recode the severity column
    data = data.rename(columns={"part_1_2": "severe"})
    data["severe"] = data["severe"].replace(2, 0)

    # Handle missing lat/lon values
    data.loc[data["lat"] == 0, "lat"] = None
    data.loc[data["lon"] == 0, "lon"] = None
    data = data.sort_values(by=["area_name", "date_occurred"]).fillna(method="ffill")

    # Filter data for firearm incidents
    data = data[data["weapon_category"] == "Firearm"]

    # Create a binary target for 'Robbery'
    data["is_robbery"] = data["crime_code_description"].apply(
        lambda x: 1 if x in ["ROBBERY", "ATTEMPTED ROBBERY"] else 0
    )

    # Bin time of day
    data["time_occurred"] = pd.to_datetime(data["time_occurred"], format="%H:%M:%S")
    data["time_occurred"] = data["time_occurred"].dt.hour
    bins = [0, 6, 12, 18, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening"]
    data["time_of_day"] = pd.cut(
        data["time_occurred"], bins=bins, labels=labels, right=False
    )

    # Identify street-related incidents
    data["is_street"] = data["premise_description"] == "STREET"

    return data
