import re
import os
import html
import time
import zipfile
import logging
import esda
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, List
from deep_translator import GoogleTranslator
from shapely.geometry import Polygon
from h3 import h3
from libpysal.weights import KNN, DistanceBand
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from tqdm import tqdm
from fast_langdetect import detect, DetectError


def build_dataset_relatedness(
        gdf: gpd.GeoDataFrame,
        max_size: int = 10000,
        frac_related: float = 0.67,
        frac_unrelated: float = 0.33,
        text_colname: str = 'text',
        device: str = 'cuda') -> gpd.GeoDataFrame:
    """Build a dataset of related and unrelated text samples.

    This function takes a GeoDataFrame containing text data,
    classifies each text sample as related or unrelated using our pre-trained text classification model,
    and returns a new GeoDataFrame with the specified fractions of related and unrelated samples.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame containing text data.
        max_size (int, optional): The maximum number of samples in the output dataset. Defaults to 10000.
        frac_related (float, optional): The fraction of related samples in the output dataset. Defaults to 0.67.
        frac_unrelated (float, optional): The fraction of unrelated samples in the output dataset. Defaults to 0.33.
        text_colname (str, optional): The name of the column containing the text data. Defaults to 'text'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cuda'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the classified text
        samples with an additional 'related' column indicating the classification (1 for related, 0 for unrelated).
    """
    total_related: int = np.round(max_size*frac_related, decimals=0)
    total_unrelated: int = np.round(max_size*frac_unrelated, decimals=0)
    num_related: int = 0
    num_unrelated: int = 0

    # shuffle the dataframe
    gdf = gdf.sample(frac=1).reset_index()

    # instanciate the classifier
    classifier = pipeline(
        'text-classification',
        model='hannybal/disaster-twitter-xlm-roberta-al',
        tokenizer='cardiffnlp/twitter-xlm-roberta-base',
        device=device,
        add_special_tokens=True,
        padding='max_length',
        max_length=512,
        truncation='longest_first'
    )

    # classify and build dataframe rows until the desired number of posts is reached
    output_gdf_rows: list[dict] = []
    with tqdm(total=max_size) as pbar:
        for index, row in gdf.iterrows():
            out = classifier(row[text_colname])[0]
            label: str = out['label']
            score: str = out['score']  # noqa
            result_dict: dict = row.to_dict()

            # incremente the counters
            if label == 'LABEL_0':
                if num_unrelated < total_unrelated:
                    result_dict['related'] = 0
                    output_gdf_rows.append(result_dict)
                    num_unrelated += 1
                    pbar.update(n=1)
            elif label == 'LABEL_1':
                if num_related < total_related:
                    result_dict['related'] = 1
                    output_gdf_rows.append(result_dict)
                    num_related += 1
                    pbar.update(n=1)

            # if we have exceeded the needed samples, break
            if num_related >= total_related and num_unrelated >= total_unrelated:
                break

    return gpd.GeoDataFrame.from_dict(output_gdf_rows, geometry='geom', crs=4326).reset_index()


def classify_disaster_relatedness(texts: list[str], device: str = 'cuda') -> list[str]:
    """Classify texts based on disaster-related content.

    Args:
        texts (list[str]): A list of texts to be classified.
        device (str, optional): The computing device for the mdoel. Defaults to 'cuda'.

    Returns:
        list[str]: A list of classification labels indicating whether each text is disaster-related (1) or not (0).
    """
    classifier = pipeline(
        'text-classification',
        model='hannybal/disaster-twitter-xlm-roberta-al',
        tokenizer='cardiffnlp/twitter-xlm-roberta-base',
        device=device,
        add_special_tokens=True,
        padding='max_length',
        max_length=512,
        truncation='longest_first'
    )

    dataset: Dataset = Dataset.from_list([{'text': x} for x in texts])
    print(dataset.shape)

    # set up empty list for sentiment classification results
    class_0_probs: list[float] = []
    class_1_probs: list[float] = []
    class_labels: list[int] = []

    for out in tqdm(classifier(KeyDataset(dataset, "text"), batch_size=16), total=dataset.shape[0]):
        if out['label'] == 'LABEL_0':
            class_0_probs.append(out['score'])
            class_1_probs.append(1-out['score'])
            class_labels.append(0)
        elif out['label'] == 'LABEL_1':
            class_1_probs.append(out['score'])
            class_0_probs.append(1-out['score'])
            class_labels.append(1)

    return class_labels


def translate_google(texts: list[str], source_language: str = 'auto', target_language: str = 'en',
                     verbose: bool = True) -> list[str]:
    """Translates a list of texts from the source language to the target language.

    Args:
        texts (list[str]): List of texts to be translated.
        source_language (str, optional): The language code of the source language. Defaults to 'auto' for automatic detection.
        target_language (str, optional): The language code of the target language. Defaults to 'en' for English.
        verbose (bool, optional): If True, displays a progress bar and retry messages. Defaults to True.

    Returns:
        list[str]: List of translated texts. If a text could not be translated after 10 attempts, None is returned for that text.
    """
    translator: GoogleTranslator = GoogleTranslator(source=source_language, target=target_language)

    translations: list[str] = []
    for text in tqdm(disable=not verbose, iterable=texts):

        # implement a retry mechanism in case of failure
        success: bool = False
        attempts: int = 0
        while not success and attempts < 10:
            try:
                translations.append(translator.translate(text))
                success = True
            except Exception as e:
                attempts += 1
                if attempts < 10:
                    print(f'Retry {attempts} for text: {text[:50]}... (Error: {e})')
                    time.sleep(1)  # Optional: Wait 1 second before retrying
                else:
                    print(f'Failed to translate text after 10 attempts: {text[:50]}...')
                    translations.append(None)
    return translations


def clean_text_bert(x: str) -> str:
    """Pre-process text for BERT-based processing.

    Args:
        x (str): The input string.

    Returns:
        str: Pre-processed string with normalised usernames and links.
    """
    x = re.sub(r"@\w+", "@user", x)  # normalise @references
    x = re.sub(r'https?://\S+|www\.\S+', 'http', x)  # normalise links
    x = x.replace("\n", "")  # remove extra white space
    x = html.unescape(x)  # remove html entities
    return x


def get_group_metrics(df, group_col='use_case', y_true_col='int_label', y_pred_col='prediction'):
    """
    Calculate classification metrics for each group in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str, optional): Column name to group by. Defaults to 'use_case'.
        y_true_col (str, optional): Column name for true labels. Defaults to 'int_label'.
        y_pred_col (str, optional): Column name for predicted labels. Defaults to 'prediction'.

    Returns:
        pd.DataFrame: DataFrame with metrics for each group.
    """
    # Get unique groups
    groups = df[group_col].unique()
    results = []
    for group in groups:
        sub_df = df[df[group_col] == group]
        y_true = sub_df[y_true_col]
        y_pred = sub_df[y_pred_col]

        # Compute metrics; using macro averages to handle multiclass problems
        metrics = {
            group_col: group,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'support': len(sub_df)
        }
        results.append(metrics)
    return pd.DataFrame(results)


def get_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes per-class precision, recall, and F1 score.
    Assumes y_true is in the column 'int_label' and y_pred in 'prediction'.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: A dataframe containing recall, precision and f1 per class.
    """
    # Determine the unique classes present in the true labels for this dataset
    classes = sorted(df['int_label'].unique())

    # Compute per-class metrics using the specified order of classes
    precision = precision_score(df['int_label'], df['prediction'], labels=classes, average=None)
    recall = recall_score(df['int_label'], df['prediction'], labels=classes, average=None)
    f1 = f1_score(df['int_label'], df['prediction'], labels=classes, average=None)

    # Assemble results into a DataFrame
    return pd.DataFrame({
        'class': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })


def h3_to_polygon(h: str) -> Polygon:
    """
    Converts an H3 hexagon index to a Shapely Polygon.

    Parameters:
    h (str): An H3 hexagon index.

    Returns:
    shapely.geometry.Polygon: A Shapely Polygon representing the boundary of the H3 hexagon.
                              The polygon is constructed using the latitude and longitude
                              coordinates of the hexagon's boundary vertices, with the coordinates
                              ordered as (longitude, latitude).

    Example:
    --------
    h3_index = '8928308280fffff'
    polygon = h3_to_polygon(h3_index)
    print(polygon)
    # Output: POLYGON ((-122.0553238 37.3615593, -122.0534897 37.3539269, ...))
    """
    boundary = h3.h3_to_geo_boundary(h)
    return Polygon([(p[1], p[0]) for p in boundary])


def polygon_to_h3_indices(h3_level: int, min_lon: float, min_lat: float,
                          max_lon: float, max_lat: float) -> List[Tuple[str, Polygon]]:
    """
    Generates H3 indices and their corresponding polygons for a given bounding box.

    This function creates a bounding box polygon using the specified latitude and longitude
    coordinates. It then computes all H3 indices that cover the bounding box at the specified
    H3 resolution level. The function returns a list of tuples where each tuple contains an
    H3 index and its corresponding Shapely Polygon representation.

    Parameters:
    -----------
    h3_level : int
        The H3 resolution level to use for generating the H3 indices.
    min_lon : float
        The minimum longitude (western boundary) of the bounding box.
    min_lat : float
        The minimum latitude (southern boundary) of the bounding box.
    max_lon : float
        The maximum longitude (eastern boundary) of the bounding box.
    max_lat : float
        The maximum latitude (northern boundary) of the bounding box.

    Returns:
    --------
    List[Tuple[str, shapely.geometry.Polygon]]
        A list of tuples where each tuple consists of:
        - An H3 index (str)
        - A Shapely Polygon representing the boundary of the H3 hexagon.

    Example:
    --------
    h3_level = 9
    min_lon, min_lat = -122.0553238, 37.3615593
    max_lon, max_lat = -122.0534897, 37.3639163

    indices = polygon_to_h3_indices(h3_level, min_lon, min_lat, max_lon, max_lat)
    for h3_index, polygon in indices:
        print(h3_index, polygon)
    """
    # Create a bounding box polygon for the entire area of interest
    bbox_polygon = Polygon([
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
        (min_lat, min_lon)
    ])

    # Get all H3 indices that cover the bounding box
    h3_indices = h3.polyfill(bbox_polygon.__geo_interface__, h3_level)

    # Initialize a set to collect intersecting H3 indices
    indices: List[Tuple[str, Polygon]] = []

    for h3_index in h3_indices:
        # convert H3 index to polygon
        h3_polygon = Polygon(h3.h3_to_geo_boundary(h3_index, geo_json=True))
        # h3_polygon = Polygon([(lon, lat) for lat, lon in h3_polygon])  # Ensure coordinates are (lon, lat)
        indices.append((h3_index, h3_polygon))

    return indices


def global_morans_i(data: gpd.GeoDataFrame, column: str, weights_matrix: str = "distance",
                    threshold: float = 1000, sample: None | int = None, k=10) -> Tuple:
    """
    Calculate Global Moran's I statistic for spatial autocorrelation.

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the spatial data.
        column (str): The name of the column in the GeoDataFrame containing the variable of interest.
        weights_matrix (str, optional): Type of spatial weights matrix to use. Options are "distance" for
            distance-based weights or "KNN" for k-nearest neighbors. Defaults to "distance".
        threshold (float, optional): Distance threshold for distance-based weights. Defaults to 1000.
        sample (None | int, optional): Number of samples to draw from the data. If None, use all data. Defaults to None.
        k (int, optional): Number of nearest neighbors for KNN weights. Defaults to 10.

    Returns:
        tuple: A tuple containing the Global Moran's I statistic, the p-value from a permutation test, and the z-score.
    """
    # sample data if specified
    if sample:
        data = data.sample(sample)

    # compute the appropriate weights matrix
    if weights_matrix == "distance":
        w = DistanceBand.from_dataframe(data, threshold=threshold)
    if weights_matrix == "KNN":
        w = KNN.from_dataframe(data, k=k)

    # now compute glo bal Moran's I
    moran = esda.Moran(data[column].values, w)

    # return global Morans's I, the p value from a permutation test and the z score
    return moran.I, moran.p_sim, moran.z_sim


def local_morans_i(data: gpd.GeoDataFrame, column: str, weights_matrix: str = "distance",
                   threshold: float = 1000, sample: None | int = None, k: int = 10):
    """
    Calculate Local Moran's I statistic for spatial autocorrelation.

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the spatial data.
        column (str): The name of the column in the GeoDataFrame containing the variable of interest.
        weights_matrix (str, optional): Type of spatial weights matrix to use. Options are "distance" for
            distance-based weights or "KNN" for k-nearest neighbors. Defaults to "distance".
        threshold (float, optional): Distance threshold for distance-based weights. Defaults to 1000.
        sample (None | int, optional): Number of samples to draw from the data. If None, use all data. Defaults to None.
        k (int, optional): Number of nearest neighbors for KNN weights. Defaults to 10.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Local Moran's I statistic, p-values, and z-scores added.
    """
    # Sample the data if specified
    if sample:
        data = data.sample(sample)

    # Create the weights matrix based on the selected method
    if weights_matrix == "distance":
        w = DistanceBand.from_dataframe(data, threshold=threshold)

    elif weights_matrix == "KNN":
        w = KNN.from_dataframe(data, k=k)

    # Calculate Local Moran's I
    moran_local = esda.Moran_Local(data[column].values, w)

    # Extract results
    Ii = moran_local.Is
    p_values = moran_local.p_sim
    z_values = moran_local.z_sim

    # Optionally add results to the GeoDataFrame (e.g., for mapping)
    data['Local_I'] = Ii
    data['p_value'] = p_values
    data['z_value'] = z_values

    return data[['message_id', 'Local_I', 'p_value', 'z_value', 'geometry']]


def point_to_h3(geom: shapely.Point, h3_level: int) -> str:
    """
    Convert a Shapely geometry point to an H3 hexagon index.

    Args:
        geom (shapely.Geometry): A Shapely geometry object representing a point.
        h3_level (int): The H3 resolution level.

    Returns:
        str: The H3 hexagon index corresponding to the given point and resolution level.
    """
    return h3.geo_to_h3(geom.y, geom.x, h3_level)


def geom_to_h3_cells(geom: shapely.Polygon | shapely.Point, h3_level: int) -> list[str]:
    """Converts a Shapely geometry (Polygon or Point) to H3 hexagon indices.

    Args:
        geom (shapely.Polygon | shapely.Point): A Shapely geometry object representing either
            a polygon or a point.
            - If a Polygon, the function will use the `polyfill` method to cover the polygon
              with H3 hexagons.
            - If a Point, the function will assign a single H3 hexagon index based on the point's
              coordinates.
        h3_level (int): The desired H3 resolution level, where higher values represent smaller
            and more precise hexagons. Ranges from 0 (largest hexagons) to 15 (smallest hexagons).

    Raises:
        ValueError: If the geometry type is unsupported (e.g., not a Polygon or Point).

    Returns:
        list[str]: A list of H3 hexagon indices as strings that either cover the polygon or
            represent the point location, depending on the geometry type. If the input is a
            Polygon, the list may contain multiple H3 indices; for a Point, it will contain one.
    """
    if isinstance(geom, Polygon):
        # Use polyfill for polygons
        # hexagons: set = set(h3.polyfill(geom.__geo_interface__, res=h3_level))
        hexagons = set()

        for j in range(4):
            hexagons.add(h3.geo_to_h3(geom.exterior.coords[j][1], geom.exterior.coords[j][0], resolution=h3_level))

    elif isinstance(geom, shapely.Point):
        # Convert Point to H3 directly
        hexagons = [h3.geo_to_h3(geom.y, geom.x, resolution=h3_level)]
    else:
        raise ValueError("Unsupported geometry type")
    return list(hexagons)


def aggregate_h3_level(data: gpd.GeoDataFrame, h3_level: int = 5,
                       true_label_col: str = 'gold', prediction_col: str = 'prediction') -> gpd.GeoDataFrame:
    """
    Aggregates data points into H3 hexagonal cells and computes evaluation metrics within each cell.

    Args:
        data (gpd.GeoDataFrame): A GeoDataFrame containing the data points with geometries.
        h3_level (int, optional): The resolution level of H3 cells. Defaults to 5.
        true_label_col (str, optional): The column name for the true labels. Defaults to 'gold'.
        prediction_col (str, optional): The column name for the predicted labels. Defaults to 'prediction'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with aggregated data and computed evaluation metrics for each H3 cell.
    """
    # reproject and clean data
    data = data.to_crs(4326)
    data = data.dropna(subset=['geometry'])

    # get all h3 cells in our covered area
    min_lon, min_lat, max_lon, max_lat = data.total_bounds
    h3_cells: List[Tuple[str, Polygon]] = polygon_to_h3_indices(h3_level, min_lon, min_lat, max_lon, max_lat)
    h3_cells: gpd.GeoDataFrame = gpd.GeoDataFrame(h3_cells, columns=['h3_index', 'geometry'], crs=4326)
    h3_cells = h3_cells.rename(columns={'h3_index': 'h3_index_polygon'})

    # aggregate h3 cells
    aggregated_dict: dict = {}
    h3_index_list: list[list] = []
    for i, row in data.iterrows():
        # get all h3 cell indices with which our data intersects
        intersections = gpd.sjoin(data.loc[[i], :], h3_cells, how='inner',
                                  predicate='intersects')
        h3_index_list.append(intersections['h3_index_polygon'].tolist())

        for cell in intersections['h3_index_polygon'].tolist():
            if cell in aggregated_dict.keys():
                aggregated_dict[cell] += 1
            else:
                aggregated_dict[cell] = 1
    aggregated: pd.DataFrame = pd.DataFrame.from_dict({'h3_index': aggregated_dict.keys(),
                                                       'total_count': aggregated_dict.values()})

    data['h3_indices'] = h3_index_list

    # compute evaluation metrics within each cell
    for i, row in aggregated.iterrows():
        subset: gpd.GeoDataFrame = data[data['h3_indices'].apply(lambda indices: row['h3_index'] in indices)]
        prec, rec, fscore, support = precision_recall_fscore_support(subset[true_label_col],
                                                                     subset[prediction_col],
                                                                     zero_division=np.nan,
                                                                     labels=[0, 1])
        acc: float = accuracy_score(subset[true_label_col], subset[prediction_col])
        aggregated.loc[i, ['precision_0', 'precision_1']] = prec
        aggregated.loc[i, ['recall_0', 'recall_1']] = rec
        aggregated.loc[i, ['fscore_0', 'fscore_1']] = fscore
        aggregated.loc[i, ['support_0', 'support_1']] = support
        aggregated.loc[i, 'accuracy'] = acc

    # add geometries
    aggregated['geometry'] = aggregated['h3_index'].apply(h3_to_polygon)
    aggregated_gdf = gpd.GeoDataFrame(aggregated, geometry='geometry', crs="EPSG:4326")
    return aggregated_gdf


def extract_files_from_zips(zip_folder: str, extract_folder: str, search_string: str):
    """Extract files from zip archives that contain a specific search string in their filenames.

    Args:
        zip_folder (str): The path to the folder containing the zip files.
        extract_folder (str): The path to the folder where the extracted files will be saved.
        search_string (str): The string to search for in the filenames within the zip files.

    This function ensures the extract folder exists, iterates through each zip file in the zip folder,
    and extracts files whose names contain the search string to the extract folder.
    """
    # Ensure extract folder exists
    os.makedirs(extract_folder, exist_ok=True)

    # Loop through zip files in the zip folder
    for zip_filename in os.listdir(zip_folder):
        if zip_filename.endswith(".zip"):
            zip_path: str = os.path.join(zip_folder, zip_filename)  # extract full zip path

            # read zipfile in memory
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # go through each file in the zip
                for file_name in zip_file.namelist():
                    with zip_file.open(file_name) as file:
                        # Filter files that contain the search string in the filename
                        matching_files: list[str] = [f for f in zip_file.namelist() if search_string in f]

                        for file_name in matching_files:
                            extract_path = os.path.join(extract_folder, os.path.basename(file_name))
                            with zip_file.open(file_name) as file, open(extract_path, "wb") as output_file:
                                output_file.write(file.read())  # Extract the file
                            print(f"Extracted: {file_name} from {zip_filename}")


def top_k_lang_distribution(df: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Return a DataFrame with the top k languages in 'tweet_lang' and their relative share
    among those top k.

    Args:
        df (pd.DataFrame): The input dataframe, must contain a 'tweet_lang' column.
        k (int): Number of languages to include.

    Returns:
        pd.DataFrame: A dataframe with the columns tweet_lang, count and rel_share_pct.
    """
    # count occurrences of each language
    counts = df['tweet_lang'].value_counts()
    total = counts.sum()
    n_langs = len(df['tweet_lang'].unique())

    if k is not None and n_langs > k:
        # more languages than k: reserve one slot for "other"
        top_counts = counts.iloc[: k-1]
        other_count = counts.iloc[k-1:].sum()

        # build DataFrame for top k-1
        result = pd.DataFrame({
            'tweet_lang': top_counts.index,
            'count':      top_counts.values,
        })

        # append the "other" row as the k-th entry
        other_df = pd.DataFrame({
            'tweet_lang': ['other'],
            'count':      [other_count]
        })
        result = pd.concat([result, other_df], ignore_index=True)

    else:
        # k or fewer languages: just take all
        top_counts = counts.iloc[:k]
        result = pd.DataFrame({
            'tweet_lang': top_counts.index,
            'count':      top_counts.values,
        })

    # compute relative share over the grand total
    result['rel_share_pct'] = result['count'] / total * 100

    return result.reset_index(drop=True)


def detect_language_fast(text: str) -> str:
    """Detects the language of an input string using fast-langdetect.

    Args:
        text (str): An input string.

    Returns:
        str: The two-letter language code; 'und' if no language could be recognised.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    try:
        return detect(str(text))['lang']
    except DetectError:
        return "und"
