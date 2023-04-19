import gzip
import json
import math
import os
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from keras.callbacks import History
from keras.models import Model
from keras.utils import plot_model
from requests.adapters import HTTPAdapter
from scipy.stats import pearsonr
from urllib3.util import Retry


def get_df(path: str) -> pd.DataFrame:
    """Load dataframe from gzip file"""

    def parse():
        file = gzip.open(path, "rb")
        for line in file:
            yield json.loads(line)

    i = 0
    dic = {}

    for data in parse():
        dic[i] = data
        i += 1

    return pd.DataFrame.from_dict(dic, orient="index")


def sampling_df(chunk: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Sampling dataframe from fraction"""

    return pd.DataFrame(chunk.values[np.random.choice(chunk.shape[0], round(chunk.shape[0] * frac), replace=False)], columns=chunk.columns)


def get_image(path: str, gzip_path: str, url_col: list[str], uid_col: str, num_workers: int = os.cpu_count() * 5):
    """Get image from url and save to file"""

    def get_url() -> tuple[list[pd.DataFrame], int]:
        dataframe = get_df(gzip_path)
        if len(url_col) > 1:
            dataframe[url_col[0]].fillna(dataframe[url_col[1]], inplace=True)

        dataframe = dataframe[[url_col[0], uid_col]].dropna()
        df_list = np.array_split(dataframe, num_workers)
        return df_list, dataframe.shape[0]

    def fetch_image(dataframe: pd.DataFrame):
        for url, uid in zip(dataframe[url_col[0]], dataframe[uid_col]):
            fullpath = path + uid + Path(url[0]).suffix
            if not Path(fullpath).exists():
                session = requests.Session()
                session.mount("https://", HTTPAdapter(max_retries=Retry(total=5)))
                response = session.get(url[0], timeout=120)
                if response.status_code == 200:
                    with open(fullpath, "wb") as file:
                        file.write(response.content)
                else:
                    print(f"Missing {uid}")

    Path(path).mkdir(parents=True, exist_ok=True)
    pool = futures.ThreadPoolExecutor(max_workers=num_workers)
    dataframe, length = get_url()

    print(f"Getting {length} images with {num_workers} workers")
    for i in range(num_workers):
        pool.submit(fetch_image(dataframe[i]))
    pool.shutdown(wait=True)


def print_struct(toggle: bool, model: Model, name: str):
    """Print model structure"""

    if toggle:
        plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)


def save_history(history: History, name: str, par: list[int | float]):
    """Save model history"""

    hist = pd.DataFrame.from_dict(history.history)
    par = "_".join(map(str, par))
    file = f"{name}_{par}.csv"
    hist.to_csv(file, index=False)


def res_evalution(y_pred: np.ndarray, y_test: pd.Series):
    """Evaluate model results"""

    y_pred = y_pred.flatten()
    y_test = y_test.to_numpy()

    MAE = np.mean(np.abs(y_test - y_pred))
    MSE = np.square(np.subtract(y_test, y_pred)).mean()
    RMSE = math.sqrt(MSE)

    print(f"Mean Absolute Error: {MAE:.5f}")
    print(f"Root Mean Square Error: {RMSE:.5f}")

    pearson = pearsonr(y_test, y_pred)
    print(f"Pearson Correlation: {pearson[0]:.5f}")
    print(f"p-value: {pearson[1]:.5f}")


def save_result(y_pred: np.ndarray, y_test: pd.Series, name: str, par: list[int | float]):
    """Save prediction results"""

    y_pred = y_pred.flatten()
    y_test = y_test.to_numpy()

    par = "_".join(map(str, par))
    file = f"{name}_{par}.csv"

    y_result = pd.DataFrame({"real": y_test, "predict": y_pred}, columns=["real", "predict"])
    y_result.to_csv(file, index=False)


def save_full_result(y_pred: np.ndarray, y_test: pd.DataFrame, name: str, par: list[int | float]):
    """Save full prediction results"""

    y_pred = y_pred.flatten()

    par = "_".join(map(str, par))
    file = f"{name}_{par}.csv"

    result = y_test.copy(deep=True)
    result["predict"] = y_pred
    result.to_csv(file, index=False)
