import gzip
import json
import os
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def get_df(path: str) -> pd.DataFrame:
    def parse():
        file = gzip.open(path, 'rb')
        for line in file:
            yield json.loads(line)

    i = 0
    dic = {}

    for data in parse():
        dic[i] = data
        i += 1

    return pd.DataFrame.from_dict(dic, orient='index')


def sampling_df(chunk: pd.DataFrame, frac: float) -> pd.DataFrame:
    return pd.DataFrame(chunk.values[np.random.choice(chunk.shape[0], round(chunk.shape[0] * frac), replace=False)], columns=chunk.columns)


def get_image(path: str, gzip_path: str, url_col: list[str], uid_col: str, num_workers: int = os.cpu_count() * 5):
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
                session.mount('https://', HTTPAdapter(max_retries=Retry(total=5)))
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
