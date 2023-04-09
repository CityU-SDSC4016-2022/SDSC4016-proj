import concurrent.futures
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def parse(path: str):
    file = gzip.open(path, 'rb')
    for line in file:
        yield json.loads(line)


def get_df(path: str) -> pd.DataFrame:
    i = 0
    dic = {}
    for data in parse(path):
        dic[i] = data
        i += 1
    dataframe = pd.DataFrame.from_dict(dic, orient='index')
    return dataframe


def get_image(path: str, gzip_path: str, url_col: str, uid_col: str, num_workers: int = 1):
    def get_url() -> list[pd.DataFrame]:
        meta = get_df(gzip_path)
        meta = meta[[url_col, uid_col]].dropna()
        df_list = np.array_split(meta, num_workers)
        return df_list

    def fetch_image(meta: pd.DataFrame):
        for url, uid in zip(meta[url_col], meta[uid_col]):
            print(url[0], uid)
            response = requests.get(url[0], timeout=60)
            with open(path + f"{uid}.jpg", "wb") as file:
                file.write(response.content)

    Path(path).mkdir(parents=True, exist_ok=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    meta = get_url()

    for i in range(num_workers):
        pool.submit(fetch_image(meta[i]))
    pool.shutdown(wait=True)
