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


def get_image(path: str, gzip_path: str, url_col: str, uid_col: str, num_workers: int = os.cpu_count() * 5):
    def get_url() -> tuple[list[pd.DataFrame], int]:
        meta = get_df(gzip_path)
        meta = meta[[url_col, uid_col]].dropna()
        df_list = np.array_split(meta, num_workers)
        return df_list, meta.shape[0]

    def fetch_image(meta: pd.DataFrame):
        for url, uid in zip(meta[url_col], meta[uid_col]):
            fullpath = path + f"{uid}.{Path(url[0]).suffix}"
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
    meta, length = get_url()

    print(f"Getting {length} images with {num_workers} workers")
    for i in range(num_workers):
        pool.submit(fetch_image(meta[i]))
    pool.shutdown(wait=True)
