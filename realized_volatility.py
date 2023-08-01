# Load Libraries
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')
import gc
import glob
import os
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from IPython.display import display
from joblib import delayed, Parallel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from darts import TimeSeries
from darts.models import TCNModel, RNNModel, TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mape, r2_score
from darts.utils.missing_values import fill_missing_values
from darts.datasets import AirPassengersDataset, SunspotsDataset, EnergyDataset
from darts.metrics import mae, rmse, mse, mape
import random
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
import shutil
from sklearn.preprocessing import MinMaxScaler
from darts.utils.statistics import check_seasonality, plot_acf
import darts.utils.timeseries_generation as tg
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.missing_values import fill_missing_values
from darts.utils.likelihood_models import GaussianLikelihood
import uuid

DATA_DIR = '/Users/pujanmaharjan/uni adelaide/uofa_research_project/datasets'
SEED = 0
NUM_WORKERS = 2 #4
MULTIPROCESSING_CONTEXT = 'spawn'
ENSEMBLE_METHOD = 'mean'

def split_df_into_train_test(df):
    train_index = int(len(df) * 0.8)
    train_data = df[:train_index]
    test_data = df[train_index:]
    # print('Train data shape ', train_data.shape)
    # print('Test data shape ', test_data.shape)
    return train_data, test_data

def split_df_into_train_val_test(df):
    # split 70, 15, 15
    train_index = int(len(df) * 0.7)
    train_data = df[:train_index]
    val_test_data = df[train_index:]
    val_index = int(len(val_test_data) * 0.5)
    val_data = val_test_data[:val_index]
    test_data = val_test_data[val_index:]
    # print('Total data shape ', df.shape)
    # print('train shape ', train_data.shape)
    # print('validation shape ', val_data.shape)
    # print('test shape ', test_data.shape)
    return train_data, val_data, test_data

@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f'[{name}] {elapsed: .3f}sec')

def print_trace(name: str = ''):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())

def make_rv_lags(df_rv_lag, number_of_periods: int):
    # add past rv
    for i in range(1, number_of_periods + 1):
        df_rv_lag[f'lag_{i}_rv'] = df_rv_lag['target'].shift(periods=i)

    df_rv_lag = df_rv_lag.dropna()
    return df_rv_lag

class DataBlock(Enum):
    TRAIN = 1
    TEST = 2
    BOTH = 3

def load_stock_data(stock_id: int, directory: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA_DIR, 'optiver-realized-volatility-prediction', directory, f'stock_id={stock_id}'))

def load_data(stock_id: int, stem: str, block: DataBlock) -> pd.DataFrame:
    if block == DataBlock.TRAIN:
        return load_stock_data(stock_id, f'{stem}_train.parquet')
    elif block == DataBlock.TEST:
        return load_stock_data(stock_id, f'{stem}_test.parquet')
    else:
        return pd.concat([
            load_data(stock_id, stem, DataBlock.TRAIN),
            load_data(stock_id, stem, DataBlock.TEST)
        ]).reset_index(drop=True)

def load_book(stock_id: int, block: DataBlock=DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, 'book', block)

def load_trade(stock_id: int, block=DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, 'trade', block)

def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def log_return(series: np.ndarray):
    return np.log(series).diff()

def log_return_df2(series: np.ndarray):
    return np.log(series).diff(2)

def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ['time_id', 'stock_id']:
            ret.append(c[0])
        else:
            ret.append('.'.join([prefix] + list(c)))
    return ret

def make_book_feature(stock_id, 
                      block,
                      add_spread_features,
                      add_statistics_features,
                      add_book_time_features):
    book = load_book(stock_id, block)

    book['wap1'] = calc_wap1(book)
    book['wap2'] = calc_wap2(book)
    book['log_return1'] = book.groupby(['time_id'], group_keys=False)['wap1'].apply(log_return)
    book['log_return2'] = book.groupby(['time_id'], group_keys=False)['wap2'].apply(log_return)
    book['log_return_ask1'] = book.groupby(['time_id'], group_keys=False)['ask_price1'].apply(log_return)
    book['log_return_ask2'] = book.groupby(['time_id'], group_keys=False)['ask_price2'].apply(log_return)
    book['log_return_bid1'] = book.groupby(['time_id'], group_keys=False)['bid_price1'].apply(log_return)
    book['log_return_bid2'] = book.groupby(['time_id'], group_keys=False)['bid_price2'].apply(log_return)

    if add_spread_features:
        book['wap_balance'] = abs(book['wap1'] - book['wap2'])
        book['price_spread'] = (book['ask_price1'] - book['bid_price1']) / ((book['ask_price1'] + book['bid_price1']) / 2)
        book['bid_spread'] = book['bid_price1'] - book['bid_price2']
        book['ask_spread'] = book['ask_price1'] - book['ask_price2']
        book['total_volume'] = (book['ask_size1'] + book['ask_size2']) + (book['bid_size1'] + book['bid_size2'])
        book['volume_imbalance'] = abs((book['ask_size1'] + book['ask_size2']) - (book['bid_size1'] + book['bid_size2']))

    features = {
        'wap1': [np.sum],
        'wap2': [np.sum],
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return_ask1': [realized_volatility],
        'log_return_ask2': [realized_volatility],
        'log_return_bid1': [realized_volatility],
        'log_return_bid2': [realized_volatility],
    }

    if add_spread_features and add_statistics_features:
        features = {
            'seconds_in_bucket': ['count'],
            'wap1': [np.sum, np.mean, np.std],
            'wap2': [np.sum, np.mean, np.std],
            'log_return1': [np.sum, realized_volatility, np.mean, np.std],
            'log_return2': [np.sum, realized_volatility, np.mean, np.std],
            'log_return_ask1': [np.sum, realized_volatility, np.mean, np.std],
            'log_return_ask2': [np.sum, realized_volatility, np.mean, np.std],
            'log_return_bid1': [np.sum, realized_volatility, np.mean, np.std],
            'log_return_bid2': [np.sum, realized_volatility, np.mean, np.std],
            'wap_balance': [np.sum, np.mean, np.std],
            'price_spread':[np.sum, np.mean, np.std],
            'bid_spread':[np.sum, np.mean, np.std],
            'ask_spread':[np.sum, np.mean, np.std],
            'total_volume':[np.sum, np.mean, np.std],
            'volume_imbalance':[np.sum, np.mean, np.std]
        }
    elif add_spread_features and not add_statistics_features:
        features = {
            'seconds_in_bucket': ['count'],
            'wap1': [np.sum],
            'wap2': [np.sum],
            'log_return1': [realized_volatility],
            'log_return2': [realized_volatility],
            'log_return_ask1': [np.sum, realized_volatility],
            'log_return_ask2': [np.sum, realized_volatility],
            'log_return_bid1': [np.sum, realized_volatility],
            'log_return_bid2': [np.sum, realized_volatility],
            'wap_balance': [np.sum],
            'price_spread':[np.sum],
            'bid_spread':[np.sum],
            'ask_spread':[np.sum],
            'total_volume':[np.sum],
            'volume_imbalance':[np.sum]
        }

    agg = book.groupby('time_id', group_keys=False).agg(features).reset_index(drop=False)
    agg.columns = flatten_name('book', agg.columns)
    agg['stock_id'] = stock_id

    if add_book_time_features:
        for time in [450, 300, 150]:
            d = book[book['seconds_in_bucket'] >= time].groupby('time_id', group_keys=False).agg(features).reset_index(drop=False)
            d.columns = flatten_name(f'book_{time}', d.columns)
            agg = pd.merge(agg, d, on='time_id', how='left')
    return agg

def make_trade_feature(stock_id, block, add_trade_time_features):
    trade = load_trade(stock_id, block)
    trade['log_return'] = trade.groupby('time_id', group_keys=False)['price'].apply(log_return)

    features = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':['count'],
        'size':[np.sum],
        'order_count':[np.mean],
    }

    agg = trade.groupby('time_id', group_keys=False).agg(features).reset_index()
    agg.columns = flatten_name('trade', agg.columns)
    agg['stock_id'] = stock_id

    if add_trade_time_features:
        for time in [450, 300, 150]:
            d = trade[trade['seconds_in_bucket'] >= time].groupby('time_id').agg(features).reset_index(drop=False)
            d.columns = flatten_name(f'trade_{time}', d.columns)
            agg = pd.merge(agg, d, on='time_id', how='left')
    return agg

def make_book_feature_v2(stock_id, block):
    book = load_book(stock_id, block)

    prices = book.set_index('time_id')[['bid_price1', 'ask_price1', 'bid_price2', 'ask_price2']]
    time_ids = list(set(prices.index))

    ticks = {}
    for tid in time_ids:
        try:
            price_list = prices.loc[tid].values.flatten()
            price_diff = sorted(np.diff(sorted(set(price_list))))
            ticks[tid] = price_diff[0]
        except Exception:
            print_trace(f'tid={tid}')
            ticks[tid] = np.nan

    dst = pd.DataFrame()
    dst['time_id'] = np.unique(book['time_id'])
    dst['stock_id'] = stock_id
    dst['tick_size'] = dst['time_id'].map(ticks)

    return dst
    

def make_features(base, block, add_spread_features, add_statistics_features, add_book_time_features, add_trade_time_features):
    stock_ids = set(base['stock_id'])
    with timer('books'):
        books = Parallel(n_jobs=-1)(delayed(make_book_feature)(stock_id, block, add_spread_features, add_statistics_features, add_book_time_features) for stock_id in stock_ids)
        book = pd.concat(books)

    with timer('trades'):
        trades = Parallel(n_jobs=-1)(delayed(make_trade_feature)(stock_id, block, add_trade_time_features) for stock_id in stock_ids)
        trade = pd.concat(trades)

    with timer('extra features'):
        df = pd.merge(base, book, on=['stock_id', 'time_id'], how='left')
        df = pd.merge(df, trade, on=['stock_id', 'time_id'], how='left')

    df = make_rv_lags(df, 1)
    
    return df

def make_features_tick_size(base, block):
    stock_ids = set(base['stock_id'])
    with timer('books(v2)'):
        books = Parallel(n_jobs=-1)(delayed(make_book_feature_v2)(i, block) for i in stock_ids)
        book_v2 = pd.concat(books)

    d = pd.merge(base, book_v2, on=['stock_id', 'time_id'], how='left')
    return d


class Neighbors:
    def __init__(self,
                 n_neighbors: int,
                 name: str,
                 pivot: pd.DataFrame,
                 p: float,
                 metric: str = 'minkowski',
                 metric_params: Optional[Dict] = None,
                 exclude_self: bool = False):
        self.name = name
        self.exclude_self = exclude_self
        self.p = p
        self.metric = metric
        self.n_neighbors = n_neighbors

        if metric == 'random':
            n_queries = len(pivot)
            self.neighbors = np.random.randint(n_queries, size=(n_queries, n_neighbors))
        else:
            print('metric ', metric)

            nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                p=p,
                metric=metric,
                metric_params=metric_params
            )

            nn.fit(pivot)
            _, self.neighbors = nn.kneighbors(pivot, return_distance=True)

        self.columns = self.index = self.feature_values = self.feature_col = None

    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        raise NotImplementedError()

    def make_nn_feature(self, n=5, agg=np.mean) -> pd.DataFrame:
        assert self.feature_values is not None, "should call rearrange_feature_values beforehand"

        start = 1 if self.exclude_self else 0

        pivot_aggs = pd.DataFrame(
            agg(self.feature_values[start:n,:,:], axis=0),
            columns=self.columns,
            index=self.index
        )

        dst = pivot_aggs.unstack().reset_index()
        dst.columns = ['stock_id', 'time_id', f'{self.feature_col}_nn{n}_{self.name}_{agg.__name__}']
        # print('Destination columns ', dst.columns)
        return dst


class TimeIdNeighbors(Neighbors):
    def __init__(self, n_neighbors: int, name: str, pivot: pd.DataFrame, p: float, metric: str = 'minkowski', metric_params: Dict | None = None, exclude_self: bool = False):
        super().__init__(n_neighbors, name, pivot, p, metric, metric_params, exclude_self)

    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        feature_pivot = df.pivot('time_id', 'stock_id', feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())
        feature_pivot.head()

        feature_values = np.zeros((self.n_neighbors, *feature_pivot.shape))

        for i in range(self.n_neighbors):
            feature_values[i, :, :] += feature_pivot.values[self.neighbors[:, i], :]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"time-id NN (name={self.name}, metric={self.metric}, p={self.p})"


class StockIdNeighbors(Neighbors):
    def __init__(self, n_neighbors: int, name: str, pivot: pd.DataFrame, p: float, metric: str = 'minkowski', metric_params: Dict | None = None, exclude_self: bool = False):
        super().__init__(n_neighbors, name, pivot, p, metric, metric_params, exclude_self)

    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """stock-id based nearest neighbor features"""
        feature_pivot = df.pivot('time_id', 'stock_id', feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        feature_values = np.zeros((self.n_neighbors, *feature_pivot.shape))

        for i in range(self.n_neighbors):
            feature_values[i, :, :] += feature_pivot.values[:, self.neighbors[:, i]]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"stock-id NN (name={self.name}, metric={self.metric}, p={self.p})"

# add_tau_features
# the tau itself is meaningless for GBDT, but useful as input to aggregate in Nearest Neighbor features
def add_tau_features(df_tau):
    df_tau['trade.tau'] = np.sqrt(1 / df_tau['trade.seconds_in_bucket.count'])
    df_tau['trade_150.tau'] = np.sqrt(1 / df_tau['trade_150.seconds_in_bucket.count'])
    df_tau['book.tau'] = np.sqrt(1 / df_tau['book.seconds_in_bucket.count'])
    df_tau['real_price'] = 0.01 / df_tau['tick_size']

    return df_tau

# build_nearest_neighbors
def build_nearest_neighbors(df_nn,
                            n_neighbors: int,
                            use_price_nn_features: bool,
                            use_volume_nn_features: bool,
                            use_size_nn_features: bool,
                            use_random_nn_features: bool):
    time_id_neighbors: List[Neighbors] = []
    stock_id_neighbors: List[Neighbors] = []

    with timer('knn fit'):
        df_pv = df_nn[['stock_id', 'time_id']].copy()
        df_pv['price'] = 0.01 / df_nn['tick_size']
        df_pv['vol'] = df_nn['book.log_return1.realized_volatility']
        df_pv['trade.tau'] = df_nn['trade.tau']
        df_pv['trade.size.sum'] = df_nn['book.total_volume.sum']

        print('USE_PRICE_NN_FEATURES ', use_price_nn_features)
        if use_price_nn_features:
            pivot = df_pv.pivot('time_id', 'stock_id', 'price')
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

            time_id_neighbors.append(
                TimeIdNeighbors(
                    n_neighbors,
                    'time_price_c',
                    pivot,
                    p=2,
                    metric='canberra',
                    exclude_self=True
                )
            )
            time_id_neighbors.append(
                TimeIdNeighbors(
                    n_neighbors,
                    'time_price_m',
                    pivot,
                    p=2,
                    metric='mahalanobis',
                    metric_params={'VI':np.linalg.inv(np.cov(pivot.values.T))}
                )
            )
            stock_id_neighbors.append(
                StockIdNeighbors(
                    n_neighbors,
                    'stock_price_l1',
                    minmax_scale(pivot.transpose()),
                    p=1,
                    exclude_self=True)
            )

        print('USE_VOL_NN_FEATURES ', use_volume_nn_features)
        if use_volume_nn_features:
            pivot = df_pv.pivot('time_id', 'stock_id', 'vol')
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

            time_id_neighbors.append(
                TimeIdNeighbors(n_neighbors, 'time_vol_l1', pivot, p=1)
            )
            stock_id_neighbors.append(
                StockIdNeighbors(
                    n_neighbors,
                    'stock_vol_l1',
                    minmax_scale(pivot.transpose()),
                    p=1,
                    exclude_self=True
                )
            )

        print('USE_SIZE_NN_FEATURES ', use_size_nn_features)
        if use_size_nn_features:
            pivot = df_pv.pivot('time_id', 'stock_id', 'trade.size.sum')
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

            time_id_neighbors.append(
                TimeIdNeighbors(
                    n_neighbors,
                    'time_size_m',
                    pivot,
                    p=2,
                    metric='mahalanobis',
                    # metric_params={'V':np.cov(pivot.values.T)}
                    metric_params={'VI':np.linalg.inv(np.cov(pivot.values.T))}
                )
            )
            time_id_neighbors.append(
                TimeIdNeighbors(
                    n_neighbors,
                    'time_size_c',
                    pivot,
                    p=2,
                    metric='canberra'
                )
            )

        print('USE_RANDOM_NN_FEATURES ', use_random_nn_features)
        if use_random_nn_features:
            pivot = df_pv.pivot('time_id', 'stock_id', 'vol')
            pivot = pivot.fillna(pivot.mean())
            pivot = pd.DataFrame(minmax_scale(pivot))

            time_id_neighbors.append(
                TimeIdNeighbors(
                    n_neighbors,
                    'time_random',
                    pivot,
                    p=2,
                    metric='random'
                )
            )
            stock_id_neighbors.append(
                StockIdNeighbors(
                    n_neighbors,
                    'stock_random',
                    pivot.transpose(),
                    p=2,
                    metric='random')
            )

    return time_id_neighbors, stock_id_neighbors

# aggregate_features_with_neighbors
# features with large changes over time are converted to relative ranks within time-id
def aggregate_features_with_neighbors(df_agg):
    df_agg['trade.order_count.mean'] = df_agg.groupby('time_id', group_keys=False)['trade.order_count.mean'].rank()
    df_agg['book.total_volume.sum']  = df_agg.groupby('time_id', group_keys=False)['book.total_volume.sum'].rank()
    df_agg['book.total_volume.mean'] = df_agg.groupby('time_id', group_keys=False)['book.total_volume.mean'].rank()
    df_agg['book.total_volume.std']  = df_agg.groupby('time_id')['book.total_volume.std'].rank()

    df_agg['trade.tau'] = df_agg.groupby('time_id', group_keys=False)['trade.tau'].rank()

    for dt in [150, 300, 450]:
        df_agg[f'book_{dt}.total_volume.sum']  = df_agg.groupby('time_id', group_keys=False)[f'book_{dt}.total_volume.sum'].rank()
        df_agg[f'book_{dt}.total_volume.mean'] = df_agg.groupby('time_id', group_keys=False)[f'book_{dt}.total_volume.mean'].rank()
        df_agg[f'book_{dt}.total_volume.std']  = df_agg.groupby('time_id', group_keys=False)[f'book_{dt}.total_volume.std'].rank()
        df_agg[f'trade_{dt}.order_count.mean'] = df_agg.groupby('time_id', group_keys=False)[f'trade_{dt}.order_count.mean'].rank()

    return df_agg


# make_nearest_neighbor_feature
def make_nearest_neighbor_feature(df_nn: pd.DataFrame, time_id_neighbors, stock_id_neighbors, use_price_nn_features) -> pd.DataFrame:
    df_nnf = df_nn.copy()

    feature_cols_stock = {
        'book.log_return1.realized_volatility': [np.mean, np.min, np.max, np.std],
        'trade.seconds_in_bucket.count': [np.mean],
        'trade.tau': [np.mean],
        'trade_150.tau': [np.mean],
        'book.tau': [np.mean],
        'trade.size.sum': [np.mean],
        'book.seconds_in_bucket.count': [np.mean],
    }

    feature_cols = {
        'book.log_return1.realized_volatility': [np.mean, np.min, np.max, np.std],
        'real_price': [np.max, np.mean, np.min],
        'trade.seconds_in_bucket.count': [np.mean],
        'trade.tau': [np.mean],
        'trade.size.sum': [np.mean],
        'book.seconds_in_bucket.count': [np.mean],
        'trade_150.tau_nn20_stock_vol_l1_mean': [np.mean],
        'trade.size.sum_nn20_stock_vol_l1_mean': [np.mean],
    }

    time_id_neigbor_sizes = [3, 5, 10, 20, 40]
    time_id_neigbor_sizes_vol = [2, 3, 5, 10, 20, 40]
    stock_id_neighbor_sizes = [10, 20, 40]

    ndf: Optional[pd.DataFrame] = None

    def _add_ndf(ndf: Optional[pd.DataFrame], dst: pd.DataFrame) -> pd.DataFrame:
        if ndf is None:
            return dst
        else:
            ndf[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
            return ndf

    # neighbor stock_id
    for feature_col in feature_cols_stock.keys():
        # print('Feature column ', feature_col)
        try:
            if feature_col not in df_nnf.columns:
                print(f"column {feature_col} is skipped")
                continue

            if not stock_id_neighbors:
                continue

            for nn in stock_id_neighbors:
                nn.rearrange_feature_values(df_nnf, feature_col)

            for agg in feature_cols_stock[feature_col]:
                for n in stock_id_neighbor_sizes:
                    try:
                        for nn in stock_id_neighbors:
                            dst = nn.make_nn_feature(n, agg)
                            ndf = _add_ndf(ndf, dst)
                    except Exception:
                        print_trace('stock-id nn')
                        pass
        except Exception:
            print_trace('stock-id nn')
            pass

    if ndf is not None:
        df_nnf = pd.merge(df_nnf, ndf, on=['time_id', 'stock_id'], how='left')
    ndf = None

    # neighbor time_id
    for feature_col in feature_cols.keys():
        # print('Feature col ', feature_col)
        try:
            # if feature_col == 'real_price':
            #     continue
            if feature_col not in df_nnf.columns:
                print(f"column {feature_col} is skipped")
                continue

            for nn in time_id_neighbors:
                # print('calling rearrange_feature_values for feature_col ', feature_col)
                nn.rearrange_feature_values(df_nnf, feature_col)

            if 'volatility' in feature_col:
                time_id_ns = time_id_neigbor_sizes_vol
            else:
                time_id_ns = time_id_neigbor_sizes

            for agg in feature_cols[feature_col]:
                for n in time_id_ns:
                    try:
                        for nn in time_id_neighbors:
                            # print('calling make_nn_feature for feature_col ', feature_col)
                            dst = nn.make_nn_feature(n, agg)
                            ndf = _add_ndf(ndf, dst)
                    except Exception:
                        print_trace('Exception in time-id nn in make_nn_feature ', feature_col)
                        pass
        except Exception:
            print_trace('Exception in neighbor time-id nn ', feature_col)

    if ndf is not None:
        df_nnf = pd.merge(df_nnf, ndf, on=['time_id', 'stock_id'], how='left')

    # features further derived from nearest neighbor features
    try:
        if use_price_nn_features:
            for sz in time_id_neigbor_sizes:
                denominator = f"real_price_nn{sz}_time_price_c"

                df_nnf[f'real_price_rankmin_{sz}']  = df_nnf['real_price'] / df_nnf[f"{denominator}_amin"]
                df_nnf[f'real_price_rankmax_{sz}']  = df_nnf['real_price'] / df_nnf[f"{denominator}_amax"]
                df_nnf[f'real_price_rankmean_{sz}'] = df_nnf['real_price'] / df_nnf[f"{denominator}_mean"]

            for sz in time_id_neigbor_sizes_vol:
                denominator = f"book.log_return1.realized_volatility_nn{sz}_time_price_c"

                df_nnf[f'vol_rankmin_{sz}'] = \
                    df_nnf['book.log_return1.realized_volatility'] / df_nnf[f"{denominator}_amin"]
                df_nnf[f'vol_rankmax_{sz}'] = \
                    df_nnf['book.log_return1.realized_volatility'] / df_nnf[f"{denominator}_amax"]

        price_cols = [c for c in df_nnf.columns if 'real_price' in c and 'rank' not in c]
        for c in price_cols:
            del df_nnf[c]

        if use_price_nn_features:
            for sz in time_id_neigbor_sizes_vol:
                tgt = f'book.log_return1.realized_volatility_nn{sz}_time_price_m_mean'
                df_nnf[f'{tgt}_rank'] = df_nnf.groupby('time_id', group_keys=False)[tgt].rank()
    except Exception:
        print_trace('nn features')

    return df_nnf

# skew correction for NN
def skew_correction_for_nn(df_skew):
    cols_to_log = [
        'trade.size.sum',
        'trade_150.size.sum',
        'trade_300.size.sum',
        'trade_450.size.sum',
        'volume_imbalance'
    ]
    for c in df_skew.columns:
        for check in cols_to_log:
            try:
                if check in c:
                    df_skew[c] = np.log(df_skew[c]+1)
                    break
            except Exception:
                print_trace('log1p')

    return df_skew

# Rolling average of RV for similar trading volume
def rolling_average_of_rv_for_similar_trading_volume(df_ra):
    try:
        df_ra.sort_values(by=['stock_id', 'book.total_volume.sum'], inplace=True)
        df_ra.reset_index(drop=True, inplace=True)

        roll_target = 'book.log_return1.realized_volatility'

        for window_size in [3, 10]:
            df_ra[f'realized_volatility_roll{window_size}_by_book.total_volume.mean'] = \
                df_ra.groupby('stock_id', group_keys=False)[roll_target].rolling(window_size, center=True, min_periods=1) \
                                                    .mean() \
                                                    .reset_index() \
                                                    .sort_values(by=['level_1'])[roll_target].values
    except Exception:
        print_trace('mean RV')

    return df_ra

# reverse engineering time-id order
@contextmanager
def timer(name):
    s = time.time()
    yield
    e = time.time() - s
    print(f"[{name}] {e:.3f}sec")

def calc_price2(df):
    tick = sorted(np.diff(sorted(np.unique(df.values.flatten()))))[0]
    return 0.01 / tick

def calc_prices(r):
    df = pd.read_parquet(r.book_path, columns=['time_id', 'ask_price1', 'ask_price2', 'bid_price1', 'bid_price2'])
    df = df.set_index('time_id')
    df = df.groupby(level='time_id', group_keys=False).apply(calc_price2).to_frame('price').reset_index()
    df['stock_id'] = r.stock_id
    return df

def sort_manifold(df, clf):
    df_ = df.set_index('time_id')
    df_ = pd.DataFrame(minmax_scale(df_.fillna(df_.mean())))

    X_compoents = clf.fit_transform(df_)

    dft = df.reindex(np.argsort(X_compoents[:,0])).reset_index(drop=True)
    return np.argsort(X_compoents[:, 0]), X_compoents

def reconstruct_time_id_order():
    print('reconstruct_time_id_order started. loading files')
    with timer('load files'):
        book_path = DATA_DIR + '/optiver-realized-volatility-prediction/book_train.parquet/**/*.parquet'
        print('book path ', book_path)
        df_files = pd.DataFrame(
            {'book_path': glob.glob(book_path)}).eval('stock_id = book_path.str.extract("stock_id=(\d+)").astype("int")', engine='python')

    print('reconstruct_time_id_order calculating prices')
    with timer('calc prices'):
        df_prices = pd.concat(Parallel(n_jobs=4, verbose=51)(delayed(calc_prices)(r) for _, r in df_files.iterrows()))
        df_prices = df_prices.pivot('time_id', 'stock_id', 'price')
        df_prices.columns = [f'stock_id={i}' for i in df_prices.columns]
        df_prices = df_prices.reset_index(drop=False)

    print('reconstruct_time_id_order tSNE')
    with timer('t-SNE(400) -> 50'):
        clf = TSNE(n_components=1, perplexity=400, random_state=0, n_iter=2000)
        order, X_compoents = sort_manifold(df_prices, clf)

        clf = TSNE(n_components=1, perplexity=50, random_state=0, init=X_compoents, n_iter=2000, method='exact')
        order, X_compoents = sort_manifold(df_prices, clf)

        df_ordered = df_prices.reindex(order).reset_index(drop=True)
        if df_ordered['stock_id=0'].iloc[0] > df_ordered['stock_id=0'].iloc[-1]:
            df_ordered = df_ordered.reindex(df_ordered.index[::-1]).reset_index(drop=True)

    return df_ordered[['time_id']]

# chek_null_columns
def chek_null_columns(X):
    xp = X.isna().any()
    xp_null = xp.loc[lambda x : x == True]
    nan_columns = list(xp_null.index)
    print('Null columns ', nan_columns)
    # X = X.drop(columns=nan_columns)
    # return X

# modal results operations
model_results = []
def get_model_results_df():
    return pd.DataFrame(model_results)

def reset_model_results():
    model_results = []

def add_model_result(model_name: str, y_true, y_pred, isDart: bool, feature: str, 
    time_taken: datetime, learning_rate: float, epochs: int,
    stock_ids: list[int]):
    if y_true is None:
        raise ValueError("y_true is None")

    if y_pred is None:
        raise ValueError("y_pred is None")

    if isDart:
        mae_value = mae(y_true, y_pred)
        rmse_value = rmse(y_true, y_pred)
        mse_value = mse(y_true, y_pred)
    else:
        mse_value = mean_squared_error(y_true, y_pred)
        rmse_value = mean_squared_error(y_true, y_pred, squared=False)
        mae_value = mean_absolute_error(y_true, y_pred)

    
    model_results.append({'model_name': model_name,
                            'mse': mse_value,
                            'rmse': rmse_value,
                            'mae': mae_value,
                            'added_date': datetime.now(),
                            'feature': feature,
                            'time_taken': time_taken.total_seconds(),
                            'learning_rate': learning_rate,
                            'epochs': epochs,
                            'stock_ids': stock_ids,
                            'stock_ids_count': len(stock_ids)
                            })

    return model_results


# light gbm
def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def feval_RMSPE(preds, train_data):
    labels = train_data.get_label()
    return 'RMSPE', round(rmspe(y_true = labels, y_pred = preds),5), False

# from: https://blog.amedama.jp/entry/lightgbm-cv-feature-importance
def plot_importance(cvbooster, figsize=(10, 10)):
    raw_importances = cvbooster.feature_importance(importance_type='gain')
    feature_name = cvbooster.boosters[0].feature_name()
    importance_df = pd.DataFrame(data=raw_importances,
                                 columns=feature_name)
    # order by average importance across folds
    sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top-n
    PLOT_TOP_N = 50
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()

def get_X(df_src):
    cols = [c for c in df_src.columns if c not in ['time_id', 'target', 'tick_size']]
    return df_src[cols]

class EnsembleModel:
    def __init__(self, models: List[lgb.Booster], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

        features = list(self.models[0].feature_name())

        for m in self.models[1:]:
            assert features == list(m.feature_name())

    def predict(self, x):
        predicted = np.zeros((len(x), len(self.models)))

        for i, m in enumerate(self.models):
            w = self.weights[i] if self.weights is not None else 1
            predicted[:, i] = w * m.predict(x)

        ttl = np.sum(self.weights) if self.weights is not None else len(self.models)
        return np.sum(predicted, axis=1) / ttl

    def feature_name(self) -> List[str]:
        return self.models[0].feature_name()

# add_results_from_light_gbm
def add_results_from_light_gbm(X_train_lgbm, y_train_lgbm, X_test_lgbm, y_test_lgbm, feature, lr=0.3):
    params = {
    'objective': 'regression',
    'verbose': 0,
    'metric': '',
    'reg_alpha': 5,
    'reg_lambda': 5,
    'min_data_in_leaf': 1000,
    'max_depth': -1,
    'num_leaves': 128,
    'colsample_bytree': 0.3,
    'learning_rate': lr
    }

    start_time = datetime.now()
    ds = lgb.Dataset(X_train_lgbm, y_train_lgbm, weight=1/np.power(y_train_lgbm, 2))

    ret = lgb.cv(params, ds, num_boost_round=8000,
                    feval=feval_RMSPE,
                    stratified=False,
                    return_cvbooster=True,
                    verbose_eval=20,
                    early_stopping_rounds=int(40*0.1/lr))

    # print(f"# overall RMSPE: {ret['RMSPE-mean'][-1]}")

    best_iteration = len(ret['RMSPE-mean'])

    # print('boosters length ', len(ret['cvbooster'].boosters))

    best_mae = None
    best_y_pred = None

    for i in range(len(ret['cvbooster'].boosters)):
        y_pred = ret['cvbooster'].boosters[i].predict(X_test_lgbm, num_iteration=best_iteration)
        # print('y_pred here ', y_pred)
        mae_value = mean_absolute_error(y_test_lgbm, y_pred)
        print('mae value ', mae_value)
        if best_mae == None:
            best_mae = mae_value

        if mae_value < best_mae:
            print('updating best mae value')
            best_mae = mae_value
            best_y_pred = y_pred

    time_taken = datetime.now() - start_time
    add_model_result('LightGBM', y_test_lgbm, best_y_pred, False, feature, time_taken, lr, 10)


# NN Training

null_check_cols = [
    'book.log_return1.realized_volatility',
    'book_150.log_return1.realized_volatility',
    'book_300.log_return1.realized_volatility',
    'book_450.log_return1.realized_volatility',
    'trade.log_return.realized_volatility',
    'trade_150.log_return.realized_volatility',
    'trade_300.log_return.realized_volatility',
    'trade_450.log_return.realized_volatility'
]

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def rmspe_metric(y_true, y_pred):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return rmspe


def rmspe_loss(y_true, y_pred):
    rmspe = torch.sqrt(torch.mean(torch.square((y_true - y_pred) / y_true)))
    return rmspe


class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))

def RMSPELoss_Tabnet(y_pred, y_true):
    return torch.sqrt(torch.mean( ((y_true - y_pred) / y_true) ** 2 )).clone()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TabularDataset(Dataset):
    def __init__(self, x_num: np.ndarray, y: Optional[np.ndarray]):
        super().__init__()
        self.x_num = x_num
        self.y = y

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx]
        else:
            return self.x_num[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self,
                 src_num_dim: int,
                 dropout: float = 0.0,
                 hidden: int = 50,
                 bn: bool = False):
        super().__init__()

        if bn:
            self.sequence = nn.Sequential(
                nn.Linear(src_num_dim, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        else:
            self.sequence = nn.Sequential(
                nn.Linear(src_num_dim, hidden),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

    def forward(self, x_num):
        x = self.sequence(x_num)
        return torch.squeeze(x)


class CNN(nn.Module):
    def __init__(self,
                 num_features: int,
                 hidden_size: int,
                 emb_dim: int = 10,
                 dropout_cat: float = 0.2,
                 channel_1: int = 256,
                 channel_2: int = 512,
                 channel_3: int = 512,
                 dropout_top: float = 0.1,
                 dropout_mid: float = 0.3,
                 dropout_bottom: float = 0.2,
                 weight_norm: bool = True,
                 two_stage: bool = True,
                 celu: bool = True,
                 kernel1: int = 5,
                 leaky_relu: bool = False):
        super().__init__()

        num_targets = 1

        cha_1_reshape = int(hidden_size / channel_1)
        cha_po_1 = int(hidden_size / channel_1 / 2)
        cha_po_2 = int(hidden_size / channel_1 / 2 / 2) * channel_3

        self.cha_1 = channel_1
        self.cha_2 = channel_2
        self.cha_3 = channel_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2
        self.two_stage = two_stage

        self.expand = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_top),
            nn.utils.weight_norm(nn.Linear(num_features, hidden_size), dim=None),
            nn.CELU(0.06) if celu else nn.ReLU()
        )

        def _norm(layer, dim=None):
            return nn.utils.weight_norm(layer, dim=dim) if weight_norm else layer

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(channel_1),
            nn.Dropout(dropout_top),
            _norm(nn.Conv1d(channel_1, channel_2, kernel_size=kernel1, stride=1, padding=kernel1 // 2, bias=False)),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=cha_po_1),
            nn.BatchNorm1d(channel_2),
            nn.Dropout(dropout_top),
            _norm(nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.ReLU()
        )

        if self.two_stage:
            self.conv2 = nn.Sequential(
                nn.BatchNorm1d(channel_2),
                nn.Dropout(dropout_mid),
                _norm(nn.Conv1d(channel_2, channel_2, kernel_size=3, stride=1, padding=1, bias=True)),
                nn.ReLU(),
                nn.BatchNorm1d(channel_2),
                nn.Dropout(dropout_bottom),
                _norm(nn.Conv1d(channel_2, channel_3, kernel_size=5, stride=1, padding=2, bias=True)),
                nn.ReLU()
            )

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        if leaky_relu:
            self.dense = nn.Sequential(
                nn.BatchNorm1d(cha_po_2),
                nn.Dropout(dropout_bottom),
                _norm(nn.Linear(cha_po_2, num_targets), dim=0),
                nn.LeakyReLU()
            )
        else:
            self.dense = nn.Sequential(
                nn.BatchNorm1d(cha_po_2),
                nn.Dropout(dropout_bottom),
                _norm(nn.Linear(cha_po_2, num_targets), dim=0)
            )

    def forward(self, x_num):
        x = self.expand(x_num)
        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)
        x = self.conv1(x)
        if self.two_stage:
            x = self.conv2(x) * x

        x = self.max_po_c2(x)
        x = self.flt(x)
        x = self.dense(x)

        return torch.squeeze(x)


def train_epoch(data_loader: DataLoader,
                model: nn.Module,
                optimizer,
                scheduler,
                device,
                clip_grad: float = 1.5):
    model.train()
    losses = AverageMeter()
    step = 0

    for x_num, y in data_loader:
        batch_size = x_num.size(0)
        x_num = x_num.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)
        loss = rmspe_loss(y, model(x_num))
        losses.update(loss.detach().cpu().numpy(), batch_size)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        step += 1

    return losses.avg


def evaluate(data_loader: DataLoader, model, device):
    model.eval()

    losses = AverageMeter()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for x_num, y in data_loader:
            batch_size = x_num.size(0)
            x_num = x_num.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            with torch.no_grad():
                output = model(x_num)

            loss = rmspe_loss(y, output)
            losses.update(loss.detach().cpu().numpy(), batch_size)

            targets = y.detach().cpu().numpy()
            output = output.detach().cpu().numpy()

            final_targets.append(targets)
            final_outputs.append(output)

    final_targets = np.concatenate(final_targets)
    final_outputs = np.concatenate(final_outputs)

    try:
        metric = rmspe_metric(final_targets, final_outputs)
    except:
        metric = None

    return final_outputs, final_targets, losses.avg, metric

def predict_nn(X_df: pd.DataFrame,
               model: Union[List[MLP], MLP],
               device,
               ensemble_method='mean'):
    if not isinstance(model, list):
        model = [model]

    for m in model:
        m.eval()

    evaluation_dataset = TabularDataset(X_df.values, None)
    evaluation_data_loader = torch.utils.data.DataLoader(evaluation_dataset,
                                               batch_size=512,
                                               shuffle=False,
                                            #    multiprocessing_context=MULTIPROCESSING_CONTEXT,
                                               num_workers=NUM_WORKERS)

    final_outputs = []

    with torch.no_grad():
        for x_num in evaluation_data_loader:
            x_num = x_num.to(device, dtype=torch.float)

            outputs = []
            with torch.no_grad():
                for m in model:
                    output = m(x_num)
                    outputs.append(output.detach().cpu().numpy())

            if ensemble_method == 'median':
                pred = np.nanmedian(np.array(outputs), axis=0)
            else:
                pred = np.array(outputs).mean(axis=0)
            final_outputs.append(pred)

    final_outputs = np.concatenate(final_outputs)
    return final_outputs

def train_nn(
             X_train_df,
             y_train_df,
             X_val_df,
             y_val_df,
             device,
             emb_dim: int = 25,
             batch_size: int = 1024,
             model_type: str = 'mlp',
             mlp_dropout: float = 0.0,
             mlp_hidden: int = 64,
             mlp_bn: bool = False,
             cnn_hidden: int = 64,
             cnn_channel1: int = 32,
             cnn_channel2: int = 32,
             cnn_channel3: int = 32,
             cnn_kernel1: int = 5,
             cnn_celu: bool = False,
             cnn_weight_norm: bool = False,
             dropout_emb: bool = 0.0,
             lr: float = 1e-3,
             weight_decay: float = 0.0,
             model_path: str = 'fold_{}.pth',
             scaler_type: str = 'standard',
             output_dir: str = 'artifacts',
             scheduler_type: str = 'onecycle',
             optimizer_type: str = 'adam',
             max_lr: float = 0.01,
             epochs: int = 30,
             seed: int = 42,
             n_pca: int = -1,
             batch_double_freq: int = 50,
             cnn_dropout: float = 0.1,
             na_cols: bool = True,
             cnn_leaky_relu: bool = False,
             patience: int = 8,
             factor: float = 0.5):
    seed_everything(seed)

    os.makedirs(output_dir, exist_ok=True)

    best_losses = []
    best_predictions = []

    cur_batch = batch_size
    best_loss = 1e10
    best_prediction = None
    train_dataset = TabularDataset(X_train_df.values, y_train_df.values)
    valid_dataset = TabularDataset(X_val_df.values, y_val_df.values)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cur_batch, shuffle=False,
                                                num_workers=NUM_WORKERS,
                                                # multiprocessing_context=MULTIPROCESSING_CONTEXT
                                                )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cur_batch, shuffle=False,
                                            #    multiprocessing_context=MULTIPROCESSING_CONTEXT,
                                                num_workers=NUM_WORKERS)

    if model_type == 'mlp':
        model = MLP(X_train_df.shape[1],
                    dropout=mlp_dropout,
                    hidden=mlp_hidden,
                    bn=mlp_bn)
    elif model_type == 'cnn':
        model = CNN(X_train_df.shape[1],
                    hidden_size=cnn_hidden,
                    emb_dim=emb_dim,
                    dropout_cat=dropout_emb,
                    channel_1=cnn_channel1,
                    channel_2=cnn_channel2,
                    channel_3=cnn_channel3,
                    two_stage=False,
                    kernel1=cnn_kernel1,
                    celu=cnn_celu,
                    dropout_top=cnn_dropout,
                    dropout_mid=cnn_dropout,
                    dropout_bottom=cnn_dropout,
                    weight_norm=cnn_weight_norm,
                    leaky_relu=cnn_leaky_relu)
    else:
        raise NotImplementedError()
    model = model.to(device)

    if optimizer_type == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()

    scheduler = epoch_scheduler = None
    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, pct_start=0.1, div_factor=1e3,
                                                        max_lr=max_lr, epochs=epochs,
                                                        steps_per_epoch=len(train_loader))
    elif scheduler_type == 'reduce':
        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                                        mode='min',
                                                                        min_lr=1e-7,
                                                                        patience=patience,
                                                                        verbose=False,
                                                                        factor=factor)

    for epoch in range(epochs):
        if epoch > 0 and epoch % batch_double_freq == 0:
            cur_batch = cur_batch * 2
            # print(f'batch: {cur_batch}')
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=cur_batch,
                                                        shuffle=False,
                                                        # multiprocessing_context=MULTIPROCESSING_CONTEXT,
                                                        num_workers=NUM_WORKERS)

        train_loss = train_epoch(train_loader, model, opt, scheduler, device)
        predictions, valid_targets, valid_loss, rmspe = evaluate(valid_loader, model, device=device)
        # print(f"epoch {epoch}, train loss: {train_loss:.3f}, valid rmspe: {rmspe:.3f}")

        if epoch_scheduler is not None:
            epoch_scheduler.step(rmspe)

        if rmspe < best_loss:
            # print(f'new best:{rmspe}')
            best_loss = rmspe
            best_prediction = predictions
            model_save_path = DATA_DIR + "/" + os.path.join(output_dir, model_path.format(0))
            torch.save(model, model_save_path)

    best_predictions.append(best_prediction)
    best_losses.append(best_loss)
    # del model, train_dataset, valid_dataset, train_loader, valid_loader, X_tr, X_va, X_tr_cat, X_va_cat, y_tr, y_va, opt
    del train_dataset, valid_dataset, train_loader, valid_loader, opt
    if scheduler is not None:
        del scheduler
    gc.collect()
    # , scaler
    return model, best_losses, best_predictions

# get_device_name
def get_device_name():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"
device = torch.device(get_device_name())
# print('device', device)

def get_top_n_models(models, scores, top_n):
    if len(models) <= top_n:
        print('number of models are less than top_n. all models will be used')
        return models
    sorted_ = [(y, x) for y, x in sorted(zip(scores, models), key=lambda pair: pair[0])]
    print(f'scores(sorted): {[y for y, _ in sorted_]}')
    return [x for _, x in sorted_][:top_n]

# add_results_for_mlp
def add_results_for_mlp(X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp, X_test_mlp, y_test_mlp, feature, 
                        epochs, learning_rate, stock_ids: list[int]):
    start_time = datetime.now()
    model_mlp, nn_losses, nn_preds = train_nn(
                                            X_train_mlp,
                                            y_train_mlp,
                                            X_val_mlp,
                                            y_val_mlp,
                                            device=device,
                                            batch_size=512,
                                            mlp_bn=True,
                                            mlp_hidden=256,
                                            mlp_dropout=0.0,
                                            emb_dim=30,
                                            epochs=epochs,
                                            lr=learning_rate,
                                            max_lr=0.0055,
                                            weight_decay=1e-7,
                                            model_path='mlp_fold_{}' + f"_seed{SEED}.pth",
                                            seed=0)

    model_mlp_preds = predict_nn(X_test_mlp, model_mlp, device, ensemble_method=ENSEMBLE_METHOD)
    time_taken = datetime.now() - start_time
    add_model_result('MLP', y_test_mlp, model_mlp_preds, False, feature, time_taken, 
                     learning_rate, epochs, stock_ids)


# add_results_for_cnn
# lr = 0.00038
def add_results_for_cnn(X_train_cnn, y_train_cnn, X_val_cnn, y_val_cnn, X_test_cnn, y_test_cnn, feature,
                        epochs, learning_rate, stock_ids: list[int]):
    start_time = datetime.now()
    model_cnn, nn_losses, nn_preds = train_nn(
                                            X_train_cnn,
                                            y_train_cnn,
                                            X_val_cnn,
                                            y_val_cnn,
                                            device=device,
                                            cnn_hidden=8*128,
                                            batch_size=1280,
                                            model_type='cnn',
                                            emb_dim=30,
                                            epochs=epochs,
                                            cnn_channel1=128,
                                            cnn_channel2=3*128,
                                            cnn_channel3=3*128,
                                            lr=learning_rate, #0.0011,
                                            max_lr=0.0013,
                                            weight_decay=6.5e-6,
                                            optimizer_type='adam',
                                            scheduler_type='reduce',
                                            model_path='cnn_fold_{}' + f"_seed{SEED}.pth",
                                            seed=0,
                                            cnn_dropout=0.0,
                                            cnn_weight_norm=False, # Note: True
                                            cnn_leaky_relu=False,
                                            patience=8,
                                            factor=0.3)

    model_cnn_preds = predict_nn(X_test_cnn, model_cnn, device, ensemble_method=ENSEMBLE_METHOD)
    time_taken = datetime.now() - start_time
    add_model_result('CNN', y_test_cnn, model_cnn_preds, False, feature, time_taken, 
                     learning_rate, epochs, stock_ids)


# create_timeseries_data
def create_timeseries_data(df):
    df_ts = TimeSeries.from_dataframe(df)
    scaler = Scaler()
    df_ts = scaler.fit_transform(df_ts).astype(np.float32)
    # print('Length of Timeseries ', len(df_ts))
    return df_ts, scaler

# add_results_for_TCN
def add_results_for_TCN(X_train_ts_tcn, y_train_ts_tcn, X_val_ts_tcn, y_val_ts_tcn, X_test_ts_tcn, y_test_ts_tcn,
    feature, epochs, learning_rate, stock_ids: list[int]):
    time_start = datetime.now()
    model_tcn = TCNModel(
        input_chunk_length=512,
        output_chunk_length=1,
        n_epochs=epochs, #500
        dropout=0.1,
        dilation_base=2,
        weight_norm=True,
        kernel_size=3,
        num_filters=3,
        random_state=0,
        optimizer_kwargs = {'lr': learning_rate}
    )

    model_tcn.fit(
        series=y_train_ts_tcn,
        past_covariates=X_train_ts_tcn,
        val_series=y_val_ts_tcn,
        val_past_covariates=X_val_ts_tcn,
        verbose=False,
        num_loader_workers=NUM_WORKERS,
    )

    predictions_tcn = model_tcn.historical_forecasts(
        series=y_test_ts_tcn,
        past_covariates=X_test_ts_tcn,
        forecast_horizon=1,
        retrain=False,
        verbose=False,
    )

    # predictions_tcn = model_tcn.predict(n = 36,
    #                                 past_covariates=X_test_ts_tcn,
    #                                 verbose=False)

    time_taken = datetime.now() - time_start
    add_model_result('TCN', y_test_ts_tcn, predictions_tcn, True, feature, time_taken, 
                     learning_rate, epochs, stock_ids)


# add_results_for_lstm
def add_results_for_lstm(X_train_ts_lstm, y_train_ts_lstm, X_val_ts_lstm, y_val_ts_lstm, X_test_ts_lstm, 
                         y_test_ts_lstm, 
                        feature, epochs, learning_rate, stock_ids: list[int]):
    start_time = datetime.now()
    model_lstm = RNNModel(
        model="LSTM",
        hidden_dim=20,
        n_rnn_layers=2,
        dropout=0.2,
        batch_size=16,
        n_epochs=epochs,
        optimizer_kwargs={"lr": learning_rate},
        random_state=0,
        training_length=72,
        input_chunk_length=512,
        output_chunk_length=1,
        likelihood=GaussianLikelihood(),
    )

    model_lstm.fit(
        series=y_train_ts_lstm,
        future_covariates=X_train_ts_lstm,
        val_series=y_val_ts_lstm,
        val_future_covariates=X_val_ts_lstm,
        verbose=False,
        num_loader_workers=NUM_WORKERS,
    )

    predictions_lstm = model_lstm.historical_forecasts(
        series=y_test_ts_lstm,
        future_covariates=X_test_ts_lstm,
        forecast_horizon=1,
        retrain=False,
        verbose=False,
    )

    # predictions_lstm = model_lstm.predict(n = 36, past_covariates = X_test_ts_lstm, verbose=True)

    time_taken = datetime.now() - start_time
    add_model_result('LSTM', y_test_ts_lstm, predictions_lstm, True, feature, time_taken, 
                     learning_rate, epochs, stock_ids)
    
# add_results_for_transformer
def add_results_for_transformer(X_train_ts_trans, y_train_ts_trans, X_val_ts_trans, y_val_ts_trans, X_test_ts_trans, 
    y_test_ts_trans, feature, epochs, learning_rate, stock_ids: list[int]):
    start_time = datetime.now()
    model_transformer = TransformerModel(
        input_chunk_length=512,
        output_chunk_length=1,
        batch_size=32,
        n_epochs=epochs,
        model_name="stock_transformer_"+feature,
        nr_epochs_val_period=10,
        d_model=16,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        random_state=0,
        save_checkpoints=True,
        force_reset=True,
        optimizer_kwargs = {'lr': learning_rate}
    )

    model_transformer.fit(
        series=y_train_ts_trans,
        past_covariates=X_train_ts_trans,
        val_series=y_val_ts_trans,
        val_past_covariates=X_val_ts_trans,
        verbose=False,
        num_loader_workers=NUM_WORKERS,
    )

    predictions_transformer = model_transformer.historical_forecasts(
        series=y_test_ts_trans,
        past_covariates=X_test_ts_trans,
        forecast_horizon=1,
        retrain=False,
        verbose=False,
    )

    # predictions_transformer = model_transformer.predict(n=36, series=y_test_ts_trans, past_covariates=X_test_ts_trans)

    time_taken = datetime.now() - start_time
    add_model_result('Transformer', y_test_ts_trans, predictions_transformer, True, feature, 
                     time_taken, learning_rate, epochs, stock_ids)
    

def add_results_for_mlp_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, 
                                                   feature, stock_ids: list[int]):
    # call this in parallel
    # learning_rates = [0.002, 0.001, 0.01]
    learning_rates = [0.002]
    # epochs = [2,5]
    epochs = [2]
    for lr in learning_rates:
        for epoch in epochs:
            print('add_results_for_mlp lr ', lr, ', epoch ', epoch)
            add_results_for_mlp(X_train, y_train, X_val, y_val, X_test, y_test, feature, epoch, lr, stock_ids)
           
            
def add_results_for_cnn_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, 
                                                   feature, stock_ids: list[int]):
    #0.00038
    # learning_rates = [0.00038, 0.001, 0.01, 0.0001]
    learning_rates = [0.0001]
    # epochs = [2,5]
    epochs = [2]
    for lr in learning_rates:
        for epoch in epochs:
            print('add_results_for_cnn lr ', lr, ', epoch ', epoch)
            add_results_for_cnn(X_train, y_train, X_val, y_val, X_test, y_test, feature, epoch, lr, stock_ids)


def add_results_for_TCN_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, 
                                                   feature, stock_ids: list[int]):
    # learning_rates = [0.002, 0.001, 0.01]
    learning_rates = [0.01]
    # epochs = [2,5]
    epochs = [2]
    for lr in learning_rates:
        for epoch in epochs:
            print('add_results_for_TCN lr ', lr, ', epoch ', epoch)
            add_results_for_TCN(X_train, y_train, X_val, y_val, X_test, y_test, feature, epoch, lr, stock_ids)


def add_results_for_lstm_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, 
                                                    feature, stock_ids: list[int]):
    # learning_rates = [0.002, 0.001, 0.01]
    learning_rates = [0.002]
    # epochs = [2,5]
    epochs = [2]
    for lr in learning_rates:
        for epoch in epochs:
            print('add_results_for_lstm lr ', lr, ', epoch ', epoch)
            add_results_for_lstm(X_train, y_train, X_val, y_val, X_test, y_test, feature, epoch, lr, 
                                 stock_ids)


def add_results_for_transformer_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, 
                                                           feature, stock_ids: list[int]):
    # learning_rates = [0.002, 0.001, 0.01]
    learning_rates = [0.001]
    # epochs = [2,5]
    epochs = [2]
    for lr in learning_rates:
        for epoch in epochs:
            print('add_results_for_transformer lr ', lr, ', epoch ', epoch)
            add_results_for_transformer(X_train, y_train, X_val, y_val, X_test, y_test, feature, epoch, 
                                        lr, stock_ids)

# perform_experiments_multivariate
def perform_experiments_multivariate(df_experiment, feature: str, stock_ids: list[int]):
    df_train, df_validation, df_test = split_df_into_train_val_test(df_experiment)

    # prepare train, validation and test data
    X_train = get_X(df_train)
    X_val = get_X(df_validation)
    X_test = get_X(df_test)

    # print('xtrain')
    # display(X_train.head())
    # sdf

    y_train = df_train['target']
    y_val = df_validation['target']
    y_test = df_test['target']

    X_train_ts, X_train_ts_scaler = create_timeseries_data(X_train)
    X_val_ts, X_val_ts_scaler = create_timeseries_data(X_val)
    X_test_ts, X_test_ts_scaler = create_timeseries_data(X_test)

    y_train_ts, y_train_ts_scaler = create_timeseries_data(y_train.to_frame())
    y_val_ts, y_val_ts_scaler = create_timeseries_data(y_val.to_frame())
    y_test_ts, y_test_ts_scaler = create_timeseries_data(y_test.to_frame())

    # X_ts = X_train_ts.append(X_val_ts)
    # y_ts = y_train_ts.append(y_val_ts)

     # models
    # reset_model_results()
    # add_results_from_light_gbm(X_train, y_train, X_val, y_val, feature, lr=0.3)

    add_results_for_mlp_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, feature, stock_ids)
    add_results_for_cnn_with_hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, feature, stock_ids)
    add_results_for_TCN_with_hyperparameter_tuning(X_train_ts, y_train_ts, X_val_ts, y_val_ts, X_test_ts, y_test_ts, feature, stock_ids)
    add_results_for_lstm_with_hyperparameter_tuning(X_train_ts, y_train_ts, X_val_ts, y_val_ts, X_test_ts, y_test_ts, feature, stock_ids)
    add_results_for_transformer_with_hyperparameter_tuning(X_train_ts, y_train_ts, X_val_ts, y_val_ts, X_test_ts, y_test_ts, feature, stock_ids)

    return get_model_results_df()


def get_all_train_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'optiver-realized-volatility-prediction', 'train_time_id_ordered.csv'))
    return train

def get_unique_stock_ids():
    train = get_all_train_data()
    return list(train['stock_id'].unique())

def get_train_data(stock_ids_to_include):
    train = get_all_train_data()
    # stock_ids = set(train['stock_id'])
    print('Train.shape all ', train.shape)
    # print('stock_ids ', len(stock_ids))

    train = train[train['stock_id'].isin(stock_ids_to_include)]
    train = train.reset_index(drop=True)
    print('Train.shape sample ', train.shape)
    # stock_ids = set(train['stock_id'])
    # print('stock_ids ', stock_ids)

    return train

# Nearest neighbor features
def perform_multivariate_experiments_with_nearest_neighbor_features(
        stock_data,
        n_neighbors,
        use_price_nn_features, 
        use_volume_nn_features, 
        use_size_nn_features, 
        use_random_nn_features,
        feature,
        stock_ids: list[int]):
    print('nearest neighbor feature ', feature)
    with_nn_df = make_features(base=stock_data,
                            block=DataBlock.TRAIN,
                            add_spread_features=True, 
                            add_statistics_features=True,
                            add_book_time_features=True,
                            add_trade_time_features=True)
    with_nn_df = make_features_tick_size(with_nn_df, DataBlock.TRAIN)
    with_nn_df = add_tau_features(with_nn_df)
    time_id_neighbors, stock_id_neighbors = build_nearest_neighbors(with_nn_df, 
                                                                    n_neighbors=n_neighbors,
                                                                    use_price_nn_features=use_price_nn_features, 
                                                                    use_volume_nn_features=use_volume_nn_features, 
                                                                    use_size_nn_features=use_size_nn_features, 
                                                                    use_random_nn_features=use_random_nn_features)
    with_nn_df = make_nearest_neighbor_feature(df_nn=with_nn_df,
                                            time_id_neighbors=time_id_neighbors,
                                            stock_id_neighbors=stock_id_neighbors,
                                            use_price_nn_features=True)

    perform_experiments_multivariate(with_nn_df, feature, stock_ids)


def perform_multivariate_experiments_for_stock(stock_ids: list[int]):
    # print('perform_multivariate_experiments_for_stock ', perform_multivariate_experiments_for_stock)
    stock_data = get_train_data(stock_ids)
    # Feature lag
    df = make_rv_lags(stock_data, 1)
    perform_experiments_multivariate(df, 'rv_lags', stock_ids)
    # Feature set: log_return
    
    df = make_features(base=stock_data,
                            block=DataBlock.TRAIN,
                            add_spread_features=False, 
                            add_statistics_features=False,
                            add_book_time_features=False,
                            add_trade_time_features=False)
    perform_experiments_multivariate(df, 'log_return', stock_ids)
    
    # Feature set: log_return + spread features
    df = make_features(base=stock_data,
                            block=DataBlock.TRAIN,
                            add_spread_features=True, 
                            add_statistics_features=False,
                            add_book_time_features=False,
                            add_trade_time_features=False)
    perform_experiments_multivariate(df, 'log_return_and_spread_features', stock_ids)

    # Feature set log_return, spread, statistics
    df = make_features(base=stock_data,
                           block=DataBlock.TRAIN,
                           add_spread_features=True, 
                           add_statistics_features=True,
                           add_book_time_features=False,
                           add_trade_time_features=False)

    perform_experiments_multivariate(df, 'log_return_spread_and_statistics_features', stock_ids)

    if len(stock_ids) > 1:
        print('Nearest neighbor features ')
        perform_multivariate_experiments_with_nearest_neighbor_features(
            stock_data=stock_data,
            n_neighbors=len(stock_ids),
            use_price_nn_features=True, 
            use_volume_nn_features=False, 
            use_size_nn_features=False, 
            use_random_nn_features=False,
            feature='price_nn_features',
            stock_ids=stock_ids)
        
        perform_multivariate_experiments_with_nearest_neighbor_features(
            stock_data=stock_data, 
            n_neighbors=len(stock_ids),
            use_price_nn_features=True, 
            use_volume_nn_features=True, 
            use_size_nn_features=False, 
            use_random_nn_features=False,
            feature='volume_nn_features',
            stock_ids=stock_ids)
        
        perform_multivariate_experiments_with_nearest_neighbor_features( 
            stock_data=stock_data,
            n_neighbors=len(stock_ids),
            use_price_nn_features=True, 
            use_volume_nn_features=False, 
            use_size_nn_features=True, 
            use_random_nn_features=False,
            feature='size_nn_features',
            stock_ids=stock_ids)
        
        perform_multivariate_experiments_with_nearest_neighbor_features(
            stock_data=stock_data,
            n_neighbors=len(stock_ids),
            use_price_nn_features=True, 
            use_volume_nn_features=False, 
            use_size_nn_features=False, 
            use_random_nn_features=True,
            feature='random_nn_features',
            stock_ids=stock_ids)
        
        perform_multivariate_experiments_with_nearest_neighbor_features(
            stock_data=stock_data,
            n_neighbors=len(stock_ids),
            use_price_nn_features=True, 
            use_volume_nn_features=True, 
            use_size_nn_features=True, 
            use_random_nn_features=True,
            feature='all_nn_features',
            stock_ids=stock_ids)


def get_top_n_results(n: int):
    df_res = get_model_results_df()
    df_res = df_res.sort_values(by=["mae"], ascending=True)
    return df_res.head(n)

if __name__ == '__main__':
    number_of_stocks = [1]
    all_stock_ids = sorted(get_unique_stock_ids())
    for i in number_of_stocks:
        print('Number of stocks processing ', i)
        stock_ids = all_stock_ids[0:i]
        print('stock_ids = ', stock_ids)
        print('len(stock_ids) = ', len(stock_ids))
        perform_multivariate_experiments_for_stock(stock_ids)
        resutls_df = get_model_results_df()
        resutls_df.to_csv(f'./results/{uuid.uuid4()}.csv')


