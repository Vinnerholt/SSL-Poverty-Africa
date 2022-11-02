from collections import defaultdict
import itertools
import os
import pickle
from pprint import pprint
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sklearn.cluster
import pandas as pd
import copy

from utils.geo_plot import plot_locs  # ignore: E402

# Fix so that fraction (and random_state) is an argument, and that function receives dataframe (or csv_path) 
def create_folds(
                 df: pd.DataFrame,
                 min_dist: float = 0.0856172535058306,
                 fold_names: Iterable[str] = ['A', 'B', 'C', 'D', 'E'],
                 random_state: int = np.random.randint(9999),
                 verbose: bool = True,
                 plot_largest_clusters: int = 0,
                 fraction: float = 1,
                 big_fold_penalty: int = 0,
                 prints: bool = False
                 ) -> Dict[str, np.ndarray]:
    '''Partitions locs into folds.

    Args
    - df: pandas.DataFrame, Dataframe of dhs data
    - locs: np.array, shape [N, 2]. NOT USED
    - min_dist: float, minimum distance between folds
    - fold_names: list of str, names of folds
    - verbose: bool
    - plot_largest_clusters: int, number of largest clusters to plot
    - fraction: float, fraction of rows of data in df to use
    - random_state: int, random_state for sampling fraction from df
    - big_fold_penalty: int, penalty for assigning cluster to large fold, higher = lower penalty

    Returns
    - folds: dict, fold name => sorted np.array of indices of locs belonging to that fold
    '''
    df = df.copy()
    #if fraction < 1:
    
    if random_state != None:
        df_frac = df.sample(frac=fraction, random_state=random_state)
    else:
        df_frac = df.sample(frac=fraction)

    
    df_frac = df_frac[['lat', 'lon', 'iwi']]
    df_frac['index'] = df_frac.index


    # there may be duplicate locs => we want to cluster based on unique locs
    locs_iwi = df_frac['iwi'].values
    locs = df_frac[['lat', 'lon']]
    unique_locs = locs.drop_duplicates().values
    locs = locs.values

    # locs_iwi = []
    # if np.shape(locs)[1] == 3:
    #     locs_iwi = locs[:,2]
    #     locs = locs[:,:2]


    #unique_locs = np.unique(locs, axis=0)  # get unique rows
    #unique_locs, index = np.unique(locs[:,:2], return_index=True, axis=0)
    #unique_locs_iwi = locs[index]
    

    # dict that maps each (lat, lon) tuple to a list of corresponding indices in
    # the locs array
    locs_to_indices = defaultdict(list)
    locs_to_original_index = defaultdict(list)
    for i, loc in enumerate(df_frac.values):
    #print(loc)
        locs_to_indices[tuple(loc[0:2])].append(i)
        locs_to_original_index[tuple(loc[0:2])].append(int(loc[3]))
    # any point within `min_dist` of another point belongs to the same cluster
    # - cluster_labels assigns a cluster index (0-indexed) to each loc
    # - a cluster label of -1 means that the point is an outlier
    _, cluster_labels = sklearn.cluster.dbscan(
        X=unique_locs, eps=min_dist, min_samples=2, metric='euclidean')

    # mapping: cluster number => list of indices of points in that cluster
    # - if cluster label is -1 (outlier), then treat that unique loc as its own cluster
    neg_counter = -1
    # clusters_dict = defaultdict(list)
    # for loc, c in zip(unique_locs, cluster_labels):
    #     indices = locs_to_indices[tuple(loc)]
    #     if c < 0:
    #         c = neg_counter
    #         neg_counter -= 1
    #     clusters_dict[c].extend(indices)

    clusters_dict = defaultdict(list)
    clusters_dict_og = defaultdict(list)

    for loc, c in zip(unique_locs, cluster_labels):
        indices = locs_to_indices[tuple(loc)]
        indices_og = locs_to_original_index[tuple(loc)]
        if c < 0:
            c = neg_counter
            neg_counter -= 1
        clusters_dict[c].extend(indices)
        clusters_dict_og[c].extend(indices_og)

    # sort clusters by descending cluster size
    sorted_clusters = sorted(clusters_dict.keys(), key=lambda c: -len(clusters_dict[c]))
    sorted_clusters_og = sorted(clusters_dict_og.keys(), key=lambda c: -len(clusters_dict_og[c]))


    # greedily assign clusters to folds
    folds: Dict[str, List[int]] = {f: [] for f in fold_names}
    #for c in sorted_clusters:
    tot_mean = df_frac['iwi'].mean()
    tot_std = df_frac['iwi'].std()
    MAX_FOLD_SIZE = int(len(df_frac.index) / len(fold_names) + 1)

    big_fold_penalty = big_fold_penalty * fraction

    folds_sum: Dict[str, List[int]] = {f: 0 for f in fold_names}
    folds_count: Dict[str, List[int]] = {f: 0 for f in fold_names}
    for c in sorted_clusters_og:
    #for c in clusters_dict_og.keys():
        # assign points in cluster c to smallest fold
        #print("C: ", c)
        #print("clusters_dict_og[c]: ", clusters_dict_og[c])
        #print("len df: ", len(df.index))
        #print("df.index: ", df.index)
        #print("df.iloc[max(df.index)]: ", df.loc[max(df.index)])
        cluster_indices = clusters_dict_og[c]
        #cluster_tot_iwi = df.loc[df.index == cluster_indices]['iwi'].sum()# ['iwi'].iloc[cluster_indices].sum()
        cluster_tot_iwi = df.loc[cluster_indices]['iwi'].sum()
        n_surveys_in_cluster = len(cluster_indices)

        # Assign cluster to fold which is most suitable based on "suitableness" criterion, defined below
        fold_suitableness: Dict[str, List[int]] = {f: 0 for f in fold_names}
        for fold in fold_names:

            #print("fold: ", fold)
            #print(tot_mean, folds_sum[fold], folds_count[fold], cluster_tot_iwi, n_surveys_in_cluster)
            
            # Suitableness 1: (dist to real mean before assigning cluster) - (dist to real mean after assigning to cluster)
            #suitableness = -abs(tot_mean - divide_safely_by_zero(folds_sum[fold], folds_count[fold])) - abs( tot_mean - ( (folds_sum[fold] + cluster_tot_iwi) / (folds_count[fold] + n_surveys_in_cluster) ) )

            # Suitableness 1.5: (dist to real mean before assigning cluster) - (dist to real mean after assigning to cluster) + penalty for assigning to large fold
            #suitableness = folds_count[fold] / big_fold_penalty - abs(tot_mean - divide_safely_by_zero(folds_sum[fold], folds_count[fold])) - abs( tot_mean - ( (folds_sum[fold] + cluster_tot_iwi) / (folds_count[fold] + n_surveys_in_cluster) ) )
            
            

            # WORKS BEST
            # Suitableness 3: dist to real mean after assigning to cluster + penalty for assigning to large fold
            #if big_fold_penalty != 0:
            #    suitableness = folds_count[fold] / big_fold_penalty + abs( tot_mean - ( (folds_sum[fold] + cluster_tot_iwi) / (folds_count[fold] + n_surveys_in_cluster) )) 
            
            #else:
                # Suitableness 2: dist to real mean after assigning to cluster
            #    suitableness = abs( tot_mean - ( (folds_sum[fold] + cluster_tot_iwi) / (folds_count[fold] + n_surveys_in_cluster) )) 

            # Suitableness 3.5: dist to real (mean and std dev) after assigning to cluster + penalty for assigning to large fold
            #tmp_fold = copy.copy(folds[fold])
            #tmp_fold.extend(clusters_dict_og[c])
            #suitableness = folds_count[fold] / big_fold_penalty + abs( tot_mean - ( (folds_sum[fold] + cluster_tot_iwi) / (folds_count[fold] + n_surveys_in_cluster) )) + abs(tot_std - df.loc[tmp_fold]['iwi'].std())


            # Suitableness 4: assign to smallest fold
                suitableness = folds_count[fold]

            #print("suitableness: ", suitableness)
                fold_suitableness[fold] = suitableness

        
        fold_suitableness_sorted = sorted(fold_suitableness.keys(), key= lambda f: fold_suitableness[f])

        if prints:
            print()
            print("MEAN IWI OF CLUSTER: ", cluster_tot_iwi / n_surveys_in_cluster, ", SIZE OF CLUSTER: ", n_surveys_in_cluster)
            print("fold_suitableness: ", fold_suitableness)
            print("fold_suitableness_sorted: ", fold_suitableness_sorted)

        # Find most suitable fold to assign cluster to which will not make it overfull
        fold_suitable_index = 0
        while folds_count[fold_suitableness_sorted[fold_suitable_index]] + n_surveys_in_cluster > MAX_FOLD_SIZE:
            fold_suitable_index += 1
        
        f = fold_suitableness_sorted[fold_suitable_index]


        folds_count[f] += n_surveys_in_cluster
        folds_sum[f] += cluster_tot_iwi


        #folds[f].extend(clusters_dict[c])
        folds[f].extend(clusters_dict_og[c])

    for f in folds:
        folds[f] = np.sort(folds[f])

    # plot the largest clusters
    for i in range(plot_largest_clusters):
        c = sorted_clusters[i]
        indices = clusters_dict[c]
        title = 'cluster {c}: {n} points'.format(c=c, n=len(indices))
        if len(locs_iwi) != 0:
            plot_locs(locs[indices], figsize=(4, 4), title=title, colors=locs_iwi[indices].tolist(), cbar_label='IWI')
        else:
            plot_locs(locs[indices], figsize=(4, 4), title=title)

    if verbose:
        _, unique_counts = np.unique(cluster_labels, return_counts=True)

        num_outliers = np.sum(cluster_labels == -1)
        outlier_offset = int(num_outliers > 0)
        max_cluster_size = np.max(unique_counts[outlier_offset:])  # exclude outliers

        print('num clusters:', np.max(cluster_labels) + 1)  # clusters are 0-indexed
        print('num outliers:', num_outliers)
        print('max cluster size (excl. outliers):', max_cluster_size)

        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)
        ax.hist(unique_counts[outlier_offset:], bins=50)  # exclude outliers
        ax.set(xlabel='cluster size', ylabel='count')
        ax.set_yscale('log')
        ax.set_title('histogram of cluster sizes (excluding outliers)')
        ax.grid(True)
        plt.show()

    return folds

def create_country_folds(
    dhs_df: pd.DataFrame,
    fold_names: Iterable[str] = ['A', 'B', 'C', 'D', 'E'],
    random_state: int = np.random.randint(9999),
    fraction: float = 1,
    big_fold_penalty: int = 0,
    ) -> Dict[str, np.ndarray]:
    '''Partitions DataFrame of locations into folds of heldout countries.

    Args
    - df: pandas.DataFrame, Dataframe of dhs data
    - fold_names: list of str, names of folds
    - fraction: float, fraction of rows of data in df to use
    - random_state: int, random_state for sampling fraction from df
    - big_fold_penalty: int, penalty for assigning cluster to large fold, higher = lower penalty

    Returns
    - folds: dict, fold name => sorted np.array of indices of locs belonging to that fold
    '''


    # Sample fraction of data from DataFrame
    dhs_df_fraction = dhs_df.sample(frac=fraction)

    # Group DataFrame by countries
    grouped = dhs_df_fraction.groupby('country')

    # Get all indices pertaining to every country 
    country_indices_dict = grouped.groups

    # Find max fold size
    MAX_FOLD_SIZE = int(1.01 * len(dhs_df_fraction.index) / len(fold_names) + 10)

    # Find mean of IWI for all clusters
    tot_mean = dhs_df['iwi'].mean()

    folds: Dict[str, List[int]] = {f: [] for f in fold_names}
    countries_in_folds: Dict[str, List[str]] = {f: [] for f in fold_names}

    folds_sum: Dict[str, List[int]] = {f: 0 for f in fold_names}
    folds_count: Dict[str, List[int]] = {f: 0 for f in fold_names}

    # Sort countries by size, in order to assign the largest countries first
    countries_sorted = sorted(country_indices_dict.keys(), key= lambda f: -len(country_indices_dict[f]))

    # Assign countries to folds
    for country in countries_sorted:

        # Get all indices for current country
        country_indices = country_indices_dict[country]

        # Calculate sum of IWI in, and number of DHS clusters in current country
        country_tot_iwi = dhs_df.loc[country_indices]['iwi'].sum()
        n_surveys_in_country = len(country_indices)


        # Find "suitableness" of assigning the current country to each fold, and assign to most suitable fold
        fold_suitableness: Dict[str, List[int]] = {f: 0 for f in fold_names}
        for fold in fold_names:
            
            # Suitableness 4: assign to smallest fold
            #suitableness = folds_count[fold]

            # Suitableness 3: dist to real mean after assigning to cluster + penalty for assigning to large fold
            if big_fold_penalty != 0:
                suitableness = folds_count[fold] / big_fold_penalty  \
                    + abs( tot_mean - ( (folds_sum[fold] + country_tot_iwi) / (folds_count[fold] + n_surveys_in_country) ))

            # Suitableness 1.5: (dist to real mean before assigning cluster) - (dist to real mean after assigning to cluster) + penalty for assigning to large fold
            #suitableness = folds_count[fold] / big_fold_penalty - abs(tot_mean - divide_safely_by_zero(folds_sum[fold], folds_count[fold])) - abs( tot_mean - ( (folds_sum[fold] + country_tot_iwi) / (folds_count[fold] + n_surveys_in_country) ) )
                
            # Suitableness 2: dist to real mean after assigning to cluster
            else:
                suitableness = abs( tot_mean - ( (folds_sum[fold] + country_tot_iwi) / (folds_count[fold] + n_surveys_in_country) )) 

            fold_suitableness[fold] = suitableness

        # Sort fold names based on their suitableness for being assigned current country
        fold_suitableness_sorted = sorted(fold_suitableness.keys(), key= lambda f: fold_suitableness[f])

        # Assign current country to most suitable fold AS LONG AS it does not fill it over the maximum fold size.
        # Otherwise, assign to next, non-full, most suitable fold
        # If "list index out of range", increase max fold size
        fold_suitable_index = 0
        while folds_count[fold_suitableness_sorted[fold_suitable_index]] + n_surveys_in_country > MAX_FOLD_SIZE:
            fold_suitable_index += 1
        fold_assigned = fold_suitableness_sorted[fold_suitable_index]

        # Increase fold counters
        folds_count[fold_assigned] += n_surveys_in_country
        folds_sum[fold_assigned] += country_tot_iwi

        # Assign indices of all clusters within country to fold
        countries_in_folds[fold_assigned].append(country)
        folds[fold_assigned].extend(country_indices)

    return folds, countries_in_folds









def divide_safely_by_zero(x, y):
    if y == 0:
        return 0
    else:
        return x / y

def verify_folds(folds: Dict[str, np.ndarray],
                 locs: np.ndarray,
                 min_dist: float,
                 max_index: Optional[int] = None
                 ) -> None:
    '''Verifies that folds do not overlap.

    Args
    - folds: dict, fold name => np.array of indices of locs belonging to that fold
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - min_dist: float, minimum distance between folds
    - max_index: int, all indices in range(max_index) should be included
    '''
    print('Size of each fold')
    pprint({f: len(indices) for f, indices in folds.items()})

    for fold, idxs in folds.items():
        assert np.all(np.diff(idxs) >= 0)  # check that indices are sorted

    # check that all indices are included
    if max_index is not None:
        assert np.array_equal(
            np.sort(np.concatenate(list(folds.values()))),
            np.arange(max_index))

    # check to ensure no overlap
    print('Minimum distance between each pair of folds')
    for a, b in itertools.combinations(folds.keys(), r=2):
        a_idxs = folds[a]
        b_idxs = folds[b]
        dists = scipy.spatial.distance.cdist(locs[a_idxs], locs[b_idxs], metric='euclidean')
        assert np.min(dists) > min_dist
        print(a, b, np.min(dists))


def create_split_folds(test_folds: Dict[str, np.ndarray],
                       fold_names: List[str],
                       ) -> Dict[str, Dict[str, np.ndarray]]:
    '''Creates a folds dict mapping each fold name (str) to another dict
    that maps each split (str) to a np.array of indices.

    folds = {
        'A': {
            'train': np.array([...]),
            'val': np.array([...]),
            'test': np.array([...])},
        ...
        'E': {...}
    }

    Args
    - test_folds: dict, fold name => sorted np.array of indices of locs belonging to that fold
    - fold_names: list of str, names of folds

    Returns
    - folds: dict, folds[f][s] is a np.array of indices for split s of fold f
    '''
    # create train/val/test splits
    folds: Dict[str, Dict[str, np.ndarray]] = {}
    for i, f in enumerate(fold_names):
        folds[f] = {}
        folds[f]['test'] = test_folds[f]

        val_f = fold_names[(i+1) % 5]
        folds[f]['val'] = test_folds[val_f]

        train_fs = [fold_names[(i+2) % 5], fold_names[(i+3) % 5], fold_names[(i+4) % 5]]
        folds[f]['train'] = np.sort(np.concatenate([test_folds[f] for f in train_fs]))

    return folds


def save_folds(folds_path: str,
               folds: Dict[str, Dict[str, np.ndarray]],
               check_exists: bool = True
               ) -> None:
    '''Saves folds dict to a pickle file at folds_path.

    Args
    - folds_path: str, path to pickle folds dict
    - folds: dict, folds[f][s] is a np.array of indices for split s of fold f
    - check_exists: bool, if True, verifies that existing pickle at folds_path
        matches the given folds
    '''
    if check_exists and os.path.exists(folds_path):
        with open(folds_path, 'rb') as p:
            existing_folds = pickle.load(p)
        assert set(existing_folds.keys()) == set(folds.keys())
        for f in existing_folds:
            for s in ['train', 'val', 'test']:
                assert np.array_equal(folds[f][s], existing_folds[f][s])
    else:
        with open(folds_path, 'wb') as p:
            pickle.dump(folds, p)
