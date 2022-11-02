import os
from glob import glob
import pandas as pd
import tensorflow as tf
import random
from batchers.mean_and_std_constants import MEANS_DICT, STD_DEVS_DICT
import numpy as np

MIN_IWI = tf.constant([-1.8341599999999998])
MAX_IWI = tf.constant([86.065248])


def create_full_tfrecords_paths(data_path):
    tfrecord_paths = []
    df = pd.read_csv(data_path + '/dhs_clusters.csv', float_precision='high', index_col=False)
    surveys = list(df.groupby(['country', 'year']).groups.keys())
    for (country, year) in surveys:
        glob_path = os.path.join(data_path + '/dhs_tfrecords/', country + '_'
                                 + str(year), '*.tfrecord.gz')
        survey_files = glob(glob_path)
        tfrecord_paths.extend(sorted(survey_files))
    return tfrecord_paths


def process_one_year_tfrecords(example_proto, labeled, n_year_composites=10, normalize=True, band_stats=None):
    '''
    Args
    - example_proto: a tf.train.Example protobuf
    Returns:
    - img: tf.Tensor, shape [224, 224, C], type float32
      - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
    - label: tf.Tensor, shape (10,1)
    - mask: one-hot mask for the true label
    '''
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    scalar_float_keys = ['lat', 'lon', 'year', 'iwi']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * n_year_composites], dtype=tf.float32)
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
    loc = tf.stack([ex['lat'], ex['lon']])
    year = tf.cast(ex.get('year', -1), tf.int32)
    iwi = tf.cast(ex.get('iwi', -1), tf.float32)
    normalized_iwi = (iwi-MIN_IWI) / (MAX_IWI-MIN_IWI)
    index = get_year_index(year)
    

    img = float('nan')
    if len(bands) > 0:
        means = band_stats['means']
        std_devs = band_stats['stds']
        for band in bands:
            ex[band] = tf.nn.relu(ex[band])
            ex[band] = tf.reshape(ex[band], (n_year_composites, 224, 224))
            if normalize:
                if band == 'NIGHTLIGHTS':
                    ex[band] = tf.cond(
                        year <= 2015,  # true = DMSP
                        true_fn=lambda: (ex[band] - means['DMSP']) / std_devs['DMSP'],
                        false_fn=lambda: (ex[band] - means['VIIRS']) / std_devs['VIIRS'])
                else:
                    ex[band] = (ex[band] - means[band]) / std_devs[band]
        img = tf.stack([ex[band] for band in bands], axis=3)
        img_one_year = get_only_images_with_labels(index, img)

    if not labeled:
        return img_one_year
    return img_one_year, normalized_iwi


def process_tfrecords(example_proto, labeled, n_year_composites=10, calculate_mean=False, normalize=True, band_stats=None):
    '''
    Args
    - example_proto: a tf.train.Example        
    
   def process_tfrecords(example_proto, labeled, n_year_composites=10, calculate_me
  
    Returns:
    - img: tf.Tensor, shape [224, 224, C], type float32
      - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
    - label: tf.Tensor, shape (10,1)
    - mask: one-hot mask for the true label
    '''
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * n_year_composites], dtype=tf.float32)
    scalar_float_keys = ['year', 'iwi'] if labeled else ['year']
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
    if labeled:
        year = tf.cast(ex.get('year', -1), tf.int32)
        iwi = tf.cast(ex.get('iwi', -1), tf.float32)
        normalized_iwi = (iwi-MIN_IWI) / (MAX_IWI-MIN_IWI)
        one_hot_year = tf.one_hot(get_year_index(year), 10)
        one_hot_label = normalized_iwi * one_hot_year

    img = float('nan')
    if len(bands) > 0:
        for band in bands:
            ex[band] = tf.nn.relu(ex[band])
            ex[band] = tf.reshape(ex[band], (n_year_composites, 224, 224))
        if normalize:
            means = band_stats['means']
            std_devs = band_stats['stds']
            for band in bands[:-1]:
                ex[band] = (ex[band] - means[band]) / std_devs[band]
            nl_band = 'NIGHTLIGHTS'
            ex[band] = tf.reshape(ex[nl_band], (n_year_composites, 224, 224))
            nl_split = tf.split(ex[band], [8, 2], axis=0)

            split1 = (nl_split[0] - means['DMSP']) / std_devs['DMSP']
            split2 = (nl_split[1] - means['VIIRS']) / std_devs['VIIRS']

            ex[nl_band] = tf.concat([split1, split2], axis=0)

        img = tf.stack([ex[band] for band in bands], axis=3)

    if labeled:
        return {"model_input": img, "outputs_mask": one_hot_year}, {"masked_outputs": one_hot_label}
    if calculate_mean:
        return img
    return {"model_input": img, "outputs_mask": tf.ones([n_year_composites])}


def choose_timespan_start_index(label_index, labeled):
    possible_start_indices = tf.constant([0])
    if tf.math.equal(label_index, tf.constant([0])):
        possible_start_indices = tf.constant([0])
    elif tf.math.equal(label_index, tf.constant([1])):
        possible_start_indices = tf.constant([0, 1])
    elif tf.math.equal(label_index, tf.constant([2])):
        possible_start_indices = tf.constant([0, 1, 2])
    elif tf.math.equal(label_index, tf.constant([3])):
        possible_start_indices = tf.constant([0, 1, 2, 3])
    elif tf.math.equal(label_index, tf.constant([4])):
        possible_start_indices = tf.constant([0, 1, 2, 3, 4])
    elif tf.math.equal(label_index, tf.constant([5])):
        possible_start_indices = tf.constant([1, 2, 3, 4, 5])
    elif tf.math.equal(label_index, tf.constant([6])):
        possible_start_indices = tf.constant([2, 3, 4, 5])
    elif tf.math.equal(label_index, tf.constant([7])):
        possible_start_indices = tf.constant([3, 4, 5])
    elif tf.math.equal(label_index, tf.constant([8])):
        possible_start_indices = tf.constant([4, 5])
    else:
        possible_start_indices = tf.constant([5])
    if labeled:
        return tf.random.shuffle(possible_start_indices)[0]
    else:
        return [0, 1, 2, 3, 4, 5]


def process_window_tfrecords(example_proto, labeled, size_of_window, calculate_mean=False, n_year_composites=10, normalize=True,
                             band_stats=None):
    '''
    Args
    - example_proto: a tf.train.Example
   def process_tfrecords(example_proto, labeled, n_year_composites=10, calculate_me
    Returns:
    - img: tf.Tensor, shape [224, 224, C], type float32
      - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
    - label: tf.Tensor, shape (10,1)
    - mask: one-hot mask for the true label
    '''
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    scalar_float_keys = ['lat', 'lon', 'year', 'iwi']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * n_year_composites], dtype=tf.float32)
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
    year = tf.cast(ex.get('year', -1), tf.int32)
    iwi = tf.cast(ex.get('iwi', -1), tf.float32)
    normalized_iwi = (iwi-MIN_IWI) / (MAX_IWI-MIN_IWI)
    year_index = get_year_index(year)
    start_index = choose_timespan_start_index(year_index, labeled)
    one_hot_year = tf.one_hot((year_index-start_index), size_of_window)
    one_hot_label = normalized_iwi * one_hot_year
    if not labeled:
        start_indices = choose_timespan_start_index(year_index, labeled)
        one_hot_years = []
        #one_hot_labels = tf.constant([])
        for start_index in start_indices:
            one_hot_year = tf.one_hot((year_index - start_index), size_of_window)
            #one_hot_label = one_hot_year*iwi
            one_hot_years.append(one_hot_year)
            #one_hot_labels = tf.concat([one_hot_labels, one_hot_label], 0)

    img = float('nan')
    if len(bands) > 0:
        for band in bands:
            ex[band] = tf.nn.relu(ex[band])
            ex[band] = tf.reshape(ex[band], (n_year_composites, 224, 224))
        if normalize:
            means = band_stats['means']
            std_devs = band_stats['stds']
            for band in bands[:-1]:
                ex[band] = (ex[band] - means[band]) / std_devs[band]
            nl_band = 'NIGHTLIGHTS'
            ex[band] = tf.reshape(ex[nl_band], (n_year_composites, 224, 224))
            nl_split = tf.split(ex[band], [8, 2], axis=0)

            split1 = (nl_split[0] - means['DMSP']) / std_devs['DMSP']
            split2 = (nl_split[1] - means['VIIRS']) / std_devs['VIIRS']

            ex[nl_band] = tf.concat([split1, split2], axis=0)

        img = tf.stack([ex[band] for band in bands], axis=3)
        if labeled:
            img_splits = tf.split(img, [start_index, size_of_window, 10-(start_index + size_of_window)], axis=0)
            new_yearspan_img = img_splits[1]
        else:
            new_imgs = []
            for start_index in start_indices:
                img_splits = tf.split(img, [start_index, size_of_window, 10 - (start_index + size_of_window)], axis=0)
                new_yearspan_img = img_splits[1]
                new_imgs.append(new_yearspan_img)
    if labeled:
        return {"model_input": new_yearspan_img, "outputs_mask": one_hot_year}, {"masked_outputs": one_hot_label}

    return {"model_input": tf.stack(new_imgs, axis=0), "outputs_mask": tf.stack(tf.ones_like(one_hot_years), axis=0)}


def process_self_supervised_tfrecords(example_proto, batch_size, normalize=True):
    '''
        Args
        - example_proto: a tf.train.Example
        Returns:
        - img: tf.Tensor, shape [224, 224, C], type float32
          - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
        - label: tf.Tensor, shape (10,1)
        - mask: one-hot mask for the true label
        '''
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    scalar_float_keys = ['lat', 'lon', 'year', 'iwi']
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * 10], dtype=tf.float32)
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
    loc = tf.stack([ex['lat'], ex['lon']])
    year = tf.cast(ex.get('year', -1), tf.int32)
    iwi = tf.cast(ex.get('iwi', -1), tf.float32)
    mask = tf.ones(10)
    label_sequence = np.arange(len(mask))
    np.random.shuffle(label_sequence)

    img = float('nan')
    if len(bands) > 0:
        means = MEANS_DICT
        std_devs = STD_DEVS_DICT
        for band in bands[:7]:
            ex[band] = tf.nn.relu(ex[band])
            ex[band] = tf.reshape(ex[band], (10, 224, 224))
            if normalize:
                ex[band] = (ex[band] - means[band]) / std_devs[band]
        if normalize:
            band = 'NIGHTLIGHTS'
            ex[band] = tf.reshape(ex[band], (10, 224, 224))
            nl_split = tf.split(ex[band], [8, 2], axis=0)

            split1 = (nl_split[0] - means['DMSP']) / std_devs['DMSP']
            split2 = (nl_split[1] - means['VIIRS']) / std_devs['VIIRS']

            ex[band] = tf.concat([split1, split2], axis=0)

            # Split per year composite in order to shuffle the images
            split_list = tf.split(ex[band], num_or_size_splits=10, axis=0)
            print(split_list)
            shuffled_list = np.array(split_list)[label_sequence]
            print(shuffled_list)
            ex[band] = tf.concat(list(shuffled_list), axis=0)
            print("band after concat ", ex[band])
        img = tf.stack([ex[band] for band in bands], axis=3)
    return {"model_input": img, "outputs_mask": mask}, {"masked_outputs": label_sequence}


def get_dataset(tfrecord_files, batch_size, n_year_composites, size_of_window, cache_file=None, one_year_model=False, labeled=True, shuffle=True,
                calculate_mean=False, normalize=True, band_stats=None, max_epochs=150):
    '''Gets the dataset preprocessed and split into batches and epochs.
    Returns
    - dataset, each sample of the form {"model_input": img, "outputs_mask": one_hot_year}, {"masked_outputs": one_hot_label}
    '''
    if normalize:
        assert band_stats != None, 'Can not normalize without band_stats'

    # convert to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type='GZIP',
        buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads=tf.data.experimental.AUTOTUNE)  # num_threads)
    # prefetch 2 batches at a time to smooth out the time taken to
    # load input files as we go through shuffling and processing
    # dataset = dataset.prefetch(buffer_size=2*batch_size)
    dataset = dataset.prefetch(buffer_size=2 * batch_size)
    if one_year_model:
        dataset = dataset.map(
            lambda x: process_one_year_tfrecords(x, labeled=labeled, n_year_composites=n_year_composites,
                                                 normalize=normalize, band_stats=band_stats),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif size_of_window < n_year_composites:
        dataset = dataset.map(lambda x: process_window_tfrecords(x, labeled=labeled, size_of_window=size_of_window, n_year_composites=n_year_composites,
                                                                 calculate_mean=calculate_mean, normalize=normalize,
                                                                 band_stats=band_stats), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if not labeled:
            return dataset

    else:
        dataset = dataset.map(lambda x: process_tfrecords(x, labeled=labeled, n_year_composites=n_year_composites,
                                                          calculate_mean=calculate_mean, normalize=normalize, band_stats=band_stats),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # batch then repeat => batches respect epoch boundaries
    # - i.e. last batch of each epoch might be smaller than batch_size
    if cache_file:
        dataset = dataset.cache(cache_file)
    if labeled:
        dataset = dataset.shuffle(2048)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if max_epochs > 1:
        dataset = dataset.repeat(max_epochs)
    return dataset


def get_self_supervised_dataset(tfrecord_files, batch_size, normalize=True):
    '''Gets the dataset preprocessed and split into batches and epochs.
    Returns
    - dataset, each sample of the form {"model_input": img, "outputs_mask": one_hot_year}, {"masked_outputs": one_hot_label}
    '''
    # convert to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type='GZIP',
        buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
        num_parallel_reads=tf.data.experimental.AUTOTUNE)  # num_threads)
    # prefetch 2 batches at a time to smooth out the time taken to
    # load input files as we go through shuffling and processing
    # dataset = dataset.prefetch(buffer_size=2*batch_size)
    dataset = dataset.prefetch(buffer_size=2 * batch_size)
    dataset = dataset.map(lambda x: process_self_supervised_tfrecords(x, 16, normalize=normalize),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # batch then repeat => batches respect epoch boundaries
    # - i.e. last batch of each epoch might be smaller than batch_size

    # dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(150)
    return dataset


def get_year_index(year):
    # Extracts the index of the 3 year-composite that contains the true label, i.e. the year the survey come from
    # label_year = int(year)
    start_year = 1990
    end_year = 2020
    span_length = 3
    # TODO: add better assert
    # assert start_year <= year , f"The year ({year} is not in the range ({start_year}-{end_year})"
    # assert year<= end_year, f"The year ({year} is not in the range ({start_year}-{end_year})"
    index = (year - start_year) // span_length
    return index


def get_only_images_with_labels(index, img):
    # Gets the slice of the tensor that contains the image with the label
    labeled_img = tf.slice(img, [index, 0, 0, 0], [1, -1, -1, -1])
    # Removes dimension of size 1