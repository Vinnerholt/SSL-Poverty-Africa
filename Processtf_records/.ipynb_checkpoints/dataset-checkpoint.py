import tensorflow as tf

def process_tfrecords(example_proto, normalize=True, band_stats=None):
    '''
    Args
    - example_proto: a tf.train.Example
  
    Returns:
    - img: tf.Tensor, shape [224, 224, C], type float32
      - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
    - mask: one-hot mask for the true label
    '''

    # Parse image from tfrecord
    bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
    n_year_composites = 10
    keys_to_features = {}
    for band in bands:
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[224 ** 2 * n_year_composites], dtype=tf.float32)
    scalar_float_keys = ['year']
    for key in scalar_float_keys:
        keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    ex = tf.io.parse_single_example(example_proto, features=keys_to_features)

    # Remove negative values, reshape to (10, 224, 224)
    for band in bands:
        ex[band] = tf.nn.relu(ex[band])
        ex[band] = tf.reshape(ex[band], (n_year_composites, 224, 224))

    # Normalize each band to mean=0, var=1
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

    # Since we're making predictions, get a mask with only ones
    outputs_mask = tf.ones([n_year_composites])
    return {"model_input": img, "outputs_mask": outputs_mask}

def get_dataset(tfrecord_files, normalize=True, band_stats=None):
    '''Gets the dataset preprocessed and split into batches and epochs.
    Returns
    - dataset, each sample of the form {"model_input": img, "outputs_mask": one_hot_year}
    '''
    if normalize:
        assert band_stats != None, 'Can not normalize without band_stats'

    # convert to files to individual records
    dataset = tf.data.TFRecordDataset(
        filenames=tfrecord_files,
        compression_type='GZIP',
        buffer_size=1024 * 1024 * 128)  # 128 MB buffer size
    
    # process individual records
    dataset = dataset.map(lambda x: process_tfrecords(x, normalize=normalize, band_stats=band_stats))
    dataset = dataset.batch(16)
                            
    return dataset