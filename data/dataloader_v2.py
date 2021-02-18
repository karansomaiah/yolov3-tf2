import tensorflow as tf


def generate_class_map(config):
    return {
        class_proto.class_name: class_proto.class_id for class_proto in config.class_map
    }


# def feature_parser(example):
#    features = {
#        'image': tf.io.FixedLenFeature([], tf.string),
#        'x': tf.io.VarLenFeature(tf.float32),
#        'y': tf.io.VarLenFeature(tf.float32),
#        'w': tf.io.VarLenFeature(tf.float32),
#        'h': tf.io.VarLenFeature(tf.float32),
#        'label': tf.io.VarLenFeature(tf.int64)
#    }
#    return features_to_annotation(tf.io.parse_single_example(example, features))
#
#
# def features_to_annotation(parsed_feature):
#    image = tf.io.decode_raw(input_bytes=parsed_feature['image'],
#                             out_type=tf.uint8)
#    image_tensor = tf.expand_dims(tf.reshape(image, ))


def build(filepath, class_map, image_height, image_width, batch_size, shuffle=True):
    def feature_parser(example):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "x": tf.io.VarLenFeature(tf.float32),
            "y": tf.io.VarLenFeature(tf.float32),
            "w": tf.io.VarLenFeature(tf.float32),
            "h": tf.io.VarLenFeature(tf.float32),
            "label": tf.io.VarLenFeature(tf.int64),
        }
        return features_to_annotation(tf.io.parse_single_example(example, features))

    def features_to_annotation(parsed_feature):
        image = tf.io.decode_raw(input_bytes=parsed_feature["image"], out_type=tf.uint8)
        image_tensor = tf.reshape(image, (image_height, image_width, 3)) / 255
        labels = tf.cast(tf.sparse.to_dense(parsed_feature["x"]), dtype=tf.float32)
        x = tf.sparse.to_dense(parsed_feature["x"])
        y = tf.sparse.to_dense(parsed_feature["y"])
        w = tf.sparse.to_dense(parsed_feature["w"])
        h = tf.sparse.to_dense(parsed_feature["h"])
        stacked = tf.stack([x, y, w, h, labels], axis=1)
        paddings = [[0, 100 - tf.shape(stacked)[0]], [0, 0]]
        annotation = tf.pad(stacked, paddings)

        return image_tensor, annotation

    dataset = tf.data.TFRecordDataset([filepath])
    dataset = dataset.map(feature_parser)

    # To-Do remove .repeat, just a hack for now
    dataset = dataset.repeat(100)

    # batch your dataset
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset
