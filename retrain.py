# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# NOTICE: This work was derived from tensorflow/examples/image_retraining
# and modified to use TensorFlow Hub modules.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

def main():
    image_dir = os.getcwd() + '\\training_images\\'  # str     path to folders of labeled images
    output_graph = os.getcwd() + '\\_graph.pb'  # str     where to save the trained graph
    intermediate_output_graphs_dir = os.getcwd() + '\\_intermediate_graph\\'  # str     where to save the intermediate graphs
    intermediate_store_frequency = 0  # int     How many steps to store intermediate graph. If "0" then will not store
    output_labels = os.getcwd() + '\\_labels.txt'  # str     where to save the trained graph's labels
    summaries_dir = os.getcwd() + '\\_summaries_dir'  # str     where to save summary logs for TensorBoard
    how_many_training_steps = 100  # int     default 4000 how many training steps to run before ending
    learning_rate = 0.01  # float   how large a learning rate to use when training
    testing_percentage = 10  # int     what percentage of images to use as a test set
    validation_percentage = 10  # int     what percentage of images to use as a validation set
    eval_step_interval = 10  # int     how often to evaluate the training results
    train_batch_size = 100  # int     how many images to train on at a time
    test_batch_size = -1  # int     how many images to test on value of -1 causes the entire test set to be used
    validation_batch_size = 100  # int     how many images to test on evaluation step value of -1 causes the entire test set to be used

    print_misclassified_test_images = False  # bool    whether to print out a list of all misclassified test images
    bottleneck_dir = os.getcwd() + '\\_bottleneck'  # str     path to cache bottleneck layer values as files
    final_tensor_name = 'final_result'  # str     the name of the output classification layer in the retrained graph

    flip_left_right = False  # bool    whether to randomly flip half of the training images horizontally
    random_crop = 0  # int     A percentage determining how much of a margin to randomly crop off the training images
    random_scale = 0  # int     A percentage determining how much to randomly scale up the size of the training images
    random_brightness = 0  # int     A percentage determining how much to randomly multiply the training image input pixels up or down

    # tf_hub_module str which TensorFlow Hub module to use
    tf_hub_module = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
    saved_model_dir = ''  # str     Where to save the exported graph

    print('main()')
    tf.logging.set_verbosity(tf.logging.INFO)

    if not image_dir:
        tf.logging.error('No image_dir location.')
        return -1
    prepare_file_system(summaries_dir, intermediate_store_frequency)
    image_lists = create_image_lists(image_dir, testing_percentage,
                                     validation_percentage)
    class_count = len(image_lists.keys())
    print('image_lists {}'.format(image_lists))

    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    do_distort_images = should_distort_images(
        flip_left_right,
        random_crop,
        random_scale,
        random_brightness
    )

    module_spec = hub.load_module_spec(tf_hub_module)
    graph, bottleneck_tensor, re_sized_image_tensor, wants_quantization = (
        create_module_graph(module_spec)
    )

    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, final_tensor_name, bottleneck_tensor, wants_quantization, learning_rate, is_training=True)

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        if do_distort_images:
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(
                flip_left_right, random_crop, random_scale,
                random_brightness, module_spec)
        else:
            cache_bottlenecks(sess, image_lists, image_dir,
                              bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, re_sized_image_tensor,
                              bottleneck_tensor, tf_hub_module)

        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)

        validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

        train_saver = tf.train.Saver()

        for i in range(how_many_training_steps):
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = get_random_distorted_bottlenecks(
                    sess, image_lists, train_batch_size, 'training',
                    image_dir, distorted_jpeg_data_tensor,
                    distorted_image_tensor, re_sized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth, _) = get_random_cached_bottlenecks(
                    sess, image_lists, train_batch_size, 'training',
                    bottleneck_dir, image_dir, jpeg_data_tensor,
                    decoded_image_tensor, re_sized_image_tensor, bottleneck_tensor,
                    tf_hub_module)
            # Feed the bottlenecks and ground truth into the graph, and run a training

            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == how_many_training_steps)
            if (i % eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                (datetime.now(), i, cross_entropy_value))
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        sess, image_lists, validation_batch_size, 'validation',
                        bottleneck_dir, image_dir, jpeg_data_tensor,
                        decoded_image_tensor, re_sized_image_tensor, bottleneck_tensor,
                        tf_hub_module))
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (datetime.now(), i, validation_accuracy * 100,
                                 len(validation_bottlenecks)))

            intermediate_frequency = intermediate_store_frequency

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                    and i > 0):
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (intermediate_output_graphs_dir +
                                          'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                save_graph_to_file(intermediate_file_name, module_spec,
                                   class_count, final_tensor_name)

        train_saver.save(sess, CHECKPOINT_NAME)
        run_final_eval(sess, module_spec, class_count, image_lists,
                       jpeg_data_tensor, decoded_image_tensor,
                       re_sized_image_tensor, bottleneck_tensor,
                       test_batch_size, bottleneck_dir, image_dir,
                       tf_hub_module, final_tensor_name, learning_rate,
                       print_misclassified_test_images)

        tf.logging.info('Save final result to : ' + output_graph)
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(output_graph, module_spec, class_count, final_tensor_name, learning_rate)
        with tf.gfile.GFile(output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if saved_model_dir:
            export_model(module_spec, class_count, saved_model_dir)

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    print('create_image_lists({}, {}, {})'.format(image_dir, testing_percentage, validation_percentage))
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # The root directory comes first, so skip it.
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg']))
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
            result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):

    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path





def create_module_graph(module_spec):
    print('create_module_graph({})'.format(module_spec))
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        re_sized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(re_sized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS
                                     for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, re_sized_input_tensor, wants_quantization


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, re_sized_input_tensor,
                            bottleneck_tensor):
  re_sized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  bottleneck_values = sess.run(bottleneck_tensor,
                               {re_sized_input_tensor: re_sized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, module_name):
    module_name = (module_name.replace('://', '~')  # URL scheme.
                 .replace('/', '~')  # URL and Unix paths.
                 .replace(':', '~').replace('\\', '~'))  # Windows paths.
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + module_name + '.txt'

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, re_sized_input_tensor,
                           bottleneck_tensor):

    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        re_sized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, re_sized_input_tensor,
                             bottleneck_tensor, module_name):

    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, module_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, re_sized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, re_sized_input_tensor,
                           bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      re_sized_input_tensor, bottleneck_tensor, module_name):


    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index,
                    image_dir, category, bottleneck_dir,
                    jpeg_data_tensor, decoded_image_tensor,
                    re_sized_input_tensor, bottleneck_tensor, module_name)
                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, re_sized_input_tensor,
                                  bottleneck_tensor, module_name):

    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                re_sized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                    image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    sess, image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    re_sized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, re_sized_input_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = tf.gfile.GFile(image_path, 'rb').read()

        distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                 {re_sized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):

    print('should_distort_images({}, {}, {}, {})'.format(
        flip_left_right,
        random_crop,
        random_scale,
        random_brightness
        )
    )
    return  (
            flip_left_right or
            (random_crop != 0) or (random_scale != 0) or (random_brightness != 0)
    )



def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, module_spec):

    print('add_input_distortions()'.format(flip_left_right,
                                           random_crop,
                                           random_scale,
                                           random_brightness,
                                           module_spec
                                           )
          )

    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name = 'DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels = input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform( shape   = [],
                                            minval  = 1.0,
                                            maxval  = resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(   shape   = [],
                                            minval  = brightness_min,
                                            maxval  = brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name = 'DistortResult')

    return jpeg_data, distort_result


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                          quantize_layer, learning_rate, is_training):

    print('add_final_retrain_ops({}, {}, {}, {}, {}, {})'.format(
        class_count, final_tensor_name, bottleneck_tensor,
        quantize_layer, learning_rate, is_training
        )
    )

    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput')

      # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
              [bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)


    if quantize_layer:
        if is_training:
          tf.contrib.quantize.create_training_graph()
        else:
          tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    print('add_evaluation_step({} {})'.format(result_tensor, ground_truth_tensor))
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    return evaluation_step, prediction


def run_final_eval(train_session, module_spec, class_count, image_lists,
                   jpeg_data_tensor, decoded_image_tensor,
                   re_sized_image_tensor, bottleneck_tensor,
                   test_batch_size, bottleneck_dir, image_dir,
                   tf_hub_module, final_tensor_name, learning_rate,
                   print_misclassified_test_images):
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            train_session, image_lists,
            test_batch_size, 'testing',
            bottleneck_dir, image_dir,
            jpeg_data_tensor, decoded_image_tensor,
            re_sized_image_tensor, bottleneck_tensor,
            tf_hub_module))

    (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
        prediction) = build_eval_session(module_spec, class_count,
                                         final_tensor_name, learning_rate)
    test_accuracy, predictions = eval_session.run(
        [evaluation_step, prediction],
        feed_dict={
          bottleneck_input: test_bottlenecks,
          ground_truth_input: test_ground_truth
        })
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
        (test_accuracy * 100, len(test_bottlenecks)))

    if print_misclassified_test_images:
        tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                tf.logging.info('%70s  %s' % (test_filename,
                    list(image_lists.keys())[predictions[i]]))


def build_eval_session(module_spec, class_count, final_tensor_name, learning_rate):
    eval_graph, bottleneck_tensor, re_sized_input_tensor, wants_quantization = (
        create_module_graph(module_spec))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        (_, _, bottleneck_input,
        ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, final_tensor_name, bottleneck_tensor,
            wants_quantization, learning_rate, is_training=False)
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor,
                                                      ground_truth_input)

    return (eval_sess, re_sized_input_tensor, bottleneck_input, ground_truth_input,
          evaluation_step, prediction)


def save_graph_to_file(graph_file_name, module_spec, class_count, final_tensor_name, learning_rate):
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count, final_tensor_name, learning_rate)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])

    with tf.gfile.GFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def prepare_file_system(summaries_dir, intermediate_store_frequency):
    print('prepare_file_system() {}'.format(summaries_dir))
    # Set up the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
        tf.gfile.MakeDirs(summaries_dir)
    if intermediate_store_frequency > 0:
        ensure_dir_exists(intermediate_output_graphs_dir)
    return


def add_jpeg_decoding(module_spec):
    print('add_jpeg_decoding({})'.format(module_spec))
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                   resize_shape_as_int)

    return jpeg_data, resized_image

def export_model(module_spec, class_count, saved_model_dir):
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    with sess.graph.as_default() as graph:
        tf.saved_model.simple_save(
            sess,
            saved_model_dir,
            inputs={'image': in_image},
            outputs={'prediction': graph.get_tensor_by_name('final_result:0')},
            legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
        )

if __name__ == '__main__':
    main()


