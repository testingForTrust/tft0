# -*- coding: utf-8 -*-
"""
code from https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_patch/AdversarialPatch.ipynb
and https://arxiv.org/abs/1712.09665
The code takes sample of images and a target class, makes adversarial pixel blocked patch for each of the image in
samples
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import tag_constants

import matplotlib.pyplot as plt
import math
import numpy as np
import PIL.Image
import scipy
import time
import random

import keras
from keras.applications import mobilenetv2
from keras import backend as k_backend
from keras.models import Model
from keras.models import load_model
from keras.layers import Input

# Declaration of some global variables that are used throughtout the code.

TARGET_LABEL = 5575  # garbage value, is the target label for which patch is to be created
PATCH_BLOCK = -1  # for 8x8 blocks , if value =8
Patch_Len = -1  # int(224/PATCH_BLOCK) = number of PATCH_BLOCK
# PATCH_SHAPE = (Patch_Len,Patch_Len,3)#(299, 299, 3)
# Ensemble of models


MODEL_NAMES = ['name you want to give to your model']  # the name u want for ur model

# Data augmentation
# Empirically found that training with a very wide scale range works well
# as a default
SCALE_MIN = 0.3  # minimum size of patch for training
SCALE_MAX = 1.5  # maximum size of patch for training

IMAGE_HEIGHT = -1  # garbage value, height of image
IMAGE_WIDTH = -1  # garbage value, width of image
CH = -1  # garbage value# no. of channels in image =3 for rgb
PATCH_SHAPE = (-1, -1, -1)  # shape of patch
BATCH_SIZE = -1  # batch size to be used for training
LEARNING_RATE = 'learning rate to train the patch'
MODEL_FILE_PATH = ''  # location of model
FILE_NAME = ''  # name of model file, for example model.h5 or model.meta
IMAGES = np.array((5, 3, 3, 3))  # 'a numpy array of images' # numpy array of Images, for 100 images of shape (32,32,3),
# its shape will be (100,32,32,3)
MAX_ROTATION = 0  # Max rotation to be used for training
NUM_CLASSES = -1  # no. of classes, i.e labels for example 1000 for imagenet
TARGET_ONEHOT = -1  # one-hot target labels
image_loader = -1  # an object for loading images and batch of images
MM = -1  # a model object for training , inference etc
NUM_IMAGES = -1  # Number of sample images to train on
TEST_SCALE = -1  # The size of patch to put on returned images
name_to_label = {}  # A dictionary mapping from names to labels
# Description is given in the string itself
Preprocess_Func = 'a func to preprocess the input array'
Rev_Preprocess = 'a func to reverse preprocess from processed input back to normal input'
INPUT_TENSOR_NAME = 'name of input tensor for ex "inp"'
LOGITS_TENSOR_NAME = 'name of logits tensor for ex "logits"'
RANGE_MODEL_INPUT = 'a tuple of floats. The minimum & maximum values of pre-processed input images fed to the model'
PATCH_APPEARANCE = 'a string (circle or rectangle). By default circle. Indicates how the patch appears on images.'


def enlarge_patch(x):
    """

    :param x: numpy array, a small patch
    :return:  numpy array, a larger patch, by concatenating small patch pixels
    """
    y = np.empty((0, IMAGE_HEIGHT, CH), dtype=np.float32)
    for i in range(0, Patch_Len):
        y1 = np.empty((PATCH_BLOCK, 0, CH), dtype=np.float32)
        for j in range(0, Patch_Len):
            y2 = np.array(x[i][j], dtype=np.float32)
            y2 = np.tile(y2, (PATCH_BLOCK, PATCH_BLOCK, 1))  # 8x8x1
            y1 = np.concatenate((y1, y2), axis=1)
        y = np.concatenate((y, y1), axis=0)
    return y


def save_transparent_png_patch(patch):
    """

    :param patch: numpy array, small patch
    :return: numpy array, a large patch patch with transparent channel
    the function takes small patch, enlarges it, adds extra transparent channel and returns
    """

    pat = Rev_Preprocess(patch)/np.max(patch)
    pat_big = enlarge_patch(pat)
    if PATCH_APPEARANCE.lower() == 'circle':
        mask = _circle_mask((IMAGE_HEIGHT, IMAGE_WIDTH, 4), PATCH_APPEARANCE)  # 4 for the rgba channels in png
        plt.imsave('patch_big.png', pat_big)
        pat_big = plt.imread('patch_big.png')
        pat_transparent = mask * pat_big
        return pat_transparent
    else:
        return pat_big


class StubImageLoader:
    """
    Class for loading images, have function get_images for traversing random images for training
    function get_batch_images is used to get batch of images for inference
    """

    def __init__(self):
        self.images = []
        assert 2*IMAGES.shape[0] >= BATCH_SIZE, "2*IMAGES.shape[0] should be >= BATCH_SIZE"

        print("\n\n image shape\n\n", IMAGES.shape)
        for i in range(IMAGES.shape[0]):
            self.images.append(IMAGES[i])

        last_batch_size = IMAGES.shape[0] % BATCH_SIZE
        extra_imgs = BATCH_SIZE - last_batch_size
        
        if last_batch_size > 0:
            for i in range(extra_imgs):
                self.images.append(IMAGES[i])

    def get_images(self):
        # index = np.random.choice(IMAGES.shape[0], BATCH_SIZE, replace=False)
        # return IMAGES[index]
        return random.sample(self.images, BATCH_SIZE)

    def get_batch_images(self, index):
        # return IMAGES[BATCH_SIZE * index:BATCH_SIZE * (index + 1)]
        return self.images[BATCH_SIZE * index:BATCH_SIZE * (index + 1)]


def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
    """
    Return transform mapping, which when used for transforming an image, scales it, rotates it and
    shifts it to some random location.
     If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
     then it maps the output point (x, y) to a transformed input point
     (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
     where k = c0 x + c1 y + 1.
     The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90. * (math.pi / 2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
         [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift / (2 * im_scale))
    b2 = y_origin_delta - (y_shift / (2 * im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


def _circle_mask(shape, patch_appearance, sharpness=40):
    """Return a circular mask of a given shape
        given a shape for ex. (32,32,3) return a largets circle
        that fits the (32,32) square, i.e has all 1's in circular shape
        and 0's outside
    """
    assert patch_appearance.lower() == 'circle' or patch_appearance.lower() == 'rectangle', \
        "The patch_appearance attribute must be set either to 'circle' or 'rectangle'"
    if patch_appearance.lower() == 'circle':
        assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape
        diameter = shape[0]
        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = (xx ** 2 + yy ** 2) ** sharpness
        mask = 1 - np.clip(z, -1, 1)
        mask = np.expand_dims(mask, axis=2)
        mask = np.broadcast_to(mask, shape).astype(np.float32)
        return mask
    else:
        mask = np.ones(shape)
        return mask.astype(np.float32)


def _gen_target_ys():
    """
    generates one_hot vector for target class
    :return: target_class one_hot vector
    """
    label = TARGET_LABEL
    y_one_hot = np.zeros(NUM_CLASSES)
    y_one_hot[label] = 1.0
    y_one_hot = np.tile(y_one_hot, (BATCH_SIZE, 1))
    return y_one_hot


class ModelContainer:
    """Encapsulates an model, and methods for interacting with it.
        for example training and inferencing
    """

    def __init__(self, model_name, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
        # Peace Mask: None, "Forward", "Backward"
        self.model_name = model_name
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.peace_mask = peace_mask
        self.patch_shape = PATCH_SHAPE
        self._peace_mask_overlay = peace_mask_overlay
        self.load_model(verbose=verbose)

    def patch(self, new_patch=None):
        """Retrieve or set the adversarial patch.

        new_patch: The new patch to set, or None to get current patch.

        Returns: Itself if it set a new patch, or the current patch."""
        if new_patch is None:
            return self._run(self._clipped_patch)

        self._run(self._assign_patch, {self._patch_placeholder: new_patch})
        return self

    def reset_patch(self):
        """Reset the adversarial patch to all zeros."""
        self.patch(np.random.uniform(RANGE_MODEL_INPUT[0], RANGE_MODEL_INPUT[1], size=self.patch_shape).astype
                   (np.float32))

    def train_step(self, images=None, target_ys=None, scale=(0.1, 1.0), dropout=None,
                   patch_disguise=None, disguise_alpha=None):
        """Train the model for one step.

        Args:
            images: A batch of images to train on, it loads one if not present.
            target_ys: Onehot target vector, defaults to TARGET_ONEHOT
            scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
            dropout: consider dropout or not
            patch_disguise: A Disguised Patch, if not None, then learned patch should be close to patch_disguise
            disguise_alpha: Weightage given to disguised_patch
        Returns: Loss on the target ys."""
        if images is None:
            images = image_loader.get_images()
        if target_ys is None:
            target_ys = TARGET_ONEHOT

        feed_dict = {self._image_input: images,
                     self._target_ys: target_ys,
                     self._learning_rate: LEARNING_RATE,
                     }

        if patch_disguise is not None:
            if disguise_alpha is None:
                raise ValueError("You need disguise_alpha")
            feed_dict[self.patch_disguise] = patch_disguise
            feed_dict[self.disguise_alpha] = disguise_alpha

        loss, _ = self._run([self._loss, self._train_op], feed_dict, scale=scale, dropout=dropout)
        return loss

    def inference_batch(self, index, images=None, target_ys=None, scale=None):
        """Report loss and label probabilities, and patched images for a batch.

        Args:
            images: A batch of images to train on, it loads if not present.
            scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
            index: Which target batch to pick for inference, for example 5th batch
            target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present."""
        global BATCH_SIZE
        if images is None:
            images = image_loader.get_batch_images(index=index)  # get_images()
        if target_ys is None:
            target_ys = TARGET_ONEHOT  # have used extra :len(images)

        print("\n\n Inferencing \n\n")

        # if BATCH_SIZE > len(images):
        #    print("\n\n\n",BATCH_SIZE,len(images),"\n\n\n")
        #    BATCH_SIZE = len(images)
        #    print("\n\n\n", BATCH_SIZE, len(images), "\n\n\n")

        feed_dict = {self._image_input: images, self._target_ys: target_ys}

        loss_per_example, ps, ims = self._run([self._loss_per_example, self._probabilities, self._patched_input],
                                              feed_dict, scale=scale)
        return loss_per_example, ps, ims

    def load_model(self, verbose=True):
        """
        calls make_model_and_ops which loads the model, make necessary changes for example adding
        patch to image inputs
        :return:
        """

        # keras_mode = False
        patch = None
        self._make_model_and_ops(patch, verbose)
        # self._make_model_and_ops(keras_mode, patch, verbose)

    def _run(self, target, feed_dict=None, scale=None, dropout=None):
        """
        generic function to run the session for obtaining the target value given
        the feed_dict
        :return: the target result from running the tensorflow session
        """
        k_backend.set_session(self.sess)
        if feed_dict is None:
            feed_dict = {}
        feed_dict[self.learning_phase] = False

        if scale is not None:
            if isinstance(scale, (tuple, list)):
                scale_min, scale_max = scale
            else:
                scale_min, scale_max = (scale, scale)
            feed_dict[self.scale_min] = scale_min
            feed_dict[self.scale_max] = scale_max

        if dropout is not None:
            feed_dict[self.dropout] = dropout
        return self.sess.run(target, feed_dict=feed_dict)

    def _make_model_and_ops(self, patch_val, verbose):
        """
        The main logic of the code that loads the model, modifies it to include
        patched images, defines the loss, training step
        """
        start = time.time()
        k_backend.set_session(self.sess)
        with self.sess.graph.as_default():
            self.learning_phase = k_backend.learning_phase()

            image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, CH)  # (299, 299, 3), change in other code also
            self._image_input = keras.layers.Input(shape=image_shape)

            self.scale_min = tf.placeholder_with_default(SCALE_MIN, [])
            self.scale_max = tf.placeholder_with_default(SCALE_MAX, [])
            self._scales = tf.random_uniform([BATCH_SIZE], minval=self.scale_min, maxval=self.scale_max)

            image_input = self._image_input
            self.patch_disguise = tf.placeholder_with_default(tf.zeros(self.patch_shape), shape=self.patch_shape)
            self.disguise_alpha = tf.placeholder_with_default(0.0, [])
            init = tf.constant(np.random.uniform(RANGE_MODEL_INPUT[0], RANGE_MODEL_INPUT[1], size=self.patch_shape).
                               astype(np.float32))
            patch = tf.get_variable("patch", initializer=init)
            self._patch_placeholder = tf.placeholder(dtype=tf.float32, shape=self.patch_shape)
            self._assign_patch = tf.assign(patch, self._patch_placeholder)
            # self._batch_size =  tf.placeholder(dtype=tf.int32,shape=[])

            modified_patch = patch

            def clip_to_valid_image(x):
                return tf.clip_by_value(x, clip_value_min=RANGE_MODEL_INPUT[0], clip_value_max=RANGE_MODEL_INPUT[1])

            self._clipped_patch = clip_to_valid_image(modified_patch)

            self.dropout = tf.placeholder_with_default(1.0, [])
            patch_with_dropout = tf.nn.dropout(modified_patch, keep_prob=self.dropout)
            patched_input = clip_to_valid_image(self._random_overlay(image_input, patch_with_dropout, image_shape))
            # patched_input = self._random_overlay(image_input, patch_with_dropout, image_shape)
            self._patched_input = patched_input

            # Labels for our attack (e.g. always a toaster)
            self._target_ys = tf.placeholder(tf.float32, shape=(None, NUM_CLASSES))
            print("patched_input--", patched_input)
            # Pre-softmax logits of our pretrained model
            # mnet = mobilenetv2.MobileNetV2(input_tensor=patched_input, weights=None, include_top=False, pooling='avg')
            # predictions = Dense(4,activation='softmax', use_bias=True,name='predictions')(mnet.output)
            # model = Model(inputs=mnet.input, outputs=predictions)
            # model.load_weights('mobilenet_rgb_stopint.h5')

            # keras tft#######
            if FILE_NAME[-3:] == '.h5':
                oldModel = load_model(MODEL_FILE_PATH + '/' + FILE_NAME)
                oldModel.layers.pop(0)
                pi = Input(tensor=patched_input, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CH))
                newOutputs = oldModel(patched_input)
                model = Model(pi, newOutputs)
                logits = model.outputs[0].op.inputs[0]
                self._probabilities = model.outputs[0]
            # tensorflow meta and ckpt model
            elif FILE_NAME[-5:] == '.meta':
                self.sess.run(tf.global_variables_initializer())
                saver = tf.train.import_meta_graph(MODEL_FILE_PATH + '/' + FILE_NAME,
                                                   input_map={INPUT_TENSOR_NAME: patched_input})
                saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_FILE_PATH))
                logits = tf.get_default_graph().get_tensor_by_name(LOGITS_TENSOR_NAME)
                self._probabilities = tf.nn.softmax(logits)
            elif FILE_NAME[-3:] == '.pb':
                self.sess.run(tf.global_variables_initializer())
                model = tf.saved_model.loader.load(sess=self.sess, export_dir=MODEL_FILE_PATH,
                                                   tags=[tag_constants.SERVING],
                                                   input_map={INPUT_TENSOR_NAME: patched_input})
                logits = self.sess.graph.get_tensor_by_name(LOGITS_TENSOR_NAME)
                self._probabilities = tf.nn.softmax(logits)
            else:
                print('\n\nBAD File Name, should end with .meta or .h5\n\n')
                exit()
            self._loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._target_ys,
                logits=logits
            )

            self._target_loss = tf.reduce_mean(self._loss_per_example)

            self._patch_loss = tf.nn.l2_loss(patch - self.patch_disguise) * self.disguise_alpha

            self._loss = self._target_loss + self._patch_loss

            # Train our attack by only training on the patch variable
            self._learning_rate = tf.placeholder(tf.float32)
            self._train_op = tf.train.GradientDescentOptimizer(self._learning_rate) \
                .minimize(self._loss, var_list=[patch])

            # self._probabilities = tf.nn.softmax(logits)#model.outputs[0]

            if patch_val is not None:
                self.patch(patch_val)
            else:
                self.reset_patch()

            elapsed = time.time() - start
            if verbose:
                print("Finished loading {}, took {:.0f}s".format(self.model_name, elapsed))

    def _random_overlay(self, imgs, patch, image_shape):
        """Augment images with random rotation, transformation.

        Image: BATCHx299x299x3
        concatenates the patch on some part of the image and returns the
        corresponding patched image
        :returns - the patched images

        """
        # Add padding

        print('\npatch is :', patch)

        def my_func(x):
            """
            exactly same as enlarge_patch function
            :param x: numpy array, a small patch
            :return:  numpy array, a larger patch, by concatenating small patch pixels
            """
            y = np.empty((0, IMAGE_HEIGHT, CH), dtype=np.float32)
            for i in range(0, Patch_Len):
                y1 = np.empty((PATCH_BLOCK, 0, CH), dtype=np.float32)
                for j in range(0, Patch_Len):
                    y2 = np.array(x[i][j], dtype=np.float32)
                    y2 = np.tile(y2, (PATCH_BLOCK, PATCH_BLOCK, 1))  # 8x8x1
                    y1 = np.concatenate((y1, y2), axis=1)
                y = np.concatenate((y, y1), axis=0)
            return y  # [:224,:224,:]

        ##############################################################################################
        #############################################################################################
        def py_func(func, inp, tout, stateful=True, name=None, grad=None):
            """
            overrides the gradient for the "func=my_func"  with the custom gradient grad
            note that initial gradient w.r.t to "func = my_func" would not be defined
            """

            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

            tf.RegisterGradient(rnd_name)(grad)
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, tout, stateful=stateful, name=name)

        # Def custom function using my_func instead of usual tf operations:
        def extend_patch(x, name=None):
            with ops.op_scope([x], name, "Extend_patch") as name:
                ext_patch = py_func(my_func,
                                    [x],
                                    [tf.float32],
                                    name=name,
                                    grad=_extendpatchgrad)  # <-- here's the call to the gradient
                return ext_patch[0]

        def _extendpatchgrad(op, grad):
            """

            :param op: used for input, but we don't use it, since we don't need the input
            :param grad: the inflowing gradients w.r.t the large patch
            :return: the gradients w.r.t the small patch
            """
            y = np.zeros((0, Patch_Len, CH), dtype=np.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            for i in range(Patch_Len):
                y1 = np.zeros((1, 0, CH), dtype=np.float32)
                y1 = tf.convert_to_tensor(y1, tf.float32)

                for j in range(Patch_Len):
                    sum = 0
                    for i1 in range(PATCH_BLOCK * i, PATCH_BLOCK * i + PATCH_BLOCK):
                        for i2 in range(PATCH_BLOCK * j, PATCH_BLOCK * j + PATCH_BLOCK):
                            sum += grad[i1][i2]
                    sum = tf.reshape(sum, [1, 1, CH])
                    # sum = sum/(PATCH_BLOCK*PATCH_BLOCK)
                    y1 = tf.concat([y1, sum], 1)
                y = tf.concat([y, y1], 0)
            return y

        patch_extend = extend_patch(patch)  # tf.py_func(my_func, [patch], tf.float32)
        # print(patch_extend,patch_extend[0])
        patch_extend.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, CH])
        image_mask = _circle_mask(image_shape, PATCH_APPEARANCE)

        image_mask = tf.stack([image_mask] * BATCH_SIZE)
        padded_patch = tf.stack([patch_extend] * BATCH_SIZE)

        transform_vecs = []

        def _random_transformation(scale_min, scale_max, width):
            """

            :param scale_min: minimum scale of image after transform
            :param scale_max: maximum scale of image after transform
            :param width: width of image
            :return: a transform vector which can be used to transform the given image
                    to a image with size as random between scale_min and scale_max  of original
                    image, some random rotation and shifted to a random part of image
                    such that image doen't come out of its boundary
            """
            im_scale = np.random.uniform(low=scale_min, high=scale_max)

            padding_after_scaling = (1 - im_scale) * width
            x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

            rot = np.random.uniform(-MAX_ROTATION, MAX_ROTATION)

            return _transform_vector(width,
                                     x_shift=x_delta,
                                     y_shift=y_delta,
                                     im_scale=im_scale,
                                     rot_in_degrees=rot)

        for _ in range(BATCH_SIZE):
            # Shift and scale the patch for each image in the batch
            random_xform_vector = tf.py_func(_random_transformation, [self.scale_min, self.scale_max, image_shape[0]],
                                             tf.float32)
            random_xform_vector.set_shape([8])

            transform_vecs.append(random_xform_vector)

        image_mask = tf.contrib.image.transform(image_mask, transform_vecs, "BILINEAR")
        padded_patch = tf.contrib.image.transform(padded_patch, transform_vecs, "BILINEAR")

        inverted_mask = (1 - image_mask)
        return imgs * inverted_mask + padded_patch * image_mask


class MetaModel:
    """
    A class for encapsulating model object, defines train and inference functions
    """
    def __init__(self, verbose=True, peace_mask=None, peace_mask_overlay=0.0):
        self.nc = {m: ModelContainer(m, verbose=verbose, peace_mask=peace_mask, peace_mask_overlay=peace_mask_overlay)
                   for m in MODEL_NAMES}
        self._patch = np.zeros(PATCH_SHAPE)
        self.patch_shape = PATCH_SHAPE

    def patch(self, new_patch=None):
        """Retrieve or set the adversarial patch.

        new_patch: The new patch to set, or None to get current patch.

        Returns: Itself if it set a new patch, or the current patch."""
        if new_patch is None:
            return self._patch

        self._patch = new_patch
        return self

    def reset_patch(self):
        """Reset the adversarial patch to uniform dist."""
        self.patch(np.random.uniform(RANGE_MODEL_INPUT[0], RANGE_MODEL_INPUT[1], size=self.patch_shape).
                   astype(np.float32))

    def train_step(self, model=None, steps=1, images=None, target_ys=None, scale=None, **kwargs):
        """Train the model for `steps` steps.

        Args:
            model: model name
            steps: number of iterations to train model
            images: A batch of images to train on, it loads one if not present.
            target_ys: Onehot target vector, defaults to TARGET_ONEHOT
            learning_rate: Learning rate for this train step.
            scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.

        Returns: Loss on the target ys."""

        if model is not None:
            to_train = [self.nc[model]]
        else:
            to_train = self.nc.values()

        losses = []
        for mc in to_train:
            mc.patch(self.patch())
            for _ in range(steps):
                loss = mc.train_step(images, target_ys, scale=scale, **kwargs)
                losses.append(loss)
            self.patch(mc.patch())
        return np.mean(losses)

    def inference_batch(self, model, index, images=None, target_ys=None, scale=None):
        """Report loss and label probabilities, and patched images for a batch.

        Args:
            model: name of the model
            index: Which target batch to pick for inference, for example 5th batch
            images: A batch of images to train on, it loads if not present.
            target_ys: The target_ys for loss calculation, TARGET_ONEHOT if not present.
            scale: Either a scalar value for the exact scale, or a (min, max) tuple for the scale range.
        """

        mc = self.nc[model]
        mc.patch(self.patch())
        return mc.inference_batch(images, target_ys, scale=scale, index=index)


def report(model, n=400, scale=(0.1, 1.0)):
    """Prints a report on how well the model is doing.


    Args:
        :param model: can be a ModelContainer instance, or a string. If it's a string, we
                lookup that model name in the MultiModel
        :param n: int, number of images for which to report the results
        :param scale: int or tuple, the fraction of image covered by patch
    """

    n_batches = int(math.ceil(float(n) / BATCH_SIZE))
   
    patched_images = np.empty((0, IMAGE_HEIGHT, IMAGE_WIDTH, CH), dtype=np.float32)
    probabs = np.empty((0, NUM_CLASSES), dtype=np.float32)  # 0,(4--> number of classes)

    for b in range(n_batches):
        if isinstance(model, str):
            loss_per_example, probs, patched_imgs = MM.inference_batch(model, scale=scale, index=b)
        else:
            loss_per_example, probs, patched_imgs = model.inference_batch(scale=scale, index=b)

        patched_images = np.concatenate((patched_images, patched_imgs), axis=0)
        probabs = np.concatenate((probabs, probs), axis=0)

    return patched_images[:n]


def train_models(steps):
    """
    A function that calls train_step for number of epochs,
    essentially train the model
    :return: a numpy array containg patched images and another numpy array storing the patch
    """
    model_targets = MODEL_NAMES  # ['mobilenetv2']
    epochs = steps  # As per mobile net model on traffic signs where we found patches to perform well
    print("Will run for epochs==>", epochs)
    regular_training_model_to_patch = {}

    for m in model_targets:
        print("Training %s" % m)
        model_con = MM.nc[m]  # ModelContainer
        model_con.reset_patch()
        for i in range(epochs):

            loss = model_con.train_step(scale=(0.1, 1.0))  # 0.7,1.2
            if i % int(epochs / 10) == 0:
                print("[%s] loss: %s" % (i, loss))

    regular_training_model_to_patch[m] = model_con.patch()

    patch_transparent = save_transparent_png_patch(model_con.patch())
    m = MM.nc[model_targets[0]]
    m.patch(regular_training_model_to_patch[model_targets[0]])
    return report(m, n=NUM_IMAGES, scale=TEST_SCALE), patch_transparent


def adv_patch(m, target_label_name, n_to_l, images, num_classes, model_file_path, file_name, input_tensor_name,
              logits_tensor_name, num_images, patch_block, range_model_input, patch_appearance, batch_size=4,
              learning_rate=5.0, scale_min=0.1, scale_max=1.0, max_rotation=22.5, model_name='model_to_test',
              test_scale=0.4):
    """

    :param m: model object
    :param target_label_name: string, the target label for example, "ostrich"
    :param n_to_l: a dictionary mapping from name to labels
    :param images: a numpy array of images
    :param num_classes: int, number of labels/classes in model
    :param model_file_path: string, path where model is stored
    :param file_name: string, name of model file for example model.h5 or model.meta
    :param input_tensor_name: string, name of input tensor for example "input:0"
    :param logits_tensor_name: string, name of logits tensor for example "logits:0"
    :param num_images: int, number of images
    :param patch_block: int, block size of pixels with same value, for example if 16 the a grid of 16x16 pixels will
                        have same value
    :param range_model_input: a tuple of floats.Range of values such that reverse preprocess gives valid original images
    :param patch_appearance: a string ('circle' or 'rectangle').By default 'circle'.Indicates how the patch appear on
    images.
    :param batch_size: int, batch_size to be used for training
    :param learning_rate: a float - learning rate for training the patch
    :param scale_min:  float [0,1], minimum patch size to be used for training
    :param scale_max:  float [0,1], maximum patch size to be used for training
    :param max_rotation: float, maximum rotation of patch to be used for training
    :param model_name: string, name u give to ur model
    :param test_scale: int or tuple, fraction of patch covering the image
    :return: a numpy array containg patched images and another numpy array storing the patch
    """
    global TARGET_LABEL, PATCH_SHAPE, BATCH_SIZE, SCALE_MAX, SCALE_MIN, MAX_ROTATION, IMAGE_HEIGHT, \
        IMAGE_WIDTH, CH, MODEL_NAMES, NUM_CLASSES, TARGET_ONEHOT
    global image_loader, MM, name_to_label, MODEL_FILE_PATH, NUM_IMAGES, TEST_SCALE, FILE_NAME, \
        Preprocess_Func, PATCH_BLOCK, Patch_Len, PATCH_SHAPE, IMAGES, RANGE_MODEL_INPUT, PATCH_APPEARANCE
    global INPUT_TENSOR_NAME, LOGITS_TENSOR_NAME, Rev_Preprocess, LEARNING_RATE
    TARGET_LABEL = int(n_to_l[target_label_name])

    name_to_label = n_to_l
    # PATCH_SHAPE = (m.image_size_height, m.image_size_width, m.num_channels)  # (299, 299, 3)
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    SCALE_MAX = scale_max
    SCALE_MIN = scale_min
    MAX_ROTATION = max_rotation
    IMAGE_HEIGHT = m.image_size_height
    IMAGE_WIDTH = m.image_size_width
    CH = m.num_channels
    INPUT_TENSOR_NAME = input_tensor_name
    LOGITS_TENSOR_NAME = logits_tensor_name
    IMAGES = images
    MODEL_NAMES = [model_name]
    PATCH_BLOCK = patch_block
    RANGE_MODEL_INPUT = range_model_input
    PATCH_APPEARANCE = patch_appearance
    Patch_Len = int(m.image_size_height / patch_block)
    PATCH_SHAPE = (Patch_Len, Patch_Len, CH)
    NUM_CLASSES = num_classes
    NUM_IMAGES = num_images
    TEST_SCALE = test_scale
    FILE_NAME = file_name
    MODEL_FILE_PATH = model_file_path
    Preprocess_Func = m.pre_process
    Rev_Preprocess = m.rev_preprocess
    TARGET_ONEHOT = _gen_target_ys()
    image_loader = StubImageLoader()
    MM = MetaModel()
    steps = np.max([int((num_images/batch_size)*4), 500])

    return train_models(steps)
