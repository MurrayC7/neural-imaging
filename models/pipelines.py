import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helpers.tf_helpers import lrelu, upsample_and_concat
from helpers.utils import upsampling_kernel, bilin_kernel, gamma_kernels
from models.tfmodel import TFModel

from models.tf_octConv import *
from models.tf_cnn_basic import *
from models.ops import *


class NIPModel(TFModel):
    """
    Abstract class for implementing neural imaging pipelines. Specific classes are expected to implement the
    'construct_model' method that builds the model, and 'parameters' method which lists its parameters. See existing
    classes for examples.
    """

    def __init__(self, sess=None, graph=None, loss_metric='L1', patch_size=None, label=None, reuse_placeholders=None,
                 **kwargs):
        """
        Base constructor with common setup.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param loss_metric: loss metric for NIP optimization (L2, L1, SSIM)
        :param patch_size: Optionally patch size can be given to fix placeholder dimensions (can be None)
        :param label: A string prefix for the model (useful when multiple NIPs are used in a single TF graph)
        :param reuse_placeholders: Give a dictionary with 'x' and 'y' keys if multiple NIPs should use the same inputs
        :param kwargs: Additional arguments for specific NIP implementations
        """
        super().__init__(sess, graph, label)

        # Initialize input placeholders and run 'construct_model' to build the model and
        # setup its output as self.y
        self.y = None  # This will be set up later by child classes
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.reg = 1e-4

        if reuse_placeholders is not None:
            self.x = reuse_placeholders['x']
            self.y_gt = reuse_placeholders['y']
        else:
            with self.graph.as_default():
                self.x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 4), name='x')
                self.y_gt = tf.placeholder(tf.float32, shape=(None, 2 * patch_size if patch_size is not None else None,
                                                              2 * patch_size if patch_size is not None else None, 3),
                                           name='y')

        self.construct_model(**kwargs)

        # Configure loss and model optimization
        self.loss_metric = loss_metric
        self.construct_loss(loss_metric)

    def construct_loss(self, loss_metric):
        with self.graph.as_default():
            with tf.name_scope('nip_optimization'):
                # Detect whether non-clipped image is available (better training stability)
                y = self.yy if hasattr(self, 'yy') else self.y

                # The loss
                if loss_metric == 'L2':
                    self.loss = tf.reduce_mean(tf.pow(255.0 * y - 255.0 * self.y_gt, 2.0))
                elif loss_metric == 'L1':
                    self.loss = tf.reduce_mean(tf.abs(255.0 * y - 255.0 * self.y_gt))
                elif loss_metric == 'SSIM':
                    self.loss = 255 * (1 - tf.image.ssim_multiscale(y, self.y_gt, 1.0))
                else:
                    raise ValueError('Unsupported loss metric!')

                # In case the model used batch norm
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # Learning rate
                    self.lr = tf.placeholder(tf.float32, name='nip_learning_rate')

                    # Create the optimizer and make sure only the parameters of the current model are updated
                    self.adam = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                       name='nip_adam{}'.format(self.scoped_name))
                    self.opt = self.adam.minimize(self.loss, var_list=self.parameters)

    def construct_model(self):
        """
        Constructs the NIP model. The method should use self.x as RAW image input, and set self.y as the model output.
        The output is expected to be clipped to [0,1]. For better optimization stability, the model can set self.yy to
        non-clipped output (will be used for gradient computation).

        A string prefix (self.scoped_name) should be used for variables / named scopes to facilitate using multiple NIPs
        in a single TF graph.
        """
        raise NotImplementedError()

    def training_step(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return the loss.
        """
        with self.graph.as_default():
            feed_dict = {
                self.x: batch_x,
                self.y_gt: batch_y,
                self.lr: learning_rate
            }
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = True

            _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
            return loss

    def process(self, batch_x, is_training=False):
        """
        Develop RAW input and return RGB image.
        """
        if batch_x.ndim == 3:
            batch_x = np.expand_dims(batch_x, 0)

        with self.graph.as_default():
            feed_dict = {self.x: batch_x}
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training

            y = self.sess.run(self.y, feed_dict=feed_dict)
            return y

    def reset_performance_stats(self):
        self.train_perf = {'loss': []}
        self.valid_perf = {'loss': [], 'psnr': [], 'ssim': []}


class UNet(NIPModel):
    """
    The UNet model, adapted from https://github.com/cchen156/Learning-to-See-in-the-Dark
    """

    def construct_model(self):
        with self.graph.as_default():
            conv1 = slim.conv2d(self.x, 32, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv1_1'.format(self.scoped_name))
            conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv1_2'.format(self.scoped_name))
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='{}/max_pool_1'.format(self.scoped_name))

            conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv2_1'.format(self.scoped_name))
            conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv2_2'.format(self.scoped_name))
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='{}/max_pool_2'.format(self.scoped_name))

            conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv3_1'.format(self.scoped_name))
            conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv3_2'.format(self.scoped_name))
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='{}/max_pool_3'.format(self.scoped_name))

            conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv4_1'.format(self.scoped_name))
            conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv4_2'.format(self.scoped_name))
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='{}/max_pool_4'.format(self.scoped_name))

            conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv5_1'.format(self.scoped_name))
            conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv5_2'.format(self.scoped_name))

            up6 = upsample_and_concat(conv5, conv4, 256, 512, name='weights',
                                      scope='{}/upsample_1'.format(self.scoped_name))
            conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv6_1'.format(self.scoped_name))
            conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv6_2'.format(self.scoped_name))

            up7 = upsample_and_concat(conv6, conv3, 128, 256, name='weights',
                                      scope='{}/upsample_2'.format(self.scoped_name))
            conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv7_1'.format(self.scoped_name))
            conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv7_2'.format(self.scoped_name))

            up8 = upsample_and_concat(conv7, conv2, 64, 128, name='weights',
                                      scope='{}/upsample_3'.format(self.scoped_name))
            conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv8_1'.format(self.scoped_name))
            conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv8_2'.format(self.scoped_name))

            up9 = upsample_and_concat(conv8, conv1, 32, 64, name='weights',
                                      scope='{}/upsample_4'.format(self.scoped_name))
            conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv9_1'.format(self.scoped_name))
            conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu,
                                scope='{}/conv9_2'.format(self.scoped_name))

            conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None,
                                 scope='{}/conv10'.format(self.scoped_name))

            with tf.name_scope('{}'.format(self.scoped_name)):
                self.yy = tf.depth_to_space(conv10, 2)
            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))


class OctUNet(NIPModel):
    """
    The OctaveUNet model, adapted from Octave Convolution
    """

    def construct_model(self):
        with self.graph.as_default():
            # conv1
            alpha = 0.25
            conv1_hf, conv1_lf = firstOctConv_BN_AC(data=self.x, alpha=alpha, num_filter_in=16, num_filter_out=32,
                                                    kernel=(3, 3), name='{}/conv1_1'.format(self.scoped_name),
                                                    pad='same')
            conv1_hf, conv1_lf = octConv_BN_AC(hf_data=conv1_hf, lf_data=conv1_lf, alpha=alpha, num_filter_in=32,
                                               num_filter_out=32,
                                               kernel=(3, 3), name='{}/conv1_2'.format(self.scoped_name), pad='same')
            pool1_hf = Pooling(data=conv1_hf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool1_hf'.format(self.scoped_name))
            pool1_lf = Pooling(data=conv1_lf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool1_lf'.format(self.scoped_name))

            # conv2
            conv2_hf, conv2_lf = octConv_BN_AC(hf_data=pool1_hf, lf_data=pool1_lf, alpha=alpha, num_filter_in=32,
                                               num_filter_out=64,
                                               kernel=(3, 3),
                                               name='{}/conv2_1'.format(self.scoped_name), pad='same')
            conv2_hf, conv2_lf = octConv_BN_AC(hf_data=conv2_hf, lf_data=conv2_lf, alpha=alpha, num_filter_in=64,
                                               num_filter_out=64,
                                               kernel=(3, 3), name='{}/conv2_2'.format(self.scoped_name), pad='same')
            pool2_hf = Pooling(data=conv2_hf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool2_hf'.format(self.scoped_name))
            pool2_lf = Pooling(data=conv2_lf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool2_lf'.format(self.scoped_name))

            # conv3
            conv3_hf, conv3_lf = octConv_BN_AC(hf_data=pool2_hf, lf_data=pool2_lf, alpha=alpha, num_filter_in=64,
                                               num_filter_out=128,
                                               kernel=(3, 3),
                                               name='{}/conv3_1'.format(self.scoped_name), pad='same')
            conv3_hf, conv3_lf = octConv_BN_AC(hf_data=conv3_hf, lf_data=conv3_lf, alpha=alpha, num_filter_in=128,
                                               num_filter_out=128,
                                               kernel=(3, 3), name='{}/conv3_2'.format(self.scoped_name), pad='same')
            pool3_hf = Pooling(data=conv3_hf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool3_hf'.format(self.scoped_name))
            pool3_lf = Pooling(data=conv3_lf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool3_lf'.format(self.scoped_name))

            # conv4
            conv4_hf, conv4_lf = octConv_BN_AC(hf_data=pool3_hf, lf_data=pool3_lf, alpha=alpha, num_filter_in=128,
                                               num_filter_out=256,
                                               kernel=(3, 3),
                                               name='{}/conv4_1'.format(self.scoped_name), pad='same')
            conv4_hf, conv4_lf = octConv_BN_AC(hf_data=conv4_hf, lf_data=conv4_lf, alpha=alpha, num_filter_in=256,
                                               num_filter_out=256,
                                               kernel=(3, 3), name='{}/conv4_2'.format(self.scoped_name), pad='same')
            pool4_hf = Pooling(data=conv4_hf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool4_hf'.format(self.scoped_name))
            pool4_lf = Pooling(data=conv4_lf, pool_type="max", kernel=(2, 2), pad="same",
                               name='{}/pool4_lf'.format(self.scoped_name))

            # conv5
            conv5_hf, conv5_lf = octConv_BN_AC(hf_data=pool4_hf, lf_data=pool4_lf, alpha=alpha, num_filter_in=256,
                                               num_filter_out=512,
                                               kernel=(3, 3),
                                               name='{}/conv5_1'.format(self.scoped_name), pad='same')
            conv5_hf, conv5_lf = octConv_BN_AC(hf_data=conv5_hf, lf_data=conv5_lf, alpha=alpha, num_filter_in=512,
                                               num_filter_out=512,
                                               kernel=(3, 3), name='{}/conv5_2'.format(self.scoped_name), pad='same')

            up6_hf = upsample_and_concat(conv5_hf, conv4_hf, 192, 384, name='weights',
                                         scope='{}/upsample_1_hf'.format(self.scoped_name))
            up6_lf = upsample_and_concat(conv5_lf, conv4_lf, 64, 128, name='weights',
                                         scope='{}/upsample_1_lf'.format(self.scoped_name))
            # conv6
            conv6_hf, conv6_lf = octConv_BN_AC(hf_data=up6_hf, lf_data=up6_lf, alpha=alpha, num_filter_in=512,
                                               num_filter_out=256,
                                               kernel=(3, 3),
                                               name='{}/conv6_1'.format(self.scoped_name), pad='same')
            conv6_hf, conv6_lf = octConv_BN_AC(hf_data=conv6_hf, lf_data=conv6_lf, alpha=alpha, num_filter_in=256,
                                               num_filter_out=256,
                                               kernel=(3, 3), name='{}/conv6_2'.format(self.scoped_name), pad='same')

            up7_hf = upsample_and_concat(conv6_hf, conv3_hf, 96, 192, name='weights',
                                         scope='{}/upsample_2_hf'.format(self.scoped_name))
            up7_lf = upsample_and_concat(conv6_lf, conv3_lf, 32, 64, name='weights',
                                         scope='{}/upsample_2_lf'.format(self.scoped_name))
            # conv7
            conv7_hf, conv7_lf = octConv_BN_AC(hf_data=up7_hf, lf_data=up7_lf, alpha=alpha, num_filter_in=256,
                                               num_filter_out=128,
                                               kernel=(3, 3),
                                               name='{}/conv7_1'.format(self.scoped_name), pad='same')
            conv7_hf, conv7_lf = octConv_BN_AC(hf_data=conv7_hf, lf_data=conv7_lf, alpha=alpha, num_filter_in=128,
                                               num_filter_out=128,
                                               kernel=(3, 3), name='{}/conv7_2'.format(self.scoped_name), pad='same')

            up8_hf = upsample_and_concat(conv7_hf, conv2_hf, 48, 96, name='weights',
                                         scope='{}/upsample_3_hf'.format(self.scoped_name))
            up8_lf = upsample_and_concat(conv7_lf, conv2_lf, 16, 32, name='weights',
                                         scope='{}/upsample_3_lf'.format(self.scoped_name))
            # conv8
            conv8_hf, conv8_lf = octConv_BN_AC(hf_data=up8_hf, lf_data=up8_lf, alpha=alpha, num_filter_in=128,
                                               num_filter_out=64,
                                               kernel=(3, 3),
                                               name='{}/conv8_1'.format(self.scoped_name), pad='same')
            conv8_hf, conv8_lf = octConv_BN_AC(hf_data=conv8_hf, lf_data=conv8_lf, alpha=alpha, num_filter_in=64,
                                               num_filter_out=64,
                                               kernel=(3, 3), name='{}/conv8_2'.format(self.scoped_name), pad='same')

            up9_hf = upsample_and_concat(conv8_hf, conv1_hf, 24, 48, name='weights',
                                         scope='{}/upsample_4_hf'.format(self.scoped_name))
            up9_lf = upsample_and_concat(conv8_lf, conv1_lf, 8, 16, name='weights',
                                         scope='{}/upsample_4_lf'.format(self.scoped_name))
            # conv9
            conv9_hf, conv9_lf = octConv_BN_AC(hf_data=up9_hf, lf_data=up9_lf, alpha=alpha, num_filter_in=64,
                                               num_filter_out=32,
                                               kernel=(3, 3),
                                               name='{}/conv9_1'.format(self.scoped_name), pad='same')
            conv9_hf, conv9_lf = octConv_BN_AC(hf_data=conv9_hf, lf_data=conv9_lf, alpha=alpha, num_filter_in=32,
                                               num_filter_out=32,
                                               kernel=(3, 3), name='{}/conv9_2'.format(self.scoped_name), pad='same')

            # conv10
            conv10 = lastOctConv_BN(hf_data=conv9_hf, lf_data=conv9_lf, alpha=alpha, num_filter_in=32,
                                    num_filter_out=12,
                                    kernel=(1, 1),
                                    name='{}/conv10'.format(self.scoped_name), pad='same')
            with tf.name_scope('{}'.format(self.scoped_name)):
                self.yy = tf.depth_to_space(conv10, 2)
            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))


class UNet3D(NIPModel):
    """
    The 3d UNet model, adapted from PSMNet
    """

    def construct_model(self):
        with self.graph.as_default():
            x = tf.split(self.x, 4, axis=3)
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]

            conv4_1 = self.CNN(x1)
            conv4_2 = self.CNN(x2, True)
            conv4_3 = self.CNN(x3, True)
            conv4_4 = self.CNN(x4, True)
            f1 = self.SPP(conv4_1)
            f2 = self.SPP(conv4_2, True)
            f3 = self.SPP(conv4_3, True)
            f4 = self.SPP(conv4_4, True)

            cost_vol = self.cost_vol(f1, f2, f3, f4)
            output = self.CNN3D(cost_vol, type="hourglass")

            with tf.name_scope('{}'.format(self.scoped_name)):
                self.yy = tf.depth_to_space(output, 2)
            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))

    def CNN(self, bottom, reuse=False):
        with tf.variable_scope('{}/CNN'.format(self.scoped_name)):
            with tf.variable_scope('conv0'):
                bottom = conv_block(tf.layers.conv2d, bottom, 32, 3, strides=1,
                                    name='conv0_1', reuse=reuse,
                                    reg=self.reg)
                for i in range(1, 3):
                    bottom = conv_block(tf.layers.conv2d, bottom, 32, 3,
                                        name='conv0_{}'.format((i + 1)), reuse=reuse,
                                        reg=self.reg)
            with tf.variable_scope('conv1'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 32, 3,
                                       name='conv1_{}'.format((i + 1)), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv2'):
                bottom = res_block(tf.layers.conv2d, bottom, 64, 3, strides=1,
                                   name='conv2_1', reuse=reuse,
                                   reg=self.reg,
                                   projection=True)
                for i in range(1, 4):
                    bottom = res_block(tf.layers.conv2d, bottom, 64, 3,
                                       name='conv2_{}'.format((i + 1)), reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv3'):
                bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2,
                                   name='conv3_1', reuse=reuse,
                                   reg=self.reg, projection=True)
                for i in range(1, 3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=2,
                                       name='conv3_{}'.format((i + 1)),
                                       reuse=reuse,
                                       reg=self.reg)
            with tf.variable_scope('conv4'):
                for i in range(3):
                    bottom = res_block(tf.layers.conv2d, bottom, 128, 3, dilation_rate=4,
                                       name='conv4_{}'.format((i + 1)),
                                       reuse=reuse,
                                       reg=self.reg)
        return bottom

    def SPP(self, bottom, reuse=False):
        with tf.variable_scope('{}/SPP'.format(self.scoped_name)):
            branches = []
            for i, p in enumerate([64, 32, 16, 8]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 32, 3,
                                           name='branch_{}'.format((i + 1)), reuse=reuse,
                                           reg=self.reg))
            # if not reuse:
            conv2_4 = tf.get_default_graph().get_tensor_by_name('{}/CNN/conv2/conv2_4/add:0'.format(self.scoped_name))
            conv4_3 = tf.get_default_graph().get_tensor_by_name('{}/CNN/conv4/conv4_3/add:0'.format(self.scoped_name))
            # else:
            #    conv2_16 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv2/conv2_16/add:0')
            #    conv4_3 = tf.get_default_graph().get_tensor_by_name('CNN_1/conv4/conv4_3/add:0')
            concat = tf.concat([conv2_4, conv4_3] + branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 128, 3, name='conv1', reuse=reuse, reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1, name='conv2', reuse=reuse, reg=self.reg)
        return fusion

    def cost_vol(self, f1, f2, f3, f4):
        with tf.variable_scope('{}/cost_vol'.format(self.scoped_name)):
            disparity_costs = []
            # shape = tf.shape(right) #(N,H,W,F)
            # cost = tf.concat([f1, f2, f3, f4], axis=3)
            for i in range(3):
                disparity_costs.append(f1)
                disparity_costs.append(f2)
                disparity_costs.append(f3)
                disparity_costs.append(f4)

            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol

    def CNN3D(self, bottom, type="basic"):
        with tf.variable_scope('{}/CNN3D'.format(self.scoped_name)):
            if type == "basic":
                bottom = conv_block(tf.layers.conv3d, bottom, 64, 3, name='3Dconv0_1', reg=self.reg)
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_2', reg=self.reg)

                _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)
                # _3Dconv1 = _3Dconv1 + bottom

                _3Dconv2 = res_block(tf.layers.conv3d, _3Dconv1, 32, 3, name='3Dconv2', reg=self.reg)
                # _3Dconv2 = _3Dconv2 + _3Dconv1

                _3Dconv3 = res_block(tf.layers.conv3d, _3Dconv2, 32, 3, name='3Dconv3', reg=self.reg)
                # _3Dconv3 = _3Dconv3 + _3Dconv2

                _3Dconv4 = res_block(tf.layers.conv3d, _3Dconv3, 32, 3, name='3Dconv4', reg=self.reg)
                # _3Dconv4 = _3Dconv4 + _3Dconv3

                output_1 = conv_block(tf.layers.conv3d, _3Dconv4, 32, 3, name='output_1_1', reg=self.reg)
                output_1 = conv_block(tf.layers.conv3d, output_1, 1, 3, name='output_1', reg=self.reg, apply_bn=False,
                                      apply_relu=False, use_bias=False)
                output_1 = tf.squeeze(output_1, axis=4)
                output_1 = tf.transpose(output_1, [0, 3, 2, 1])
                output = tf.depth_to_space(output_1, 2)
            elif type == "hourglass":
                bottom = conv_block(tf.layers.conv3d, bottom, 64, 3, name='3Dconv0_1', reg=self.reg)
                bottom = conv_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv0_2', reg=self.reg)

                _3Dconv1 = res_block(tf.layers.conv3d, bottom, 32, 3, name='3Dconv1', reg=self.reg)

                _3Dstack = [hourglass('3d', _3Dconv1, [64, 64, 64, 32], [3, 3, 3, 3], [None, None, -2, _3Dconv1],
                                      name='3Dstack1', reg=self.reg)]
                for i in range(1, 3):
                    _3Dstack.append(hourglass('3d', _3Dstack[-1][-1], [64, 64, 64, 32], [3, 3, 3, 3],
                                              [_3Dstack[-1][-2], None, _3Dstack[0][0], _3Dconv1],
                                              name='3Dstack%d' % (i + 1),
                                              reg=self.reg))
                output_1 = conv_block(tf.layers.conv3d, _3Dstack[0][3], 32, 3, name='output_1_1', reg=self.reg)
                output_1 = conv_block(tf.layers.conv3d, output_1, 1, 3, name='output_1', reg=self.reg, apply_bn=False,
                                      apply_relu=False, use_bias=False)
                output_1 = tf.squeeze(output_1, axis=4)
                output_1 = tf.transpose(output_1, [0, 3, 2, 1])
                output = tf.depth_to_space(output_1, 2)
        return output


class INet(NIPModel):
    """
    A neural pipeline which replicates the steps of a standard imaging pipeline.
    """

    def construct_model(self, random_init=False, kernel=5, trainable_upsampling=False, cfa_pattern='gbrg'):
        self.trainable_upsampling = trainable_upsampling
        self.cfa_pattern = cfa_pattern

        with self.graph.as_default():
            with tf.variable_scope('{}'.format(self.scoped_name)):

                # Initialize the upsampling kernel
                upk = upsampling_kernel(cfa_pattern)

                if random_init:
                    # upk = np.random.normal(0, 0.1, (4, 12))
                    dmf = np.random.normal(0, 0.1, (kernel, kernel, 3, 3))
                    gamma_d1k = np.random.normal(0, 0.1, (3, 12))
                    gamma_d1b = np.zeros((12,))
                    gamma_d2k = np.random.normal(0, 0.1, (12, 3))
                    gamma_d2b = np.zeros((3,))
                    srgbk = np.eye(3)
                else:
                    # Prepare demosaicing kernels (bilinear)
                    dmf = bilin_kernel(kernel)

                    # Prepare gamma correction kernels (obtained from a pre-trained toy model)
                    gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b = gamma_kernels()

                    # Example sRGB conversion table
                    srgbk = np.array([[1.82691061, -0.65497452, -0.17193617],
                                      [-0.00683982, 1.33216381, -0.32532394],
                                      [0.06269717, -0.40055895, 1.33786178]]).transpose()

                # Up-sample the input back the full resolution
                with tf.variable_scope('upsampling'):
                    h12 = tf.layers.conv2d(self.x, 12, 1, kernel_initializer=tf.constant_initializer(upk),
                                           use_bias=False, activation=None, name='conv_h12',
                                           trainable=trainable_upsampling)

                # Demosaicing
                with tf.variable_scope('demosaicing'):
                    pad = (kernel - 1) // 2
                    bayer = tf.depth_to_space(h12, 2)
                    bayer = tf.pad(bayer, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
                    rgb = tf.layers.conv2d(bayer, 3, kernel, kernel_initializer=tf.constant_initializer(dmf),
                                           use_bias=False, activation=None, name='conv_demo', padding='VALID')

                # Color space conversion
                with tf.variable_scope('rgb2sRGB'):
                    srgb = tf.layers.conv2d(rgb, 3, 1, kernel_initializer=tf.constant_initializer(srgbk),
                                            use_bias=False, activation=None, name='conv_sRGB')

                # Gamma correction
                with tf.variable_scope('gamma'):
                    rgb_g0 = tf.layers.conv2d(srgb, 12, 1, kernel_initializer=tf.constant_initializer(gamma_d1k),
                                              bias_initializer=tf.constant_initializer(gamma_d1b), use_bias=True,
                                              activation=tf.nn.tanh, name='conv_encode')
                    self.yy = tf.layers.conv2d(rgb_g0, 3, 1, kernel_initializer=tf.constant_initializer(gamma_d2k),
                                               bias_initializer=tf.constant_initializer(gamma_d2b), use_bias=True,
                                               activation=None, name='conv_decode')

            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))


class DNet(NIPModel):
    """
    Neural imaging pipeline adapted from a joint demosaicing-&-denoising model:
    Gharbi, Michaël, et al. "Deep joint demosaicking and denoising." ACM Transactions on Graphics (TOG) 35.6 (2016): 191.
    """

    def construct_model(self, n_layers=15, kernel=3, n_features=64):
        with self.graph.as_default():
            with tf.name_scope('{}'.format(self.scoped_name)):
                k_initializer = tf.variance_scaling_initializer

                # Initialize the upsampling kernel
                upk = upsampling_kernel()

                # Padding size
                pad = (kernel - 1) // 2

                # Convolutions on the sub-sampled input tensor
                deep_x = self.x
                for r in range(n_layers):
                    deep_y = tf.layers.conv2d(deep_x, 12 if r == n_layers - 1 else n_features, kernel,
                                              activation=tf.nn.relu, name='{}/conv{}'.format(self.scoped_name, r),
                                              padding='VALID', kernel_initializer=k_initializer)  #
                    print('CNN layer out: {}'.format(deep_y.shape))
                    deep_x = tf.pad(deep_y, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

                # Up-sample the input
                h12 = tf.layers.conv2d(self.x, 12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False,
                                       activation=None, name='{}/conv_h12'.format(self.scoped_name), trainable=False)
                bayer = tf.depth_to_space(h12, 2, name="{}/upscaled_bayer".format(self.scoped_name))

                # Upscale the conv. features and concatenate with the input RGB channels
                features = tf.depth_to_space(deep_x, 2, name='{}/upscaled_features'.format(self.scoped_name))
                bayer_features = tf.concat((features, bayer), axis=3)

                print('Final deep X: {}'.format(deep_x.shape))
                print('Bayer shape: {}'.format(bayer.shape))
                print('Features shape: {}'.format(features.shape))
                print('Concat shape: {}'.format(bayer_features.shape))

                # Project the concatenated 6-D features (R G B bayer from input + 3 channels from convolutions)
                pu = tf.layers.conv2d(bayer_features, n_features, kernel, kernel_initializer=k_initializer,
                                      use_bias=True, activation=tf.nn.relu,
                                      name='{}/conv_postupscale'.format(self.scoped_name), padding='VALID',
                                      bias_initializer=tf.zeros_initializer)

                print('Post upscale: {}'.format(pu.shape))

                # Final 1x1 conv to project each 64-D feature vector into the RGB colorspace
                pu = tf.pad(pu, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
                rgb = tf.layers.conv2d(pu, 3, 1, kernel_initializer=tf.ones_initializer, use_bias=False,
                                       activation=None, name='{}/conv_final'.format(self.scoped_name), padding='VALID')

                print('RGB affine: {}'.format(rgb.shape))

                self.yy = rgb
                print('Y: {}'.format(self.yy.shape))

            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))
