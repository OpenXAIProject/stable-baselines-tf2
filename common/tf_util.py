import os
import collections
import functools
import multiprocessing

import numpy as np
import tensorflow as tf
from collections import deque


def is_image(tensor):
    """
    Check if a tensor has the shape of
    a valid image for tensorboard logging.
    Valid image: RGB, RGBD, GrayScale

    :param tensor: (np.ndarray or tf.placeholder)
    :return: (bool)
    """

    return len(tensor.shape) == 3 and tensor.shape[-1] in [1, 3, 4]


# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(tensor, delta=1.0):
    """
    Reference: https://en.wikipedia.org/wiki/Huber_loss

    :param tensor: (TensorFlow Tensor) the input value
    :param delta: (float) huber loss delta value
    :return: (TensorFlow Tensor) huber loss output
    """
    return tf.compat.v1.where(
        tf.abs(tensor) < delta,
        tf.square(tensor) * 0.5,
        delta * (tf.abs(tensor) - 0.5 * delta)
    )


# ================================================================
# Global session
# ================================================================

def make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.compat.v1.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.compat.v1.Session(config=tf_config, graph=graph)


def single_threaded_session(make_default=False, graph=None):
    """
    Returns a session which will only use a single CPU

    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    return make_session(num_cpu=1, make_default=make_default, graph=graph)


def in_session(func):
    """
    wrappes a function so that it is in a TensorFlow Session

    :param func: (function) the function to wrap
    :return: (function)
    """

    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        with tf.compat.v1.Session():
            func(*args, **kwargs)

    return newfunc


ALREADY_INITIALIZED = set()


def initialize(sess=None):
    """
    Initialize all the uninitialized variables in the global scope.

    :param sess: (TensorFlow Session)
    """
    if sess is None:
        sess = tf.compat.v1.get_default_session()
    new_variables = set(tf.compat.v1.global_variables()) - ALREADY_INITIALIZED
    sess.run(tf.compat.v1.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Theano-like Function
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """
    Take a bunch of tensorflow placeholders and expressions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be fed to the input's placeholders and produces the values of the expressions
    in outputs. Just like a Theano function.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).

    Example:
       >>> x = tf.placeholder(tf.int32, (), name="x")
       >>> y = tf.placeholder(tf.int32, (), name="y")
       >>> z = 3 * x + 2 * y
       >>> lin = function([x, y], z, givens={y: 0})
       >>> with single_threaded_session():
       >>>     initialize()
       >>>     assert lin(2) == 6
       >>>     assert lin(x=3) == 9
       >>>     assert lin(2, 2) == 10

    :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
    :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
    :param givens: (dict) the values known for the output
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        func = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), func(*args, **kwargs)))
    else:
        func = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        """
        Theano like function

        :param inputs: (TensorFlow Tensor or Object with make_feed_dict) list of input arguments
        :param outputs: (TensorFlow Tensor) list of outputs or a single output to be returned from function. Returned
            value will also have the same shape.
        :param updates: ([tf.Operation] or tf.Operation)
        list of update functions or single update function that will be run whenever
        the function is called. The return is ignored.
        :param givens: (dict) the values known for the output
        """
        for inpt in inputs:
            if not hasattr(inpt, 'make_feed_dict') and not (isinstance(inpt, tf.Tensor)and len(inpt.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    @classmethod
    def _feed_input(cls, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = value

    def __call__(self, *args, sess=None, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = sess.run(self.outputs_update, feed_dict=feed_dict, **kwargs)[:-1]
        return results


# ================================================================
# Flat vectors
# ================================================================

def var_shape(tensor):
    """
    get TensorFlow Tensor shape

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: ([int]) the shape
    """
    out = tensor.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(tensor):
    """
    get TensorFlow Tensor's number of elements

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: (int) the number of elements
    """
    return intprod(var_shape(tensor))


def intprod(tensor):
    """
    calculates the product of all the elements in a list

    :param tensor: ([Number]) the list of elements
    :return: (int) the product truncated
    """
    return int(np.prod(tensor))


def flatgrad(loss, var_list, clip_norm=None):
    """
    calculates the gradient and flattens it

    :param loss: (float) the loss value
    :param var_list: ([TensorFlow Tensor]) the variables
    :param clip_norm: (float) clip the gradients (disabled if None)
    :return: ([TensorFlow Tensor]) flattend gradient
    """
    grads = tf.gradients(ys=loss, xs=var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32, sess=None):
        """
        Set the parameters from a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param dtype: (type) the type for the placeholder
        :param sess: (TensorFlow Session)
        """
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.compat.v1.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, _var) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.compat.v1.assign(_var, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.operation = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        if self.sess is None:
            return tf.compat.v1.get_default_session().run(self.operation, feed_dict={self.theta: theta})
        else:
            return self.sess.run(self.operation, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list, sess=None):
        """
        Get the parameters as a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param sess: (TensorFlow Session)
        """
        self.operation = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if self.sess is None:
            return tf.compat.v1.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)


# ================================================================
# retrieving variables
# ================================================================

def get_trainable_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name):
    """
    returns the trainable variables

    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)


def outer_scope_getter(scope, new_scope=""):
    """
    remove a scope layer for the getter

    :param scope: (str) the layer to remove
    :param new_scope: (str) optional replacement name
    :return: (function (function, str, ``*args``, ``**kwargs``): Tensorflow Tensor)
    """
    def _getter(getter, name, *args, **kwargs):
        name = name.replace(scope + "/", new_scope, 1)
        val = getter(name, *args, **kwargs)
        return val
    return _getter


def sample(logits):
    """
    Creates a sampling Tensor for non deterministic policies
    when using categorical distribution.
    It uses the Gumbel-max trick: http://amid.fish/humble-gumbel

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The sampled action
    """
    noise = tf.random.uniform(tf.shape(input=logits))
    return tf.argmax(input=logits - tf.math.log(-tf.math.log(noise)), axis=1)


def calc_entropy(logits):
    """
    Calculates the entropy of the output values of the network

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The Entropy of the output values of the network
    """
    # Compute softmax
    a_0 = logits - tf.reduce_max(input_tensor=logits, axis=1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    z_0 = tf.reduce_sum(input_tensor=exp_a_0, axis=1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(input_tensor=p_0 * (tf.math.log(z_0) - a_0), axis=1)


def calc_entropy_softmax(action_proba):
    """
    Calculates the softmax entropy of the output values of the network

    :param action_proba: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The softmax entropy of the output values of the network
    """
    return - tf.reduce_sum(input_tensor=action_proba * tf.math.log(action_proba + 1e-6), axis=1)


def mse(pred, target):
    """
    Returns the Mean squared error between prediction and target

    :param pred: (TensorFlow Tensor) The predicted value
    :param target: (TensorFlow Tensor) The target value
    :return: (TensorFlow Tensor) The Mean squared error between prediction and target
    """
    return tf.reduce_mean(input_tensor=tf.square(pred - target))


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def conv(input_tensor, scope, *, n_filters, filter_size, stride,
         pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, \
            "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.compat.v1.variable_scope(scope):
        weight = tf.compat.v1.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.compat.v1.get_variable("b", bias_var_shape, initializer=tf.compat.v1.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input=input_tensor, filters=weight, strides=strides, padding=pad, data_format=data_format)


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.compat.v1.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.compat.v1.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.compat.v1.get_variable("b", [n_hidden], initializer=tf.compat.v1.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies

    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies
    """
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])


def lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False):
    """
    Creates an Long Short Term Memory (LSTM) cell for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param layer_norm: (bool) Whether to apply Layer Normalization or not
    :return: (TensorFlow Tensor) LSTM cell
    """
    _, n_input = [v.value for v in input_tensor[0].get_shape()]
    with tf.compat.v1.variable_scope(scope):
        weight_x = tf.compat.v1.get_variable("wx", [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
        weight_h = tf.compat.v1.get_variable("wh", [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
        bias = tf.compat.v1.get_variable("b", [n_hidden * 4], initializer=tf.compat.v1.constant_initializer(0.0))

        if layer_norm:
            # Gain and bias of layer norm
            gain_x = tf.compat.v1.get_variable("gx", [n_hidden * 4], initializer=tf.compat.v1.constant_initializer(1.0))
            bias_x = tf.compat.v1.get_variable("bx", [n_hidden * 4], initializer=tf.compat.v1.constant_initializer(0.0))

            gain_h = tf.compat.v1.get_variable("gh", [n_hidden * 4], initializer=tf.compat.v1.constant_initializer(1.0))
            bias_h = tf.compat.v1.get_variable("bh", [n_hidden * 4], initializer=tf.compat.v1.constant_initializer(0.0))

            gain_c = tf.compat.v1.get_variable("gc", [n_hidden], initializer=tf.compat.v1.constant_initializer(1.0))
            bias_c = tf.compat.v1.get_variable("bc", [n_hidden], initializer=tf.compat.v1.constant_initializer(0.0))

    cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
    for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
                    + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
        in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        input_tensor[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
    return input_tensor, cell_state_hidden


def _ln(input_tensor, gain, bias, epsilon=1e-5, axes=None):
    """
    Apply layer normalisation.

    :param input_tensor: (TensorFlow Tensor) The input tensor for the Layer normalization
    :param gain: (TensorFlow Tensor) The scale tensor for the Layer normalization
    :param bias: (TensorFlow Tensor) The bias tensor for the Layer normalization
    :param epsilon: (float) The epsilon value for floating point calculations
    :param axes: (tuple, list or int) The axes to apply the mean and variance calculation
    :return: (TensorFlow Tensor) a normalizing layer
    """
    if axes is None:
        axes = [1]
    mean, variance = tf.nn.moments(x=input_tensor, axes=axes, keepdims=True)
    input_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)
    input_tensor = input_tensor * gain + bias
    return input_tensor


def lnlstm(input_tensor, mask_tensor, cell_state, scope, n_hidden, init_scale=1.0):
    """
    Creates a LSTM with Layer Normalization (lnlstm) cell for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
    :param cell_state: (TensorFlow Tensor) The state tensor for the LSTM cell
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :return: (TensorFlow Tensor) lnlstm cell
    """
    return lstm(input_tensor, mask_tensor, cell_state, scope, n_hidden, init_scale, layer_norm=True)


def conv_to_fc(input_tensor):
    """
    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network

    :param input_tensor: (TensorFlow Tensor) The convolutional input tensor
    :return: (TensorFlow Tensor) The fully connected output tensor
    """
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor


def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]

def make_path(path):
    """
    For a given path, create the folders if they do not exist

    :param path: (str) The path
    :return: (bool) Whether or not it finished correctly
    """
    return os.makedirs(path, exist_ok=True)


def constant(_):
    """
    Returns a constant value for the Scheduler

    :param _: ignored
    :return: (float) 1
    """
    return 1.


def linear_schedule(progress):
    """
    Returns a linear value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress
    """
    return 1 - progress


def middle_drop(progress):
    """
    Returns a linear value with a drop near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress if (1 - progress) >= 0.75 else 0.075
    """
    eps = 0.75
    if 1 - progress < eps:
        return eps * 0.1
    return 1 - progress


def double_linear_con(progress):
    """
    Returns a linear value (x2) with a flattened tail for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress*2 if (1 - progress*2) >= 0.125 else 0.125
    """
    progress *= 2
    eps = 0.125
    if 1 - progress < eps:
        return eps
    return 1 - progress


def double_middle_drop(progress):
    """
    Returns a linear value with two drops near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) if 0.75 <= 1 - p: 1 - p, if 0.25 <= 1 - p < 0.75: 0.75, if 1 - p < 0.25: 0.125
    """
    eps1 = 0.75
    eps2 = 0.25
    if 1 - progress < eps1:
        if 1 - progress < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - progress


SCHEDULES = {
    'linear': linear_schedule,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):
    def __init__(self, initial_value, n_values, schedule):
        """
        Update a value every iteration, with a specific curve

        :param initial_value: (float) initial value
        :param n_values: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.step = 0.
        self.initial_value = initial_value
        self.nvalues = n_values
        self.schedule = SCHEDULES[schedule]

    def value(self):
        """
        Update the Scheduler, and return the current value

        :return: (float) the current value
        """
        current_value = self.initial_value * self.schedule(self.step / self.nvalues)
        self.step += 1.
        return current_value

    def value_steps(self, steps):
        """
        Get a value for a given step

        :param steps: (int) The current number of iterations
        :return: (float) the value for the current number of iterations
        """
        return self.initial_value * self.schedule(steps / self.nvalues)


class EpisodeStats:
    def __init__(self, n_steps, n_envs):
        """
        Calculates the episode statistics

        :param n_steps: (int) The number of steps to run for each environment
        :param n_envs: (int) The number of environments
        """
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
        self.len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.n_steps = n_steps
        self.n_envs = n_envs

    def feed(self, rewards, masks):
        """
        Update the latest reward and mask

        :param rewards: ([float]) The new rewards for the new step
        :param masks: ([float]) The new masks for the new step
        """
        rewards = np.reshape(rewards, [self.n_envs, self.n_steps])
        masks = np.reshape(masks, [self.n_envs, self.n_steps])
        for i in range(0, self.n_envs):
            for j in range(0, self.n_steps):
                self.episode_rewards[i].append(rewards[i][j])
                if masks[i][j]:
                    reward_length = len(self.episode_rewards[i])
                    reward_sum = sum(self.episode_rewards[i])
                    self.len_buffer.append(reward_length)
                    self.rewbuffer.append(reward_sum)
                    self.episode_rewards[i] = []

    def mean_length(self):
        """
        Returns the average length of each episode

        :return: (float)
        """
        if self.len_buffer:
            return np.mean(self.len_buffer)
        else:
            return 0  # on the first params dump, no episodes are finished

    def mean_reward(self):
        """
        Returns the average reward of each episode

        :return: (float)
        """
        if self.rewbuffer:
            return np.mean(self.rewbuffer)
        else:
            return 0


# For ACER
def get_by_index(input_tensor, idx):
    """
    Return the input tensor, offset by a certain value

    :param input_tensor: (TensorFlow Tensor) The input tensor
    :param idx: (int) The index offset
    :return: (TensorFlow Tensor) the offset tensor
    """
    assert len(input_tensor.get_shape()) == 2
    assert len(idx.get_shape()) == 1
    idx_flattened = tf.range(0, input_tensor.shape[0]) * input_tensor.shape[1] + idx
    offset_tensor = tf.gather(tf.reshape(input_tensor, [-1]),  # flatten input
                              idx_flattened)  # use flattened indices
    return offset_tensor


def check_shape(tensors, shapes):
    """
    Verifies the tensors match the given shape, will raise an error if the shapes do not match

    :param tensors: ([TensorFlow Tensor]) The tensors that should be checked
    :param shapes: ([list]) The list of shapes for each tensor
    """
    i = 0
    for (tensor, shape) in zip(tensors, shapes):
        assert tensor.get_shape().as_list() == shape, "id " + str(i) + " shape " + str(tensor.get_shape()) + str(shape)
        i += 1


def avg_norm(tensor):
    """
    Return an average of the L2 normalization of the batch

    :param tensor: (TensorFlow Tensor) The input tensor
    :return: (TensorFlow Tensor) Average L2 normalization of the batch
    """
    return tf.reduce_mean(input_tensor=tf.sqrt(tf.reduce_sum(input_tensor=tf.square(tensor), axis=-1)))


def gradient_add(grad_1, grad_2, param, verbose=0):
    """
    Sum two gradients

    :param grad_1: (TensorFlow Tensor) The first gradient
    :param grad_2: (TensorFlow Tensor) The second gradient
    :param param: (TensorFlow parameters) The trainable parameters
    :param verbose: (int) verbosity level
    :return: (TensorFlow Tensor) the sum of the gradients
    """
    if verbose > 1:
        print([grad_1, grad_2, param.name])
    if grad_1 is None and grad_2 is None:
        return None
    elif grad_1 is None:
        return grad_2
    elif grad_2 is None:
        return grad_1
    else:
        return grad_1 + grad_2


def q_explained_variance(q_pred, q_true):
    """
    Calculates the explained variance of the Q value

    :param q_pred: (TensorFlow Tensor) The predicted Q value
    :param q_true: (TensorFlow Tensor) The expected Q value
    :return: (TensorFlow Tensor) the explained variance of the Q value
    """
    _, var_y = tf.nn.moments(x=q_true, axes=[0, 1])
    _, var_pred = tf.nn.moments(x=q_true - q_pred, axes=[0, 1])
    check_shape([var_y, var_pred], [[]] * 2)
    return 1.0 - (var_pred / var_y)


def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.compat.v1.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                    summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc
