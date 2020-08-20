import numpy as np

from keras import backend as K
from keras import activations, initializers
from keras.initializers import Constant, Initializer
from keras.layers import Layer

from scipy import signal
from scipy import linalg as la
import math
import tensorflow as tf


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    if measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B

forward_aliases   = ['euler', 'forward_euler', 'forward', 'forward_diff']
backward_aliases  = ['backward', 'backward_diff', 'backward_euler']
bilinear_aliases = ['bilinear', 'tustin', 'trapezoidal', 'trapezoid']
zoh_aliases       = ['zoh']

class HippoTCell(Layer):

    def __init__(self,
                 units,
                 memory_order,
                 theta,  # relative to dt=1
                 measure='legt',
                 method='zoh',
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer='lecun_uniform',
                 hidden_encoders_initializer='lecun_uniform',
                 memory_encoders_initializer=Constant(0),  # 'lecun_uniform',
                 input_kernel_initializer='glorot_normal',
                 hidden_kernel_initializer='glorot_normal',
                 memory_kernel_initializer='glorot_normal',
                 hidden_activation='tanh',
                 **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.memory_order = memory_order
        self.theta = theta
        self.method = method
        self.trainable_input_encoders = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel = trainable_input_kernel
        self.trainable_hidden_kernel = trainable_hidden_kernel
        self.trainable_memory_kernel = trainable_memory_kernel
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        self.input_encoders_initializer = initializers.get(
            input_encoders_initializer)
        self.hidden_encoders_initializer = initializers.get(
            hidden_encoders_initializer)
        self.memory_encoders_initializer = initializers.get(
            memory_encoders_initializer)
        self.input_kernel_initializer = initializers.get(
            input_kernel_initializer)
        self.hidden_kernel_initializer = initializers.get(
            hidden_kernel_initializer)
        self.memory_kernel_initializer = initializers.get(
            memory_kernel_initializer)

        self.hidden_activation = activations.get(hidden_activation)

        A, B = transition(measure, memory_order)
        # Construct A and B matrices
        C = np.ones((1, memory_order))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=1./theta, method=method)

        self._A = dA - np.eye(memory_order)  # puts into form: x += Ax
        self._B = dB

        self.state_size = (self.units, self.memory_order)
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.input_encoders = self.add_weight(
            name='input_encoders',
            shape=(input_dim, 1),
            initializer=self.input_encoders_initializer,
            trainable=self.trainable_input_encoders)

        self.hidden_encoders = self.add_weight(
            name='hidden_encoders',
            shape=(self.units, 1),
            initializer=self.hidden_encoders_initializer,
            trainable=self.trainable_hidden_encoders)

        self.memory_encoders = self.add_weight(
            name='memory_encoders',
            shape=(self.memory_order, 1),
            initializer=self.memory_encoders_initializer,
            trainable=self.trainable_memory_encoders)

        self.input_kernel = self.add_weight(
            name='input_kernel',
            shape=(input_dim, self.units),
            initializer=self.input_kernel_initializer,
            trainable=self.trainable_input_kernel)

        self.hidden_kernel = self.add_weight(
            name='hidden_kernel',
            shape=(self.units, self.units),
            initializer=self.hidden_kernel_initializer,
            trainable=self.trainable_hidden_kernel)

        self.memory_kernel = self.add_weight(
            name='memory_kernel',
            shape=(self.memory_order, self.units),
            initializer=self.memory_kernel_initializer,
            trainable=self.trainable_memory_kernel)

        self.AT = self.add_weight(
            name='AT',
            shape=(self.memory_order, self.memory_order),
            initializer=Constant(self._A.T),  # note: transposed
            trainable=self.trainable_A)

        self.BT = self.add_weight(
            name='BT',
            shape=(1, self.memory_order),  # system is SISO
            initializer=Constant(self._B.T),  # note: transposed
            trainable=self.trainable_B)

        self.built = True

    def call(self, inputs, states):
        h, m = states

        u = (K.dot(inputs, self.input_encoders) +
             K.dot(h, self.hidden_encoders) +
             K.dot(m, self.memory_encoders))

        m = m + K.dot(m, self.AT) + K.dot(u, self.BT)

        h = self.hidden_activation(
             K.dot(inputs, self.input_kernel) +
             K.dot(h, self.hidden_kernel) +
             K.dot(m, self.memory_kernel))

        return h, [h, m]

class HippoSCell(Layer):

    def __init__(self,
                 units,
                 memory_order,
                 measure='legt',
                 method='zoh',
                 max_length=256,
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer='lecun_uniform',
                 hidden_encoders_initializer='lecun_uniform',
                 memory_encoders_initializer=Constant(0),  # 'lecun_uniform',
                 input_kernel_initializer='glorot_normal',
                 hidden_kernel_initializer='glorot_normal',
                 memory_kernel_initializer='glorot_normal',
                 hidden_activation='tanh',
                 gate=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.units                     = units
        self.memory_order              = memory_order
        self.method                    = method
        self.max_length                = max_length
        self.trainable_input_encoders  = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel    = trainable_input_kernel
        self.trainable_hidden_kernel   = trainable_hidden_kernel
        self.trainable_memory_kernel   = trainable_memory_kernel
        self.trainable_A               = trainable_A
        self.trainable_B               = trainable_B
        self.gate                      = gate

        self.input_encoders_initializer = initializers.get(
            input_encoders_initializer)
        self.hidden_encoders_initializer = initializers.get(
            hidden_encoders_initializer)
        self.memory_encoders_initializer = initializers.get(
            memory_encoders_initializer)
        self.input_kernel_initializer = initializers.get(
            input_kernel_initializer)
        self.hidden_kernel_initializer = initializers.get(
            hidden_kernel_initializer)
        self.memory_kernel_initializer = initializers.get(
            memory_kernel_initializer)

        self.hidden_activation = activations.get(hidden_activation)

        A, B = transition(measure, memory_order)
        # Construct A and B matrices

        A_stacked = np.empty((max_length, memory_order, memory_order), dtype=A.dtype)
        B_stacked = np.empty((max_length, memory_order), dtype=B.dtype)
        B = B[:,0]
        N = memory_order
        for t in range(1, max_length + 1):
            At = A / t
            Bt = B / t
            # if discretization in forward_aliases:
            if method in forward_aliases:
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            # elif discretization in backward_aliases:
            elif method in backward_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif method in bilinear_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            elif method in zoh_aliases:
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(memory_order)  # puts into form: x += Ax
        self._A = A_stacked - np.eye(memory_order)  # puts into form: x += Ax
        self._B = B_stacked

        self.state_size = (self.units, self.memory_order, 1)
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.input_encoders = self.add_weight(
            name='input_encoders',
            shape=(input_dim, 1),
            initializer=self.input_encoders_initializer,
            trainable=self.trainable_input_encoders)

        self.hidden_encoders = self.add_weight(
            name='hidden_encoders',
            shape=(self.units, 1),
            initializer=self.hidden_encoders_initializer,
            trainable=self.trainable_hidden_encoders)

        self.memory_encoders = self.add_weight(
            name='memory_encoders',
            shape=(self.memory_order, 1),
            initializer=self.memory_encoders_initializer,
            trainable=self.trainable_memory_encoders)

        self.input_kernel = self.add_weight(
            name='input_kernel',
            shape=(input_dim, self.units),
            initializer=self.input_kernel_initializer,
            trainable=self.trainable_input_kernel)

        if self.trainable_hidden_kernel:
            self.hidden_kernel = self.add_weight(
                name='hidden_kernel',
                shape=(self.units, self.units),
                initializer=self.hidden_kernel_initializer,
                trainable=self.trainable_hidden_kernel)
        else:
            self.hidden_kernel = self.add_weight(
                name='hidden_kernel',
                shape=(self.units, self.units),
                initializer=Constant(0.),
                trainable=False)

        self.memory_kernel = self.add_weight(
            name='memory_kernel',
            shape=(self.memory_order, self.units),
            initializer=self.memory_kernel_initializer,
            trainable=self.trainable_memory_kernel)

        self.A = self.add_weight(
            name='A',
            shape=(self.max_length, self.memory_order, self.memory_order),
            initializer=Constant(self._A),  # note: transposed
            trainable=self.trainable_A)

        self.B = self.add_weight(
            name='B',
            shape=(self.max_length, self.memory_order, 1),  # system is SISO
            initializer=Constant(self._B),  # note: transposed
            trainable=self.trainable_B)
 
        if self.gate:
            self.W_gate = self.add_weight(
                name='gate',
                shape=(self.units+self.memory_order, self.units),  # system is SISO
                initializer=initializers.get('glorot_normal'),  # note: transposed
                trainable=True)

        self.built = True

    def call(self, inputs, states):
        h, m, t = states
        tt = tf.cast(t, tf.int32)
        tt = tt[0,0]

        tt = tf.math.minimum(tt, self.max_length-1)
        u = (K.dot(inputs, self.input_encoders) +
             K.dot(h, self.hidden_encoders) +
             K.dot(m, self.memory_encoders))

        m = m + K.dot(m, tf.transpose(self.A[tt])) + K.dot(u, tf.transpose(self.B[tt]))

        new_h = self.hidden_activation(
             K.dot(inputs, self.input_kernel) +
             K.dot(h, self.hidden_kernel) +
             K.dot(m, self.memory_kernel))
        if self.gate:
            g = tf.sigmoid(K.dot(tf.concat([h, m], axis=-1), self.W_gate))
            h = (1.-g)*h + g*new_h
        else:
            h = new_h

        return h, [h, m, t+1]
