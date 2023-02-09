import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
from tensorflow.python.layers.convolutional import conv1d as Conv1D
from tensorflow.python.layers.core import dense as Dense
from tensorflow.python.layers.core import flatten as Flatten

import matplotlib.pyplot as plt
import datetime
import csv

NX = 8   #number of sites 64
NP = 8   #number of particle 64
NSAMPLE = 1024  #number of sample
STEP = 1000

J = -1.0  # hopping strength
U = 6.0   # on-site interaction energy
V = 1.0  # nearest_neighbor interaction


# tensorflow model
class Network:
    hidden_layer = 3
    channel = 3
    kernel_size = [5, 3, 3]

    def __init__(self):
        self.prepare_model()
        self.prepare_session()

    def prepare_model(self):

        global x_conv_0, x_conv_1
        hidden_layer = Network.hidden_layer
        channel = Network.channel
        kernel_size = Network.kernel_size

        # input unit
        x = tf.placeholder(tf.float32, [None, NX])

        x_reshape_0 = tf.reshape(x, [-1, NX, 1])


        for i in range(hidden_layer):

            lower_pad = x_reshape_0[:, :(kernel_size[i] - 1), :]

            x_pad = tf.concat([x_reshape_0, lower_pad], axis=1)

            x_conv_0 = Conv1D(inputs=x_pad, filters=channel, kernel_size=kernel_size[i], padding="valid", activation=tf.nn.relu)


        x_dense = Dense(inputs=x_conv_0, units=256, use_bias=True, activation=tf.nn.relu)

        x_flat = Flatten(x_dense)

        output = Dense(inputs=x_flat, units=1, use_bias=False, activation=None)

        # ground state energy
        eloc = tf.placeholder(tf.float32, [None, 1])
        # the reduce state energy
        ene = tf.reduce_mean(eloc)
        # loss function
        loss = tf.reduce_sum(output * (eloc - ene))
        # optimization
        train_step = tf.train.AdamOptimizer().minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


        self.x, self.output = x, output
        self.eloc, self.ene, self.loss = eloc, ene, loss
        self.train_step = train_step

    def prepare_session(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess

    def forward(self, num):
        return self.sess.run(self.output, feed_dict={self.x: num}).ravel()

    def optimize(self, state, eloc):
        eloc = eloc.reshape(NSAMPLE, 1)
        self.sess.run(self.train_step,
                      feed_dict={self.x: state.num, self.eloc:  eloc})

class SampledState:
    thermalization_n = 1024

    def __init__(self, net):
        self.num = np.zeros(NSAMPLE * NX)
        self.num = self.num.reshape(NSAMPLE, NX)
        for i in range(NSAMPLE):
            for j in range(NP):
                self.num[i][j % NX] += 1
        self.lnpsi = net.forward(self.num)

    def try_flip(self, net):
        num_tmp = np.copy(self.num)
        for i in range(NSAMPLE):
            p0 = np.random.randint(NX)
            p1 = np.random.randint(NX)
            if num_tmp[i][p0] > 0 and p0 != p1:
                num_tmp[i][p0] -= 1
                num_tmp[i][p1] += 1
        lnpsi_tmp = net.forward(num_tmp)
        r = np.random.rand(NSAMPLE)
        isflip = r < np.exp(2 * (lnpsi_tmp - self.lnpsi))
        for i in range(NSAMPLE):
            if isflip[i]:
                self.num[i] = num_tmp[i]
                self.lnpsi[i] = lnpsi_tmp[i]

    def thermalize(self, net):
        for i in range(SampledState.thermalization_n):
            self.try_flip(net)

#-----------------------------------

def LocalEnergy(net, state):
    st = np.zeros((NSAMPLE, NX, 2, NX))
    st += state.num.reshape(NSAMPLE, 1, 1, NX)
    for b in range(NSAMPLE):
        for j in range(NX):
            if state.num[b][j] > 0:
                st[b][j][0][j] -= 1
                st[b][j][0][(j+1) % NX] += 1
                st[b][j][1][j] -= 1
                st[b][j][1][(j-1+NX)%NX] += 1
    st = st.reshape(NSAMPLE * NX * 2, NX)
    lnpsi2 = net.forward(st).reshape(NSAMPLE, NX, 2)
    eloc = np.zeros(NSAMPLE)
    for b in range(NSAMPLE):
        onsite = hopping = nearest_neighbor = 0
        for j in range(NX):

            if state.num[b][j] > 0:
                onsite += 0.5 * U * state.num[b][j] * (state.num[b][j] - 1)

                hopping += J * np.sqrt(state.num[b][j] * (state.num[b][(j + 1) % NX] + 1)) * np.exp(
                    lnpsi2[b][j][0] - state.lnpsi[b])
                hopping += J * np.sqrt(state.num[b][j] * (state.num[b][(j - 1 + NX) % NX] + 1)) * np.exp(
                    lnpsi2[b][j][1] - state.lnpsi[b])

                nearest_neighbor += 0.5 * V * state.num[b][j] * state.num[b][(j + 1) % NX]

                nearest_neighbor += 0.5 * V * state.num[b][j] * state.num[b][(j - 1 + NX) % NX]

        eloc[b] = hopping + onsite + nearest_neighbor

    return eloc

def Correlation(net, state):
    st = np.zeros((NSAMPLE, NX, 2, NX))
    st += state.num.reshape(NSAMPLE, 1, 1, NX)
    sf = np.zeros(NSAMPLE)
    r = 0
    for b in range(NSAMPLE):
        for j in range(NX):
            if state.num[b][j] > 0:
                st[b][j][0][j] -= 1
                st[b][j][0][(j + r) % NX] += 1
    st = st.reshape(NSAMPLE * NX * 2, NX)
    lnpsi2 = net.forward(st).reshape(NSAMPLE, NX, 2)
    for b in range(NSAMPLE):
        SF = 0
        for j in range(NX):
            if r == 0:
                SF += np.sqrt(state.num[b][j] * (state.num[b][(j + r) % NX])) * np.exp(
                    lnpsi2[b][j][0] - state.lnpsi[b])
            else:
                SF += np.sqrt(state.num[b][j] * (state.num[b][(j + r) % NX] + 1)) * np.exp(
                    lnpsi2[b][j][0] - state.lnpsi[b])
        sf[b] = (SF / NX)
    return sf
# -------------- main -----------------


energy_ground_state = 0

net = Network()

state = SampledState(net)

state.thermalize(net)




energy_history = np.zeros(STEP)
print('U', '=', U, 'V', '=', V)
for counter in range(STEP):
    # try_flip state
    for i in range(32):
        state.try_flip(net)
    # got LocalEnergy
    eloc = LocalEnergy(net, state)
    # optimize LocalEnergy
    net.optimize(state, eloc)

    #got correlation
    sf = Correlation(net, state)
    net.optimize(state, sf)

    # Output data every 10 times

    # print("step: {:>6d} SF.mean: {:7.5f} eloc.mean: {:7.5f}".format(counter + 1, sf.mean(), eloc.mean(), flush=True))
    # print("step: {:>6d} SF.mean: {:7.5f}".format(counter + 1, sf.mean(), flush=True))

    if counter % 1 == 0:
        with open(r'/home/huang/pyvenv/Program/NQS/eBHM/eloc_U_6.0_V_1.0_R_0.txt', 'a+') as tf:
            tf.write(str(counter))
            tf.write(' ')
            tf.write(str(eloc.mean()))
            tf.write('\n')
            tf.close()
        with open(r'/home/huang/pyvenv/Program/NQS/eBHM/SF_U_6.0_V_1.0_R_0.txt', 'a+') as tf:
            tf.write(str(counter))
            tf.write(' ')
            tf.write(str(sf.mean()))
            tf.write('\n')
            tf.close()
        # print(counter, eloc.mean(), flush=True)
        # print(counter, sf.mean(), flush=True)

    counter += 1


