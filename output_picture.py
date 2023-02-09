import matplotlib.pyplot as plt
class Drawing(object):
    def __init__(self, xcounter=[], mean_tanh=[], mean_sigmoid=[], mean_softmax=[], mean_softsign=[], mean_swish=[],
                 mean_relu=[],  mean_selu=[], mean_relu_1=[], mean_relu_2=[], mean_relu_cnn=[], mean_relu_cnn_dense=[],
                 mean_relu_cnn_j_0=[], mean_relu_cnn_j_1=[], mean_relu_cnn_channel_up=[], mean_relu_cnn_channel_down=[],
                 mean_cnn_adadelta=[], mean_cnn_adagrad=[], mean_cnn=[], mean_cnn_fnn=[], mean_2cnn_inside=[], mean_2cnn_outside=[], mean_cnn_8=[], mean_cnn_draft=[] ):
        self.xcounter = xcounter
        self.mean_tanh = mean_tanh
        self.mean_sigmoid = mean_sigmoid
        self.mean_softmax = mean_softmax
        self.mean_softsign = mean_softsign
        self.mean_swish = mean_swish
        self.mean_relu = mean_relu
        self.mean_selu = mean_selu
        self.mean_relu_1 = mean_relu_1
        self.mean_relu_2 = mean_relu_2
        self.mean_relu_cnn = mean_relu_cnn
        self.mean_relu_cnn_dense = mean_relu_cnn_dense
        self.mean_relu_cnn_j_0 = mean_relu_cnn_j_0
        self.mean_relu_cnn_j_1 = mean_relu_cnn_j_1
        self.mean_relu_cnn_channel_up = mean_relu_cnn_channel_up
        self.mean_relu_cnn_channel_down = mean_relu_cnn_channel_down
        self.mean_cnn_adadelta = mean_cnn_adadelta
        self.mean_cnn_adagrad = mean_cnn_adagrad
        self.mean_cnn = mean_cnn
        self.mean_cnn_fnn = mean_cnn_fnn
        self.mean_2cnn_inside = mean_2cnn_inside
        self.mean_2cnn_outside = mean_2cnn_outside
        self.mean_cnn_8 = mean_cnn_8
        self.mean_cnn_draft = mean_cnn_draft

    def data(self):
        # tanh
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/tanh.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    b1 = int(a[0])
                    b2 = float(a[1].strip())
                    self.xcounter.append(b1)
                    self.mean_tanh.append(b2)
        #sigmoid
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/sigmoid.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_sigmoid.append(b2)
        #softmax
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/softmax.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_softmax.append(b2)
        #softsign
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/softsign.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_softsign.append(b2)
        #swish
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/swish.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_swish.append(b2)
        #relu
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_relu.append(b2)
        #selu
        with open(r'/home/huang/pyvenv/Program/NQS/ouput/selu.txt') as tf:
            for i in range(100):
                text = tf.readline()
                if (i % 4) == 0:
                    a = text.split(' ')
                    # b1 = int(a[0])
                    b2 = float(a[1].strip())
                    # self.xcounter.append(b1)
                    self.mean_selu.append(b2)
        # #relu_1
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_1.txt') as tf:
        #     for i in range(500):
        #         text = tf.readline()
        #         if (i % 10) == 0:
        #             a = text.split(' ')
        #             b1 = int(a[0])
        #             b2 = float(a[1].strip())
        #             self.xcounter.append(b1)
        #             self.mean_relu_1.append(b2)
        # #relu_2
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_1.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_2.append(b2)
        # #relu_cnn
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn.txt') as tf:
        #     for i in range(400):
        #         text = tf.readline()
        #         if (i % 10) == 0:
        #             a = text.split(' ')
        #             b1 = int(a[0])
        #             b2 = float(a[1].strip())
        #             self.xcounter.append(b1)
        #             self.mean_relu_cnn.append(b2)
        # # relu_cnn_dense
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn_dense.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_cnn_dense.append(b2)
        # # relu_cnn_j_0
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn_j_0.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_cnn_j_0.append(b2)
        # # relu_cnn_j_1
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn_j_1.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_cnn_j_1.append(b2)
        # # relu_cnn_channel_up
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn_channel_up.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_cnn_channel_up.append(b2)
        # # relu_cnn_channel_down
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/relu_cnn_channel_down.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_relu_cnn_channel_down.append(b2)
        # # mean_cnn_adadelta
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn_adadelta.txt') as tf:
        #     for i in range(400):
        #         text = tf.readline()
        #         if (i % 10) == 0:
        #             a = text.split(' ')
        #             # b1 = int(a[0])
        #             b2 = float(a[1].strip())
        #             # self.xcounter.append(b1)
        #             self.mean_cnn_adadelta.append(b2)
        # # mean_cnn_adagrad
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn_adagrad.txt') as tf:
        #     for i in range(400):
        #         text = tf.readline()
        #         if (i % 10) == 0:
        #             a = text.split(' ')
        #             # b1 = int(a[0])
        #             b2 = float(a[1].strip())
        #             # self.xcounter.append(b1)
        #             self.mean_cnn_adagrad.append(b2)
        # # mean_cnn
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn.txt') as tf:
        #     for i in range(500):
        #         text = tf.readline()
        #         if (i % 10) == 0:
        #             a = text.split(' ')
        #             # b1 = int(a[0])
        #             b2 = float(a[1].strip())
        #             # self.xcounter.append(b1)
        #             self.mean_cnn.append(b2)
        # # mean_cnn_fnn
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn+fnn.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_cnn_fnn.append(b2)
        # # mean_2cnn_inside
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn+fnn_2.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_2cnn_inside.append(b2)
        # # mean_2cnn_outside
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn+fnn_3.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_2cnn_outside.append(b2)
        # # mean_cnn_8
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/cnn_8.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_cnn_8.append(b2)
        # mean_cnn_draft
        # with open(r'/home/huang/pyvenv/Program/NQS/ouput/draft.txt') as tf:
        #     for i in range(101):
        #         text = tf.readline()
        #         a = text.split(' ')
        #         # b1 = int(a[0])
        #         b2 = float(a[1].strip())
        #         # self.xcounter.append(b1)
        #         self.mean_cnn_draft.append(b2)

    def draw(self):
        x = self.xcounter
        y1 = self.mean_tanh
        y2 = self.mean_sigmoid
        y3 = self.mean_softmax
        y4 = self.mean_softsign
        y5 = self.mean_swish
        y6 = self.mean_relu
        y7 = self.mean_selu
        y8 = self.mean_relu_1
        y9 = self.mean_relu_2
        y10 = self.mean_relu_cnn
        y11 = self.mean_relu_cnn_dense
        y12 = self.mean_relu_cnn_j_0
        y13 = self.mean_relu_cnn_channel_up
        y14 = self.mean_relu_cnn_channel_down
        y15 = self.mean_relu_cnn_j_1
        y16 = self.mean_cnn_adadelta
        y17 = self.mean_cnn_adagrad
        y18 = self.mean_cnn
        y19 = self.mean_cnn_fnn
        y20 = self.mean_2cnn_inside
        y21 = self.mean_2cnn_outside
        y22 = self.mean_cnn_8
        y23 = self.mean_cnn_draft

        plt.plot(x, y1, color='tab:red', label='tanh', marker='d')
        plt.plot(x, y2, color='tab:blue', label='sigmoid', marker='d')
        plt.plot(x, y3, color='tab:green', label='softmax', marker='d')
        plt.plot(x, y4, color='black', label='softsign', marker='d')
        plt.plot(x, y5, color='tab:pink', label='swish', marker='d')
        plt.plot(x, y6, color='tab:orange', label='relu', marker='d')
        plt.plot(x, y7, color='tab:olive', label='selu', marker='d')
        # plt.plot(x, y8, color='tab:red', label='Fully Connected Network', marker='^')
        # plt.plot(x, y9, color='black', label='relu_cov_2')
        # plt.plot(x, y10, color='tab:blue', label='Adam', marker='^')
        # plt.plot(x, y11, color='tab:red', label='cnn_no_dense')
        # plt.plot(x, y12, color='tab:red', label='cnn_j_0')
        # plt.plot(x, y13, color='tab:green', label='cnn_channel_6')
        # plt.plot(x, y14, color='tab:red', label='cnn_channel_0')
        # plt.plot(x, y15, color='black', label='cnn_j_1')
        # plt.plot(x, y16, color='tab:orange', label='AdaDelta', marker='s')
        # plt.plot(x, y17, color='tab:red', label='AdaGrad', marker='d')
        # plt.plot(x, y18, color='black', label='Convolutional Network', marker='d')
        # plt.plot(x, y19, color='tab:blue', label='fnn+cnn')
        # plt.plot(x, y20, color='tab:orange', label='2cnn_inside')
        # plt.plot(x, y21, color='tab:olive', label='2cnn_outside')
        # plt.plot(x, y22, color='tab:blue', label='cnn')
        # plt.plot(x, y23, color='tab:red', label='cnn_cnn_ad')
        plt.legend(loc='best')
        plt.title('Different Activation Function Comparation\n'
                  '(sites=64;bosons=64;U=1;J=-1)')
        plt.xlabel('counter')
        plt.ylabel('Ground Energy')

        plt.show()
if __name__ == '__main__' :
    d = Drawing()
    d.data()
    d.draw()


