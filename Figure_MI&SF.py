import matplotlib.pyplot as plt
import numpy as np
class Drawing(object):
    def __init__(self, U_0_0=[], U_1_0=[], U_2_0=[], U_3_0=[], U_4_0=[], U_5_0=[], U_6_0=[], U_7_0=[], U_8_0=[],
                 U_0_5=[], U_1_5=[], U_2_5=[], U_3_5=[], U_4_5=[], U_5_5=[], U_6_5=[], U_7_5=[],
                 Co_U_1_0_R_4_0 = [], Co_U_1_5_R_4_0=[], Co_U_2_0_R_4_0=[], Co_U_2_5_R_4_0=[],
                 Co_U_3_0_R_4_0=[], Co_U_3_5_R_4_0=[], Co_U_4_0_R_4_0=[], Co_U_4_5_R_4_0=[],  Co_U_5_0_R_4_0=[],
                 Co_U_0_0_V_1=[], Co_U_0_5_V_1=[], Co_U_1_0_V_1=[], Co_U_1_5_V_1=[], Co_U_2_0_V_1=[], Co_U_2_5_V_1=[], Co_U_3_0_V_1=[],
                 Co_U_3_5_V_1=[], Co_U_4_0_V_1=[], Co_U_4_5_V_1=[], Co_U_5_0_V_1=[], Co_U_5_5_V_1=[], Co_U_6_0_V_1=[], Co_U_6_5_V_1=[],
                 Co_U_7_0_V_1=[], Co_U_7_5_V_1=[], Co_U_8_0_V_1=[], ):
        self.U_0_0 = U_0_0
        self.U_1_0 = U_1_0
        self.U_2_0 = U_2_0
        self.U_3_0 = U_3_0
        self.U_4_0 = U_4_0
        self.U_5_0 = U_5_0
        self.U_6_0 = U_6_0
        self.U_7_0 = U_7_0
        self.U_8_0 = U_8_0
        self.U_0_5 = U_0_5
        self.U_1_5 = U_1_5
        self.U_2_5 = U_2_5
        self.U_3_5 = U_3_5
        self.U_4_5 = U_4_5
        self.U_5_5 = U_5_5
        self.U_6_5 = U_6_5
        self.U_7_5 = U_7_5
        self.Co_U_1_0_R_4_0 = Co_U_1_0_R_4_0
        self.Co_U_1_5_R_4_0 = Co_U_1_5_R_4_0
        self.Co_U_2_0_R_4_0 = Co_U_2_0_R_4_0
        self.Co_U_2_5_R_4_0 = Co_U_2_5_R_4_0
        self.Co_U_3_0_R_4_0 = Co_U_3_0_R_4_0
        self.Co_U_3_5_R_4_0 = Co_U_3_5_R_4_0
        self.Co_U_4_0_R_4_0 = Co_U_4_0_R_4_0
        self.Co_U_4_5_R_4_0 = Co_U_4_5_R_4_0
        self.Co_U_5_0_R_4_0 = Co_U_5_0_R_4_0

        self.Co_U_0_0_V_1 = Co_U_0_0_V_1
        self.Co_U_0_5_V_1 = Co_U_0_5_V_1
        self.Co_U_1_0_V_1 = Co_U_1_0_V_1
        self.Co_U_1_5_V_1 = Co_U_1_5_V_1
        self.Co_U_2_0_V_1 = Co_U_2_0_V_1
        self.Co_U_2_5_V_1 = Co_U_2_5_V_1
        self.Co_U_3_0_V_1 = Co_U_3_0_V_1
        self.Co_U_3_5_V_1 = Co_U_3_5_V_1
        self.Co_U_4_0_V_1 = Co_U_4_0_V_1
        self.Co_U_4_5_V_1 = Co_U_4_5_V_1
        self.Co_U_5_0_V_1 = Co_U_5_0_V_1
        self.Co_U_5_5_V_1 = Co_U_5_5_V_1
        self.Co_U_6_0_V_1 = Co_U_6_0_V_1
        self.Co_U_6_5_V_1 = Co_U_6_5_V_1
        self.Co_U_7_0_V_1 = Co_U_7_0_V_1
        self.Co_U_7_5_V_1 = Co_U_7_5_V_1
        self.Co_U_8_0_V_1 = Co_U_8_0_V_1


    def data(self):
        # # Co_U_1_0_R_4_0 site=32 bosons=32 J=-1 U=1 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_1.0_R_4.0.txt') as tf:
        #         text = tf.readlines()
        #         for i in range(999, 1000):
        #             a = text[i].split(' ')
        #             b1 = 1.0
        #             b2 = float(a[1].strip())
        #             # print(b1)
        #             self.U_1_0.append(b1)
        #             self.Co_U_1_0_R_4_0.append(b2)
        # # Co_U_1_5_R_4_0 site=32 bosons=32 J=-1 U=1.5 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_1.5_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 1.5
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_1_5.append(b1)
        #         self.Co_U_1_5_R_4_0.append(b2)
        # # Co_U_2_0_R_4_0 site=32 bosons=32 J=-1 U=2.0 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_2.0_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 2.0
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_2_0.append(b1)
        #         self.Co_U_2_0_R_4_0.append(b2)
        # # Co_U_2_5_R_4_0 site=32 bosons=32 J=-1 U=2.5 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_2.5_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 2.5
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_2_5.append(b1)
        #         self.Co_U_2_5_R_4_0.append(b2)
        # # Co_U_3_0_R_4_0 site=32 bosons=32 J=-1 U=3.0 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_3.0_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 3.0
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_3_0.append(b1)
        #         self.Co_U_3_0_R_4_0.append(b2)
        # # Co_U_3_5_R_4_0 site=32 bosons=32 J=-1 U=3.5 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_3.5_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 3.5
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_3_5.append(b1)
        #         self.Co_U_3_5_R_4_0.append(b2)
        # # Co_U_4_0_R_4_0 site=32 bosons=32 J=-1 U=4.0 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_4.0_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 4.0
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_4_0.append(b1)
        #         self.Co_U_4_0_R_4_0.append(b2)
        # # Co_U_4_5_R_4_0 site=32 bosons=32 J=-1 U=4.0 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_4.5_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 4.5
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_4_5.append(b1)
        #         self.Co_U_4_5_R_4_0.append(b2)
        # # Co_U_5_0_R_4_0 site=32 bosons=32 J=-1 U=5.0 V=1 r=4
        # with open(r'/home/huang/pyvenv/Program/NQS/BHM/MI&SF_V_1.0/Co_U_5.0_R_4.0.txt') as tf:
        #     text = tf.readlines()
        #     for i in range(999, 1000):
        #         a = text[i].split(' ')
        #         b1 = 5.0
        #         b2 = float(a[1].strip())
        #         # print(b1)
        #         self.U_5_0.append(b1)
        #         self.Co_U_5_0_R_4_0.append(b2)

#------------------------------------------------------------------------------------------------------------------------#

        # Co_U_0_0_V_1 site=32 bosons=32 J=-1 U=0.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_0.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 0.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_0_0.append(b1)
                self.Co_U_0_0_V_1.append(b2)
        # Co_U_0_5_V_1 site=32 bosons=32 J=-1 U=0.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_0.5_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 0.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_0_5.append(b1)
                self.Co_U_0_5_V_1.append(b2)
        # Co_U_1_0_V_1 site=32 bosons=32 J=-1 U=1.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_1.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 1.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_1_0.append(b1)
                self.Co_U_1_0_V_1.append(b2)
        # Co_U_1_5_V_1 site=32 bosons=32 J=-1 U=1.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_1.5_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 1.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_1_5.append(b1)
                self.Co_U_1_5_V_1.append(b2)
        # Co_U_2_0_V_1 site=32 bosons=32 J=-1 U=2.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_2.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 2.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_2_0.append(b1)
                self.Co_U_2_0_V_1.append(b2)
        # Co_U_2_5_V_1 site=32 bosons=32 J=-1 U=2.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_2.5_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 2.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_2_5.append(b1)
                self.Co_U_2_5_V_1.append(b2)
        # Co_U_3_0_V_1 site=32 bosons=32 J=-1 U=3.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_3.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 3.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_3_0.append(b1)
                self.Co_U_3_0_V_1.append(b2)
        # Co_U_3_5_V_1 site=32 bosons=32 J=-1 U=3.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_3.5_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 3.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_3_5.append(b1)
                self.Co_U_3_5_V_1.append(b2)
        # Co_U_4_0_V_1 site=32 bosons=32 J=-1 U=4.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_4.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 4.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_4_0.append(b1)
                self.Co_U_4_0_V_1.append(b2)
        # Co_U_4_5_V_1 site=32 bosons=32 J=-1 U=4.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_4.5_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 4.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_4_5.append(b1)
                self.Co_U_4_5_V_1.append(b2)
        # Co_U_5_0_V_1 site=32 bosons=32 J=-1 U=5.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_5.0_V_1.0_R_4.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 5.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_5_0.append(b1)
                self.Co_U_5_0_V_1.append(b2)
        # Co_U_5_5_V_1 site=32 bosons=32 J=-1 U=5.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_5.5_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 5.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_5_5.append(b1)
                self.Co_U_5_5_V_1.append(b2)
        # Co_U_6_0_V_1 site=32 bosons=32 J=-1 U=6.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_6.0_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 6.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_6_0.append(b1)
                self.Co_U_6_0_V_1.append(b2)
        # Co_U_6_5_V_1 site=32 bosons=32 J=-1 U=6.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_6.5_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 6.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_6_5.append(b1)
                self.Co_U_6_5_V_1.append(b2)
        # Co_U_7_0_V_1 site=32 bosons=32 J=-1 U=7.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_7.0_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 7.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_7_0.append(b1)
                self.Co_U_7_0_V_1.append(b2)
        # Co_U_7_5_V_1 site=32 bosons=32 J=-1 U=7.5 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_7.5_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 7.5
                b2 = float(a[1].strip())
                # print(b1)
                self.U_7_5.append(b1)
                self.Co_U_7_5_V_1.append(b2)
        # Co_U_8_0_V_1 site=32 bosons=32 J=-1 U=8.0 V=1 r=4
        with open(r'/home/huang/pyvenv/Program/NQS/BHM/s_32_b_32/V_1.0_R_4/U_8.0_V_1.0_R_4_copy.txt') as tf:
            text = tf.readlines()
            for i in range(999, 1000):
                a = text[i].split(' ')
                b1 = 8.0
                b2 = float(a[1].strip())
                # print(b1)
                self.U_8_0.append(b1)
                self.Co_U_8_0_V_1.append(b2)



    def draw(self):
        U_0_0 = self.U_0_0
        U_1_0 = self.U_1_0
        U_2_0 = self.U_2_0
        U_3_0 = self.U_3_0
        U_4_0 = self.U_4_0
        U_5_0 = self.U_5_0
        U_6_0 = self.U_6_0
        U_7_0 = self.U_7_0
        U_8_0 = self.U_8_0
        U_0_5 = self.U_0_5
        U_1_5 = self.U_1_5
        U_2_5 = self.U_2_5
        U_3_5 = self.U_3_5
        U_4_5 = self.U_4_5
        U_5_5 = self.U_5_5
        U_6_5 = self.U_6_5
        U_7_5 = self.U_7_5
        Co_U_1_0_R_4_0 = self.Co_U_1_0_R_4_0
        Co_U_1_5_R_4_0 = self.Co_U_1_5_R_4_0
        Co_U_2_0_R_4_0 = self.Co_U_2_0_R_4_0
        Co_U_2_5_R_4_0 = self.Co_U_2_5_R_4_0
        Co_U_3_0_R_4_0 = self.Co_U_3_0_R_4_0
        Co_U_3_5_R_4_0 = self.Co_U_3_5_R_4_0
        Co_U_4_0_R_4_0 = self.Co_U_4_0_R_4_0
        Co_U_4_5_R_4_0 = self.Co_U_4_5_R_4_0
        Co_U_5_0_R_4_0 = self.Co_U_5_0_R_4_0

        Co_U_0_0_V_1 = self.Co_U_0_0_V_1
        Co_U_0_5_V_1 = self.Co_U_0_5_V_1
        Co_U_1_0_V_1 = self.Co_U_1_0_V_1
        Co_U_1_5_V_1 = self.Co_U_1_5_V_1
        Co_U_2_0_V_1 = self.Co_U_2_0_V_1
        Co_U_2_5_V_1 = self.Co_U_2_5_V_1
        Co_U_3_0_V_1 = self.Co_U_3_0_V_1
        Co_U_3_5_V_1 = self.Co_U_3_5_V_1
        Co_U_4_0_V_1 = self.Co_U_4_0_V_1
        Co_U_4_5_V_1 = self.Co_U_4_5_V_1
        Co_U_5_0_V_1 = self.Co_U_5_0_V_1
        Co_U_5_5_V_1 = self.Co_U_5_5_V_1
        Co_U_6_0_V_1 = self.Co_U_6_0_V_1
        Co_U_6_5_V_1 = self.Co_U_6_5_V_1
        Co_U_7_0_V_1 = self.Co_U_7_0_V_1
        Co_U_7_5_V_1 = self.Co_U_7_5_V_1
        Co_U_8_0_V_1 = self.Co_U_8_0_V_1

        fig, (ax1) = plt.subplots(nrows=1, figsize=(10, 5))

        ax1.set_title('(site=32 bosons=32/V=1.0 R=4.0)', fontweight='bold')
        ax1.set_xlabel('U')
        ax1.set_ylabel('Correlation')
        ax1.plot(U_0_0, Co_U_0_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_0_5, Co_U_0_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_1_0, Co_U_1_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_1_5, Co_U_1_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_2_0, Co_U_2_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_2_5, Co_U_2_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_3_0, Co_U_3_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_3_5, Co_U_3_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_4_0, Co_U_4_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_4_5, Co_U_4_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_5_0, Co_U_5_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_5_5, Co_U_5_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_6_0, Co_U_6_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_6_5, Co_U_6_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_7_0, Co_U_7_0_V_1, color='tab:green', marker='^')
        ax1.plot(U_7_5, Co_U_7_5_V_1, color='tab:green', marker='^')
        ax1.plot(U_8_0, Co_U_8_0_V_1, color='tab:green', marker='^')
        # fig.legend((c1, c2, c3, c4, c5, c6, c7, c8, c9), ('U=0.0', 'U=1.0', 'U=2.0', 'U=3.0', 'U=4.0', 'U=5.0', 'U=6.0', 'U=7.0', 'U=8.0'), loc='upper right',
        #            bbox_to_anchor=(0.9, 0.88), fontsize='x-small')


        # ax2.set_title('Superfluid(sites=32 bosons=32)', fontweight='bold')
        # ax2.set_xlabel('number of updates')
        # ax2.set_ylabel('SF.mean')
        # ax2.plot(x, y17, color='tab:olive', label='U=1 V=1 r=0', marker='^')
        # v1, = ax2.plot(x, y1, color='tab:blue', label='U=1 V=1 r=2', marker='s')
        # v2_3, = ax2.plot(x, y5, color='tab:red', label='U=2.3 V=1 r=2', marker='8')
        # plt.plot(x, y6, color='tab:pink', label='U=2.5 V=1 r=2')
        # plt.plot(x, y3, color='tab:orange', label='U=5 V=1 r=2')
        # plt.plot(x, y4, color='tab:orange', label='U=5 V=1 r=2')
        # r3, = ax2.plot(x, y7, color='tab:red', label='U=1 V=1 r=3', marker='8')
        # r5, = ax2.plot(x, y9, color='black', label='U=1 V=1 r=5', marker='s')
        # plt.plot(x, y11, color='tab:cyan', label='U=10 V=1 r=2')
        # plt.plot(x, y12, color='tab:cyan', label='U=10 V=1 r=2')
        # plt.plot(x, y13, color='tab:purple', label='U=4 V=4 r=2')
        # plt.plot(x, y14, color='tab:purple', label='U=4 V=4 r=2')
        # v3, = ax2.plot(x, y15, color='black', label='U=3 V=1 r=2', marker='^')
        # plt.plot(x, y16, color='tab:red', label='U=3 V=1 r=2')
        # fig.legend((r3, r5), ('r=3', 'r=29',), loc='upper right',
        #            bbox_to_anchor=(0.9, 0.88), title='U=1.0 / V=1.0', fontsize='medium')

        plt.show()

if __name__ == '__main__' :
    d = Drawing()
    d.data()
    d.draw()