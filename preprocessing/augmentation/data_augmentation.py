
# coding: utf-8

# # Data augmentation for time-series data

# #### This is a simple example to apply data augmentation to time-series data (e.g. wearable sensor data). If it helps your research, please cite the below paper. 

# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220. 

# https://dl.acm.org/citation.cfm?id=3136817
# 
# https://arxiv.org/abs/1706.00527

# @inproceedings{TerryUm_ICMI2017,
#  author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
#  title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
#  booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
#  series = {ICMI 2017},
#  year = {2017},
#  isbn = {978-1-4503-5543-8},
#  location = {Glasgow, UK},
#  pages = {216--220},
#  numpages = {5},
#  doi = {10.1145/3136755.3136817},
#  acmid = {3136817},
#  publisher = {ACM},
#  address = {New York, NY, USA},
#  keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
# } 

# #### You can freely modify this code for your own purpose. However, please leave the above citation information untouched when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Terry Taewoong Um (terry.t.um@gmail.com)
# 
# https://twitter.com/TerryUm_ML
# 
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from utils.filter import get_highest_freq_fft


class DataAugmentation:
    def __init__(self, x, not_section):
        self.x = x
        self.not_section = not_section

    def get_knot(self,):
        highest_freq = get_highest_freq_fft(self.x[:, 3])
        highest_freq = 5
        knot = highest_freq * self.not_section  # each knob should be devided to 5 section
        return knot

    def jitterning(self, sigma=0.05):
        '''
        # ## 1. Jittering
        # #### Hyperparameters :  sigma = standard devitation (STD) of the noise
        :param sigma:
        :return:
        '''
        myNoise = np.random.normal(loc=0, scale=sigma, size=self.x.shape)
        return self.x+myNoise

    def scaling(self, sigma=0.1):
        '''
        # ## 2. Scaling
        # #### Hyperparameters :  sigma = STD of the zoom-in/out factor
        :param sigma:
        :return:
        '''
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, self.x.shape[1])) # shape=(1,3)
        myNoise = np.matmul(np.ones((self.x.shape[0],1)), scalingFactor)
        return self.x*myNoise

    def generaterandomcurve(self, sigma=0.2, knot=10):
        '''
        this is for timewrap --> distorttimesteps, shift or squeeze all variable with a single dt
        :param sigma:
        :param knot:
        :return:
        '''
        xx = (np.ones((self.shape[1], 1))*(np.arange(0, self.shape[0], (self.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, self.shape[1]))
        x_range = np.arange(self.shape[0])
        cs_all = []
        for i in range(xx.shape[1]):
            cs = CubicSpline(xx[:, i], yy[:, 0])  # by changing the i value of yy[:,i] to a single number, you can produce constant changes
            cs_all.append(cs(x_range))
        return np.array(cs_all).transpose()


    def generaterandomcurves(self, sigma=0.2, knot=10):
        xx = (np.ones((self.shape[1], 1))*(np.arange(0, self.shape[0], (self.shape[0]-1)/(knot+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, self.shape[1]))
        x_range = np.arange(self.shape[0])
        cs_all = []
        for i in range(xx.shape[1]):
            cs = CubicSpline(xx[:, i], yy[:, i])  # by changing the i value of yy[:,i] to a single number, you can produce constant changes
            cs_all.append(cs(x_range))
        return np.array(cs_all).transpose()

    def da_magwarp(self, sigma=0.2):
        '''
        # ## 3. Magnitude Warping
        # #### Hyperparameters :  sigma = STD of the random knots for generating curves
        # #### knot = # of knots for the random curves (complexity of the curves)
        # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".
        # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
        :param sigma:
        :return:
        '''
        try:
            knot = self.get_knot()
        except:
            knot = 10
        return self.x * DataAugmentation.generaterandomcurves(self.x, sigma, knot)

    def da_magoffset(self, std=5):
        '''
        # ## 3. Magnitude Warping
        # #### Hyperparameters :  sigma = STD of the random knots for generating curves
        # #### knot = # of knots for the random curves (complexity of the curves)
        # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".
        # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
        :param sigma:
        :return:
        '''
        delta = np.random.normal(loc=0, scale=std)
        return self.x + delta

    def da_magwarppoffset(self, sigma=0.2, std=5):
        '''
        # ## 3. Magnitude Warping
        # #### Hyperparameters :  sigma = STD of the random knots for generating curves
        # #### knot = # of knots for the random curves (complexity of the curves)
        # "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".
        # "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
        :param sigma:
        :return:
        '''
        knot = self.get_knot()
        delta = np.random.normal(loc=0, scale=std)
        return self.x + delta*DataAugmentation.generaterandomcurves(self.x, sigma, knot)

    def distorttimesteps(self, sigma=0.2, knot=10):
        tt = DataAugmentation.generaterandomcurve(self, sigma, knot)  # Regard these samples around 1 as time intervals
        tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
        # Make the last value to have X.shape[0]
        t_scale = []
        for i in range(tt_cum.shape[1]):
            t_scale.append((self.shape[0] - 1) / tt_cum[-1, i])

        for j in range(tt_cum.shape[1]):
            tt_cum[:, j] = tt_cum[:, j] * t_scale[j]
        return tt_cum

    def da_timewarp(self, sigma=0.1):
        '''
         ## 4. Time Warping
        # #### Hyperparameters :  sigma = STD of the random knots for generating curves
        # #### knot = # of knots for the random curves (complexity of the curves)
        sigma = 0.4
        knot = 4
        :param sigma:
        :return:
        '''
        try:
            knot = self.get_knot()
        except:
            knot = 10
        tt_new = DataAugmentation.distorttimesteps(self.x, sigma, knot)
        X_new = np.zeros(self.x.shape)
        x_range = np.arange(self.x.shape[0])
        for i in range(tt_new.shape[1]):
            X_new[:, i] = np.interp(x_range, tt_new[:, i], self.x[:, i])
        return X_new

    def da_rotation(self):
        '''
        # ## 5. Rotation
        # #### Hyperparameters :  N/A
        :return:
        '''
        axis = np.random.uniform(low=-1, high=1, size=self.x.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(self.x, axangle2mat(axis, angle))

    def da_permutation(self, nPerm=4, minSegLength=10):
        '''
        # ## 6. Permutation
        # #### Hyperparameters :  nPerm = # of segments to permute
        # #### minSegLength = allowable minimum length for each segment
        :param nPerm:  nPerm = 4
        :param minSegLength:  minSegLength = 100
        :return:
        '''
        X_new = np.zeros(self.x.shape)
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm + 1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, self.x.shape[0] - minSegLength, nPerm - 1))
            segs[-1] = self.x.shape[0]
            if np.min(segs[1:] - segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = self.x[segs[idx[ii]]:segs[idx[ii] + 1], :]
            X_new[pp:pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return (X_new)

    def RandSampleTimesteps(self, nSample=1000):
        X_new = np.zeros(self.shape)
        tt = np.zeros((nSample, self.shape[1]), dtype=int)
        for i in range(tt.shape[1]):
            tt[1:-1, i] = np.sort(np.random.randint(1, self.shape[0]-1, nSample-2))
        tt[-1, :] = self.shape[0]-1
        return tt

    def da_randsampling(self, nSample=1000):
        '''
        # ## 7. Random Sampling
        # #### Hyperparameters :  nSample = # of subsamples (nSample <= X.shape[0])
        # This approach is similar to TimeWarp, but will use only subsamples (not all samples) for interpolation. (Using TimeWarp is more recommended)
        :param self:
        :param nSample:
        :return:
        '''
        tt = DataAugmentation.RandSampleTimesteps(self.x, nSample)
        X_new = np.zeros(self.x.shape)
        for i in range(tt.shape[1]):
            X_new[:, i] = np.interp(np.arange(self.x.shape[0]), tt[:, i], self.x[tt[:, i], i])
        return X_new


