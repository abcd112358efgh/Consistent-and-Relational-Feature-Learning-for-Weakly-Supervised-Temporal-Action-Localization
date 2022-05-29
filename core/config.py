# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()
cfg.E_THETA= 1
cfg.E_G = 0.2
cfg.SIGMA = 1
cfg.E_ALPHA=0.00000003
cfg.DROP_OUT=0.1
cfg.BETA=0.0000001
cfg.temp = 0.2
cfg.THETA = 0.00007
cfg.L_P = 0.00001
cfg.L_G2 = 0.0001
cfg.L_D1 = 0.001
cfg.S2 = 0.05
cfg.RH2 = 50
cfg.mm = 5
cfg.MM=12
cfg.GAMA=0.018
cfg.GPU_ID = '0'
cfg.LR_G2 = 0.00001
cfg.LR_D2 = 0.00001
cfg.LR_D1 = 0.00001


cfg.LR = '[0.000102]*5000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 20
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048
cfg.BATCH_SIZE = 12
cfg.DATA_PATH = './data/THUMOS14'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 50
cfg.PRINT_FREQ = 100
cfg.CLASS_THRESH = 0.2
cfg.NMS_THRESH = 0.6
cfg.CAS_THRESH = np.arange(0.0, 0.25, 0.025)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.UP_SCALE = 24
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')
cfg.SEED = 0
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 750
cfg.CLASS_DICT = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 
                  'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 
                  'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 
                  'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 
                  'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 
                  'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 
                  'ThrowDiscus': 18, 'VolleyballSpiking': 19}
