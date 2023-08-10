import os

class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        self.workspace_dir = 'workspace_test'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = 'tensorboard_test'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir    # Directory for saving other models pre-trained networks
        self.eval_dir = 'eval_test'    # Base directory for saving the evaluations. 
        self.log_dir = 'log_test'
        self.llff = '/data/zhangyichi/zhangyichi/nhaplus/sparf/data/llff'
        self.dtu = '/data/zhangyichi/zhangyichi/nhaplus/sparf/data/set0'
        self.dtu_depth = ''
        self.dtu_mask = '/data/zhangyichi/zhangyichi/nhaplus/sparf/data/set0'
        self.replica = ''
