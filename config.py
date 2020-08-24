__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

from torch.autograd import Variable
import torch
import torch.nn as nn



class HParameters:

    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.l2_req = 0.00001
        self.lr_epochs = [0]
        self.lr = [0.00005]

        self.epochs_max = 5
        self.train_batch_size = 1

        self.output_dir = 'ex-10'

        self.root = ''
        self.datasets=['/content/trans-vsumm/datasets/eccv16_dataset_summe_google_pool5.h5',
                       '/content/trans-vsumm/datasets/eccv16_dataset_tvsum_google_pool5.h5',
                       '/content/trans-vsumm/datasets/eccv16_dataset_ovp_google_pool5.h5',
                       '/content/trans-vsumm/datasets/eccv16_dataset_youtube_google_pool5.h5']

        self.splits = ['/content/trans-vsumm/splits/tvsum_splits.json',
                        '/content/trans-vsumm/splits/summe_splits.json']

        self.splits += ['/content/trans-vsumm/splits/tvsum_aug_splits.json',
                        '/content/trans-vsumm/splits/summe_aug_splits.json']


        #### ses2seq network initialization #################################

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.INPUT_DIM = 1024
        self.OUTPUT_DIM = 1024
        self.HID_DIM = 1024 # 256
        self.ENC_LAYERS = 1 # TODO Changed the Encoder layer from 3 to 1
        self.DEC_LAYERS = 1 # TODO Changedd the Decoder layer from 3 to 1
        self.ENC_HEADS = 8
        self.DEC_HEADS = 8
        self.ENC_PF_DIM =  1024 # TODO Changed from 512 to 1024
        self.DEC_PF_DIM =  1024 # TODO Changed from 512 to 1024
        self.ENC_DROPOUT = 0.1
        self.DEC_DROPOUT = 0.1

        self.SRC_PAD_IDX = 0
        self.TRG_PAD_IDX = 0


        return


    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()

                setattr(self, key, val)

    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str


if __name__ == "__main__":

    # Tests
    hps = HParameters()
    print(hps)

    args = {'root': 'root_dir',
            'datasets': 'set1,set2,set3',
            'splits': 'split1, split2',
            'new_param_float': 1.23456
            }

    hps.load_from_args(args)
    print(hps)
