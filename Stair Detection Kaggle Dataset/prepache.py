import argparse
import sys

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import TrafficLightDataset
from util.logconf import logging
# from .model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)
def collate_fn(batch):
    return tuple(zip(*batch))

class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=64,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=2,
            type=int,
        )
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        for mode in ["train", "test"]:
            log.info("Prepping {} dataset")
            log.info(f"Mode {mode}")
            self.prep_dl = DataLoader(
            TrafficLightDataset(dataset_path=r"c:\users\alex\.cache\kagglehub\datasets\wjybuqi\traffic-light-detection-dataset\versions\4",
                                mode=mode
                               
                
            ),
                  batch_size=self.cli_args.batch_size,
                  num_workers=self.cli_args.num_workers,
                  collate_fn=collate_fn
        )

            batch_iter = enumerateWithEstimate(
                   self.prep_dl,
                   "Stuffing cache",
                   start_ndx=self.prep_dl.num_workers,

        )
            log.info("Dataset length:" + str(len(self.prep_dl)))
            for batch_ndx, batch_tup in batch_iter:
                  pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
