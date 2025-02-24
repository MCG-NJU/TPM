import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import tpm.tasks as tasks
from tpm.common.config import Config
from tpm.common.dist_utils import get_rank, init_distributed_mode
from tpm.common.logger import setup_logger
from tpm.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from tpm.common.registry import registry
from tpm.common.utils import now

# imports modules for registration
from tpm.datasets.builders import *
from tpm.models import *
from tpm.processors import *
from tpm.runners import *
from tpm.tasks import *

import mask_model


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", default = 'xxx/TPM/train_configs/policy_learning.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
   
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)

    model = mask_model.policy()

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
