import os
import sys
import time
import random
import logging

import yaml
import easydict
import jinja2

from torch import distributed as dist

from torchdrug.utils import comm


def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)

    return cfg

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger

def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    if cfg.task['class'] in ["ImageGeneration"]:
        output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))
    elif cfg.task['class'] in ["PriorReweighing"]:
        output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.vae_model["class"] + cfg.task.adv_model["class"] 
                              + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        raise NotImplementedError

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    if comm.get_rank() != 0:
        if local_rank is not None and comm.get_rank() != local_rank:
            pass
        else:
            with open(file_name, "r") as fin:
                output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    if local_rank is not None and comm.get_rank() != local_rank:
        pass
    else:
        os.chdir(output_dir)
    return output_dir

