import os
import sys
import math
import pprint
import shutil
import argparse
import numpy as np

import torch
import torchdrug
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from NCprior import dataset, model, task, util

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="./config/vae_training.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    return parser.parse_known_args()[0]


def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # NOTE
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def build_solver(cfg, logger):

    # build dataset
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    if "test_split" in cfg:
        train_set, valid_set, test_set = _dataset.split(['train', 'valid', cfg.test_split])
    else:
        train_set, valid_set, test_set = _dataset.split()
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))
    
    #if cfg.dataset['class'] in ["MNIST"]:
    if cfg.task['class'] in ["ImageGeneration"]:
        cfg.task.model.in_channels = _dataset.in_channels
        cfg.task.model.image_rows = _dataset.image_rows
        cfg.task.model.image_cols = _dataset.image_cols
    elif cfg.task['class'] in ["PriorReweighing"]:
        cfg.task.vae_model.in_channels = _dataset.in_channels
        cfg.task.vae_model.image_rows = _dataset.image_rows
        cfg.task.vae_model.image_cols = _dataset.image_cols
        cfg.task.adv_model.in_channels = _dataset.in_channels
    # build task model
    task = core.Configurable.load_config_dict(cfg.task)

    # build solver
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)

    if "vae_checkpoint" in cfg:
        assert cfg.task['class'] in ["PriorReweighing"]
        checkpoint = os.path.expanduser(cfg.checkpoint)
        state = torch.load(checkpoint, map_location=solver.device)
        model_dict = solver.model.state_dict()
        keys = [k for k in state['model'].keys()]
        for k in keys:
            #new_k = k.replace("model.", "")
            new_k = "vae_" + k
            state['model'][new_k] = state['model'][k]
            state['model'].pop(k)
        solver.model.load_state_dict(state['model'], strict=False)
    
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)

    return solver

def train_and_validate(cfg, solver):

    step = math.ceil(cfg.train.num_epoch / 100)
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch
    
    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        if "test_batch_size" in cfg:
            solver.batch_size = cfg.test_batch_size
        print(solver.batch_size)
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        solver.batch_size = cfg.engine.batch_size

        if solver.epoch % 5 == 0:
            solver.save("model_epoch_%d.pth" % solver.epoch)

        '''score = []
        for k, v in metric.items():
            if k.startswith(cfg.eval_metric):
                if "root mean squared error" in cfg.eval_metric:
                    score.append(-v)
                else:
                    score.append(v)
        score = sum(score) / len(score)
        if score > best_score:
            best_score = score
            best_epoch = solver.epoch
            solver.save("best_valid_epoch.pth")'''

    #solver.load("best_valid_epoch.pth")
    return solver, best_epoch

def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")

    return


if __name__ == "__main__":

    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = util.load_config(args.config)

    set_seed(args.seed)
    output_dir = util.create_working_directory(cfg)
    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))
        logger.warning("Backup: from %s to %s" % (args.config, os.path.basename(args.config)))
    os.chdir(output_dir)

    solver = build_solver(cfg, logger)
    solver, best_epoch = train_and_validate(cfg, solver)
    if comm.get_rank() == 0:
        logger.warning("Best epoch on valid: %d" % best_epoch)
    test(cfg, solver)

    if "generate" in cfg:
        def generate_image(N=3):
            from matplotlib import pyplot as plt
            data = solver.model.sample(N*N)
            data = data.detach().cpu().repeat(1,3,1,1).numpy().transpose(0,2,3,1)
            data2 = solver.model.vae_model.sample(N*N)
            data2 = data2.detach().cpu().repeat(1,3,1,1).numpy().transpose(0,2,3,1)
            fig, axarr = plt.subplots(2*N, N)
            for i in range(N):
                for j in range(N):
                    axarr[i][j].imshow(data[i*N+j])
                    axarr[i+N][j].imshow(data2[i*N+j])
            plt.savefig("/home/xinyu402/projects/def-bengioy/xinyu402/Homeworks/IFT6269/NCprior/figs/gen.png")
        generate_image()
        ipdb.set_trace()