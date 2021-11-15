# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, open_dict

from hydra.core.config_loader import ConfigLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction

from IPython import embed
try:
    import pynvml as nvidia_smi
except:
    print("Nvidia is not supported!!!")
import time
# from multiprocessing import Process
# import multiprocessing as mp
import multiprocess as mp # used since this module does not use pickle to serialize the function (problems with hydra wrappers)
# Remember!!!! You get error if you try to import a module before the main decorator which is accessing to torch devices...
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

log = logging.getLogger(__name__)


def worker(fn, cfg, idx):
    logging.info(f'Worker id:  {idx}')
    # time.sleep(30)
    return fn(cfg)


@dataclass
class BasicLauncherConf:
    _target_: str = "hydra_plugins.parallel_launcher.ParallelLauncher"


ConfigStore.instance().store(
    group="hydra/launcher", name="gpu_parallel", node=BasicLauncherConf, provider="hydra"
)

class ParallelLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__()
        self.config: Optional[DictConfig] = None
        self.config_loader: Optional[ConfigLoader] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(
        self,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        setup_globals()
        assert self.config is not None
        assert self.task_function is not None
        assert self.config_loader is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = self.config.hydra.sweep.dir
        Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
        log.info(f"Launching {len(job_overrides)} jobs locally")

        #Setup parallel run. gpus_id is a list of (absolute) ids of GPUs
        jobs_started, runs = [], []

        nvidia_smi.nvmlInit()
        try:
            gpus_id = list(map(int, self.config.multirun.devices.split(",")))
        except:
            if self.config.multirun.devices == "all":
                gpus_id = range(nvidia_smi.nvmlDeviceGetCount())
            else:
                raise RuntimeError("Use correct yaml to choose GPUs")

        # TODO the following can be improved. The list of objects gpus should contain also the ids
        gpus_real_stats = []
        for gpu_id in gpus_id:
            gpus_real_stats.append(nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id))

        def get_gpus_usage(gpus_real_stats, cfg):
            usage = []
            if cfg.multirun.gpu_bottleneck == "usage":
                for g in gpus_real_stats:
                    usage.append(nvidia_smi.nvmlDeviceGetUtilizationRates(g).gpu)
            elif cfg.multirun.gpu_bottleneck == "memory":
                for g in gpus_real_stats:
                    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(g)
                    usage.append(100 * (mem_res.used / mem_res.total))
            elif cfg.multirun.gpu_bottleneck == "tasks":
                for g in gpus_real_stats:
                    usage.append(len(nvidia_smi.nvmlDeviceGetComputeRunningProcesses(g)))
            else:
                raise NotImplementedError("The scheduler does not support this bottleneck!!!")

            return usage

        def free_gpu(gpus_real_stats, cfg, gpus_id):
            usage = get_gpus_usage(gpus_real_stats, cfg)
            min_usage = usage.index(min(usage))
            if usage[min_usage] >= cfg.multirun.max_usage:
                return -1
            else:
                return gpus_id[min_usage]

        # Setup loading threshold
        self.config.multirun.max_usage = int(self.config.multirun.max_usage)
        while job_overrides:
            next_free_gpu = free_gpu(gpus_real_stats, self.config, gpus_id)
            print(f"Next gpus is {next_free_gpu}")
            started_programs = []
            if next_free_gpu == -1:
                time.sleep(self.config.multirun.wait_for_next_available_gpu)
                continue
            else:
                idx, overrides = len(jobs_started), job_overrides[0]
                log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))
                sweep_config = self.config_loader.load_sweep_config(self.config, list(overrides))
                with open_dict(sweep_config):
                    sweep_config.hydra.job.id = idx
                    sweep_config.hydra.job.num = idx
                HydraConfig().set_config(sweep_config)

                # Insert the free GPU that is going to be used
                sweep_config.device = f"cuda:{next_free_gpu}"

                # p = run_job(sweep_config,
                #           self.task_function,
                #           "hydra.sweep.dir",
                #           "hydra.sweep.subdir",)

                # # self.task_function(sweep_config)
                # print(self.task_function)
                # print(worker)
                # p = mp.Process(target=self.task_function, args=(sweep_config,))
                # p.start()

                p = mp.Process(target=run_job, args=(sweep_config,
                                                  self.task_function,
                                                  "hydra.sweep.dir",
                                                  "hydra.sweep.subdir",))
                p.start()

                # print('function name: ', self.task_function.__name__)
                # p = mp.Process(target=worker, args=(self.task_function, sweep_config, idx,))
                # p.start()

                started_programs.append(p)
                jobs_started.append(job_overrides.pop(0))
                configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
                # time.sleep(15)
                time.sleep(self.config.multirun.wait_to_start_new_job)

        for p in started_programs:
            p.join()
        return runs



        # print("parallel_launcher_luca")
        # embed()
        #
        # #####################
        # runs: List[JobReturn] = []
        # for idx, overrides in enumerate(job_overrides):
        #     idx = initial_job_idx + idx
        #     lst = " ".join(filter_overrides(overrides))
        #     log.info(f"\t#{idx} : {lst}")
        #     sweep_config = self.config_loader.load_sweep_config(
        #         self.config, list(overrides)
        #     )
        #     with open_dict(sweep_config):
        #         sweep_config.hydra.job.id = idx
        #         sweep_config.hydra.job.num = idx
        #     HydraConfig.instance().set_config(sweep_config)
        #     ret = run_job(
        #         config=sweep_config,
        #         task_function=self.task_function,
        #         job_dir_key="hydra.sweep.dir",
        #         job_subdir_key="hydra.sweep.subdir",
        #     )
        #     runs.append(ret)
        #     configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        # return runs
