# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

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
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from hydra.core.singleton import Singleton

log = logging.getLogger(__name__)


@dataclass
class BasicLauncherConf:
    _target_: str = "hydra_plugins.parallel_launcher_joblib.ParallelLauncher"


ConfigStore.instance().store(
    group="hydra/launcher", name="gpu_parallel_joblib", node=BasicLauncherConf, provider="hydra"
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

        print('asd', job_overrides)
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

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

        def execute_job(
                idx: int,
                overrides: Sequence[str],
                config_loader: ConfigLoader,
                config: DictConfig,
                task_function: TaskFunction,
                singleton_state: Dict[Any, Any],
        ) -> JobReturn:
            """Calls `run_job` in parallel"""
            setup_globals()
            Singleton.set_state(singleton_state)

            sweep_config = config_loader.load_sweep_config(config, list(overrides))
            with open_dict(sweep_config):
                sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
                sweep_config.hydra.job.num = idx
            HydraConfig.instance().set_config(sweep_config)

            ret = run_job(
                config=sweep_config,
                task_function=task_function,
                job_dir_key="hydra.sweep.dir",
                job_subdir_key="hydra.sweep.subdir",
            )

            return ret

        # Setup loading threshold
        self.config.multirun.max_usage = int(self.config.multirun.max_usage)
        while job_overrides:
            next_free_gpu = free_gpu(gpus_real_stats, self.config, gpus_id)
            print(f"Next gpus is {next_free_gpu}")
            started_programs = []
            if next_free_gpu == -1:
                time.sleep(60)
                continue
            else:
                idx, overrides = len(jobs_started), job_overrides[0]
                log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))
                sweep_config = self.config_loader.load_sweep_config(self.config, list(overrides))
                with open_dict(sweep_config):
                    sweep_config.hydra.job.id = idx
                    sweep_config.hydra.job.num = idx
                HydraConfig().set_config(sweep_config)

                singleton_state = Singleton.get_state()

                # Insert the free GPU that is going to be used
                sweep_config.device = f"cuda:{next_free_gpu}"
                print(sweep_config.device)
                print(sweep_config.keys())
                p = mp.Process(target=execute_job, args=(
                                                  initial_job_idx + idx,
                                                  overrides,
                                                  self.config_loader,
                                                  # self.config,
                                                  sweep_config,
                                                  self.task_function,
                                                  singleton_state,
                                                  # sweep_config,
                                                  # self.task_function,
                                                  # "hydra.sweep.dir",
                                                  # "hydra.sweep.subdir",
                ))
                p.start()

                started_programs.append(p)
                jobs_started.append(job_overrides.pop(0))
                configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
                time.sleep(15)

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
