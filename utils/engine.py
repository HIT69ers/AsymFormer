import os
import torch
from utils.utils import save_ckpt_new


class Engine(object):
    def __init__(self, logger):
        self.checkpoint_state = []
        self.logger = logger

        ############# Distributed Training Config
        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1

        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.devices = [0, 1]
        else:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.devices = [0, 1]  # parse_devices(self.args.devices)

    def save_and_remove(self, epoch, miou, ckpt_dir, model, optimizer, global_step):
        self.checkpoint_state.append(dict(epoch=epoch, miou=miou))
        self.checkpoint_state.sort(key=lambda x: x["miou"], reverse=True)
        if len(self.checkpoint_state) > 5:
            try:
                ckpt_model_filename = f"epoch-{self.checkpoint_state[-1]['epoch']}_miou-{self.checkpoint_state[-1]['miou']}.pth"
                ckpt_path = os.path.join(ckpt_dir, ckpt_model_filename)
                os.remove(ckpt_path)
                self.logger.info(f"remove inferior checkpoint: {self.checkpoint_state[-1]}")
            except:
                pass
            self.checkpoint_state.pop()
        save_ckpt_new(ckpt_dir, model, optimizer, global_step, epoch, 0, 1, miou)