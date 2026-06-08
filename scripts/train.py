"""
Training script for GAP.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import hydra
import torch
from omegaconf import OmegaConf
import pathlib

GAP_ROOT = str(pathlib.Path(__file__).parent.parent)

sys.path.append(GAP_ROOT)
sys.path.append(os.path.join(GAP_ROOT, "gap_policy"))

from torch.utils.data import DataLoader
import copy

import wandb
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import random

from gap_policy.dataset.gap_dataset import GAPDataset
from gap_policy.common.pytorch_util import dict_apply
from gap_policy.model.diffusion.ema_model import EMAModel
from gap_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="../gap_policy/config",
    config_name="GAP",
)
def main(cfg: OmegaConf):
    # Set random seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get task name and setting
    task_name = cfg.task_name
    setting = cfg.get("setting", "demo_clean")
    expert_data_num = cfg.expert_data_num
    observation_chunk = cfg.observation_chunk
    interval = cfg.interval
    model_3d = cfg.model_3d

    data_root = cfg.get("data_root", os.environ.get("GAP_DATA_ROOT", "data"))
    if not os.path.isabs(data_root):
        data_root = os.path.join(GAP_ROOT, data_root)
    zarr_path = os.path.join(
        data_root,
        f"{task_name}-{setting}-{expert_data_num}-{model_3d}-{observation_chunk}-{interval}.zarr",
    )

    cprint(f"[GAP Training]", "cyan", attrs=["bold"])
    cprint(f"  Task: {task_name}", "cyan")
    cprint(f"  Setting: {setting}", "cyan")
    cprint(f"  Expert data: {expert_data_num}", "cyan")
    cprint(f"  Zarr path: {zarr_path}", "cyan")
    cprint(f"  Seed: {seed}", "cyan")

    # Create dataset
    dataset = GAPDataset(
        zarr_path=zarr_path,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        seed=seed,
        val_ratio=0.0,
        max_train_episodes=expert_data_num,
        task_name=task_name,
    )

    # Get normalizer
    normalizer = dataset.get_normalizer()

    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=cfg.dataloader.shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    # Create policy
    cprint("\nCreating GAP policy...", "green")
    policy = hydra.utils.instantiate(cfg.policy)

    # Set normalizer
    policy.set_normalizer(normalizer)

    # Move to device
    device = torch.device(cfg.training.device)
    policy = policy.to(device)

    # Create EMA model
    ema: EMAModel = None
    if cfg.training.use_ema:
        ema_policy = copy.deepcopy(policy)
        ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)

    # Create optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
    )

    # Initialize wandb
    wandb_mode = cfg.logging.get("mode", "online")
    use_wandb = (not cfg.training.debug) and wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.logging.project,
            name=f"{cfg.name}_{task_name}_{setting}_{expert_data_num}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )
    if cfg.training.debug:
        cfg.training.num_epochs = 100
        cfg.training.max_train_steps = 10
        cfg.training.max_val_steps = 3
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1

    # Training loop
    cprint("\nStarting training...", "green", attrs=["bold"])
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        policy.train()

        epoch_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps:
                    break
                # Move batch to device
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # Forward pass
                loss, loss_dict = policy.compute_loss(batch)

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                    # Clip gradients
                    if cfg.training.get("clip_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            cfg.training.clip_grad_norm
                        )

                    # Update parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update EMA
                    if ema is not None:
                        ema.step(policy)

                    global_step += 1

                # Logging
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Log to wandb
                if use_wandb and (batch_idx % 10 == 0):
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    # Add loss_dict items with train/ prefix
                    for key, value in loss_dict.items():
                        log_dict[f"train/{key}"] = value
                    wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        cprint(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}", "yellow")

        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch,
            }, step=global_step)

        # Save checkpoint
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", f'{task_name}_{setting}_{expert_data_num}')
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch+1}.ckpt")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                'normalizer': normalizer.state_dict(),
            }

            if ema is not None:
                checkpoint["ema"] = ema.averaged_model.state_dict()

            torch.save(checkpoint, checkpoint_path)
            cprint(f"Saved checkpoint to {checkpoint_path}", "green")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
