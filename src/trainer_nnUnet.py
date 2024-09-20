# File: nnUNetTrainerV2_WandB.py
# This file should be placed in the nnunet/training/network_training/ directory

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import wandb

class nnUNetTrainerV2_WandB(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.wandb_run = None

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        if training:
            self.wandb_run = wandb.init(project="BraTS2024_nnUNet", name=f"Task{self.dataset_directory.split('/')[-1]}_fold{self.fold}")

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.wandb_run:
            wandb.log({
                "epoch": self.epoch,
                "train_loss": self.all_tr_losses[-1],
                "val_loss": self.all_val_losses[-1],
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            if len(self.all_val_eval_metrics) > 0:
                wandb.log({f"val_metric_{i}": v for i, v in enumerate(self.all_val_eval_metrics[-1])})

    def run_training(self):
        try:
            super().run_training()
        finally:
            if self.wandb_run:
                wandb.finish()

# Main script for running the training
# This script should be run instead of using nnUNet_train directly

import os
from nnunet.run.run_training import run_training

if __name__ == "__main__":
    task = "Task901_BraTS2024_PostSurgery"  # Replace with your actual task name
    fold = 0  # Replace with the fold you want to run
    trainer = "nnUNetTrainerV2_WandB"
    
    os.environ['RESULTS_FOLDER'] = '/path/to/results'  # Replace with your results folder path
    
    run_training(task, fold, trainer)

# After training, you can use nnUNet's built-in evaluation scripts
# For example, you might run something like this in a separate script or command:
# nnUNet_evaluate_folder -ref /path/to/ground_truth -pred /path/to/predictions -l 1 2 3
