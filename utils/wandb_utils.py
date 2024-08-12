"""
Utilities for Weight and Biases.
"""
import wandb

def log_model_performance(model_name, metrics):
    wandb.init(project="brain-tumor-segmentation", name=f"{model_name}_evaluation")
    wandb.log(metrics)
    wandb.finish()

def log_image(image_name, image):
    wandb.log({image_name: wandb.Image(image)})