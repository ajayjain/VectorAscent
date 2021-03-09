"""
Author: Ajay Jain
Wrapper functions for OpenAI's CLIP model.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import clip


USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_NORMALIZE = torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD)  # normalize an image that is already scaled to [0, 1]


clip_model_vit, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
clip_model_rn, _ = clip.load("RN50", device=DEVICE, jit=False)
clip_model_vit.eval()
clip_model_rn.eval()


@torch.no_grad()
def embed_text(text: str):
    # Embed text
    assert isinstance(text, str)
    text = clip.tokenize(text).to(DEVICE)
    text_features_vit = clip_model_vit.encode_text(text)  # [1, 512]
    text_features_rn = clip_model_rn.encode_text(text)  # [1, 512]
    return torch.cat([text_features_vit, text_features_rn], dim=-1)


def rgba_to_rgb(rgba_image):
    # TODO: Try just taking the first 3 channels
    return rgba_image[:, :, 3:4] * rgba_image[:, :, :3] + torch.ones(rgba_image.shape[0], rgba_image.shape[1], 3, device=DEVICE) * (1 - rgba_image[:, :, 3:4])


def embed_image(image):
    # Convert and normalize image
    image = rgba_to_rgb(image)
    assert image.shape[0] == 224 and image.shape[1] == 224
    image = image.permute(2, 0, 1).unsqueeze(0)  # [224, 224, 3] to [1, 3, 224, 224]
    image = CLIP_NORMALIZE(image.to(DEVICE))

    # Embed
    image_features_vit = clip_model_vit.encode_image(image)  # [1, 512]
    image_features_rn = clip_model_rn.encode_image(image)  # [1, 512]
    return torch.cat([image_features_vit, image_features_rn], dim=-1)


def plot_losses(losses, dir):
    plt.figure()
    plt.plot(-np.array(losses))
    plt.xlabel('Iteration')
    plt.ylabel('Cosine similarity')
    plt.savefig(os.path.join(dir, 'cosine_sim.pdf'))
    plt.savefig(os.path.join(dir, 'cosine_sim.png'))
