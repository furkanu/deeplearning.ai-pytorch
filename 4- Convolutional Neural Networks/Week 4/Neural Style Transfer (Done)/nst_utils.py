import torch.nn as nn
from torchvision import models
import numpy as np
import imageio



class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    STYLE_IMAGE = 'images/stone_style.jpg' # Style image to use.
    CONTENT_IMAGE = 'images/content300.jpg' # Content image to use.
    OUTPUT_DIR = 'output/'


class VGG19_StyleTransfer(nn.Module):
    def __init__(self, layers=[None]):
        super().__init__()
        self.layers = layers
        layers_needed = max(layers)+1 if max(layers) is not None else None
        features = list(models.vgg19(pretrained=True).features)[:layers_needed]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = {}
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.layers:
                results[i] = x

        return results



def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.COLOR_CHANNELS, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """

    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))

    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS

    # Tranpose image to match the PyTorch format: N, C, H, W
    image = np.transpose(image, (0, 3, 1, 2))

    return image

