import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, 7, stride=1, padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        model += [nn.Conv2d(out_features, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0)

def deprocess_image(tensor):
    inv_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage(),
    ])
    image = tensor.squeeze(0)
    return inv_transform(image)

def split_image(image, patch_size=256, overlap=32):
    w, h = image.size
    patches = []
    for i in range(0, h, patch_size - overlap):
        for j in range(0, w, patch_size - overlap):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches

def weighted_blend(patch1, patch2, alpha):
    np_patch1 = np.array(patch1).astype(float)
    np_patch2 = np.array(patch2).astype(float)
    blended_patch = np_patch1 * (1 - alpha) + np_patch2 * alpha
    return Image.fromarray(np.uint8(blended_patch))

def combine_patches(patches, image_size, patch_size=256, overlap=32):
    w, h = image_size
    new_image = Image.new('RGB', (w, h))
    patch_count_x = (w + patch_size - 1) // (patch_size - overlap)
    patch_count_y = (h + patch_size - 1) // (patch_size - overlap)
    
    patch_index = 0
    for i in range(patch_count_y):
        for j in range(patch_count_x):
            x = j * (patch_size - overlap)
            y = i * (patch_size - overlap)
            patch = patches[patch_index]
            patch_index += 1
            
            patch_width = min(patch_size, w - x)
            patch_height = min(patch_size, h - y)
            patch = patch.crop((0, 0, patch_width, patch_height))
                
            if j > 0:
                left_patch = new_image.crop((x, y, x + overlap, y + patch_height))
                blended_patch = weighted_blend(left_patch, patch.crop((0, 0, overlap, patch_height)), np.linspace(0, 1, overlap).reshape((1, overlap, 1)))
                new_image.paste(blended_patch, (x, y))
                
            if i > 0:
                top_patch = new_image.crop((x, y, x + patch_width, y + overlap))
                blended_patch = weighted_blend(top_patch, patch.crop((0, 0, patch_width, overlap)), np.linspace(0, 1, overlap).reshape((overlap, 1, 1)))
                new_image.paste(blended_patch, (x, y))
                
            new_image.paste(patch, (x, y, x + patch_width, y + patch_height))
    
    return new_image

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
print(f"Using device: {device}")

input_shape = (3, 256, 256)
G = GeneratorResNet(input_shape, num_residual_blocks=9)

checkpoint_path = 'models/cycle_gan_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

state_dict = checkpoint['G_state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

G.load_state_dict(new_state_dict)
G.to(device)
G.eval()

def process_with_cyclegan(input_image: Image.Image) -> Image.Image:
    patches = split_image(input_image)

    processed_patches = []
    for patch in patches:
        input_patch = preprocess_image(patch)
        input_patch = input_patch.to(device)  

        with torch.no_grad():
            output_patch = G(input_patch)

        output_patch = deprocess_image(output_patch.cpu())
        processed_patches.append(output_patch)

    output_image = combine_patches(processed_patches, input_image.size)

    output_image = np.array(output_image)
    image_after_filter = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return Image.fromarray(cv2.cvtColor(image_after_filter, cv2.COLOR_BGR2RGB))



