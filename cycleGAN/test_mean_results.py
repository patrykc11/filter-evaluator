import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Definicja modelu
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

        # Initial convolution block
        out_features = 64
        model = [
            nn.Conv2d(channels, out_features, 7, stride=1, padding=3),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features

        # Output layer
        model += [nn.Conv2d(out_features, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Ustawienia
input_shape = (3, 256, 256)
cuda = torch.cuda.is_available()

# Tworzenie instancji modelu
G = GeneratorResNet(input_shape, num_residual_blocks=9)

# Wczytywanie checkpointu
checkpoint_path = 'checkpoint_epoch_197_batch_0.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Wyciąganie tylko stanu generatora i ładowanie go do modelu
G.load_state_dict(checkpoint['G_state_dict'])
# Ustawienie modelu w tryb ewaluacji
G.eval()

# Przykład przetwarzania obrazu
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform(image).unsqueeze(0)  # Dodanie wymiaru batcha

def deprocess_image(tensor):
    inv_transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage(),
    ])
    image = tensor.squeeze(0)  # Usunięcie wymiaru batcha
    return inv_transform(image)

def split_image(image, patch_size=256, overlap=64):
    w, h = image.size
    patches = []
    for i in range(0, h, patch_size - overlap):
        for j in range(0, w, patch_size - overlap):
            box = (j, i, min(j + patch_size, w), min(i + patch_size, h))
            patch = image.crop(box)
            patches.append((patch, box))
    return patches

def combine_patches(patches, image_size, patch_size=256, overlap=64):
    w, h = image_size
    new_image = Image.new('RGB', (w, h))
    count_map = np.zeros((h, w), dtype=np.float32)
    for patch, (x, y, _, _) in patches:
        patch_array = np.array(patch)
        patch_w, patch_h = patch_array.shape[1], patch_array.shape[0]
        new_image.paste(patch, (x, y))
        count_map[y:y + patch_h, x:x + patch_w] += 1
    
    new_image_array = np.array(new_image, dtype=np.float32)
    new_image_array /= count_map[..., None]
    return Image.fromarray(new_image_array.astype(np.uint8))

# Przetwarzanie obrazu
input_image_path = '../images/original_night/image_1.png'
input_image = Image.open(input_image_path).convert('RGB')
patches = split_image(input_image)

processed_patches = []
for patch, box in patches:
    input_patch = preprocess_image(patch)
    if cuda:
        input_patch = input_patch.cuda()
    
    with torch.no_grad():
        output_patch = G(input_patch)
    
    output_patch = deprocess_image(output_patch.cpu())
    processed_patches.append((output_patch, box))

output_image = combine_patches(processed_patches, input_image.size)

# Wyświetlenie obrazów oryginalnego i przetworzonego obok siebie
def show_images(original, processed):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(processed)
    axs[1].set_title('Processed Image')
    axs[1].axis('off')
    
    plt.show()

# Wyświetlenie obrazów
show_images(input_image, output_image)