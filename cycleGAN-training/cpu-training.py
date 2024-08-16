import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import itertools
import os
from PIL import Image

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

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        # PatchGAN discriminator
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files_X = sorted(os.listdir(os.path.join(root, f'{mode}/night')))
        self.files_Y = sorted(os.listdir(os.path.join(root, f'{mode}/day')))
        self.root = root
        self.mode = mode

    def __getitem__(self, index):
        image_X = Image.open(os.path.join(self.root, f'{self.mode}/night/{self.files_X[index % len(self.files_X)]}')).convert('RGB')
        image_Y = Image.open(os.path.join(self.root, f'{self.mode}/day/{self.files_Y[index % len(self.files_Y)]}')).convert('RGB')

        if self.transform:
            image_X = self.transform(image_X)
            image_Y = self.transform(image_Y)

        return {'X': image_X, 'Y': image_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(dataloader))
    real_X = imgs['X']
    fake_Y = G(real_X)
    real_Y = imgs['Y']
    fake_X = F(real_Y)
    # Save sample images
    # Example: save_image(fake_Y, 'images/fake_Y_%d.png' % batches_done, nrow=5, normalize=True)
    pass

if __name__ == '__main__':
    transform = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Paths to night and day images
    dataloader = DataLoader(ImageDataset('images_subs', transforms_=transform), batch_size=8, shuffle=True, num_workers=4)

    input_shape = (3, 256, 256)

    # Initialize generator and discriminator
    G = GeneratorResNet(input_shape, num_residual_blocks=9)
    F = GeneratorResNet(input_shape, num_residual_blocks=9)
    D_X = Discriminator(input_shape)
    D_Y = Discriminator(input_shape)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Buffers of previously generated samples
    fake_X_buffer = []
    fake_Y_buffer = []

    # Calculate the output shape of D_X
    with torch.no_grad():
        example_data = torch.randn(1, *input_shape)
        output_shape = D_X(example_data).shape[1:]

    total_images_processed = 0
    # Training
    for epoch in range(500):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_X = batch['X']
            real_Y = batch['Y']

            # Adversarial ground truths
            valid = torch.ones((real_X.size(0), *output_shape), requires_grad=False)
            fake = torch.zeros((real_X.size(0), *output_shape), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_X = criterion_identity(F(real_X), real_X)
            loss_id_Y = criterion_identity(G(real_Y), real_Y)

            loss_identity = (loss_id_X + loss_id_Y) / 2

            # GAN loss
            fake_Y = G(real_X)
            loss_GAN_XY = criterion_GAN(D_Y(fake_Y), valid)
            fake_X = F(real_Y)
            loss_GAN_YX = criterion_GAN(D_X(fake_X), valid)

            loss_GAN = (loss_GAN_XY + loss_GAN_YX) / 2

            # Cycle loss
            recovered_X = F(fake_Y)
            loss_cycle_XYX = criterion_cycle(recovered_X, real_X)
            recovered_Y = G(fake_X)
            loss_cycle_YXY = criterion_cycle(recovered_Y, real_Y)

            loss_cycle = (loss_cycle_XYX + loss_cycle_YXY) / 2

            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator X
            # -----------------------

            optimizer_D_X.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_X(real_X), valid)
            # Fake loss (on batch of previously generated samples)
            fake_X = fake_X_buffer.pop(0) if len(fake_X_buffer) > 0 else fake_X
            loss_fake = criterion_GAN(D_X(fake_X.detach()), fake)

            # Total loss
            loss_D_X = (loss_real + loss_fake) / 2

            loss_D_X.backward()
            optimizer_D_X.step()

            # -----------------------
            #  Train Discriminator Y
            # -----------------------

            optimizer_D_Y.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_Y(real_Y), valid)
            # Fake loss (on batch of previously generated samples)
            fake_Y = fake_Y_buffer.pop(0) if len(fake_Y_buffer) > 0 else fake_Y
            loss_fake = criterion_GAN(D_Y(fake_Y.detach()), fake)

            # Total loss
            loss_D_Y = (loss_real + loss_fake) / 2

            loss_D_Y.backward()
            optimizer_D_Y.step()

            # Buffer fake samples
            fake_X_buffer.append(fake_X)
            fake_Y_buffer.append(fake_Y)

            total_images_processed += real_X.size(0)
            total_batches = len(dataloader)
            print(f'[Epoch {epoch}/{200}] [Batch {i+1}/{total_batches}] '
                  f'[D loss: {loss_D_X.item() + loss_D_Y.item()}] '
                  f'[G loss: {loss_G.item()}] '
                  f'[Total images processed: {total_images_processed}]')

            
            if i % 100 == 0:
                sample_images(epoch * len(dataloader) + i)


                