import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.UNet import UNet
from utils.Diffuser import DDPMDiffuser
from utils.lib import load_model, train_model, generate_samples
from utils.FID import calculate_fid_for_dataset_and_generated_images

if __name__ == "__main__":
    model_dir = './fashion_ddpm_models'
    image_dir = './fashion_ddpm_images'
    batch_size = 128
    num_timesteps = 1000
    epochs = 100
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.FashionMNIST(root='./data', download=True, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    diffuser = DDPMDiffuser(num_timesteps, device=device)
    model = UNet()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    model_weight_file = os.path.join(model_dir, 'model_epoch_latest.pth')

    pretrained = load_model(model, model_weight_file, device)
    
    if pretrained:
        print('Generating samples using pre-trained model weights...')
        generated_samples = generate_samples(model, diffuser, device, image_dir, pretrained=True)
    else:
        train_model(model, dataloader, diffuser, optimizer, epochs, device, num_timesteps, model_dir, image_dir)
        generated_samples = generate_samples(model, diffuser, device, image_dir, pretrained=False)

    # FIDの計算
    fid = calculate_fid_for_dataset_and_generated_images(dataset, generated_samples, diffuser)
    print(f'FID: {fid}')
