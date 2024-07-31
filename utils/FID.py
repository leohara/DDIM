import torch
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

def calculate_fid(real_images, generated_images, batch_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    inception.eval()
    real_activations = get_activations(real_images, inception, batch_size, device)
    generated_activations = get_activations(generated_images, inception, batch_size, device)
    
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)
    return fid

def get_activations(images, model, batch_size, device):
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)
    activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if batch.size(1) == 1:  # グレースケールの修正
                batch = batch.repeat(1, 3, 1, 1)
            pred = model(batch)
            pred = pred.detach().cpu().numpy()
            activations.append(pred)
    
    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

def calculate_fid_for_dataset_and_generated_images(dataset, generated_samples, diffuser, batch_size=50):
    preprocess = transforms.Compose([
        transforms.Resize((75, 75)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print('Calculating FID...')
    real_images = [dataset.data[i].numpy() for i in range(len(dataset))]
    real_images = [Image.fromarray(img, mode='L').convert('RGB') for img in real_images]  # 3チャンネルに変換
    real_images = [preprocess(img) for img in real_images]

    generated_images = [img.convert('RGB') for img in diffuser.reverse_to_img(generated_samples)]  # 3チャンネルに変換
    generated_images = [preprocess(img) for img in generated_images]

    fid = calculate_fid(real_images, generated_images, batch_size=batch_size)
    return fid