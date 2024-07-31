import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def show_images(images, rows=5, cols=10, save_path=None):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_model(model, model_weight_file, device):
    if os.path.exists(model_weight_file):
        print(f'Loading model weights from {model_weight_file}')
        model.load_state_dict(torch.load(model_weight_file, map_location=device))
        model.eval()
        return True
    else:
        print('No pre-trained model weights found. Training from scratch.')
        return False

def train_model(model, dataloader, diffuser, optimizer, epochs, device, num_timesteps, model_dir, image_dir):
    losses = []
    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0

        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch {epoch} | Loss: {loss_avg}')

        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch}.pth'))
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_epoch_latest.pth'))

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(image_dir, 'losses.png'))
    plt.show()

def generate_samples(model, diffuser, device, image_dir, pretrained=False):
    x_samples = diffuser.sample(model, x_shape=(5000, 1, 28, 28))
    images = diffuser.reverse_to_img(x_samples[:50])

    if pretrained:
        image_save_path = os.path.join(image_dir, 'generated_images_pretrained.png')
    else:
        image_save_path = os.path.join(image_dir, 'generated_images.png')

    show_images(images, save_path=image_save_path)
    return x_samples
