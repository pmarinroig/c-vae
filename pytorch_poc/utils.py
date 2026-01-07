import torch
import matplotlib.pyplot as plt
import os

def visualize_reconstruction(model, dataloader, device, n=8):
    model.eval()
    with torch.no_grad():
        data = next(iter(dataloader)).to(device)
        recon, _, _ = model(data)
        
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().permute(1, 2, 0).numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("Original")

            # Reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon[i].cpu().permute(1, 2, 0).numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0: ax.set_title("Reconstructed")
        plt.show()

def generate_samples(model, device, latent_dim, n=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device)
        z_projected = model.decoder_input(z)
        sample = model.decoder(z_projected).cpu()
        
        plt.figure(figsize=(8, 8))
        for i in range(n):
            plt.subplot(4, 4, i + 1)
            plt.imshow(sample[i].permute(1, 2, 0).numpy())
            plt.axis('off')
        plt.suptitle("Generated Samples from Latent Space")
        plt.show()

def interpolate_samples(model, dataloader, device, steps=10):
    model.eval()
    with torch.no_grad():
        # Get two real images
        data = next(iter(dataloader)).to(device)
        img1 = data[0].unsqueeze(0)
        img2 = data[1].unsqueeze(0)
        
        # Encode them to latent space
        _, mu1, _ = model(img1)
        _, mu2, _ = model(img2)
        
        # Reconstruct originals to see how well the model captured them
        recon1_z = model.decoder_input(mu1)
        recon1 = model.decoder(recon1_z).cpu()
        
        recon2_z = model.decoder_input(mu2)
        recon2 = model.decoder(recon2_z).cpu()
        
        total_plots = steps + 4 # Orig1, Recon1, [steps], Recon2, Orig2
        
        plt.figure(figsize=(2 * total_plots, 4))
        
        # Plot Original 1
        ax = plt.subplot(1, total_plots, 1)
        plt.imshow(img1[0].cpu().permute(1, 2, 0).numpy())
        ax.set_title("Start (Orig)")
        plt.axis('off')
        
        # Plot Reconstruction 1
        ax = plt.subplot(1, total_plots, 2)
        plt.imshow(recon1[0].permute(1, 2, 0).numpy())
        ax.set_title("Start (Recon)")
        plt.axis('off')

        # Interpolate
        for i in range(steps):
            alpha = (i + 1) / (steps + 1) # distribute steps evenly between 0 and 1 exclusive
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            
            # Decode
            z_projected = model.decoder_input(z_interp)
            recon_interp = model.decoder(z_projected).cpu()
            
            ax = plt.subplot(1, total_plots, i + 3)
            plt.imshow(recon_interp[0].permute(1, 2, 0).numpy())
            plt.axis('off')
            
        # Plot Reconstruction 2
        ax = plt.subplot(1, total_plots, total_plots - 1)
        plt.imshow(recon2[0].permute(1, 2, 0).numpy())
        ax.set_title("End (Recon)")
        plt.axis('off')
        
        # Plot Original 2
        ax = plt.subplot(1, total_plots, total_plots)
        plt.imshow(img2[0].cpu().permute(1, 2, 0).numpy())
        ax.set_title("End (Orig)")
        plt.axis('off')
            
        plt.suptitle("Interpolation in Latent Space")
        plt.show()

def get_average_embedding(model, dataset, keyword, device):
    """Computes the average latent vector (mu) for all images matching the keyword."""
    model.eval()
    images = dataset.get_images_by_keyword(keyword)
    if not images:
        print(f"No images found for keyword: {keyword}")
        return None
        
    # Stack images into a batch
    batch = torch.stack(images).to(device)
    
    with torch.no_grad():
        _, mu, _ = model(batch)
        # Compute mean across the batch dimension
        avg_mu = torch.mean(mu, dim=0)
        
    return avg_mu

def latent_arithmetic(model, dataset, device, base_item_name, minus_keyword, plus_keyword):
    """
    Performs: Base Item - Avg(Minus) + Avg(Plus)
    Example: Iron Chestplate - Iron + Diamond -> Diamond Chestplate?
    """
    model.eval()
    
    # Get embeddings
    base_img = dataset.get_image_by_name(base_item_name)
    if base_img is None:
        print(f"Base item not found: {base_item_name}")
        return

    base_batch = base_img.unsqueeze(0).to(device)
    
    minus_vec = get_average_embedding(model, dataset, minus_keyword, device)
    plus_vec = get_average_embedding(model, dataset, plus_keyword, device)
    
    if minus_vec is None or plus_vec is None:
        return

    with torch.no_grad():
        _, base_mu, _ = model(base_batch)
        base_mu = base_mu.squeeze(0)
        
        # Arithmetic
        result_vec = base_mu - minus_vec + plus_vec
        
        # Decode all vectors for visualization
        # Base
        base_recon = model.decoder(model.decoder_input(base_mu.unsqueeze(0))).cpu()
        
        # Minus Avg (Visualizing what the "concept" looks like)
        minus_recon = model.decoder(model.decoder_input(minus_vec.unsqueeze(0))).cpu()
        
        # Plus Avg
        plus_recon = model.decoder(model.decoder_input(plus_vec.unsqueeze(0))).cpu()
        
        # Result
        result_recon = model.decoder(model.decoder_input(result_vec.unsqueeze(0))).cpu()
        
        # Plotting
        plt.figure(figsize=(16, 4))
        
        # Base Original
        plt.subplot(1, 5, 1)
        plt.imshow(base_img.permute(1, 2, 0).numpy())
        plt.title(f"Base:\n{base_item_name}")
        plt.axis('off')
        
        # Base Recon
        plt.subplot(1, 5, 2)
        plt.imshow(base_recon[0].permute(1, 2, 0).numpy())
        plt.title("Base (Recon)")
        plt.axis('off')
        
        # Minus Concept
        plt.subplot(1, 5, 3)
        plt.imshow(minus_recon[0].permute(1, 2, 0).numpy())
        plt.title(f"Minus:\n{minus_keyword}")
        plt.axis('off')
        
        # Plus Concept
        plt.subplot(1, 5, 4)
        plt.imshow(plus_recon[0].permute(1, 2, 0).numpy())
        plt.title(f"Plus:\n{plus_keyword}")
        plt.axis('off')
        
        # Result
        plt.subplot(1, 5, 5)
        plt.imshow(result_recon[0].permute(1, 2, 0).numpy())
        plt.title("Result")
        plt.axis('off')
        
        plt.show()

def save_model(model, path="vae_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="vae_model.pth", device="cpu"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"Model loaded from {path}")
        return True
    else:
        print(f"Model file {path} not found.")
        return False
