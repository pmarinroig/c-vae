import torch.optim as optim
from model import loss_function

def train_vae(model, dataloader, epochs, lr, device):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    train_loss = []
    
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            overall_loss += loss.item()
        
        # Calculate average loss over the dataset
        # overall_loss is sum of batch averages, so divide by number of batches
        avg_loss = overall_loss / len(dataloader)
        train_loss.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
    return train_loss
