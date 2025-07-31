import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, random_split
from ..architecture.geneattentionnet import GeneAttentionNet
from .datasets import GeneExpressionData
import pandas as pd
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        print(f"\n Epoch {epoch+1}/{num_epochs} -----------------------------------------------", flush=True)
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.7)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}', flush=True)
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}', flush=True)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Losses")
    plt.plot(val_losses, label="Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefg("training_plot.png")
    plt.show()
    
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

def train():
    # Hyperparameters
    df = pd.read_csv("processsed.csv")
    actual_num_genes = df.shape[1]-1
    print(f"Acutla num genes: {actual_num_genes}")
    num_genes = 1247  # Should match your actual gene count
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0001
    val_split = 0.2  # 20% for validation
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create pathway masks (placeholder - replace with real data)
    pathway_masks = {
        'inflammation': (torch.rand(num_genes) > 0.7).to(device),  # Example pathway
        'apoptosis': (torch.rand(num_genes) > 0.7).to(device),
        'oxidative_stress': (torch.rand(num_genes) > 0.7).to(device)
    }
    
    # Create PPI mask (placeholder - replace with real PPI data)
    ppi_mask = torch.rand(num_genes, num_genes) > 0.9  # Sparse connections
    ppi_mask = ppi_mask.float().to(device)
    
    # Initialize model
    model = GeneAttentionNet(
        num_genes=num_genes,
        pathway_masks=pathway_masks,
        ppi_mask=ppi_mask
    ).to(device)
    
    # Load and split dataset
    full_dataset = GeneExpressionData('processsed.csv', num_genes)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs, 
        learning_rate, 
        device
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), 'final_model.pth')
    print("Training complete. Models saved.")

if __name__ == '__main__':
    train()