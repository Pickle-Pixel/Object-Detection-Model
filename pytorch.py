import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import torch.onnx

# ---- Check GPU ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Check PyTorch installation or configuration.")

# ---- Configuration ----
BATCH_SIZE = 512  # Start with a large batch size
IMG_HEIGHT, IMG_WIDTH = 100, 100
NUM_WORKERS = 4
EPOCHS = 500
MODEL_SAVE_PATH = "trained_model.pth"
ONNX_MODEL_PATH = "trained_model.onnx"  # Path for saving ONNX model

# ---- Data Transforms (Including Augmentation) ----
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

# ---- Load Datasets ----
train_dataset = datasets.ImageFolder(root="DataSets", transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_dataset.classes
print(f"Class names: {class_names}")

# ---- Define Model with Dropout ----
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer to reduce overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# ---- Instantiate Model ----
model = ImageClassifier(len(class_names)).to(device)
print(model)

# ---- Define Loss, Optimizer, and Learning Rate Scheduler ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# ---- Mixed Precision Training (Optional) ----
scaler = GradScaler()

# ---- Training Loop with Early Stopping ----
best_val_loss = float('inf')
patience = 10  # Stop after 10 epochs with no improvement
patience_counter = 0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss, train_correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    # Validation phase
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()

    # Calculate average losses and accuracies
    train_loss /= len(train_loader)
    train_acc = train_correct / len(train_loader.dataset)
    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved best model at epoch {epoch+1}")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Learning rate scheduler
    scheduler.step(val_loss)

# Load the best model (if early stopping was triggered)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f"Final model loaded from {MODEL_SAVE_PATH}")

# ---- Convert PyTorch Model to ONNX ----
# Export the model to ONNX format
dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, verbose=True)
print(f"PyTorch model exported to {ONNX_MODEL_PATH}")
