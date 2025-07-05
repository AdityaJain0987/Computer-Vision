import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image loader
imsize = 356
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# VGG model for intermediate feature outputs
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "21", "28"]
        self.model = models.vgg19(pretrained=True).features[:29]

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# Load input images
original_img = load_image("content.png")
style_img = load_image("style.jpg")
generated = original_img.clone().to(device).requires_grad_(True)

# Hyperparameters
total_steps = 6000
learning_rate = 0.007
alpha = 1
beta = 1e6

# Style layer weights
style_weights = {
    "0": 0.2,
    "5": 0.2,
    "10": 0.2,
    "19": 0.2,
    "28": 0.2
}
content_layer = "21"

# Model and optimizer
model = VGG().to(device).eval()
optimizer = optim.Adam([generated], lr=learning_rate)

# Create output directory
os.makedirs("output", exist_ok=True)

# Training loop
for step in range(total_steps):
    generated_features = model(generated)
    original_features = model(original_img)
    style_features = model(style_img)

    # Content loss from conv4_2
    content_index = model.chosen_features.index(content_layer)
    gen_content = generated_features[content_index]
    orig_content = original_features[content_index]
    _, c, h, w = gen_content.shape
    content_loss = (1 / (4 * c * h * w)) * torch.sum((gen_content - orig_content) ** 2)

    # Style loss from multiple layers
    style_loss = 0
    for i, (gen_feat, style_feat) in enumerate(zip(generated_features, style_features)):
        layer = model.chosen_features[i]
        if layer not in style_weights:
            continue

        _, c, h, w = gen_feat.shape
        G = gen_feat.view(c, h * w)
        A = style_feat.view(c, h * w)

        gram_G = G @ G.t()
        gram_A = A @ A.t()

        factor = 1 / (4 * (c * h * w) ** 2)
        style_loss += style_weights[layer] * factor * torch.sum((gram_G - gram_A) ** 2)

    # Total loss
    total_loss = alpha * content_loss + beta * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item():.4f}")
        save_image(generated.detach().cpu(), f"output/generated_{step}.png")

# Save final output
save_image(generated.detach().cpu(), "output/final_generated_image.png")
