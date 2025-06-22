import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28 + 10, 400)
        self.fc21 = torch.nn.Linear(400, 20)
        self.fc22 = torch.nn.Linear(400, 20)
        self.fc3 = torch.nn.Linear(20 + 10, 400)
        self.fc4 = torch.nn.Linear(400, 28 * 28)

    def encode(self, x, label):
        x = torch.cat([x, label], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, label):
        z = torch.cat([z, label], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def generate(self, digit, n=5):
        device = torch.device("cpu")
        z = torch.randn(n, 20).to(device)
        one_hot = torch.nn.functional.one_hot(torch.tensor([digit] * n), num_classes=10).float().to(device)
        imgs = self.decode(z, one_hot)
        return imgs.view(-1, 28, 28).detach().cpu().numpy()

# Load the model
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location="cpu"))
model.eval()

# Web UI
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit", list(range(10)))

if st.button("Generate Images"):
    with st.spinner("Generating..."):
        images = model.generate(digit, 5)
        fig, axs = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axs[i].imshow(images[i], cmap="gray")
            axs[i].axis('off')
        st.pyplot(fig)
