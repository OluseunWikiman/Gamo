
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from google.colab import files

# Upload files manually
print("📁 Please upload KDDTrain+.txt and KDDTest+.txt")
uploaded = files.upload()

# Confirm files
for fname in uploaded.keys():
    print(f"✅ Uploaded: {fname}")

# Load dataset
def load_nsl_kdd(train_path, test_path):
    print(f"Loading: {train_path}, {test_path}")
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    train_df.columns = [f'f{i}' for i in range(41)] + ['label']
    test_df.columns = [f'f{i}' for i in range(41)] + ['label']
    return train_df, test_df

# Preprocess NSL-KDD data
def preprocess_data(df):
    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    attack_map = {'normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
    y = y.map(lambda v: 'normal' if v == 'normal' else (
        'DoS' if v in ['neptune', 'smurf', 'back', 'teardrop', 'pod'] else (
        'Probe' if v in ['satan', 'ipsweep', 'nmap', 'portsweep'] else (
        'R2L' if v in ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy'] else 'U2R')))
    )
    y = y.map(attack_map)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

# Build Models
class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], dim=1)
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.model(x)

# Train
def train_gamo(train_data, input_dim, num_classes, epochs=20, batch_size=128):
    dataset = TensorDataset(*train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(input_dim, 10, num_classes)
    D = Discriminator(input_dim, num_classes)
    M = Classifier(input_dim, num_classes)

    adv_loss = nn.BCELoss()
    cls_loss = nn.CrossEntropyLoss()
    opt_G = optim.Adam(G.parameters(), lr=0.0005)
    opt_D = optim.Adam(D.parameters(), lr=0.001)
    opt_M = optim.Adam(M.parameters(), lr=0.001)

    for epoch in range(epochs):
        g_loss_total = d_loss_total = m_loss_total = 0
        for real_x, real_y in loader:
            bs = real_x.size(0)
            valid = torch.full((bs, 1), 0.9)
            fake = torch.full((bs, 1), 0.1)

            z = torch.randn(bs, 10)
            gen_y = torch.randint(0, num_classes, (bs,))
            gen_x = G(z, gen_y)

            d_real = D(real_x, real_y)
            d_fake = D(gen_x.detach(), gen_y)
            d_loss = (adv_loss(d_real, valid) + adv_loss(d_fake, fake)) / 2
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            g_pred = D(gen_x, gen_y)
            div_loss = -torch.var(gen_x, dim=0).mean()
            g_loss = adv_loss(g_pred, valid) + 0.1 * div_loss
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            pred = M(real_x)
            m_loss = cls_loss(pred, real_y)
            opt_M.zero_grad(); m_loss.backward(); opt_M.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
            m_loss_total += m_loss.item()

        print(f"Epoch {epoch+1}/{epochs} G:{g_loss_total:.2f} D:{d_loss_total:.2f} M:{m_loss_total:.2f}")

    return M

# Evaluate
def evaluate(model, test_data):
    model.eval()
    with torch.no_grad():
        X_test, y_test = test_data
        preds = model(X_test)
        y_pred = preds.argmax(dim=1).numpy()
        print(classification_report(y_test.numpy(), y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test.numpy(), y_pred))

# Execute
train_df, test_df = load_nsl_kdd("KDDTrain+.txt", "KDDTest+.txt")
train_data = preprocess_data(train_df)
test_data = preprocess_data(test_df)
input_dim = train_data[0].shape[1]

model = train_gamo(train_data, input_dim, num_classes=5, epochs=10)
evaluate(model, test_data)
