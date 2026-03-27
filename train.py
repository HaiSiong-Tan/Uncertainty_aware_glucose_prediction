import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from data import prepare_dataset, compute_means_variances, normalize_features, BGDataset
from utils import evidential_data_loss, kl_reg_loss_term, amini_reg_loss_term
from model import e_Transformers
from configs import (PATIENT_IDS, FEATURES, COLUMN_MAP, BATCH_SIZE, DATA_DIR, HORIZON_LENGTH,
                      NUM_EPOCHS, LEARNING_RATE, LAMBDA_REG,
                      REG_TERM, MAX_LEN, D_MODEL, FF_DIM,
                      N_HEADS, NUM_LAYERS, MODEL_SAVE_PATH, LOSS_SAVE_PATH)




def train_evidential_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=400,
    lr=1e-4,
    lambda_reg=0.01,
    reg_term='kl',
    scheduler_lambda=None,
    loss_save_path=None,
    model_save_path=None
):
    """
    Train and save an evidential Transformer model 

    Returns:
        model: trained PyTorch model
        training_loss, validation_loss: lists of loss per epoch
        rel_tol: relative tolerance list
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=scheduler_lambda if scheduler_lambda else (lambda e: 1.0)
    )

    training_loss = []
    validation_loss = []

    for epoch in range(1, num_epochs + 1):
        # --- training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # model forward
            L_p_alpha, L_p_beta, L_p_nu, L_pred = model(X_batch)

            # evidential loss
            loss = evidential_data_loss(L_pred, y_batch, L_p_nu, L_p_alpha, L_p_beta)

            # regularization
            if reg_term == 'kl':
                reg_loss = kl_reg_loss_term(y_batch, L_pred, L_p_alpha, L_p_beta)
            else:
                reg_loss = amini_reg_loss_term(y_batch, L_pred, L_p_nu, L_p_alpha)

            total_loss = loss + lambda_reg * reg_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        training_loss.append(train_loss)
        scheduler.step()

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                L_v_alpha, L_v_beta, L_v_nu, L_v_pred = model(X_val)

                v_loss = evidential_data_loss(L_v_pred, y_val, L_v_nu, L_v_alpha, L_v_beta)

                if reg_term == 'kl':
                    vreg_loss = kl_reg_loss_term(y_val, L_v_pred, L_v_alpha, L_v_beta)
                else:
                    vreg_loss = amini_reg_loss_term(y_val, L_v_pred, L_v_nu, L_v_alpha)

                total_v_loss = v_loss + lambda_reg * vreg_loss
                val_loss += total_v_loss.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)
        validation_loss.append(val_loss)

        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # --- relative tolerance ---
    rel_tol = [np.abs((validation_loss[k] - validation_loss[k+1]) / validation_loss[k])
               for k in range(len(validation_loss)-1)]
    m_rel = np.median(rel_tol[-10:])

    # --- parameter count ---
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(training_loss, label='Training Loss')
    axes[0].plot(validation_loss, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'N_param = {n_p:.3e}', fontsize=9)
    axes[0].legend()

    axes[1].plot(rel_tol[50:])
    axes[1].set_title(f'Rel. Tol. Val., median of last 10 = {m_rel:.3e}', fontsize=9)

    plt.tight_layout()
    if loss_save_path:
        plt.savefig(loss_save_path)
    plt.show()

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    return model, training_loss, validation_loss, m_rel

# ================================
# Data preparation for training
# ================================

# Load CSVs for all patients and extract features/targets
data_splits = prepare_dataset(
    data_dir=DATA_DIR,
    patient_ids=PATIENT_IDS,
    feature_names=FEATURES,
    column_map=COLUMN_MAP,
    L=36,
    H=HORIZON_LENGTH
)

# --- Compute feature means and variances using training data only ---
mu_g, sigma_g, mu_gen, sigma_gen = compute_means_variances(
    data_dir=DATA_DIR,
    patient_ids=PATIENT_IDS,
    feature_names=FEATURES,
    column_map=COLUMN_MAP,
    L=36,
    H=HORIZON_LENGTH)


# --- Normalize features and targets ---
data_norm = {}
for key, (X, Y) in data_splits.items():
    X_n = normalize_features(X, mu_gen, sigma_gen)
    Y_n = normalize_features(Y, np.array([mu_g]), np.array([sigma_g]))
    data_norm[key] = (X_n, Y_n)

# --- Helper function to construct DataLoaders ---
def make_loader(X, Y, batch_size, shuffle=False):
    dataset = BGDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- Construct PyTorch datasets and dataloaders ---
train_input_n, train_output_n       = data_norm["train"]
val_input_n, val_output_n           = data_norm["val"]
test_input_n, test_output_n         = data_norm["test"]
train_val_input_n, train_val_output_n = data_norm["train_val"]

train_loader     = make_loader(train_input_n, train_output_n, BATCH_SIZE, shuffle=True)
val_loader       = make_loader(val_input_n, val_output_n, BATCH_SIZE)
test_loader      = make_loader(test_input_n, test_output_n, BATCH_SIZE)
train_val_loader = make_loader(train_val_input_n, train_val_output_n, BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = e_Transformers(
    input_dim=len(FEATURES),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    output_dim=HORIZON_LENGTH,
    max_len=MAX_LEN
)

model, train_loss, val_loss, rel_tol = train_evidential_model(
    model,
    train_loader=train_val_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    lambda_reg=LAMBDA_REG,
    reg_term=REG_TERM,
    scheduler_lambda=lambda e: 1.0 if e < 300 else 0.1,
    loss_save_path=LOSS_SAVE_PATH,
    model_save_path=MODEL_SAVE_PATH
)

