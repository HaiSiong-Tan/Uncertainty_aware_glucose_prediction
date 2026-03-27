import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, t as student_t
import matplotlib.pyplot as plt

from data import prepare_dataset, compute_means_variances, normalize_features, BGDataset
from utils import (sensitivity_metric, CI_calculation, DTS_error_zone_count,
                   f_auc_hypo_score, f_auc_hyper_score, f_brier_hypo_score, f_brier_hyper_score)
from model import e_Transformers
from configs import (PATIENT_IDS, FEATURES, COLUMN_MAP, BATCH_SIZE, DATA_DIR, HORIZON_LENGTH,
                     MAX_LEN, D_MODEL, FF_DIM, N_HEADS, NUM_LAYERS,
                     MODEL_SAVE_PATH, ECP_GRAPH_SAVE_PATH)


def evaluate_model_evidential(model, loader, sigma_g, mu_g, ecp_graph_save_path):
    """
    Evaluate an evidential Transformer model on a test dataset.

    Parameters:
    - model: PyTorch model (e_Transformers)
    - loader: DataLoader for evaluation set e.g. testing dataset
    - sigma_g, mu_g: float, for unnormalizing glucose predictions
    - horizon_L: int, prediction horizon
    - overall_ecp_graph_save_path: str, path to save ECP plot

    Returns:
    - metrics_dict: dictionary with MARD, Brier/AUC, ECP, Spearman correlations
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_targets, all_alpha, all_beta, all_nu = [], [], [], [], []

    # --- Collect predictions and evidential parameters ---
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            alpha, beta, nu, preds = model(x_batch)

            all_preds.append(preds.cpu())
            all_alpha.append(alpha.cpu())
            all_beta.append(beta.cpu())
            all_nu.append(nu.cpu())
            all_targets.append(y_batch.cpu())

    # --- Concatenate all batches ---
    all_preds = torch.cat(all_preds).numpy()
    all_alpha = torch.cat(all_alpha).numpy()
    all_beta = torch.cat(all_beta).numpy()
    all_nu = torch.cat(all_nu).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # --- Unnormalize ---
    all_preds_real = (all_preds * sigma_g + mu_g).reshape(-1)
    all_beta_real = (all_beta * sigma_g * sigma_g).reshape(-1)
    all_alpha_real = all_alpha.reshape(-1)
    all_nu_real = all_nu.reshape(-1)
    all_targets_real = (all_targets * sigma_g + mu_g).reshape(-1)
    t_std = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))

    # --- MARD ---
    MARD = np.mean(np.abs(all_preds_real - all_targets_real) / all_targets_real) * 100

    # --- DTS zone accuracies
    arr = DTS_error_zone_count(all_targets_real, all_preds_real)

    arr_rho, arr_pvalue = spearmanr(t_std, arr)

    counts = np.bincount(arr, minlength=5)
    percentages = 100 * counts / counts.sum()

    # --- sensitivity to hypo/hyperglycemia ---
    Llower = all_preds_real + CI_calculation(0.68, all_alpha_real, 
           all_beta_real, all_nu_real, all_preds_real)

    Lupper = all_preds_real - CI_calculation(0.68, all_alpha_real, 
           all_beta_real, all_nu_real, all_preds_real)

    pred_level_70_hypo = (Llower < 70).astype(int)
    target_level_70_hypo = (all_targets_real < 70).astype(int)

    pred_level_180_hyper = (Lupper > 180).astype(int)
    target_level_180_hyper = (all_targets_real > 180).astype(int)

    level_70_sensitivity = sensitivity_metric(target_level_70_hypo, pred_level_70_hypo)
    level_180_sensitivity = sensitivity_metric(target_level_180_hyper, pred_level_180_hyper)


    # --- Brier and AUC scores ---
    f_brier_hypo = f_brier_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
    f_brier_hyper = f_brier_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
    f_auc_hypo = f_auc_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)
    f_auc_hyper = f_auc_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real)

    # --- Spearman correlation of predicted std vs error ---
    pred_std = np.sqrt(all_beta_real * (1 + all_nu_real) / (all_alpha_real * all_nu_real))
    pred_error = np.abs(all_preds_real - all_targets_real)
    rho_uncertainty, _ = spearmanr(pred_std.flatten(), pred_error.flatten())

    # --- Error Calibration Plot (ECP vs NCP) ---
    ecp_list = []
    dis_list = []

    for j in range(99):
        interval_CI = (j+1) * 0.01
        neg_CI_delta = student_t.interval(interval_CI, 2*all_alpha_real, loc=0, scale=pred_std)[1]
        acc = np.sum(np.abs(all_targets_real - all_preds_real) < np.abs(neg_CI_delta))
        empirical_ecp = acc / len(all_targets_real)
        discrepancy = np.abs(empirical_ecp - interval_CI)
        ecp_list.append(empirical_ecp)
        dis_list.append(discrepancy)

    ecp_list.append(1.0)
    MCE = np.mean(dis_list)

    # --- Plot ECP vs NCP ---
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(0.01,1.01,0.01), ecp_list, label=f'ECP (MCE = {MCE:.2e})')
    plt.plot([0,1],[0,1],'r--', label='Nominal Coverage')
    plt.xlabel('Nominal coverage prob.')
    plt.ylabel('Empirical coverage prob.')
    plt.legend(fontsize=9)
    plt.title('Error Calibration Plot', fontsize=10)
    plt.tight_layout()
    plt.savefig(ecp_graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Collect metrics ---
    metrics_dict = {
        "Zone A accuracy": percentages[0],
        "Zone B accuracy": percentages[1],
        "Zone C accuracy": percentages[2],
        "Zone D accuracy": percentages[3],
        "Zone E accuracy": percentages[4],
        "MARD": MARD,
        "Brier_Hypo": f_brier_hypo,
        "Brier_Hyper": f_brier_hyper,
        "Level 70 Sensitivity": level_70_sensitivity,
        "Level 180 Sensitivity": level_180_sensitivity,
        "AUC_Hypo": f_auc_hypo,
        "AUC_Hyper": f_auc_hyper,
        "Correlation with error": rho_uncertainty,
        "Correlation with risk zones": arr_rho,
        "MCE": MCE
    }

    return metrics_dict

# ================================
# Loading and processing data
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


# ================================
# Evaluating model
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = e_Transformers(input_dim=len(FEATURES), d_model=D_MODEL, n_heads=N_HEADS,
                       num_layers=NUM_LAYERS, ff_dim=FF_DIM, output_dim=HORIZON_LENGTH,
                       max_len=MAX_LEN).to(device)

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

metrics = evaluate_model_evidential(model, test_loader, sigma_g, mu_g, ECP_GRAPH_SAVE_PATH)

for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

