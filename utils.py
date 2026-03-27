import torch
import numpy as np
from sklearn.metrics import average_precision_score
from scipy.stats import t as student_t


def kl_reg_loss_term(u_obs, gamma, alpha, beta):
    """
    Regularization function of Evidential Regression by H.S.Tan et al.
    """
    euler_constant = 0.5772156649015329
    beta_r = u_obs.max()**2
    kl_div = alpha*torch.log(beta_r/beta) + torch.lgamma(alpha) + \
              euler_constant*(alpha - 1) + (beta - beta_r)/beta_r

    return torch.mean(torch.abs(u_obs - gamma)*kl_div)


def amini_reg_loss_term(u_obs, gamma, nu, alpha):

    """
    Regularization function of Evidential Regression by Amini et al.
    """
    reg_term = torch.abs(u_obs - gamma)*(2*nu + alpha)

    return torch.mean(reg_term)


def evidential_data_loss(u_obs, gamma, nu, alpha, beta ):

    """
    Loss function of Evidential Regression

    Args:
        u_obs : observed target
        alpha, beta, nu, gamma : evidential model's outputs with 'gamma' being mean

    Returns:
        the main loss term of evidential regression.
    """

    twoBlambda = 2*(beta)*(1+nu)

    nll = 0.5*torch.log(torch.pi/(nu))  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(nu*(u_obs-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll)



def sensitivity_metric(y_true, y_pred):
    
    """ Computes sensitivity metric """

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    sensitivity = TP / (TP + FN + 1e-5)

    return sensitivity


def CI_calculation(interval_CI, alpha, beta, nu, gamma):

    """
    Function that computes the confidence interval for the model's output

    Args:
        interval_CI : confidence interval, e.g. 0.95 for 95% interval
        alpha, beta, nu, gamma : evidential model's outputs with 'gamma' being mean

    Returns:
        lower bound of the confidence interval assuming mean centered at 0
    """

    scale_sq = beta*(1+nu)/(alpha*nu)
    scale = np.sqrt(scale_sq)

    I_delta = []

    for i in range(len(alpha)):
      a,b= student_t.interval(interval_CI, df=2*alpha[i], loc=0, scale=scale[i])
      I_delta.append(a)

    return np.array(I_delta)



def f_auc_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  """
  Computes the auc score for evidential hypoglycemia prediction.

  Args:
      (i)all_preds_real: evidential mean
      (ii)all_beta_real: evidential beta
      (iii)all_nu_real: evidential nu
      (iv)all_alpha_real: evidential alpha
      (v)all_targets_real: targets

  Returns: PR-AUC score for evidential hypoglycemia prediction.

  """
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (70 - t_mean) / t_sigma
  p_hypo = student_t.cdf(z, df=deg_of_freedom)
  target_hypo = (all_targets_real <= 70).astype(int)
  auc = average_precision_score(target_hypo.flatten(), p_hypo.flatten())

  return auc



def f_auc_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  """
  Computes the auc score for evidential hyperglycemia prediction.

  Args:
      (i)all_preds_real: evidential mean
      (ii)all_beta_real: evidential beta
      (iii)all_nu_real: evidential nu
      (iv)all_alpha_real: evidential alpha
      (v)all_targets_real: targets

  Returns: PR-AUC score for evidential hyperglycemia prediction.
  """
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (180 - t_mean) / t_sigma
  p_lt_180 = student_t.cdf(z, df=deg_of_freedom)
  p_hyper = 1 - p_lt_180
  target_hyper = (all_targets_real > 180).astype(int)
  auc = average_precision_score(target_hyper.flatten(), p_hyper.flatten())

  return auc



def DTS_error_zone_count(act,pred):
  """
  This function outputs the DTS Error Grid region, based on the article
  https://pubmed.ncbi.nlm.nih.gov/39369312/

  Arguments:
  act : actual value
  pred : predicted value

  """

  def DTS_error_zone_detailed(act, pred):

    """
    Arguments:
    act : actual value
    pred : predicted value

    Returns:
    DTS Error Grid region: A(0), B(1), C(2), D(3), E(4)
    """

    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def dts_error(act, pred):
        # Zone A
        if (pred < 60 and act < 62.5) or (act > 50 and above_line(62.5, 50, 600, 480) \
                and below_line(50, 60, 500, 600)):
            return 0

        # Zone B
        if (pred < 86.5 and act < 97.5) or (act > 50 and above_line(97.5, 50, 600, 307) \
                and below_line(50, 86.5, 347, 600)):
            return 1

        # Zone C
        if (pred < 124 and act < 153) or (act > 50 and above_line(153, 50, 600, 197) \
                and below_line(50, 124, 241, 600)):
            return 2

        # Zone D
        if (pred < 179 and act < 238) or (act > 50 and above_line(238, 50, 600, 126) \
                and below_line(50, 179, 167, 600)):
            return 3

        # Zone E
        return 4

    return dts_error(act, pred)

  DTS_error_zone_detailed = np.vectorize(DTS_error_zone_detailed)
  zones = DTS_error_zone_detailed(act, pred)

  return zones


def DTS_zones(targets, preds):
  """
  Computes the DTS zone grid accuracies from zone A to E

  Args:
    targets : actual values
    preds : predicted values

  Returns: a list of zone accuracies from A to E
  """
  zones = DTS_error_zone_count(targets, preds)
  counts = np.bincount(zones, minlength=5)
  percentages = 100 * counts / counts.sum()

  return percentages



def f_brier_hypo_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  '''

  Computes the Brier score for hypoglycemia prediction.

  '''
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (70 - t_mean) / t_sigma
  p_no_hypo = 1 - student_t.cdf(z, df=deg_of_freedom)
  target_no_hypo = (all_targets_real > 70).astype(int)
  b_score = np.mean((target_no_hypo - p_no_hypo)**2)

  return b_score



def f_brier_hyper_score(all_preds_real, all_beta_real, all_nu_real, all_alpha_real, all_targets_real):
  '''

  Computes the Brier score for hyperglycemia prediction.

  '''
  t_mean = all_preds_real
  t_sigma = np.sqrt(all_beta_real*(1+all_nu_real)/(all_alpha_real*all_nu_real))
  deg_of_freedom = 2*all_alpha_real
  z = (180 - t_mean) / t_sigma
  p_lt_180 = student_t.cdf(z, df=deg_of_freedom)
  p_no_hyper = p_lt_180
  target_no_hyper = (all_targets_real <= 180).astype(int)
  b_score = np.mean((target_no_hyper - p_no_hyper)**2)

  return b_score