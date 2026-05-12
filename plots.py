import numpy as np
import matplotlib.pyplot as plt
 
#  Task 1.1
 
def plot_task1_1(cost, gradient, consensus, title="Task 1.1 – Gradient Tracking"):
    """
    Three-panel plot for the distributed quadratic minimisation.
 
    Parameters
    ----------
    cost       : array-like, shape (K,)
    gradient   : array-like, shape (K,)
    consensus  : array-like, shape (K,)
    """
    fig, axes = plt.subplots(ncols=3, figsize=(13, 4))
    fig.suptitle(title)
 
    axes[0].plot(cost[:-1])
    axes[0].set_title("Cost")
    axes[0].set_xlabel("Iterations")
    axes[0].grid(True)
 
    axes[1].semilogy(np.abs(gradient[:-1]))
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iterations")
    axes[1].grid(True)
 
    axes[2].semilogy(consensus[:-1])
    axes[2].set_title("Consensus Error")
    axes[2].set_xlabel("Iterations")
    axes[2].grid(True)
 
    plt.tight_layout()
    return fig

#  Task 1.2

def plot_task1_2(cost_hist, grad_norm_hist, map_name=""):
    """
    Two-panel training-curve plot for centralised logistic regression.
 
    Parameters
    ----------
    cost_hist      : list of floats
    grad_norm_hist : list of floats
    map_name       : str, e.g. 'Parabola'
    """
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    fig.suptitle(f"Task 1.2 – Training Curves ({map_name})")
 
    axes[0].plot(cost_hist)
    axes[0].set_title("Cost Evolution")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
 
    axes[1].semilogy(grad_norm_hist)
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("‖∇L‖")
    axes[1].grid(True)
 
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────
#  Task 1.3
# ──────────────────────────────────────────────────────────────
 
def plot_task1_3(cost_history, grad_norm_history, consensus_history,
                 title="Task 1.3 – Distributed Logistic Regression"):
    """
    Three-panel plot for the distributed gradient-tracking algorithm.
 
    Parameters
    ----------
    cost_history      : list of floats
    grad_norm_history : list of floats
    consensus_history : list of floats
    title             : str
    """
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    fig.suptitle(title)
 
    axes[0].plot(cost_history)
    axes[0].set_title("Total Cost")
    axes[0].set_xlabel("Iterations")
    axes[0].grid(True)
 
    axes[1].semilogy(grad_norm_history)
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Iterations")
    axes[1].grid(True)
 
    axes[2].semilogy(consensus_history)
    axes[2].set_title("Consensus Error")
    axes[2].set_xlabel("Iterations")
    axes[2].grid(True)
 
    plt.tight_layout()
    return fig
