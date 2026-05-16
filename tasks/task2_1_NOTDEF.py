import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- LE TUE FUNZIONI RIGOROSAMENTE VETTORIALI (2D) AGGIORNATE ---

def local_cost(zi, sigma, r_i, b_i, gamma_i, beta_i):
    """ℓᵢ(zᵢ, σ) = γᵢ‖zᵢ−rᵢ‖² + βᵢ‖zᵢ − σ − bᵢ‖²"""
    return gamma_i * np.dot(zi - r_i, zi - r_i) + beta_i * np.dot(zi - sigma - b_i, zi - sigma - b_i)
 
def grad1_li(zi, sigma, r_i, b_i, gamma_i, beta_i):
    """∇₁ℓᵢ  w.r.t. zᵢ = 2γᵢ(zᵢ − rᵢ) + 2βᵢ(zᵢ − σ − bᵢ)"""
    return 2.0 * gamma_i * (zi - r_i) + 2.0 * beta_i * (zi - sigma - b_i)
 
def grad2_li(zi, sigma, b_i, beta_i):
    """∇₂ℓᵢ  w.r.t. σ = -2βᵢ(zᵢ − σ − bᵢ)"""
    return -2.0 * beta_i * (zi - sigma - b_i)

def phi(z_i):
    return z_i

def grad_phi(z_i):
    return 1.0


# --- ALGORITMO AGGREGATIVE TRACKING (IN 2D) CON PLOT E ANIMAZIONE ---

def task_aggregative_2d(graph_type: str = "star", N: int = 10):
    stepsize = 0.01
    maxK = 2000
    
    # Parametri: Trade-off tra Target Privati e Formazione Rigida
    r0 = np.array([0.0, 0.0]) # Mantenuto ESCLUSIVAMENTE per il plot (stella nera)                    
    gamma = np.ones(N) * 1  # Peso (alpha_i nel testo) per i target privati
    beta = np.ones(N) * 0.1  # Peso (beta_i nel testo) per la formazione geometrica

    # Generazione Vettori di Offset b_i (Poligono regolare per garantire sum(b_i) = 0)
    b = np.zeros((N, 2))
    radius = 6
    for i in range(N):
        angle = 2.0 * np.pi * i / N
        b[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Generazione Grafo e pesi
    G = nx.star_graph(N-1) if graph_type == "star" else nx.path_graph(N)
    Adj = nx.adjacency_matrix(G).toarray()
    
    weightedAdj = np.zeros((N, N)) 
    for i in range(N):
        N_i = np.nonzero(Adj[i])[0]
        deg_i = len(N_i)
        for j in N_i:
            N_j = np.nonzero(Adj[j])[0]
            deg_j = len(N_j)
            weightedAdj[i, j] = 1.0 / (1 + max(deg_i, deg_j))
    weightedAdj += np.eye(N) - np.diag(weightedAdj.sum(axis=0))

    # Target privati (r_i) sparsi
    np.random.seed(0)
    r = 10 * (np.random.rand(N, 2) - 0.5) 

    # Inizializzazione matrici di stato
    z = np.zeros((maxK, N, 2))
    s = np.zeros((maxK, N, 2))
    v = np.zeros((maxK, N, 2))

    for ii in range(N):
        z[0, ii] = np.array([-5.0, 5.0])  
        s[0, ii] = phi(z[0, ii])  
        v[0, ii] = grad2_li(z[0, ii], s[0, ii], b[ii], beta[ii]) 

    # --- CICLO PRINCIPALE ---
    for k in range(maxK - 1):
        for ii in range(N):
            g1 = grad1_li(z[k, ii], s[k, ii], r[ii], b[ii], gamma[ii], beta[ii])
            g_phi = grad_phi(z[k, ii])
            z[k+1, ii] = z[k, ii] - stepsize * (g1 + g_phi * v[k, ii])

        for ii in range(N):
            N_ii = np.nonzero(Adj[ii])[0]
            s[k+1, ii] = weightedAdj[ii, ii] * s[k, ii]
            for jj in N_ii:
                s[k+1, ii] += weightedAdj[ii, jj] * s[k, jj]
            s[k+1, ii] += phi(z[k+1, ii]) - phi(z[k, ii])

        for ii in range(N):
            N_ii = np.nonzero(Adj[ii])[0]
            v[k+1, ii] = weightedAdj[ii, ii] * v[k, ii]
            for jj in N_ii:
                v[k+1, ii] += weightedAdj[ii, jj] * v[k, jj]
            
            grad2_new = grad2_li(z[k+1, ii], s[k+1, ii], b[ii], beta[ii])
            grad2_old = grad2_li(z[k, ii], s[k, ii], b[ii], beta[ii])
            v[k+1, ii] += grad2_new - grad2_old

    # ==========================================
    # 2. CALCOLO METRICHE (COSTO E GRADIENTE)
    # ==========================================
    J_cost = np.zeros(maxK)
    grad_norm = np.zeros(maxK)
    
    for k in range(maxK):
        sigma_k = np.mean(z[k], axis=0)
        cost_k = 0.0
        grad_sq_sum = 0.0
        
        # Pre-calcolo della sommatoria delle derivate parziali per la regola della catena
        sum_grad2 = np.sum([-2.0 * beta[j] * (z[k, j] - sigma_k - b[j]) for j in range(N)], axis=0)
        
        for ii in range(N):
            cost_k += local_cost(z[k, ii], sigma_k, r[ii], b[ii], gamma[ii], beta[ii])
            
            # ∇_{z_i} J = ∇_1 l_i + (1/N) * sum_j(∇_2 l_j)
            grad_i = 2.0 * gamma[ii] * (z[k, ii] - r[ii]) + 2.0 * beta[ii] * (z[k, ii] - sigma_k - b[ii]) + (1.0 / N) * sum_grad2
            grad_sq_sum += np.dot(grad_i, grad_i)
            
        J_cost[k] = cost_k
        grad_norm[k] = np.sqrt(grad_sq_sum)

    # --- PLOT 1: Evoluzione Costo e Gradiente ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(J_cost, 'b-', linewidth=2)
    plt.title('Evolution of Global Cost Function', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration $k$')
    plt.ylabel('Cost $J(z, \sigma)$')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Usiamo scala logaritmica sull'asse Y per apprezzare meglio la convergenza a zero
    plt.semilogy(grad_norm, 'r-', linewidth=2) 
    plt.title('Norm of the Global Gradient', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration $k$')
    plt.ylabel('$|\| \\nabla J \||$ (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() # Chiudi questa finestra per far partire l'animazione

    # ==========================================
    # 3. ANIMAZIONE DEL COMPORTAMENTO DEL TEAM
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Animated Visualization of Team Behaviour", fontsize=14, fontweight='bold')
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    # Disegna gli obiettivi fissi
    ax.scatter(r0[0], r0[1], c='black', marker='*', s=300, label='Global Target $r_0$', zorder=3)
    ax.scatter(r[:, 0], r[:, 1], c='red', marker='x', s=80, label='Private Targets $r_i$', zorder=2)
    
    # Inizializza i punti per gli agenti e per il baricentro
    agents_scatter = ax.scatter(z[0, :, 0], z[0, :, 1], c='blue', s=60, alpha=0.7, label='Agents $z_i$', zorder=4)
    barycenter_scatter = ax.scatter(np.mean(z[0, :, 0]), np.mean(z[0, :, 1]), c='green', marker='D', s=100, label='Barycenter $\sigma(z)$', zorder=5)
    
    # Calcola dinamicamente i limiti degli assi per inquadrare tutto
    all_x = np.concatenate([z[:, :, 0].flatten(), r[:, 0], [r0[0]]])
    all_y = np.concatenate([z[:, :, 1].flatten(), r[:, 1], [r0[1]]])
    pad = 1.5
    ax.set_xlim(np.min(all_x) - pad, np.max(all_x) + pad)
    ax.set_ylim(np.min(all_y) - pad, np.max(all_y) + pad)
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Funzione di aggiornamento frame per matplotlib.animation
    def update(frame):
        # Aggiorna la posizione degli agenti
        agents_scatter.set_offsets(z[frame])
        
        # Aggiorna la posizione del baricentro
        curr_barycenter = np.mean(z[frame], axis=0)
        barycenter_scatter.set_offsets(curr_barycenter)
        
        # Opzionale: aggiunge il numero di iterazione nel titolo
        ax.set_title(f"Animated Visualization - Iteration $k={frame}$", fontsize=14, fontweight='bold')
        
        return agents_scatter, barycenter_scatter

    # Crea l'animazione (salta qualche frame se maxK è molto grande per fluidità)
    frame_step = max(1, maxK // 150)
    frames_to_render = np.arange(0, maxK, frame_step)
    
    ani = animation.FuncAnimation(fig, update, frames=frames_to_render, interval=40, blit=False)
    
    plt.show()

    return z, s, v

if __name__ == "__main__":
    task_aggregative_2d()