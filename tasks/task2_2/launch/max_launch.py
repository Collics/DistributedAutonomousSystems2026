from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config = os.path.join(
    get_package_share_directory('task2_2'),
    'config.rviz'
    )
    # --- PARAMETRI GLOBALI DEL TASK 2.1 ---
    NN = 10         # Numero di agenti
    maxK = 2000     
    stepsize = 0.01
    radius = 6.0
    
    # Target privati (r_i) sparsi con lo stesso seed del Task 2.1
    np.random.seed(0)
    R_targets = 10 * (np.random.rand(NN, 2) - 0.5) 
    
    # Inizializzazione (z_i al tempo 0) per tutti a [-5.0, 5.0]
    Z_init = np.array([[-5.0, 5.0] for _ in range(NN)])

    # Generazione Vettori di Offset b_i
    B_offsets = np.zeros((NN, 2))
    for i in range(NN):
        angle = 2.0 * np.pi * i / NN
        B_offsets[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    # Generazione Grafo e pesi (Stesso codice del tuo Task 2.1)
    G = nx.star_graph(NN-1) # o nx.path_graph(NN)
    Adj = nx.adjacency_matrix(G).toarray()
    
    weightedAdj = np.zeros((NN, NN)) 
    for i in range(NN):
        N_i = np.nonzero(Adj[i])[0]
        deg_i = len(N_i)
        for j in N_i:
            N_j = np.nonzero(Adj[j])[0]
            deg_j = len(N_j)
            weightedAdj[i, j] = 1.0 / (1 + max(deg_i, deg_j))
    
    # La matrice di peso include anche l'autopeso sulla diagonale
    weightedAdj += np.eye(NN) - np.diag(weightedAdj.sum(axis=0))

    node_list = []
    package_name = "task2_2"

    for ii in range(NN):
        # Estrae i vicini dall'adiacenza
        N_ii = np.nonzero(Adj[ii])[0].tolist()
        
        # Estrae SOLO i pesi corrispondenti ai vicini (W_ij) e il proprio autopeso (W_ii)
        weights_ii = [weightedAdj[ii, j] for j in N_ii]
        self_weight = weightedAdj[ii, ii]

        node_list.append(
            Node(
                package=package_name,
                namespace=f"agent_{ii}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": ii,
                        "stepsize": stepsize,
                        "maxK": maxK,
                        "gamma": 1.0, 
                        "beta": 0.1,   
                        "neighbors": N_ii,
                        "weights": weights_ii,
                        "self_weight": float(self_weight),
                        "b": B_offsets[ii].tolist(),
                        "r": R_targets[ii].tolist(),
                        "xzero": Z_init[ii].tolist(),
                    }
                ],
                output="screen",
            )
        )
        node_list.append(
            Node(
                package=package_name,
                executable="visualizer",
                namespace=f"viz_{ii}",
                parameters=[{
                    "agent_id": ii,
                    "b": B_offsets[ii].tolist(),
                    "r": R_targets[ii].tolist(),
                }],
                output="screen",
            )
        )
    
    node_list.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
        )
    )

    return LaunchDescription(node_list)