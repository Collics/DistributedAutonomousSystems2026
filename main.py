import matplotlib.pyplot as plt
import Parameters as par
 
from plots import (
    plot_task1_1,
    plot_task1_1_comparison,
    plot_task1_1_network,
    plot_task1_2,
    plot_task1_2_dataset,
    plot_task1_3,
    plot_task1_3_comparison,
    plot_task1_3_data_split,
    plot_task2_1_metrics,
    plot_task2_1_trajectories,
    plot_task2_1_comparison,
    animate_task2_1
)
from tasks.task1_1_DEF import task1_1, _build_graph
from tasks.task1_2_DEF import task1_2
from tasks.task1_3_DEF import task1_3
from tasks.task2_1_DEF import run_task2_1

 
 

def main():
    # ── Task 1.1 ────────────────────────────────────────────
    if par.TASK_1_1:
        print("=" * 55)
        print(" TASK 1.1 – Distributed Gradient Tracking (Quadratic)")
        print("=" * 55)
 
        # Determine active topology from Parameters.py
        if par.TASK_1_1_Path:
            graph_type = "path"
        elif par.TASK_1_1_Star:
            graph_type = "star"
        elif par.TASK_1_1_Cycle:
            graph_type = "cycle"
        else:
            raise ValueError("No graph topology flag set in Parameters.py "
                             "for Task 1.1.")
 
        weightedAdj, metrics = task1_1(graph_type=graph_type)
 
        # Single-topology convergence plot
        plot_task1_1(
            metrics["cost"], metrics["gradient"], metrics["consensus"],
            title=f"Task 1.1 – Gradient Tracking  |  Topology: "
                  f"{graph_type.capitalize()}",
        )
 
        # Network graph
        G = _build_graph(graph_type, par.TASK_1_1_N)
        plot_task1_1_network(G, graph_type)
 
        plt.show()

    # ── Task 1.2 ────────────────────────────────────────────
    if par.TASK_1_2:
        print("\n" + "=" * 55)
        print(" TASK 1.2 – Centralised Logistic Regression")
        print("=" * 55)
 
        results = task1_2()
 
        for res in results:
            # Training curves
            plot_task1_2(
                res["cost_hist"], res["grad_norm_hist"],
                map_name=res["map_name"],
            )
            # Dataset + decision boundary (if weights are available)
            if "X" in res and "labels" in res:
                plot_task1_2_dataset(
                    res["X"], res["labels"],
                    phi_fn=res.get("phi_fn"),
                    wb=res.get("wb"),
                    map_name=res["map_name"],
                )
 
        plt.show()

    # ── Task 1.3 ────────────────────────────────────────────
    if par.TASK_1_3:
        print("\n" + "=" * 55)
        print(" TASK 1.3 – Distributed Logistic Regression")
        print("=" * 55)
 
        results = task1_3()
 
        # Per-run convergence plots
        for res in results:
            plot_task1_3(
                res["cost_history"],
                res["grad_norm_history"],
                res["consensus_history"],
                title=res["title"],
            )
 
        # Multi-run comparison (all topologies × dataset sizes on one figure)
        if len(results) > 1:
            plot_task1_3_comparison(results)
 
        plt.show()

    # ── Task 2.1 ────────────────────────────────────────────

    if par.RUN_TASK_2_1:
        print("\n" + "=" * 55)
        print(" TASK 2.1 – Aggregative Tracking (Multi-Robot)")
        print("=" * 55)
        
 
        scenarios = run_task2_1()
 
        # Per-scenario: metrics + trajectories
        for sc in scenarios:
            plot_task2_1_metrics(sc)
            plot_task2_1_trajectories(sc)
 
        # Comparison across all scenarios
        plot_task2_1_comparison(scenarios)
 
        # Animation for the first (baseline) scenario
        if par.TASK_2_1_ANIMATE:
            anim, fig_anim = animate_task2_1(scenarios[0])
 
        plt.show()

    # ── Task 2.1 ────────────────────────────────────────────
    # if par.RUN_TASK_2_1:
    #     print("\n" + "=" * 55)
    #     print(" TASK 2.1 – Aggregative Tracking (Multi-Robot)")
    #     print("=" * 55)
 
    #     scenarios = run_task2_1()
 
    #     # Per-scenario: convergence metrics + trajectories
    #     for sc in scenarios:
    #         plot_task2_1_metrics(sc)
    #         plot_task2_1_trajectories(sc)
 
    #     # Comparison across all scenarios (meaningful only if len > 1)
    #     if len(scenarios) > 1:
    #         plot_task2_1_comparison(scenarios)
 
    #     # Animation for the first (baseline) scenario
    #     if par.TASK_2_1_ANIMATE:
    #         animate_task2_1(
    #             scenarios[0],
    #             skip_frames=getattr(par, "TASK_2_1_SKIP_FRAMES", 5),
    #             save_mp4=getattr(par, "TASK_2_1_SAVE_MP4", False),
    #         )
 
    #     plt.show()
 
if __name__ == "__main__":
    main()