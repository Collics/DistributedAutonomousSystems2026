import matplotlib.pyplot as plt
import Parameters as par
from plots import plot_task1_1, plot_task1_2, plot_task1_3
from tasks.task1_1_DEF import task1_1
from tasks.task1_2_DEF import task1_2
from tasks.task1_3_DEF import task1_3
from tasks.task2_1_NOTDEF import run_task2_1
from plots_task2 import (plot_task2_1_metrics,plot_task2_1_trajectories,plot_task2_1_animation,plot_task2_1_comparison)


def main():
    # ── Task 1.1 ────────────────────────────────────────────
    if par.TASK_1_1:
        print("=" * 55)
        print(" TASK 1.1 – Distributed Gradient Tracking (Quadratic)")
        print("=" * 55)
        
        _, metrics = task1_1()
        plot_task1_1(metrics["cost"], metrics["gradient"], metrics["consensus"])
        plt.show()

    # ── Task 1.2 ────────────────────────────────────────────
    if par.TASK_1_2:
        print("\n" + "=" * 55)
        print(" TASK 1.2 – Centralised Logistic Regression")
        print("=" * 55)
        
        results = task1_2()
        for res in results:
            plot_task1_2(res["cost_hist"], res["grad_norm_hist"], map_name=res["map_name"])
        plt.show()

    # ── Task 1.3 ────────────────────────────────────────────
    if par.TASK_1_3:
        print("\n" + "=" * 55)
        print(" TASK 1.3 – Distributed Logistic Regression")
        print("=" * 55)
        
        results = task1_3()
        for res in results:
            plot_task1_3(res["cost_history"], res["grad_norm_history"], res["consensus_history"], title=res["title"])
        plt.show()

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
            anim, fig_anim = plot_task2_1_animation(scenarios[0])
 
        plt.show()
 
if __name__ == "__main__":
    main()