import matplotlib.pyplot as plt
import Parameters as par
 
from plots import (
    plot_task2_1_metrics,
    plot_task2_1_trajectories,
    plot_task2_1_comparison,
    animate_task2_1
)
from tasks.task1_1_DEF import task1_1
from tasks.task1_2_DEF import task1_2
from tasks.task1_3_DEF import task1_3
from tasks.task2_1_DEF import run_task2_1

 
 

def main():
    # ── Task 1.1 ────────────────────────────────────────────
    if par.TASK_1_1:
        print("\n" + "=" * 55)
        print(" TASK 1.1 – Distributed Optimization")
        print("=" * 55)

        task1_1()
    
    # ── Task 1.2 ────────────────────────────────────────────
    if par.TASK_1_2:
        print("\n" + "=" * 55)
        print(" TASK 1.2 – Centralized Classification")
        print("=" * 55)

        task1_2()

    # ── Task 1.3 ────────────────────────────────────────────
    if par.TASK_1_3:
        print("\n" + "=" * 55)
        print(" TASK 1.3 – Distributed Logistic Regression")
        print("=" * 55)

        task1_3()

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