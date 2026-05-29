import matplotlib.pyplot as plt
import Parameters as par

from plots import (
    plot_task2_1_metrics,
    plot_task2_1_trajectories,
    plot_task2_1_comparison,
    animate_task2_1,
)

from tasks.task1_1_DEF import task1_1
from tasks.task1_2_DEF import task1_2
from tasks.task1_3_DEF import task1_3
from tasks.task2_1_DEF import run_task2_1
from tasks.task2_3_DEF import run_task_2_3

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
    if getattr(par, "RUN_TASK_2_1", False):
        print("\n" + "=" * 55)
        print(" TASK 2.1 – Aggregative Tracking (Multi-Robot)")
        print("=" * 55)
        
        scenarios = run_task2_1()
 
        for sc in scenarios:
            plot_task2_1_metrics(sc)
            plot_task2_1_trajectories(sc)
 
        if len(scenarios) > 1:
            plot_task2_1_comparison(scenarios)
 
        if getattr(par, "TASK_2_1_ANIMATE", False):
            anim, fig_anim = animate_task2_1(scenarios[0])
 
        plt.show()

    # ── Task 2.3 ────────────────────────────────────────────
    if getattr(par, "RUN_TASK_2_3", False):
        # The printing, execution, and plotting are all handled inside run_task_2_3()
        run_task_2_3()
 
if __name__ == "__main__":
    main()