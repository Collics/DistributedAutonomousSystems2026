import matplotlib.pyplot as plt
import parameters as par
from plots import plot_task1_1, plot_task1_2, plot_task1_3


def main():
    # ── Task 1.1 ────────────────────────────────────────────
    if par.TASK_1_1:
        print("=" * 55)
        print(" TASK 1.1 – Distributed Gradient Tracking (Quadratic)")
        print("=" * 55)
        from tasks.task1_1_DEF import task1_1
        _, metrics = task1_1()
        plot_task1_1(metrics["cost"], metrics["gradient"], metrics["consensus"])
        plt.show()

    # ── Task 1.2 ────────────────────────────────────────────
    if par.TASK_1_2:
        print("\n" + "=" * 55)
        print(" TASK 1.2 – Centralised Logistic Regression")
        print("=" * 55)
        from tasks.task1_2_DEF import task1_2
        results = task1_2()
        for res in results:
            plot_task1_2(res["cost_hist"], res["grad_norm_hist"], map_name=res["map_name"])
        plt.show()

    # ── Task 1.3 ────────────────────────────────────────────
    if par.TASK_1_3:
        print("\n" + "=" * 55)
        print(" TASK 1.3 – Distributed Logistic Regression")
        print("=" * 55)
        from tasks.task1_3_DEF import task1_3
        results = task1_3()
        for res in results:
            plot_task1_3(res["cost_history"], res["grad_norm_history"], res["consensus_history"], title=res["title"])
        plt.show()
 
if __name__ == "__main__":
    main()