"""
Distributed Autonomous Systems
Main
Authors: Ivan Colangelo, Nicholas Gioia, Alexandru Zaporojanu
Bologna, 09/06/26
"""
import Parameters as par

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
        # Printing, execution, and saving are all handled inside run_task2_1()
        run_task2_1()

    # ── Task 2.3 ────────────────────────────────────────────
    if getattr(par, "RUN_TASK_2_3", False):
        # The printing, execution, and plotting are all handled inside run_task_2_3()
        run_task_2_3()
 
if __name__ == "__main__":
    main()