import sqlite3
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Find all .db files in the current directory
db_files = glob.glob("./deprecated/paramdb/*.db")

fig, ax = plt.subplots(figsize=(12, 7))

for db_path in sorted(db_files):
    label = os.path.splitext(os.path.basename(db_path))[0]
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # Optuna stores trials in 'trials' table; values in 'trial_values'
        cur.execute("""
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.number
        """)
        rows = cur.fetchall()
        con.close()

        if not rows:
            print(f"  No complete trials found in {db_path}, skipping.")
            continue

        trial_numbers = [r[0] for r in rows]
        values = [r[1] for r in rows]

        # Plot raw loss
        ax.plot(trial_numbers, values, alpha=0.35, linewidth=0.8)

        # Overlay running best (min so far)
        best_so_far = np.minimum.accumulate(values)
        line, = ax.plot(trial_numbers, best_so_far, linewidth=2, label=label)

        print(f"  {label}: {len(rows)} trials, best = {min(values):.4f}")

    except Exception as e:
        print(f"  Error reading {db_path}: {e}")

ax.set_xlabel("Trial number", fontsize=13)
ax.set_ylabel("Loss (objective value)", fontsize=13)
ax.set_title("Optuna optimisation — loss vs trial", fontsize=14)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("optuna_loss_curves.png", dpi=150)
plt.show()
print("Saved to optuna_loss_curves.png")