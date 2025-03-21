import matplotlib.pyplot as plt
import numpy as np
import os

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Main plot
fig, ax = plt.subplots()
ax.plot(x, y1, label='Sin(x)', color='blue')

# Overlay plot using `axes` for positioning
inset_ax = fig.add_axes([0.5, 0.5, 0.4, 0.4])  # [left, bottom, width, height]
inset_ax.plot(x, y2, label='Cos(x)', color='red')

# Customizing both plots
ax.legend()
inset_ax.legend()
ax.set_title("Main Plot with Overlapping Inset")

# Save and show
plots_dir = "plots"
plot_path = os.path.join(plots_dir, f"overlap_plot4.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=450)

plt.show()