import matplotlib.pyplot as plt

# Data from the tables
methods = ['RWI-RF [5]', 'MTL-MGAN [6]', 'Deep Learning [10]', 'SegChaNet [11]', 'Proposed Model']

# Accuracy data (Table 2)
accuracy = [99.6535, 94.2, 97.8, 98.90, 99.89]
accuracy_colors = ['blue'] * (len(methods) - 1) + ['red']

# F1 Score data (Table 3)
f1_score = [99.7898, 94.4, 98.1, 98.49, 99.832]
f1_score_colors = ['pink'] * (len(methods) - 1) + ['red']

# Sensitivity data (Table 4)
sensitivity = [99.9889, 94.7, 98.1, 92.82, 100]
sensitivity_colors = ['green'] * (len(methods) - 1) + ['red']

# Precision data (Table 5)
precision = [99.5894, 93.9, 97.5, 96.66, 99.832]
precision_colors = ['violet'] * (len(methods) - 1) + ['red']

# Specificity data (Table 6)
specificity = [94.7458, 93.4, 97.3, 94.08, 99.7883]
specificity_colors = ['yellow'] * (len(methods) - 1) + ['red']

# Create a figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3 rows, 2 columns (5 plots total)
fig.suptitle('Performance Comparison of Methods', fontsize=16, fontweight='bold')

# Plot data
data_list = [accuracy, f1_score, sensitivity, precision, specificity]
color_list = [accuracy_colors, f1_score_colors, sensitivity_colors, precision_colors, specificity_colors]
titles = ['Accuracy Comparison (Fig. 8)', 'F1 Score Comparison (Fig. 9)', 
          'Sensitivity Comparison (Fig. 10)', 'Precision Comparison (Fig. 11)', 
          'Specificity Comparison (Fig. 12)']
ylabels = ['Accuracy (%)', 'F1 Score (%)', 'Sensitivity (%)', 'Precision (%)', 'Specificity (%)']

for i, ax in enumerate(axs.flat):
    if i < len(data_list):  # Ensure we don’t exceed the number of plots
        ax.bar(methods, data_list[i], color=color_list[i])  # Use defined colors
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabels[i], fontsize=10)
        ax.set_ylim(85, 105)
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=8)
        for j, v in enumerate(data_list[i]):
            ax.text(j, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)

# Remove the extra subplot (since we have 5 plots but 6 slots in a 3x2 grid)
fig.delaxes(axs[2, 1])

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
plt.savefig('all_metrics_comparison.png')
plt.show()
