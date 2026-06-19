import matplotlib.pyplot as plt
import numpy as np

def plot_scaling_laws():
    """
    Plots the validation F1 score vs training data fraction.
    Update the dictionary below with the final macro F1 scores
    obtained from your experiments (e.g., from Weights & Biases).
    """
    
    # Fill in the macro F1 scores you got for each fraction.
    # For example: 0.25: 0.45, 0.5: 0.52, 0.75: 0.58, 1.0: 0.61
    linear_baseline_f1_scores = {
        0.25: 0.0, 
        0.50: 0.0,
        0.75: 0.0,
        1.00: 0.0
    }
    
    temporal_model_f1_scores = {
        0.25: 0.0, 
        0.50: 0.0,
        0.75: 0.0,
        1.00: 0.0
    }

    fractions = [0.25, 0.50, 0.75, 1.00]
    
    # Extract data for plotting
    lin_f1 = [linear_baseline_f1_scores[f] for f in fractions]
    temp_f1 = [temporal_model_f1_scores[f] for f in fractions]

    plt.figure(figsize=(8, 6))
    
    # Plot Linear Baseline
    plt.plot(fractions, lin_f1, marker='o', linestyle='-', linewidth=2, label='Linear Baseline')
    
    # Plot Temporal Model
    plt.plot(fractions, temp_f1, marker='s', linestyle='-', linewidth=2, label='Temporal Model')

    plt.title('Scaling Laws: Validation F1 Score vs. Training Data Fraction')
    plt.xlabel('Training Data Fraction')
    plt.ylabel('Macro F1 Score')
    plt.xticks(fractions, labels=['25%', '50%', '75%', '100%'])
    plt.ylim(0, 1.0) # Assuming F1 scores are between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    output_filename = "scaling_laws_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    plot_scaling_laws()
