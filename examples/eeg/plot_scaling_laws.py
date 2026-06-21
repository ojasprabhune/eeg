import matplotlib.pyplot as plt


def plot_scaling_laws():
    """
    Plots the validation F1 score vs training data fraction for the linear
    baseline model predicting gesture classes from EEG data.
    """

    linear_baseline_4classes_f1_scores = {
        0.25: 0.26939,
        0.50: 0.2624,
        0.75: 0.26425,
        1.00: 0.25811,
    }

    linear_baseline_3classes_f1_scores = {
        0.25: 0.51257,
        0.50: 0.5049,
        0.75: 0.50238,
        1.00: 0.50052,
    }

    fractions = [0.25, 0.50, 0.75, 1.00]

    # extract data for plotting
    lin_4_f1 = [linear_baseline_4classes_f1_scores[f] for f in fractions]
    lin_3_f1 = [linear_baseline_3classes_f1_scores[f] for f in fractions]

    plt.figure(figsize=(8, 6))

    # plot Linear Baseline 3 classes
    plt.plot(
        fractions,
        lin_4_f1,
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Linear Baseline 4 classes",
    )

    # plot Linear Baseline 4 classes
    plt.plot(
        fractions,
        lin_3_f1,
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Linear Baseline 3 classes",
    )

    plt.title("Scaling Laws: Validation F1 Score vs. Training Data Fraction")
    plt.xlabel("Training Data Fraction")
    plt.ylabel("Macro F1 Score")
    plt.xticks(fractions, labels=["25%", "50%", "75%", "100%"])
    plt.ylim(0, 1.0)  # Assuming F1 scores are between 0 and 1
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()

    output_filename = "figures/scaling_laws_plot.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot successfully saved to {output_filename}")


if __name__ == "__main__":
    plot_scaling_laws()
