from nn_simulator import NeuralNetwork, Architecture
from nn_simulator.engine.utils import initialize_weights
import matplotlib.pyplot as plt
import numpy as np


def generate_graph(
    neural_network,
    show_weights=False,
    show_layer_names=True,
    gamma=2.5
):

    layers = [
        neural_network.architecture.input_vector_size,
        *neural_network.architecture.layers
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []

    # Generate neuron positions
    for i, n_neurons in enumerate(layers):

        y = np.linspace(
            -(n_neurons - 1) / 2,
            (n_neurons - 1) / 2,
            n_neurons
        )

        x = np.full(n_neurons, i)

        positions.append(np.column_stack((x, y)))

    weights_initialized = all(
        w is not None for w in neural_network.weights
    )

    # Draw connections
    for layer_idx in range(len(layers) - 1):

        if weights_initialized:

            weights = neural_network.weights[layer_idx]
            weights = weights[:, :layers[layer_idx]]

            max_weight = np.max(np.abs(weights))

            normalized_weights = (
                np.abs(weights) / max_weight if max_weight > 0
                else np.zeros_like(weights)
            )

            intensities = normalized_weights ** gamma

        for pre_idx, pre in enumerate(positions[layer_idx]):
            for cur_idx, cur in enumerate(positions[layer_idx + 1]):

                if weights_initialized:
                    weight = weights[cur_idx, pre_idx]
                    intensity = intensities[cur_idx, pre_idx]

                    color = "tab:blue" if weight >= 0 else "tab:red"

                    alpha = 0.1 + 0.9 * intensity
                    linewidth = 0.3 + 4 * intensity

                else:
                    color = "gray"
                    alpha = 0.5
                    linewidth = 1
                    weight = None

                ax.plot(
                    [pre[0], cur[0]],
                    [pre[1], cur[1]],
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    zorder=1
                )

                # Display weight values
                if show_weights and weight is not None:

                    mid_x = (pre[0] + cur[0]) / 2
                    mid_y = (pre[1] + cur[1]) / 2

                    ax.text(
                        mid_x,
                        mid_y,
                        f"{weight:.2f}",
                        fontsize=8,
                        ha="center",
                        va="center",
                        color=color,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.7,
                            pad=1
                        ),
                        zorder=2
                    )

    # Draw neurons
    for i, layer_positions in enumerate(positions):

        ax.scatter(
            layer_positions[:, 0],
            layer_positions[:, 1],
            s=500,
            zorder=3
        )

        # Only number input neurons
        if i == 0:
            for j, (x, y) in enumerate(layer_positions):
                ax.text(
                    x, y, str(j),
                    ha="center",
                    va="center",
                    zorder=4
                )

        # Layer names
        if show_layer_names:

            if i == 0:
                layer_name = "Input layer"

            elif i == len(layers) - 1:
                layer_name = "Output layer"

            else:
                layer_name = f"Hidden layer {i}"

            ax.text(
                i,
                min(layer_positions[:, 1]) - 0.4,
                layer_name,
                ha="center",
                va="top",
                fontsize=18
            )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def generate_plot():
    pass

def visualize():
    pass

arc = Architecture(
    input_vector_size=5,
    layers=[
        (3, "linear"),
        (1, "linear")
    ]
)
nn = NeuralNetwork(
    architecture=arc
)

initialize_weights(architecture=arc, weights=nn.weights)

generate_graph(nn)
