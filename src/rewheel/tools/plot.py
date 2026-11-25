"Moduel containing functions to easely create different useful plots."
import math
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(
        loss: list[float],
        valid_loss: list[float] | None = None,
        file_path: str | None = None,
        **kwargs
        ) -> None:
    """
    Plot the training and validation loss over epochs.

    Parameters
    ----------
    loss : list of float
        Training loss values, one for each epoch.
    valid_loss : list of float or None, optional
        Validation loss values, one for each epoch.
        If None, only the training loss is plotted.
    file_path : str or None, optional
        Path where the plot should be saved.
        If None, the plot is displayed instead.
    **kwargs
        Additional keyword arguments for plot customization:
        - figsize : tuple of float, optional
            Figure dimensions in inches. Defaults to (10, 6).
        - show_grid : bool, optional
            Whether to display the plot grid. Defaults to True.
        - linestyle : str, optional
            Grid line style. Defaults to '--'.
    """
    figsize = kwargs.get("figsize", (10, 6))
    show_grid = kwargs.get("show_grid", True)
    linestyle = kwargs.get("linestyle", "--")

    epochs = np.arange(1, len(loss) + 1)

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.plot(epochs, loss, label="Training loss")

    if valid_loss is not None:
        valid_epochs = np.linspace(
            1, len(loss), len(valid_loss)
        )
        ax.plot(valid_epochs, valid_loss, label="Validation loss")
        ax.set_title("Training and Validation Loss")
    else:
        ax.set_title("Training Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.grid(visible=show_grid, linestyle=linestyle, alpha=0.5)
    ax.legend()

    fig.tight_layout()

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_confusion(
        matrix: np.ndarray,
        file_path: str | None = None,
        **kwargs
        ) -> None:
    """
    Plot a confusion matrix or a collection of multilabel confusion matrices.

    Parameters
    ----------
    matrix : numpy.ndarray
        Confusion matrix. Can be either:
        - A 2D square array representing a standard confusion matrix.
        - A 3D array containing one 2x2 confusion matrix per class.
    file_path : str or None, optional
        Path where the plot should be saved.
        If None, the plot is displayed instead.
    **kwargs
        Additional keyword arguments for plot customization:
        - figsize : tuple of float, optional
            Figure dimensions in inches.
        - cmap : str, optional
            Colormap used for the heatmaps. Defaults to 'Blues'.
        - annot : bool, optional
            Whether to display values inside the cells. Defaults to True.
        - fmt : str, optional
            Number format for cell annotations. Defaults to 'd'.
    """
    cmap = kwargs.get("cmap", "Blues")
    annot = kwargs.get("annot", True)
    fmt = kwargs.get("fmt", "d")

    def draw_heatmap(ax, data, title=None):
        """Helper function that creates the matrix heatmap."""
        im = ax.imshow(data, cmap=cmap)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))

        if annot:
            threshold = data.max() / 2
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(
                        j, i, format(data[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if data[i, j] > threshold else "black"
                    )

        return im

    if matrix.ndim == 2:
        fig, ax = plt.subplots(
            figsize=kwargs.get("figsize", (8, 6))
        )
        im = draw_heatmap(ax, matrix, "Confusion Matrix")
        fig.colorbar(im, ax=ax)
    else:
        n_classes = matrix.shape[0]
        n_cols = math.ceil(math.sqrt(n_classes))
        n_rows = math.ceil(n_classes / n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=kwargs.get("figsize", (5 * n_cols, 4 * n_rows)),
            squeeze=False
        )

        for i, ax in enumerate(axes.flat):
            if i < n_classes:
                im = draw_heatmap(
                    ax,
                    matrix[i],
                    f"Class {i}"
                )
            else:
                ax.set_visible(False)

        fig.colorbar(im, ax=axes.ravel().tolist())

    fig.tight_layout()

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
