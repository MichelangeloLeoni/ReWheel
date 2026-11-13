"""Module defining the Architecture dataclass."""
from dataclasses import dataclass, field
from .. import activation_functions as af

@dataclass
class Architecture:
    """
    Store and validate the architecture of a neural network.

    Parameters
    ----------
    input_vector_size : int
        Number of features in the input vectors.
    layers : list of tuple[int, str]
        Network layer specifications. Each tuple contains the
        number of neurons in the layer and the name of its
        activation function.

    Attributes
    ----------
    input_vector_size : int
        Number of features accepted by the input layer.
    layers : list of int
        Number of neurons in each network layer.
    act_funcs : list of str
        Activation function associated with each network layer.
        Function names are stored in uppercase.

    Raises
    ------
    ValueError
        If one or more activation functions are not supported.

    Notes
    -----
    The input layer is not included in ``layers``. Its size is
    stored separately in ``input_vector_size``.

    The activation function names are case-insensitive and are
    normalized to uppercase during initialization.

    Examples
    --------
    >>> architecture = Architecture(
    ...     input_vector_size=2,
    ...     layers=[
    ...         (3, "Linear"),
    ...         (2, "Sigmoid"),
    ...         (1, "SoftMax"),
    ...     ],
    ... )
    >>> architecture.layers
    [3, 2, 1]
    >>> architecture.act_funcs
    ['LINEAR', 'SIGMOID', 'SOFTMAX']
    """
    input_vector_size: int
    layers: list[tuple]
    act_funcs: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.act_funcs = []
        layer_specs = self.layers
        self.layers = []

        for layer in layer_specs:
            act_func_name = layer[1].upper()
            if act_func_name not in af.af_list:
                raise ValueError(
                    f"Activation function {layer[1]} is not supported. "
                    f"Supported functions are: {af.af_list}"
                )

            self.act_funcs.append(act_func_name)
            self.layers.append(layer[0])
