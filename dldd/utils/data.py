"""
Just a collection of different useful functions, data structures and helpers.
"""

from typing import Any

from torch_geometric.data import Data


class TwoGraphData(Data):
    """
    Subclass of torch_geometric.data.Data for protein and drug data.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __inc__(self, key: str, value: Any) -> int:
        """How to increment values during batching.
        When we are concatenating many graphs together we need to increase the edge_index
        by the number of nodes in a graph for each next graph.

        Returns:
            int
        """
        if not key.endswith("edge_index"):
            return super().__inc__(key, value)

        lenedg = len("edge_index")
        prefix = key[:-lenedg]
        return self.__dict__[prefix + "x"].size(0)

    def nnodes(self, prefix: str) -> int:
        """Number of nodes

        Args:
            prefix (str): prefix for which to count ('prot_', 'drug_')

        Returns:
            int: Number of nodes
        """
        return self.__dict__[prefix + "x"].size(0)

    def numfeats(self, prefix: str) -> int:
        """
        Calculate the feature dimension of one of the graphs.
        If the features are index-encoded (dtype long, single number for each node, for use with Embedding),
        then return the max. Otherwise return size(1)
        :param prefix: str for prefix "drug_", "prot_" or else
        """
        x = self.__dict__[prefix + "x"]
        if len(x.size()) == 1:
            return x.max().item() + 1
        if len(x.size()) == 2:
            return x.size(1)
        raise ValueError("Too many dimensions in input features")
