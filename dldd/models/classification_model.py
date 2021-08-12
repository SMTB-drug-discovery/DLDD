import torch.nn.functional as F
from torch.functional import Tensor
from torch.nn import Embedding
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj
from torchmetrics.functional import accuracy, auroc

from ..layers import MLP, GINConvNet
from ..utils.data import TwoGraphData
from .base_model import BaseModel
import torch


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(
        self,
        drug_feats=119,
        prot_feats=20,
        drug_hidden_dim=64,
        prot_hidden_dim=64,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.prot_feat_embed = Embedding(prot_feats, prot_hidden_dim)
        self.drug_feat_embed = Embedding(drug_feats, drug_hidden_dim)
        self.prot_node_embed = GINConvNet(prot_hidden_dim, prot_hidden_dim)
        self.drug_node_embed = GINConvNet(drug_hidden_dim, drug_hidden_dim)
        self.mlp = MLP(drug_hidden_dim + prot_hidden_dim, out_dim=1)

    def forward(
        self,
        prot_x: Tensor,
        drug_x: Tensor,
        prot_edge_index: Adj,
        drug_edge_index: Adj,
        prot_batch: Tensor,
        drug_batch: Tensor,
        *args,
    ) -> Tensor:
        """Forward pass of the model

        Args:
            prot_x (Tensor): Protein node features
            drug_x (Tensor): Drug node features
            prot_edge_index (Adj): Protein edge info
            drug_edge_index (Adj): Drug edge info
            prot_batch (Tensor): Protein batch
            drug_batch (Tensor): Drug batch

        Returns:
            (Tensor): Final prediction
        """
        prot_x = self.prot_feat_embed(prot_x)
        drug_x = self.drug_feat_embed(drug_x)
        prot_x = self.prot_node_embed(prot_x, prot_edge_index, prot_batch)
        drug_x = self.drug_node_embed(drug_x, drug_edge_index, drug_batch)
        prot_embed = global_mean_pool(prot_x, prot_batch)
        drug_embed = global_mean_pool(drug_x, drug_batch)
        combined = torch.cat([prot_embed, drug_embed])
        prediction = self.mlp(combined)
        return torch.sigmoid(prediction)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        output = self.forward(
            data.prot_x,
            data.drug_x,
            data.prot_edge_index,
            data.drug_edge_index,
            data.prot_x_batch,
            data.drug_x_batch,
        )
        labels = data.label.unsqueeze(1)
        loss = F.binary_cross_entropy(output, labels.float())
        acc = accuracy(output, labels)
        auc = auroc(output, labels)
        return {
            "loss": loss,
            "acc": acc,
            "auroc": auc,
        }
