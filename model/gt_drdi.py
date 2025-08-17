import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features

"""
import dgl
from model.graph_transformer_layer import GraphTransformerLayer


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dims, hidden_dim, out_dim, n_heads, dropout):
        super(GraphTransformer, self).__init__()

        self.device = device
        self.layer_norm = True
        node_types = ['drug','disease']
        self.node_types=node_types
        self.batch_norm = False
        self.residual = True
        # self.linear_h = nn.Linear(node_dim, hidden_dim)
        # 为每种节点类型创建不同的线性层
        self.linear_h = nn.ModuleDict({
            ntype: nn.Linear(node_dims[ntype], hidden_dim)
            for ntype in node_types
        })
        # self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm,
                                                           self.batch_norm, self.residual)
                                     for _ in range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))

    def forward(self, g):

        g = g.to(self.device)

        # 分别处理每种节点类型的特征
        new_h = {}
        for ntype in self.node_types:
            h = g.ndata['h'][ntype].float().to(self.device)
            h = self.linear_h[ntype](h)
            if ntype == 'drug':
                g.nodes[ntype].data['Q_h'] =h

            # 遍历图变换层
            for layer in self.layers:
                # 假设GraphTransformerLayer已经能够处理整个图g和特定类型的特征h
                # 如果需要，可以在GraphTransformerLayer中添加对节点类型的支持
                h = layer(g, h)

            new_h[ntype] = h

            # 更新图的节点数据（如果需要）
        # 注意：这通常不是必要的，除非您需要在后续的图处理步骤中使用这些特征
        # for ntype in self.node_types:
        #     g.ndata['h'][ntype] = new_h[ntype]

        # 返回所有节点类型的处理后的特征（或者根据需要返回特定类型的特征）
        return new_h
