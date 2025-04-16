import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease, gt_drdi
from model.graph_transformer_layer import GraphTransformerLayer
from torch_geometric.nn.dense.linear import Linear
from model.GAT import GraphAttentionLayer
from data_preprocess import *
import torch.nn.functional as F
from torch_geometric.nn import conv

device = torch.device('cuda')


class GHTMDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.liner = nn.Linear(867, args.gt_out_dim)
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim,
                                                    args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                          args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_drdi = gt_drdi.GraphTransformer(device, args.gt_layer, {'drug': 663, 'disease': 409}, args.gt_out_dim,
                                                args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head), args.hgt_head,
                                                   3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3,
                                                        args.dropout)
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.GraphTransformerLayer = GraphTransformerLayer(1072, args.gt_out_dim, args.gt_head, args.dropout,
                                                           self.layer_norm, self.batch_norm,
                                                           self.residual)
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer - 1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        # encoder_layer2 = nn.TransformerEncoderLayer(d_model=867, nhead=args.tr_head)
        # self.drdi_trans = nn.TransformerEncoder(encoder_layer2, num_layers=1)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3,
                                      num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3,
                                         num_decoder_layers=3, batch_first=True)

        # self.proj = Linear(2249, 128, weight_initializer='glorot', bias=True)
        class CustomSequential(torch.nn.Sequential):
            def reset_parameters(self):
                for layer in self:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

        self.proj = CustomSequential(
            torch.nn.Linear(2249, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128, bias=True)
        )

        if self.proj is not None:
            self.proj.reset_parameters()
        if self.proj is not None:
            self.proj.reset_parameters()
        self.gat = GraphAttentionLayer(2249, 128, 3, residual=True)
        self.g = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
                               nn.ReLU(inplace=True)).to(device)
        self.g_1 = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_equidim, bias=False),
                                 nn.ReLU(inplace=True)).to(device)

        self.p_1 = nn.Sequential(nn.Linear(self.args.g_equidim, self.args.p_equidim, bias=False),
                                 nn.ReLU(inplace=True)).to(device)

        self.gcn_1 = conv.GCNConv(2249, 512)
        self.gcn_2 = conv.GCNConv(512, 256)
        self.gcn_3 = conv.GCNConv(256, 128)

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim*2, 1024),  # + 256
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, drdr_graph, didi_graph, het_mat, edge_idx, adj_mat,
                sample):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        cnn_embd_hetro = self.proj(het_mat) if self.proj is not None else het_mat
        gat_embd = self.gat(het_mat, edge_idx)
        embd_mlp = cnn_embd_hetro
        distance = pairwise_distance(adj_mat).to(device)
        emb_het, emb_hom = hero(embd_mlp, gat_embd, distance)
        embs_P1 = self.g(emb_het)  # (2249,256)
        embs_P2 = self.g(emb_hom)  # (2249,256)

        # The second term in Eq. (10): uniformity loss
        intra_c = (embs_P1).T @ (embs_P1).contiguous()
        intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
        loss_uni = torch.log(intra_c).mean()

        intra_c_2 = (embs_P2).T @ (embs_P2).contiguous()
        intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
        loss_uni += torch.log(intra_c_2).mean()

        #######################################################################
        # The first term in Eq. (10): invariance loss
        inter_c = embs_P1.T @ embs_P2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv = -torch.diagonal(inter_c).sum()

        #######################################################################
        # Projection and Transformation
        embs_Q2 = self.g_1(emb_het)
        embs_Q1 = self.g_1(emb_hom)
        embs_Q1_trans = self.p_1(embs_Q1)

        # The first term in Eq. (11)
        inter_c = embs_Q1_trans.T @ embs_Q2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_inv = -torch.diagonal(inter_c).sum()

        #######################################################################
        # The second term in Eq. (11)
        inter_c = embs_Q1_trans.T @ embs_Q1
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_nontrival_1 = torch.diagonal(inter_c).sum()

        inter_c_1 = embs_Q1_trans.T @ embs_P2
        inter_c_1 = F.normalize(inter_c_1, p=2, dim=1)
        loss_spe_nontrival_2 = torch.diagonal(inter_c_1).sum()
        ########################################################################

        loss_consistency = loss_inv + self.args.gamma * loss_uni
        loss_specificity = loss_spe_inv + self.args.eta * (loss_spe_nontrival_1 + loss_spe_nontrival_2)
        cl_loss1 = contrastive_loss(embs_P1, embs_P2)

        # loss_h = loss_consistency + self.args.lambbda * loss_specificity  #
        loss_h = loss_consistency + loss_specificity + cl_loss1
        # loss_h = loss_consistency + loss_specificity

        h_concat = torch.cat((emb_het, emb_hom), 1)

        # drdi_fea = self.drdi_trans(drdi_dgl)
        # X = self.liner(drdi_fea)
        dr_x = h_concat[:self.args.drug_number]
        di_X = h_concat[self.args.drug_number:]

        cl_loss2 = contrastive_loss(dr_x, dr_sim)
        cl_loss3 = contrastive_loss(di_X, di_sim)
        loss = loss_h + 0.5 * cl_loss2 + 0.5 * cl_loss3
        # loss = loss_h

        dr = torch.cat((dr_sim, dr_x), dim=1)
        di = torch.cat((di_sim, di_X), dim=1)




        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        return loss, output
