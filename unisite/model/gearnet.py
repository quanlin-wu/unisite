from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import layers
from torchdrug.layers.geometry import GraphConstruction, AlphaCarbonNode, SequentialEdge, SpatialEdge, KNNEdge

from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class GearNetTransformer(nn.Module):
    def __init__(self, model_conf):
        super(GearNetTransformer, self).__init__()
        self._model_conf = model_conf
        self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(
            input_dim = 21,
            hidden_dims = [512, 512, 512, 512, 512, 512],
            batch_norm = True,
            concat_hidden = True,
            short_cut = True,
            readout = 'sum',
            num_relation = 7,
            edge_input_dim = 59,
            num_angle_bin = 8,
        )
        if model_conf.gearnet_weight != "":
            self.gearnet.load_state_dict(torch.load(model_conf.gearnet_weight, weights_only=False), strict=True)
        self.graph_construction = GraphConstruction(
            node_layers=[AlphaCarbonNode()],
            edge_layers=[
                SequentialEdge(max_distance=2),
                SpatialEdge(radius=10.0, min_distance=5),
                KNNEdge(k=10, min_distance=5),
            ],
            edge_feature="gearnet",
        )
        d_model = model_conf.node_embed_size
        encoder_layer = TransformerEncoderLayer(d_model, model_conf.no_heads, model_conf.dim_feedforward,
                                                model_conf.dropout, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, model_conf.enc_layers, norm=None)
        decoder_layer = TransformerDecoderLayer(d_model, model_conf.no_heads, model_conf.dim_feedforward,
                                                model_conf.dropout, "relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, model_conf.dec_layers, decoder_norm,
                                          return_intermediate=True)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.proj = nn.Linear(512 * 6 + d_model, d_model)
    
    def forward(self, init_node_embed, input_feats, mask, query_embed, pos_embed=None):
        """
        Args:
            init_node_embed: [N, L, C],
            input_feats
            mask: [N, L], bool, 'true' means the value should be ignored
            query_embed: [num_query, C],
            pos_embed: [N, L, C] or None
        Returns:
            hs: [N, num_query, C], the output of the decoder
            memory: [N, L, C], the output of the encoder
        """
        # NOTE: the residues of graph may change after graph construction
        graph = self.graph_construction(input_feats["graph"])
        with graph.residue():
            graph.input = graph.node_feature.float()
        num_residues = graph.num_residues.tolist()                       # residues of each pdb
        node_feat_pdb = self.gearnet(graph, graph.input)["node_feature"]
        node_feat_pdb = torch.split(node_feat_pdb, num_residues, dim=0)  # [[l_i, C], ...]
        N, L, _ = init_node_embed.shape
        struct_feats = torch.zeros((N, L, 3072), device=init_node_embed.device)
        for i in range(N):
            map_dict = input_feats["map_dict"][i]
            residue_number = graph[i].residue_number
            # map residue number to uniprot successfully
            success_map = [(str(r.item()) in map_dict) for r in residue_number]
            success_map = torch.tensor(success_map, dtype=torch.bool, device=init_node_embed.device)
            # filter out the residues that are not mapped to uniprot
            pdb2uniprot = [int(map_dict[str(r.item())]) - 1 for i, r in enumerate(residue_number) if success_map[i]]
            assert len(success_map) == len(node_feat_pdb[i])
            assert max(pdb2uniprot) < L, input_feats["name"][i]
            struct_feats[i, pdb2uniprot] = node_feat_pdb[i][success_map]            
        
        if pos_embed is not None:
            pos_embed = pos_embed.permute(1, 0, 2)
        seq_feats = self.encoder(init_node_embed.permute(1, 0, 2), src_key_padding_mask=mask, pos=pos_embed).permute(1, 0, 2)

        node_embed = torch.cat([seq_feats, struct_feats], dim=-1)  # [N, L, C + 3072]
        node_embed = self.proj(node_embed)  # [N, L, C]

        query_embed = query_embed.unsqueeze(1).repeat(1, N, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, node_embed.permute(1, 0, 2), memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed).transpose(1, 2)
        return hs, node_embed


class GeometryAwareRelationalGraphNeuralNetwork(nn.Module):
    """
    Geometry Aware Relational Graph Neural Network proposed in
    `Protein Representation Learning by Geometric Structure Pretraining`_.

    .. _Protein Representation Learning by Geometric Structure Pretraining:
        https://arxiv.org/pdf/2203.06125.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_angle_bin (int, optional): number of bins to discretize angles between edges.
            The discretized angles are used as relations in edge message passing.
            If not provided, edge message passing is disabled.
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, num_angle_bin=None,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GeometryAwareRelationalGraphNeuralNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.edge_dims = [edge_input_dim] + self.dims[:-1]
        self.num_relation = num_relation
        self.num_angle_bin = num_angle_bin
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeometricRelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                   None, batch_norm, activation))
        if num_angle_bin:
            self.spatial_line_graph = layers.SpatialLineGraph(num_angle_bin)
            self.edge_layers = nn.ModuleList()
            for i in range(len(self.edge_dims) - 1):
                self.edge_layers.append(layers.GeometricRelationalGraphConv(
                    self.edge_dims[i], self.edge_dims[i + 1], num_angle_bin, None, batch_norm, activation))

        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        if self.num_angle_bin:
            line_graph = self.spatial_line_graph(graph)
            edge_input = line_graph.node_feature.float()

        for i in range(len(self.layers)):
            hidden = self.layers[i](graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            if self.num_angle_bin:
                edge_hidden = self.edge_layers[i](line_graph, edge_input)
                edge_weight = graph.edge_weight.unsqueeze(-1)
                node_out = graph.edge_list[:, 1] * self.num_relation + graph.edge_list[:, 2]
                update = scatter_add(edge_hidden * edge_weight, node_out, dim=0,
                                     dim_size=graph.num_node * self.num_relation)
                update = update.view(graph.num_node, self.num_relation * edge_hidden.shape[1])
                update = self.layers[i].linear(update)
                update = self.layers[i].activation(update)
                hidden = hidden + update
                edge_input = edge_hidden
            if self.batch_norm:
                hidden = self.batch_norms[i](hidden)
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
