import inspect
from collections import OrderedDict

from torch.nn import Module, ModuleDict
from torch_geometric.utils import remove_self_loops
from torch import is_tensor, bool, index_select, empty_like, bernoulli, zeros

msg_special_args = {"edge_index", "edge_index_i", "edge_index_j", "size", "size_i", "size_j"}
aggr_special_args = {"index", "dim_size"}


def __process_size__(size):
    if isinstance(size, int):
        return [size, size]
    if is_tensor(size):
        return size.tolist()
    return list(size) if size else [None, None]


def __distribute__(params, kwargs):
    return {key: kwargs.get(key, param.default) for key, param in params.items()}


REQUIRED_QUANTIZER_KEYS = ("message_low",
                           "message_high",
                           "update_low",
                           "update_high",
                           "aggregate_low",
                           "aggregate_high",
                           )


class MessagePassingMultiQuant(Module):
    def __init__(self, aggr="sum", flow="source_to_target", node_dim=0, mp_quantizers=None):
        super(MessagePassingMultiQuant, self).__init__()

        assert aggr in ["sum", "prod", "mean", "amax", "amin"], f"{aggr} is not a valid aggregation method."
        assert flow in ["source_to_target", "target_to_source"], f"{flow} is not a valid flow direction."
        assert node_dim >= 0 and isinstance(node_dim, int), "node_dim must be non-negative integer."

        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

        self.__msg_params__ = OrderedDict(inspect.signature(self.message).parameters)
        self.__aggr_params__ = OrderedDict(inspect.signature(self.aggregate).parameters)
        self.__aggr_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        self.__args__ = msg_args.union(aggr_args)

        assert mp_quantizers is not None
        self.mp_quant_fns = mp_quantizers

    def reset_parameters(self):
        self.mp_quantizers = ModuleDict()
        for key in REQUIRED_QUANTIZER_KEYS:
            self.mp_quantizers[key] = self.mp_quant_fns[key]()

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            idx = ij.get(arg[-2:])
            data = kwargs.get(arg[:-2] if idx is not None else arg)

            if idx is not None and is_tensor(data):
                size[idx] = data.shape[self.node_dim]
                out[arg] = data.index_select(self.node_dim, edge_index[idx])
            else:
                out[arg] = data

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        out.update({"edge_index": edge_index,
                    "edge_index_i": edge_index[i],
                    "edge_index_j": edge_index[j],
                    "size": size,
                    "size_i": size[i],
                    "size_j": size[j],
                    "index": edge_index[i],
                    "dim_size": size[i]})
        return out

    def propagate(self, edge_index, mask, size=None, **kwargs):
        size = __process_size__(size)
        kwargs = self.__collect__(edge_index, size, kwargs)
        msg = self.message(**__distribute__(self.__msg_params__, kwargs))
        if self.training:
            edge_mask = index_select(mask, 0, edge_index[0])
            out = empty_like(msg)
            out[edge_mask] = self.mp_quantizers["message_high"](msg[edge_mask])
            out[~edge_mask] = self.mp_quantizers["message_low"](msg[~edge_mask])
        else:
            out = self.mp_quantizers["message_low"](msg)

        aggrs = self.aggregate(out, **__distribute__(self.__aggr_params__, kwargs))
        if self.training:
            out = empty_like(aggrs)
            out[mask] = self.mp_quantizers["aggregate_high"](aggrs[mask])
            out[~mask] = self.mp_quantizers["aggregate_low"](aggrs[~mask])
        else:
            out = self.mp_quantizers["aggregate_low"](aggrs)

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size):
        num_features = inputs.shape[1]
        index = index.view(-1, 1).expand(-1, num_features) if inputs.dim() > 1 else index
        return zeros((dim_size, num_features),
                     device=inputs.device,
                     dtype=inputs.dtype).scatter_reduce_(self.node_dim, index, inputs, self.aggr)


def evaluate_prob_mask(data):
    return bernoulli(data.prob_mask).to(bool)


class GINConvMultiQuant(MessagePassingMultiQuant):
    def __init__(self, nn, mp_quantizers=None, **kwargs):
        super(GINConvMultiQuant, self).__init__(aggr="sum", mp_quantizers=mp_quantizers, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()

    def forward(self, x, edge_index, mask):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x, mask=mask)
        return self.nn(x + out)

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)
