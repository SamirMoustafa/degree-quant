import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Identity, BatchNorm1d, ModuleList
from torch_geometric.nn import global_max_pool

from dq.quantization import IntegerQuantizer
from dq.q_linear import QuantLinear
from dq.dq_message_passing import evaluate_prob_mask, GINConvMultiQuant


def integer_quantizer(qypte, momentum, percentile, signed, sample_prop):
    return lambda: IntegerQuantizer(4 if qypte == "INT4" else 8,
                                    signed=signed,
                                    use_momentum=momentum,
                                    percentile=percentile,
                                    sample=sample_prop,
                                    )


def make_quantizers(qypte, sign_input, momentum, percentile, sample_prop):
    linear_quantizers = {"inputs": integer_quantizer(qypte, momentum, percentile, sign_input, sample_prop),
                         "weights": integer_quantizer(qypte, momentum, percentile, True, sample_prop),
                         "features": integer_quantizer(qypte, momentum, percentile, True, sample_prop),
                         }
    mp_quantizers = {"message_low": integer_quantizer(qypte, momentum, percentile, True, sample_prop),
                     "message_high": Identity,
                     "update_low": integer_quantizer(qypte, momentum, percentile, True, sample_prop),
                     "update_high": Identity,
                     "aggregate_low": integer_quantizer(qypte, momentum, percentile, True, sample_prop),
                     "aggregate_high": Identity,
                     }
    return linear_quantizers, mp_quantizers


class ResettableSequential(Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


class DegreeQuantGIN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 num_layers,
                 qypte,
                 momentum,
                 percentile,
                 sample_prop,
                 ):
        super(DegreeQuantGIN, self).__init__()

        lq, mq = make_quantizers(qypte, False, momentum=momentum, percentile=percentile, sample_prop=sample_prop)
        lq_signed, _ = make_quantizers(qypte, True, momentum=momentum, percentile=percentile, sample_prop=sample_prop)

        self.convs = ModuleList()
        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hidden_channels
            self.convs += [GINConvMultiQuant(nn=ResettableSequential(QuantLinear(input_dim, hidden_channels, linear_quantizers=lq_signed),
                                                                     ReLU(),
                                                                     QuantLinear(hidden_channels, hidden_channels, linear_quantizers=lq),
                                                                     ReLU(),
                                                                     BatchNorm1d(hidden_channels),
                                                                     ),
                                             mp_quantizers=mq)]
        self.fc1 = QuantLinear(hidden_channels, hidden_channels, linear_quantizers=lq)
        self.fc2 = QuantLinear(hidden_channels, out_channels, linear_quantizers=lq)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mask = evaluate_prob_mask(data) if hasattr(data, "prob_mask") and data.prob_mask is not None else None

        for conv in self.convs:
            x = conv(x, edge_index, mask)
        x = global_max_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
