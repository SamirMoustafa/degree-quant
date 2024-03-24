from torch.nn import Linear, ModuleDict
from torch.nn.functional import linear


class QuantLinear(Linear):
    def __init__(self, in_features, out_features, linear_quantizers):
        self.layer_quant_fns = linear_quantizers
        super(QuantLinear, self).__init__(in_features, out_features, False)

    def reset_parameters(self):
        super().reset_parameters()
        self.layer_quant = ModuleDict()
        for key in ["inputs", "features", "weights"]:
            self.layer_quant[key] = self.layer_quant_fns[key]()

    def forward(self, input):
        input_q = self.layer_quant["inputs"](input)
        w_q = self.layer_quant["weights"](self.weight)
        out = linear(input_q, w_q)
        out = self.layer_quant["features"](out)
        return out
