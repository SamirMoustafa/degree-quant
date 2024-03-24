from typing import Optional

from torch import bool, tensor, empty, bernoulli, finfo, float32, min as torch_min, max as torch_max, kthvalue
from torch.autograd.function import InplaceFunction
from torch.nn import Module


def get_qparams(max_val, min_val, num_bits, signed, eps, symmetric):
    max_val, min_val = float(max_val), float(min_val)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)

    qmin = -(2.0 ** (num_bits - 1)) if signed else 0.0
    qmax = qmin + 2.0 ** num_bits - 1

    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:

        if symmetric:
            scale = 2 * max(abs(min_val), max_val) / (qmax - qmin)
            zero_point = 0.0 if signed else 128.0
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale = max(scale, eps)
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
            zero_point = zero_point

    return qmin, qmax, zero_point, scale


class FakeQuantizeFunction(InplaceFunction):
    @classmethod
    def forward(cls, ctx, x, max_val, min_val, num_bits, signed, eps, symmetric):
        # compute qparams
        qmin, qmax, zero_point, scale = get_qparams(max_val, min_val, num_bits, signed, eps, symmetric)

        q_x = zero_point + x / scale
        q_x = q_x.clamp(qmin, qmax).round()
        dq_x = scale * (q_x - zero_point)

        ctx.save_for_backward(x)
        ctx.scale, ctx.zero_point = scale, zero_point
        ctx.qmin, ctx.qmax = qmin, qmax
        return dq_x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale, zero_point = ctx.scale, ctx.zero_point
        qmin, qmax = ctx.qmin, ctx.qmax
        q_x = zero_point + x / scale
        grad_x = grad_output * ((q_x >= qmin) & (q_x <= qmax)).float()  # STE for input
        return grad_x, None, None, None, None, None, None, None


fake_quantize = FakeQuantizeFunction.apply


SAMPLE_CUTOFF = 1000


def sample_tensor(prop, x):
    if x.numel() < SAMPLE_CUTOFF:
        return x

    cutoff_prop = SAMPLE_CUTOFF / x.numel()
    if cutoff_prop > prop:
        prop = cutoff_prop

    x = x.view(-1)
    probs = tensor([prop], device=x.device).expand_as(x)
    out = empty(probs.shape, dtype=bool, device=probs.device)
    mask = bernoulli(probs, out=out)
    return x[mask]


class IntegerQuantizer(Module):
    def __init__(
        self,
        num_bits: int,
        signed: bool,
        use_momentum: bool,
        symmetric: bool = False,
        momentum: float = 0.01,
        percentile: Optional[float] = None,
        sample: Optional[float] = None,
    ):
        super(IntegerQuantizer, self).__init__()
        self.register_buffer("min_val", tensor([]))
        self.register_buffer("max_val", tensor([]))
        self.momentum = momentum
        self.num_bits = num_bits
        self.signed = signed
        self.symmetric = symmetric
        self.eps = finfo(float32).eps

        self.momentum_min_max = use_momentum

        if percentile is None:
            self.min_fn = torch_min
            self.max_fn = torch_max
        else:
            self.min_fn = lambda t: kthvalue(t.flatten(), max(1, min(t.numel(), int(t.numel() * percentile))))[0]
            self.max_fn = lambda t: kthvalue(t.flatten(), min(t.numel(), max(1, int(t.numel() * (1 - percentile)))),)[0]

        if sample is None:
            self.sample_fn = lambda x: x
        else:
            assert percentile is not None
            self.sample_fn = lambda x: sample_tensor(sample, x)
        pass

    def update_ranges(self, x):
        min_val = self.min_val
        max_val = self.max_val

        x = self.sample_fn(x)
        current_min = self.min_fn(x)
        current_max = self.max_fn(x)

        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = current_min
            max_val = current_max
        else:
            if self.momentum_min_max:
                min_val = min_val + self.momentum * (current_min - min_val)
                max_val = max_val + self.momentum * (current_max - max_val)
            else:
                min_val = torch_min(current_min, min_val)
                max_val = torch_max(current_max, max_val)

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        if self.training:
            self.update_ranges(x.detach())

        return fake_quantize(x,
                             self.max_val,
                             self.min_val,
                             self.num_bits,
                             self.signed,
                             self.eps,
                             self.symmetric,
                             )
