from statistics import variance
import warnings
import torch
import torch.nn as nn
from torch._six import inf
from typing import Union, Iterable, final


# _tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

# def get_param_plot_stats(
#         parameters: _tensor_or_tensors, norm_type: float = 2.0, prefix: str = "",
#         error_if_nonfinite: bool = False) -> torch.Tensor:
#     r"""Calculate gradient norm of an iterable of parameters.

#     The norm is computed over all gradients together, as if they were
#     concatenated into a single vector.

#     Args:
#         parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
#             single Tensor that will have gradients normalized
#         max_norm (float or int): max norm of the gradients
#         norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
#             infinity norm.
#         error_if_nonfinite (bool): if True, an error is thrown if the total
#             norm of the gradients from :attr:`parameters` is ``nan``,
#             ``inf``, or ``-inf``. Default: False (will switch to True in the future)

#     Returns:
#         Total norm of the parameters (viewed as a single vector).
#     """
#     if isinstance(parameters, torch.Tensor):
#         parameters = [parameters]
#     parameters = [p for p in parameters if p.grad is not None]
#     norm_type = float(norm_type)
#     if len(parameters) == 0:
#         return torch.tensor(0.)
#     device = parameters[0].grad.device
#     if norm_type == inf:
#         raise NotImplementedError("Infinity norm is not implemented yet")
#         # norms = [p.grad.detach().abs().max().to(device) for p in parameters]
#         # total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
#     else:
#         grad_norms = []
#         params_norm = []
#         max_singular_v = []
#         for idx,p in enumerate(parameters):
#             grad_norms.append(torch.norm(p.grad.detach(), norm_type).to(device))
#             params_norm.append(torch.norm(p.detach(), norm_type).to(device))
#             if len(p.shape)==2:
#                 s = torch.linalg.svdvals(p)
#                 max_singular_v.append(("max_singular_values/"+prefix+str(idx), s[0]))
#         total_norm = torch.norm(torch.stack(grad_norms), norm_type)
#         total_param_norm = torch.norm(torch.stack(params_norm), norm_type)
    
#     if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
#         raise RuntimeError(
#             f'The total norm of order {norm_type} for gradients from '
#             '`parameters` is non-finite, so it cannot be clipped. To disable '
#             'this error and scale the gradients by the non-finite norm anyway, '
#             'set `error_if_nonfinite=False`')

#     return total_norm, total_param_norm, max_singular_v


def get_weight_stats(weight_matrix, matrix_name):
    final_res = []
    s = torch.linalg.svdvals(weight_matrix)
    final_res.append(("max_singular_values/"+matrix_name, s[0]))
    explained_variance = s**2 / torch.sum(s**2)
    explained_variance = torch.cumsum(explained_variance, dim=0)
    variances = [0.5,0.8,0.9,0.95,0.99]
    for var in variances:
        final_res.append((f"explained_variance/{var}/"+matrix_name, torch.searchsorted(explained_variance, var)))
    return final_res


def get_param_plot_stats_sn(
    module: nn.Module, norm_type: float = 2.0, prefix: str = "",
        error_if_nonfinite: bool = False) -> torch.Tensor:
    norm_type = float(norm_type)
    if norm_type == inf:
        raise NotImplementedError("Infinity norm is not implemented yet")
    else:
        grad_norms = []
        params_norm = []
        singular_v_stats = []
        for name, mod in module.named_children():
            if type(mod) is nn.Linear:
                grad_norms.append(torch.norm(mod.weight.grad.detach(), norm_type))
                params_norm.append(torch.norm(mod.weight.detach(), norm_type))
                if len(mod.weight.shape)==2:
                    singular_v_stats += get_weight_stats(mod.weight.detach(), prefix+name)
            elif type(mod) is nn.GRUCell:
                grad_norms.append(torch.norm(mod.weight_hh.grad.detach(), norm_type))
                grad_norms.append(torch.norm(mod.weight_ih.grad.detach(), norm_type))
                params_norm.append(torch.norm(mod.weight_ih.detach(), norm_type))
                params_norm.append(torch.norm(mod.weight_hh.detach(), norm_type))
                if len(mod.weight_ih.shape)==2:
                    singular_v_stats += get_weight_stats(mod.weight_ih.detach(), prefix+name+"_ih")
                    singular_v_stats += get_weight_stats(mod.weight_hh.detach(), prefix+name+"_hh")
            else:
                try:
                    grad_norms.append(torch.norm(mod.parametrizations.weight.original.grad.detach(), norm_type))
                    params_norm.append(torch.norm(mod.weight.detach(), norm_type))
                    if len(mod.weight.shape)==2:
                        singular_v_stats += get_weight_stats(mod.parametrizations.weight.original.detach(), prefix+name)
                except:
                    grad_norms.append(torch.norm(mod.parametrizations.weight_ih.original.grad.detach(), norm_type))
                    grad_norms.append(torch.norm(mod.parametrizations.weight_hh.original.grad.detach(), norm_type))
                    params_norm.append(torch.norm(mod.weight_ih.detach(), norm_type))
                    params_norm.append(torch.norm(mod.weight_hh.detach(), norm_type))
                    if len(mod.weight_ih.shape)==2:
                        singular_v_stats += get_weight_stats(mod.parametrizations.weight_ih.original.detach(), prefix+name+"_ih")
                        singular_v_stats += get_weight_stats(mod.parametrizations.weight_hh.original.detach(), prefix+name+"_hh")
        total_norm = torch.norm(torch.stack(grad_norms), norm_type)
        total_param_norm = torch.norm(torch.stack(params_norm), norm_type)
    
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')

    return total_norm, total_param_norm, singular_v_stats
