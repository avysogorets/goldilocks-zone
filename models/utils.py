from .model_base import ClassificationModelBase
from collections import OrderedDict
import torch


def get_trainable_parameters(
            model: ClassificationModelBase) -> torch.Tensor:
    
    """ Returns a detached flattened tensor
        containing all trainable parameters.
    """
    parameters = []
    for module in model.modules():
        attr = []
        if hasattr(module, 'registered_parameters_name'):
            for name in module.registered_parameters_name:
                param = getattr(module, name)
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                attr.append(param.detach().reshape(-1))
        else:
            for _,param in module._parameters.items():
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                attr.append(param.detach().reshape(-1))
        if len(attr)<1:
            continue
        attr = torch.cat(attr)
        parameters.append(attr)
    parameters = torch.cat(parameters).reshape(1,-1)
    return parameters


def flip_parameters_to_tensors(
            model: ClassificationModelBase) -> None:
    
    """ Removes all parameters from the model and substitutes
        them for tensors. Cannot be called on the same model once.
    """
    parameters = get_trainable_parameters(model)
    for module in model.modules():
        if hasattr(module, 'registered_parameters_name'):
            err_msg = "looks like this model has been flipped before..."
            raise AttributeError(err_msg)
        attr = []
        for named_param in module._parameters.items():
            attr.append(named_param)
        module._parameters = OrderedDict()
        setattr(module, 'registered_parameters_name', [])
        for i in attr:
            if i[1] is not None:
                is_tracked = i[1].requires_grad
                param = torch.zeros(i[1].shape, requires_grad=is_tracked)
                setattr(module, i[0], param)
            else:
                setattr(module, i[0], None)
            module.registered_parameters_name.append(i[0])
    set_trainable_parameters(model, parameters)


def set_trainable_parameters(
            model: ClassificationModelBase,
            theta: torch.Tensor) -> None:
    
    """ Sets theta as trainable parameters of the model.
    
        Warning: this method sets trainable parameters only!
        Note that batchnorm statistics are not trainable.
        Not valid for model copying, use state_dict instead.
        
        Warning: When the model is flipped, this method sets
        potentially non-leaf tensors for model parameters.
        Thus, training / retriving gradients of the model
        parameters may be hindered.
        
        Warning: When the model is not flipped, any prior
        modifications made to theta are untracked because
        parameters of the model are updated in an untrackable
        fashion as param.data = theta[...]. 
    """
    theta = theta.reshape(1,-1)
    if not theta.requires_grad:
        theta = theta.requires_grad_(True)
    count = 0  
    for module in model.modules():
        if hasattr(module, 'registered_parameters_name'):
            for name in module.registered_parameters_name:
                param = getattr(module, name)
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                a = count
                b = a + param.numel()
                t = torch.reshape(theta[0,a:b], param.shape)
                setattr(module, name, t)
                count += param.numel()
        else:
            for named_param in module._parameters.items():
                name, param = named_param
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                a = count
                b = a + param.numel()
                t = torch.reshape(theta[0,a:b], param.shape)
                module._parameters[name].data = t
                count += param.numel()


def get_current_gradients(
            model: ClassificationModelBase) -> torch.Tensor:
    
    """ Returns current gradients of the model, flattened.
    """
    w = get_trainable_parameters(model).squeeze()
    gradients = torch.zeros_like(w)
    count = 0
    for module in model.modules():
        for named_param in module._parameters.items():
            _,param = named_param
            if param is None:
                continue
            if param.grad is None:
                continue
            curr_layer_grad = param.grad
            a = count
            b = a+curr_layer_grad.numel()
            curr_layer_grad = curr_layer_grad.reshape(-1)
            gradients[a:b] = curr_layer_grad
            count += curr_layer_grad.numel()
    return gradients


def set_trainable_parameters_in_layer(
            model: ClassificationModelBase,
            theta: torch.Tensor,
            layer: int) -> None:
    
    """ Same as <set_trainable_parameters> for a particular model layer.
        See warnings of <set_trainable_parameters>.
    """
    theta = theta.reshape(1,-1).to(model.dtype)
    if not theta.requires_grad:
        theta = theta.requires_grad_(True)
    count = 0
    layer_idx = -1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            layer_idx += 1
        if layer_idx == layer:
            if hasattr(module, 'registered_parameters_name'):
                for name in module.registered_parameters_name:
                    param = getattr(module, name)
                    if param is None:
                        continue
                    if not param.requires_grad:
                        continue
                    a = count
                    b = a + param.numel()
                    t = torch.reshape(theta[0,a:b], param.shape)
                    setattr(module, name, t)
                    count += param.numel()
            else:
                for named_param in module._parameters.items():
                    name, param = named_param
                    if param is None:
                        continue
                    if not param.requires_grad:
                        continue
                    a = count
                    b = a + param.numel()
                    t = torch.reshape(theta[0,a:b], param.shape)
                    module._parameters[name].data = t
                    count += param.numel()


def get_trainable_parameters_in_layer(
            model: ClassificationModelBase,
            layer: int) -> torch.Tensor:
    
    """ Same as <set_trainable_parameters> for a particular model layer.
        See warnings of <set_trainable_parameters>.
    """
    parameters = []
    layer_idx = -1
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            layer_idx += 1
        if layer_idx == layer:
            attr = []
            if hasattr(module, 'registered_parameters_name'):
                for name in module.registered_parameters_name:
                    param = getattr(module, name)
                    if param is None:
                        continue
                    if not param.requires_grad:
                        continue
                    attr.append(param.detach().reshape(-1))
            else:
                for _,param in module._parameters.items():
                    if param is None:
                        continue
                    if not param.requires_grad:
                        continue
                    attr.append(param.detach().reshape(-1))
            if len(attr)<1:
                continue
            attr = torch.cat(attr)
            parameters.append(attr)
    parameters = torch.cat(parameters).reshape(1,-1)
    return parameters


def get_layer_idxs(model):
    idxs = [(0,0)]
    for module in model.modules():
        s_idx = idxs[-1][-1]
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            count = 0
            if hasattr(module, 'weight'):
                count += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                count += module.bias.numel()    
            e_idx = s_idx + count
            idxs.append((s_idx, e_idx))
    return idxs[1:]


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses