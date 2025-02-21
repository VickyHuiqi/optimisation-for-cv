import torch
import torch.nn as nn
import numpy as np

def get_layers(model):
    """ 
    Recursively extract all layers from a model.

    Args:
        model (nn.Module): The PyTorch model from which to extract layers.

    Returns:
        list: A list of model layers in depth-first order.
    """
    layers = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            layers.extend(get_layers(module))  # Recursively add layers
        else:
            layers.append(module)
    return layers

def load_yolov2_weights(model, weights_path):
    """
    Load YOLOv2 weights into the Darknet19Detector model.

    Args:
        model (nn.Module): The PyTorch model to load weights into.
        weights_path (str): Path to the binary weights file.

    Returns:
        nn.Module: The model with loaded weights.
    """
    # Load raw weights from file
    with open(weights_path, "rb") as f:
        weights = np.fromfile(f, dtype=np.float32)
    
    ptr = 0
    layers = get_layers(model)  # Extract all layers in correct order

    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            # Check if followed by BatchNorm
            next_idx = layers.index(layer) + 1
            if next_idx < len(layers) and isinstance(layers[next_idx], nn.BatchNorm2d):
                bn = layers[next_idx]

                # Load BatchNorm weights: [bias, weight, running_mean, running_var]
                num_bn_biases = bn.bias.numel()
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_bn_biases])); ptr += num_bn_biases
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_bn_biases])); ptr += num_bn_biases
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + num_bn_biases])); ptr += num_bn_biases
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + num_bn_biases])); ptr += num_bn_biases

                # Load Conv2d weights
                num_weights = layer.weight.numel()
                layer.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(layer.weight))
                ptr += num_weights
            else:
                # No BatchNorm, load Conv bias first
                num_biases = layer.bias.numel()
                layer.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_biases])); ptr += num_biases

                # Load Conv2d weights
                num_weights = layer.weight.numel()
                layer.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_weights]).view_as(layer.weight))
                ptr += num_weights

    return model