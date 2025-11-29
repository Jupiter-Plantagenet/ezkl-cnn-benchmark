"""
PyTorch model definitions for single-layer and composite CNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialActivation(nn.Module):
    """Polynomial activation function (x^degree)"""
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
    
    def forward(self, x):
        return torch.pow(x, self.degree)


class SingleLayerNet(nn.Module):
    """Wrapper for isolating a single CNN layer for profiling"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return self.layer(x)


def create_single_layer_model(layer_name, config):
    """
    Create a single-layer model based on configuration.
    
    Args:
        layer_name: Name of the layer (e.g., 'ReLU', 'Conv2d_k3_s1')
        config: Configuration dictionary with 'type' and 'params'
    
    Returns:
        SingleLayerNet wrapping the specified layer
    """
    layer_type = config['type']
    params = config['params']
    
    if layer_type == 'activation':
        if layer_name == 'ReLU':
            layer = nn.ReLU()
        elif layer_name == 'SiLU':
            layer = nn.SiLU()
        elif layer_name == 'Tanh':
            layer = nn.Tanh()
        elif layer_name == 'Poly':
            layer = PolynomialActivation(degree=params.get('degree', 2))
        else:
            raise ValueError(f"Unknown activation: {layer_name}")
    
    elif layer_type == 'pooling':
        kernel_size = params['kernel_size']
        if 'MaxPool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=kernel_size)
        elif 'AvgPool' in layer_name:
            layer = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            raise ValueError(f"Unknown pooling: {layer_name}")
    
    elif layer_type == 'convolution':
        if 'Depthwise' in layer_name:
            layer = nn.Conv2d(
                in_channels=params['in_channels'],
                out_channels=params['out_channels'],
                kernel_size=params['kernel_size'],
                stride=params.get('stride', 1),
                padding=params.get('padding', 0),
                groups=params['groups']
            )
        else:
            layer = nn.Conv2d(
                in_channels=params['in_channels'],
                out_channels=params['out_channels'],
                kernel_size=params['kernel_size'],
                stride=params.get('stride', 1),
                padding=params.get('padding', 0)
            )
    
    elif layer_type == 'normalization':
        if 'BatchNorm' in layer_name:
            layer = nn.BatchNorm2d(num_features=params['num_features'])
        elif 'LayerNorm' in layer_name:
            layer = nn.LayerNorm(normalized_shape=params['normalized_shape'])
        else:
            raise ValueError(f"Unknown normalization: {layer_name}")
    
    elif layer_type == 'linear':
        layer = nn.Linear(
            in_features=params['in_features'],
            out_features=params['out_features']
        )
    
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return SingleLayerNet(layer)


class CompositeCNN(nn.Module):
    """Generic composite CNN architecture for CIFAR-10"""
    def __init__(self, architecture_spec):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for layer_type, params in architecture_spec:
            if layer_type == 'conv':
                self.layers.append(nn.Conv2d(**params))
            elif layer_type == 'relu':
                self.layers.append(nn.ReLU())
            elif layer_type == 'poly':
                self.layers.append(PolynomialActivation(degree=params.get('degree', 2)))
            elif layer_type == 'maxpool':
                self.layers.append(nn.MaxPool2d(**params))
            elif layer_type == 'avgpool':
                self.layers.append(nn.AvgPool2d(**params))
            elif layer_type == 'flatten':
                self.layers.append(nn.Flatten())
            elif layer_type == 'linear':
                self.layers.append(nn.Linear(**params))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def create_composite_model(architecture_name, architecture_config):
    """
    Create a composite CNN model for CIFAR-10.
    
    Args:
        architecture_name: Name of the architecture (e.g., 'CNN_ReLU')
        architecture_config: Configuration dictionary with 'architecture' spec
    
    Returns:
        CompositeCNN model
    """
    architecture_spec = architecture_config['architecture']
    return CompositeCNN(architecture_spec)


if __name__ == '__main__':
    # Test model creation
    from config.experiment_config import LAYER_CONFIGS, COMPOSITE_ARCHITECTURES
    
    print("Testing single-layer models...")
    for layer_name, config in LAYER_CONFIGS.items():
        model = create_single_layer_model(layer_name, config)
        dummy_input = torch.randn(*config['input_shape'])
        output = model(dummy_input)
        print(f"{layer_name}: Input {dummy_input.shape} -> Output {output.shape}")
    
    print("\nTesting composite models...")
    for arch_name, arch_config in COMPOSITE_ARCHITECTURES.items():
        model = create_composite_model(arch_name, arch_config)
        dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 input
        output = model(dummy_input)
        print(f"{arch_name}: Input {dummy_input.shape} -> Output {output.shape}")
