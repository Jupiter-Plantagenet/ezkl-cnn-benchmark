"""
Experiment configuration for EZKL CNN benchmarking.
Defines the 40 experiments as outlined in the paper methodology.
"""

# Scale settings for all experiments (EZKL uses input_scale/param_scale)
# Lower scale = fewer bits of precision, smaller circuits, faster proving
# Higher scale = more bits of precision, larger circuits, slower proving
# Scale 7 = ~7 bits (efficiency mode)
# Scale 10 = ~10 bits (accuracy mode)
SCALE_SETTINGS = {
    'efficiency': 7,   # Lower precision, smaller circuit
    'accuracy': 10     # Higher precision, larger circuit  
}

# Legacy mapping for backwards compatibility during experiments
TOLERANCE_VALUES = [0.5, 2.0]  # 0.5 -> scale 10, 2.0 -> scale 7

# Standard input configuration for core layer benchmarking
STANDARD_INPUT_SHAPE = (1, 32, 64, 64)  # Batch=1, Channels=32, H=64, W=64

# Scaling study input shapes (for Conv2d and Dense)
SCALING_SHAPES = {
    'small': (1, 16, 32, 32),
    'standard': (1, 32, 64, 64),
    'large': (1, 64, 128, 128)
}

# 12 representative layer types
LAYER_CONFIGS = {
    # Activations (4 types)
    'ReLU': {
        'type': 'activation',
        'input_shape': (1, 64, 64),  # 1D for activations
        'params': {}
    },
    'SiLU': {
        'type': 'activation',
        'input_shape': (1, 64, 64),
        'params': {}
    },
    'Tanh': {
        'type': 'activation',
        'input_shape': (1, 64, 64),
        'params': {}
    },
    'Poly': {
        'type': 'activation',
        'input_shape': (1, 64, 64),
        'params': {'degree': 2}  # x^2
    },
    
    # Pooling (2 types)
    'MaxPool2d_k2': {
        'type': 'pooling',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {'kernel_size': 2}
    },
    'AvgPool2d_k2': {
        'type': 'pooling',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {'kernel_size': 2}
    },
    
    # Convolution (3 types)
    'Conv2d_k3_s1': {
        'type': 'convolution',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {
            'in_channels': 32,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
    },
    'Conv2d_k3_s2': {
        'type': 'convolution',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {
            'in_channels': 32,
            'out_channels': 64,
            'kernel_size': 3,
            'stride': 2,
            'padding': 1
        }
    },
    'DepthwiseConv2d': {
        'type': 'convolution',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {
            'in_channels': 32,
            'out_channels': 32,
            'kernel_size': 3,
            'groups': 32,
            'stride': 1,
            'padding': 1
        }
    },
    
    # Normalization (2 types)
    'BatchNorm2d': {
        'type': 'normalization',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {'num_features': 32}
    },
    'LayerNorm': {
        'type': 'normalization',
        'input_shape': STANDARD_INPUT_SHAPE,
        'params': {'normalized_shape': [32, 64, 64]}
    },
    
    # Linear (1 type)
    'Dense': {
        'type': 'linear',
        'input_shape': (1, 256),
        'params': {
            'in_features': 256,
            'out_features': 128
        }
    }
}

# Scaling study configurations (for Conv2d and Dense only)
SCALING_CONFIGS = {
    'Conv2d_Scaling_Small': {
        'base_layer': 'Conv2d_k3_s1',
        'input_shape': SCALING_SHAPES['small'],
        'params': {
            'in_channels': 16,
            'out_channels': 32,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
    },
    'Conv2d_Scaling_Large': {
        'base_layer': 'Conv2d_k3_s1',
        'input_shape': SCALING_SHAPES['large'],
        'params': {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        }
    },
    'Dense_Scaling_Small': {
        'base_layer': 'Dense',
        'input_shape': (1, 128),
        'params': {
            'in_features': 128,
            'out_features': 64
        }
    },
    'Dense_Scaling_Large': {
        'base_layer': 'Dense',
        'input_shape': (1, 1024),
        'params': {
            'in_features': 1024,
            'out_features': 512
        }
    }
}

# Composite CNN architectures for CIFAR-10
COMPOSITE_ARCHITECTURES = {
    'CNN_ReLU': {
        'description': 'Conv -- ReLU -- MaxPool -- Conv -- ReLU -- MaxPool -- Dense',
        'architecture': [
            ('conv', {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}),
            ('relu', {}),
            ('maxpool', {'kernel_size': 2}),
            ('conv', {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}),
            ('relu', {}),
            ('maxpool', {'kernel_size': 2}),
            ('flatten', {}),
            ('linear', {'in_features': 64 * 8 * 8, 'out_features': 10})
        ]
    },
    'CNN_Poly': {
        'description': 'Conv -- Poly -- AvgPool -- Conv -- Poly -- AvgPool -- Dense',
        'architecture': [
            ('conv', {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}),
            ('poly', {'degree': 2}),
            ('avgpool', {'kernel_size': 2}),
            ('conv', {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}),
            ('poly', {'degree': 2}),
            ('avgpool', {'kernel_size': 2}),
            ('flatten', {}),
            ('linear', {'in_features': 64 * 8 * 8, 'out_features': 10})
        ]
    },
    'CNN_Mixed': {
        'description': 'Conv -- Poly -- AvgPool -- Conv -- ReLU -- AvgPool -- Dense',
        'architecture': [
            ('conv', {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}),
            ('poly', {'degree': 2}),
            ('avgpool', {'kernel_size': 2}),
            ('conv', {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}),
            ('relu', {}),
            ('avgpool', {'kernel_size': 2}),
            ('flatten', {}),
            ('linear', {'in_features': 64 * 8 * 8, 'out_features': 10})
        ]
    },
    'CNN_Strided': {
        'description': 'Conv (stride 2) -- ReLU -- Conv (stride 2) -- ReLU -- Dense',
        'architecture': [
            ('conv', {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('relu', {}),
            ('conv', {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}),
            ('relu', {}),
            ('flatten', {}),
            ('linear', {'in_features': 64 * 8 * 8, 'out_features': 10})
        ]
    }
}

# EZKL configuration
EZKL_CONFIG = {
    'scales': [7, 8, 9, 10],
    'scale_rebase_multipliers': [0.0, 0.5, 1.0, 2.0],
    'bits': 16,
    'logrows': 17
}

# Training configuration for composite models
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'optimizer': 'adam'
}

# Experiment counts
EXPERIMENT_COUNTS = {
    'core_layers': len(LAYER_CONFIGS) * len(TOLERANCE_VALUES),  # 12 * 2 = 24
    'scaling_study': len(SCALING_CONFIGS) * len(TOLERANCE_VALUES),  # 4 * 2 = 8
    'composite': len(COMPOSITE_ARCHITECTURES) * len(TOLERANCE_VALUES),  # 4 * 2 = 8
    'total': 40
}
