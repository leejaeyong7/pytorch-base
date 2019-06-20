# -*- coding: utf-8 -*-
"""Boilerplate Model Class definition.

Defines example model that can be used as a base for actual model.

Model extends torch.nn.Module which should override __init__ and forward
functions.
"""

import torch.nn as nn

class FooModel(nn.Module):
    def __init__(self):
        """Model Constructor.

        Setup attributes and networks architectures
        """
        super(FooModel, self).__init__()
        # TODO: Implement model architecture below
        pass

    def to(self, device):
        """Model device setup

        While model.to(device) usually suffices, this allows device setup
        for additional variables that are dynamically created.
        """
        super(FooModel, self).to(device)
        self.device = device
        # TODO: Add dynamically created variables here
        return self

    def forward(self, x):
        """Model forward pass.

        Setup input => output flow here.
        """
        # TODO: Implement model forward pass below
        pass

