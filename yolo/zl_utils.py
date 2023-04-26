import os
import sys
import argparse
from collections import OrderedDict

import numpy as np

def is_pan_arch(cfg_file_path):
    """Determine whether the yolo model is with PAN architecture."""
    with open(cfg_file_path, 'r') as f:
        cfg_lines = [l.strip() for l in f.readlines()]
    yolos_or_upsamples = [l for l in cfg_lines
                            if l in ['[yolo]', '[upsample]']]
    yolo_count = len([l for l in yolos_or_upsamples if l == '[yolo]'])
    upsample_count = len(yolos_or_upsamples) - yolo_count
    assert yolo_count in (2, 3, 4)  # at most 4 yolo layers
    assert upsample_count == yolo_count - 1 or upsample_count == 0
    # the model is with PAN if an upsample layer appears before the 1st yolo
    return yolos_or_upsamples[0] == '[upsample]'

def get_output_convs(layer_configs):
    """Find output conv layer names from layer configs.
    The output conv layers are those conv layers immediately proceeding
    the yolo layers.
    # Arguments
        layer_configs: output of the DarkNetParser, i.e. a OrderedDict of
                       the yolo layers.
    """
    output_convs = []
    previous_layer = None
    for current_layer in layer_configs.keys():
        if previous_layer is not None and current_layer.endswith('yolo'):
            assert previous_layer.endswith('convolutional')
            activation = layer_configs[previous_layer]['activation']
            if activation == 'linear':
                output_convs.append(previous_layer)
            elif activation == 'logistic':
                output_convs.append(previous_layer + '_lgx')
            else:
                raise TypeError('unexpected activation: %s' % activation)
        previous_layer = current_layer
    return output_convs

def get_category_num(cfg_file_path):
    """Find number of output classes of the yolo model."""
    with open(cfg_file_path, 'r') as f:
        cfg_lines = [l.strip() for l in f.readlines()]
    classes_lines = [l for l in cfg_lines if l.startswith('classes=')]
    assert len(set(classes_lines)) == 1
    return int(classes_lines[-1].split('=')[-1].strip())

def get_anchors(cfg_file_path):
    """Get anchors of all yolo layers from the cfg file."""
    with open(cfg_file_path, 'r') as f:
        cfg_lines = f.readlines()
    yolo_lines = [l.strip() for l in cfg_lines if l.startswith('[yolo]')]
    mask_lines = [l.strip() for l in cfg_lines if l.startswith('mask')]
    anch_lines = [l.strip() for l in cfg_lines if l.startswith('anchors')]
    assert len(mask_lines) == len(yolo_lines)
    assert len(anch_lines) == len(yolo_lines)
    anchor_list = eval('[%s]' % anch_lines[0].split('=')[-1])
    mask_strs = [l.split('=')[-1] for l in mask_lines]
    masks = [eval('[%s]' % s)  for s in mask_strs]
    anchors = []
    for mask in masks:
        curr_anchors = []
        for m in mask:
            curr_anchors.append(anchor_list[m * 2])
            curr_anchors.append(anchor_list[m * 2 + 1])
        anchors.append(curr_anchors)
    return anchors

class DarkNetParser(object):
    """Definition of a parser for DarkNet-based YOLO model."""

    def __init__(self, supported_layers=None):
        """Initializes a DarkNetParser object.
        Keyword argument:
        supported_layers -- a string list of supported layers in DarkNet naming convention,
        parameters are only added to the class dictionary if a parsed layer is included.
        """

        # A list of YOLO layers containing dictionaries with all layer
        # parameters:
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers if supported_layers else \
                                ['net', 'convolutional', 'maxpool', 'shortcut',
                                 'route', 'upsample', 'yolo']
        self.layer_counter = 0

    def parse_cfg_file(self, cfg_file_path):
        """Takes the yolov?.cfg file and parses it layer by layer,
        appending each layer's parameters as a dictionary to layer_configs.
        Keyword argument:
        cfg_file_path
        """
        with open(cfg_file_path, 'r') as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs

    def _next_layer(self, remainder):
        """Takes in a string and segments it by looking for DarkNet delimiters.
        Returns the layer parameters and the remaining string after the last delimiter.
        Example for the first Conv layer in yolo.cfg ...
        [convolutional]
        batch_normalize=1
        filters=32
        size=3
        stride=1
        pad=1
        activation=leaky
        ... becomes the following layer_dict return value:
        {'activation': 'leaky', 'stride': 1, 'pad': 1, 'filters': 32,
        'batch_normalize': 1, 'type': 'convolutional', 'size': 3}.
        '001_convolutional' is returned as layer_name, and all lines that follow in yolo.cfg
        are returned as the next remainder.
        Keyword argument:
        remainder -- a string with all raw text after the previously parsed layer
        """
        remainder = remainder.split('[', 1)
        while len(remainder[0]) > 0 and remainder[0][-1] == '#':
            # '#[...' case (the left bracket is proceeded by a pound sign),
            # assuming this layer is commented out, so go find the next '['
            remainder = remainder[1].split('[', 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            # no left bracket found in remainder
            return None, None, None
        remainder = remainder.split(']', 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            # no right bracket
            raise ValueError('no closing bracket!')
        if layer_type not in self.supported_layers:
            raise ValueError('%s layer not supported!' % layer_type)

        out = remainder.split('\n[', 1)
        if len(out) == 2:
            layer_param_block, remainder = out[0], '[' + out[1]
        else:
            layer_param_block, remainder = out[0], ''
        layer_param_lines = layer_param_block.split('\n')
        # remove empty lines
        layer_param_lines = [l.lstrip() for l in layer_param_lines if l.lstrip()]
        # don't parse yolo layers
        if layer_type == 'yolo':  layer_param_lines = []
        skip_params = ['steps', 'scales'] if layer_type == 'net' else []
        layer_name = str(self.layer_counter).zfill(3) + '_' + layer_type
        layer_dict = dict(type=layer_type)
        for param_line in layer_param_lines:
            param_line = param_line.split('#')[0]
            if not param_line:  continue
            assert '[' not in param_line
            param_type, param_value = self._parse_params(param_line, skip_params)
            layer_dict[param_type] = param_value
        self.layer_counter += 1
        return layer_dict, layer_name, remainder

    def _parse_params(self, param_line, skip_params=None):
        """Identifies the parameters contained in one of the cfg file and returns
        them in the required format for each parameter type, e.g. as a list, an int or a float.
        Keyword argument:
        param_line -- one parsed line within a layer block
        """
        param_line = param_line.replace(' ', '')
        param_type, param_value_raw = param_line.split('=')
        assert param_value_raw
        param_value = None
        if skip_params and param_type in skip_params:
            param_type = None
        elif param_type == 'layers':
            layer_indexes = list()
            for index in param_value_raw.split(','):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = param_value_raw[0] == '-' and \
                param_value_raw[1:].isdigit()
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)
        return param_type, param_value


def get_h_and_w(layer_configs):
    """Find input height and width of the yolo model from layer configs."""
    net_config = layer_configs['000_net']
    return net_config['height'], net_config['width']
