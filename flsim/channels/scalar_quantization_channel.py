#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# pyre-ignore-all-errors[16]

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from flsim.channels.base_channel import FLChannelConfig, IdentityChannel, Message
from flsim.utils.config_utils import fullclassname, init_self_cfg
from torch.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


class ScalarQuantizationChannel(IdentityChannel):
    """
    Implements a channel that emulates scalar quantization from 1 to 8
    bits per weight (8 bits per weight corresponds to int8 quantization).
    We simulate this by successively quantizing and dequantizing. This way,
    the rest of the training is transparent for aggregators, reducers,
    trainers and so on.

    Notes:
        - We can perform either per_tensor quantization (same scale and
          zero_point for every parameters in a weight matrix) or per_channel
          quantization (each channel has its own scale and zero_point). Set
          quantize_per_tensor = False to perform per_channel quantization.
        - We rely on the very simple MinMax observers for both per_tensor
          and per_channel quantization. This can be refined by leveraging the
          HistogramObserver for instance.
        - We do not quantize the biases for the moment since their compression
          overhead is very small.
        - We arbitrarily choose to set the int_repr() of a quantized tensor
          to [-(2 ** (n_bits - 1)), (2 ** (n_bits - 1)) - 1]; symmetric around 0.
        - All the quantized tensors share the same type, ie `torch.qint8`.
          However, when quantizing to less than 8 bits, this is not memory
          efficient since each element is stored over 1 byte anyway. Since
          we are interested only in emulation for the moment, that's good.
          We could also have relied on the fake_quantize primitives but we
          prefer to rely on the true quantization operators.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ScalarQuantizationChannelConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        
        if not (1 <= self.cfg.n_bits_client_to_server <= 8):
            raise ValueError(
                "ScalarQuantizationChannel expects n_bits_client_to_server between 1 and 8 (included)."
            )
        if not (1 <= self.cfg.n_bits_server_to_client <= 8):
            raise ValueError(
                "ScalarQuantizationChannel expects n_bits_server_to_client between 1 and 8 (included)."
            )
        if self.cfg.qscheme not in ("affine", "symmetric"):
            raise ValueError(
                "ScalarQuantizationChannel qscheme should be either affine or symmetric."
            )

        self.quant_min_client_to_server = -(2 ** (self.cfg.n_bits_client_to_server - 1))
        self.quant_max_client_to_server = (2 ** (self.cfg.n_bits_client_to_server - 1)) - 1
        
        self.quant_min_server_to_client = -(2 ** (self.cfg.n_bits_server_to_client - 1))
        self.quant_max_server_to_client = (2 ** (self.cfg.n_bits_server_to_client - 1)) - 1

        self.observer_client_to_server, self.quantizer_client_to_server = self.get_observers_and_quantizers(self.cfg.n_bits_client_to_server)
        self.observer_server_to_client, self.quantizer_server_to_client = self.get_observers_and_quantizers(self.cfg.n_bits_server_to_client)
        
        self.use_shared_qparams = self.cfg.use_shared_qparams or self.cfg.sec_agg_mode

    def _calc_message_size_client_to_server(self, message: Message):
        """
        We compute the size of the compressed message as follows:
            - for the weights (compressed): n_bits / 8 bytes per element
              - use an additional bit to account for ooverflow during sec agg.
            - for the biases (not compressed): 4 bytes per element
            - for the scales (one for each layer or one for each layer channel
              depending on quantize_per_tensor): 8 bytes / element (fp64)
            - for the zero_points (one for each layer or one for each layer channel
              depending on quantize_per_tensor): 4 bytes / element (int32)
            - NOTE: scales and zero_points are not sent to the server in shared qparams mode.
        """
        message_size_bytes = 0
        for param in message.model_state_dict.values():
            if param.ndim > 1:  # non-bias params are in int representation
                # we need an additional bits to accomdate for possible overflow after adding one-time pad
                n_bits = (
                    self.cfg.sec_agg_n_bits
                    if self.cfg.sec_agg_mode
                    else self.cfg.n_bits
                )
                message_size_bytes += param.numel() * n_bits / 8
                if self.use_shared_qparams:
                    continue  # qparams are not sent to the server if shared qparams are used.
                # size of scale(s) (fp64) and zero_point(s) (int32)
                if self.cfg.quantize_per_tensor:
                    message_size_bytes += ScalarQuantizationChannel.BYTES_PER_FP64
                    message_size_bytes += ScalarQuantizationChannel.BYTES_PER_FP32
                else:
                    n_scales = param.q_per_channel_scales().numel()
                    n_zero_points = param.q_per_channel_zero_points().numel()
                    message_size_bytes += (
                        ScalarQuantizationChannel.BYTES_PER_FP64 * n_scales
                    )
                    message_size_bytes += (
                        ScalarQuantizationChannel.BYTES_PER_FP32 * n_zero_points
                    )
            else:
                message_size_bytes += 4 * param.numel()
        return message_size_bytes

    def _calc_message_size_server_to_client(self, message: Message):
        message_size_bytes = super()._calc_message_size_server_to_client(message)
        if self.use_shared_qparams:
            for param in message.model_state_dict.values():
                if param.ndim > 1:  # non-bias params are in int representation
                    if self.cfg.quantize_per_tensor:
                        message_size_bytes += ScalarQuantizationChannel.BYTES_PER_FP64
                        message_size_bytes += (
                            0  # zero point is fixed at 0 for symmetric qscheme and hence need not be sent
                            if self.cfg.qscheme == "symmetric"
                            else ScalarQuantizationChannel.BYTES_PER_FP32
                        )
                    else:
                        n_scales = param.q_per_channel_scales().numel()
                        n_zero_points = param.q_per_channel_zero_points().numel()
                        message_size_bytes += (
                            ScalarQuantizationChannel.BYTES_PER_FP64 * n_scales
                        )
                        message_size_bytes += (
                            0  # zero point is fixed at 0 for symmetric qscheme and hence need not be sent
                            if self.cfg.qscheme == "symmetric"
                            else (
                                ScalarQuantizationChannel.BYTES_PER_FP32 * n_zero_points
                            )
                        )
        return message_size_bytes

    def get_observers_and_quantizers(self, n_bits: int):
        quant_min = -(2 ** (n_bits - 1))
        quant_max = (2 ** (n_bits - 1)) - 1

        if self.cfg.quantize_per_tensor:
            qscheme = (
                torch.per_tensor_symmetric
                if self.cfg.qscheme == "symmetric"
                else torch.per_tensor_affine
            )
            observer = MinMaxObserver(
                dtype=torch.qint8,
                qscheme=qscheme,
                quant_min=quant_min,
                quant_max=quant_max,
                reduce_range=False,
            )
            quantizer = torch.quantize_per_tensor
        else:
            qscheme = (
                torch.per_channel_symmetric
                if self.cfg.qscheme == "symmetric"
                else torch.per_channel_affine
            )
            observer = PerChannelMinMaxObserver(
                dtype=torch.qint8,
                qscheme=qscheme,
                quant_min=quant_min,
                quant_max=quant_max,
                reduce_range=False,
                ch_axis=0,
            )
            quantizer = torch.quantize_per_channel
        return observer, quantizer

    def _quantize(self, name: str, x: torch.Tensor, message: Message, mode: str) -> torch.Tensor:
        observer = self.observer_client_to_server if mode == 'client_to_server' else self.observer_server_to_client
        quantizer = self.quantizer_client_to_server if mode == 'client_to_server' else self.quantizer_server_to_client
        
        # important to reset values, otherwise takes running min and max
        observer.reset_min_max_vals()

        # forward through the observer to get scale(s) and zero_point(s)
        _ = observer(x)
        scale, zero_point = observer.calculate_qparams()
        # use shared qparams from server
        if self.use_shared_qparams and mode == 'client_to_server':
            if message.qparams is None:
                raise ValueError(
                    "global_qparams is necessary when shared qparams is enabled in channel."
                )
            scale, zero_point = message.qparams[name]

        # Simulate quantization. Not a no-op since we lose precision when quantizing.
        if self.cfg.quantize_per_tensor:
            xq = quantizer(x, float(scale), int(zero_point), dtype=torch.qint8)
        else:
            scale = scale.to(x.device)
            zero_point = zero_point.to(x.device)
            xq = quantizer(x, scale, zero_point, axis=0, dtype=torch.qint8)
        return xq

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        We quantize the weights but do not quantize the biases since
        the overhead is very small. We copy the state dict since the
        tensor format changes.
        """
        message.populate_state_dict()
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                new_state_dict[name] = self._quantize(name, param.data, message, mode='client_to_server')
            else:
                new_state_dict[name] = param.data

        message.model_state_dict = new_state_dict
        return message

    def _on_server_after_reception(self, message: Message) -> Message:
        """
        We dequantize the weights and do not dequantize the biases
        since they have not been quantized in the first place. We
        copy the state dict since the tensor format changes.
        """
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                new_state_dict[name] = (
                    # in sec agg mode, we have to perform a few more steps before
                    # dequantizing. We will, therefore, send int representations
                    # and will defer dequantization to the server.
                    param.data.int_repr()
                    if self.cfg.sec_agg_mode
                    # in non sec agg mode, we can dequantize right here as this is
                    # the first step.
                    else param.data.dequantize()
                )
            else:
                new_state_dict[name] = param.data
        message.model_state_dict = new_state_dict
        message.update_model_()
        return message
    
    def _on_client_after_reception(self, message: Message) -> Message:
        """
        We dequantize the weights we received from the server and do not dequantize the biases
        since they have not been quantized in the first place. We copy the state dict since 
        the tensor format changes.
        """
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                new_state_dict[name] = param.data.dequantize()
            else:
                new_state_dict[name] = param.data
        message.model_state_dict = new_state_dict
        message.update_model_()
        return message
    
    def _on_server_before_transmission(self, message: Message) -> Message:
        """
        Quantize the message before sending it from the server to the client.
        """
        message.populate_state_dict()
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                new_state_dict[name] = self._quantize(name, param.data, message, mode='server_to_client')
            else:
                new_state_dict[name] = param.data

        message.model_state_dict = new_state_dict
        return message


@dataclass
class ScalarQuantizationChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(ScalarQuantizationChannel)
    n_bits_client_to_server: int = 8
    n_bits_server_to_client: int = 8
    quantize_per_tensor: bool = True
    qscheme: str = "affine"
    use_shared_qparams: bool = False
    qparams_refresh_freq: int = 1
    sec_agg_mode: bool = False
    sec_agg_n_bits: int = 8
