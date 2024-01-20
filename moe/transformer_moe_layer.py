# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, List, Optional

from fairseq.modules.quant_noise import quant_noise
# changed from TransformerDecoderLayer, TransformerEncoderLayer to TransformerDecoderLayerBase, TransformerEncoderLayerBase
# from fairseq.models.transformer import TransformerDecoderLayer, TransformerEncoderLayer
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase, TransformerEncoderLayerBase
from moe.gates import TopKGate, MOESARTGate

def use_expert(layer_idx):
    return layer_idx % 2 == 0

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, q_noise, qn_block_size, dropout):
        nn.Module.__init__(self)

        # first layer
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = quant_noise(nn.Linear(input_dim, hidden_dim), q_noise, qn_block_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[self.activation_fn]
#         else:
#             self.intermediate_act_fn = self.activation_fn
        self.activation_fn = nn.ReLU()

        self.activation_dropout_module = nn.Dropout(dropout)

        # second layer
        # self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc2 = quant_noise(nn.Linear(hidden_dim, input_dim), q_noise, qn_block_size)

    def forward(self, hidden_states: Tensor):
        input_tensor = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout_module(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class MoETransformerEncoderLayer(TransformerEncoderLayerBase):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """
    def __init__(self, args, layer_idx=-1):
        self.num_experts = args.num_experts
        self.inference_level = args.inference_level
        self.use_expert = use_expert(layer_idx)
        self.route_method = args.route_method # New parameters
        self.trimmed_lasso_reg = args.trimmed_lasso_reg # New parameters
        self.dropout = args.dropout # New parameters
        self.k = args.k # New parameters
        self.tau = 1.0 # New parameters
        self.biasness = "small-random" # New parameters
        self.replace = False # New parameters
        super().__init__(args)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            self.experts = nn.ModuleList([FeedForward(input_dim, output_dim, q_noise, qn_block_size, self.dropout) for i in range(self.num_experts)])
            # build gate
            if self.route_method == "topk":
                config = {
                    "nb_experts": self.num_experts,
                    "k": self.k,
                    "jitter": False,
                    "input_dim": input_dim,
                }
                self.gate = TopKGate(config)
            elif self.route_method == "moesart":
                config = {
                    "nb_experts": self.num_experts,
                    "k": self.k,
                    "jitter": False,
                    "input_dim": input_dim,
                    "tau": 1.0,
                    "trimmed_lasso_reg": self.trimmed_lasso_reg,
                    'biasness': 'small-random',
                    'replace': False,
                }
                self.gate = MOESARTGate(config)
#                 return nn.ModuleList(
#                     [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
#                      for _ in range(self.num_experts)]
#                 )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

        # build experts
        
    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            pass
#             return nn.ModuleList(
#                 [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
#                  for _ in range(self.num_experts)]
#             )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    # TODO: Need to modify
    def _forward_sparse_gate(self, x):
        # TOFILL
        bsz, seq_len, dim = x.size()  # bsz is batch size, seq_len is sequence length, dim is hidden size
        x_squashed = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        # pass the hidden states to the gate
        sparse_weights, regularization_loss = self.gate.forward(x_squashed)
        weights, selected_experts = torch.topk(sparse_weights, self.k)
        y_agg = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            y_agg[batch_idx] += weights[batch_idx, nth_expert, None] * expert.forward(
                x_squashed[batch_idx]
            )
        y_agg.view_as(x)
        x = y_agg.view(bsz, seq_len, dim)
        return x, regularization_loss

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        expert_num: Optional[int] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

#         expert = None
        gate_loss = 0.0
        if self.use_expert:
#             if self.training:
#                 if expert_num is None:
#                     expert = torch.randint(low=0, high=self.num_experts, size=(1,)).item()  # selected expert
#                 else:
#                     expert = expert_num
#                 x = self.activation_fn(self.fc1[expert](x))
#                 x = self.activation_dropout_module(x)
#                 x = self.fc2[expert](x)
#                 x, gate_loss, gate_load = self._forward_sparse_gate(x)
#             else:
#                 result = []
#                 for expert in range(self.num_experts):
#                     temp = self.activation_fn(self.fc1[expert](x))
#                     temp = self.activation_dropout_module(temp)
#                     temp = self.fc2[expert](temp)
#                     result.append(temp)
#                 result = torch.stack(result, dim=0)
#                 if self.inference_level == 0:  # token level
#                     mask = torch.randint(0, self.num_experts,
#                                          size=(result.size(1), result.size(2)), device=result.device)
#                     for i in range(self.num_experts):
#                         expert_mask = mask.eq(i)
#                         result[i] *= expert_mask.unsqueeze(-1)
#                     x = result.sum(0)
#                 elif self.inference_level == 1:  # sentence level
#                     mask = torch.randint(0, self.num_experts,
#                                          size=(result.size(1),), device=result.device)
#                     for i in range(self.num_experts):
#                         expert_mask = mask.eq(i)
#                         result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
#                     x = result.sum(0)
            x, gate_loss = self._forward_sparse_gate(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, gate_loss


class MoETransformerDecoderLayer(TransformerDecoderLayerBase):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1,
    ):
        self.num_experts = args.num_experts
        self.inference_level = args.inference_level
        self.use_expert = use_expert(layer_idx)
        self.route_method = args.route_method # New parameters
        self.trimmed_lasso_reg = args.trimmed_lasso_reg # New parameters
        self.k = args.k # New parameters
        self.dropout = args.dropout # New parameters
        self.tau = 1.0 # New parameters
        self.biasness = "small-random" # New parameters
        self.replace = False # New parameters
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            self.experts = nn.ModuleList([FeedForward(input_dim, output_dim, q_noise, qn_block_size, self.dropout) for i in range(self.num_experts)])
            # build gate
            if self.route_method == "topk":
                config = {
                    "nb_experts": self.num_experts,
                    "k": self.k,
                    "jitter": False,
                    "input_dim": input_dim,
                }
                self.gate = TopKGate(config)
            elif self.route_method == "moesart":
                config = {
                    "nb_experts": self.num_experts,
                    "k": self.k,
                    "jitter": False,
                    "input_dim": input_dim,
                    "tau": 1.0,
                    "trimmed_lasso_reg": self.trimmed_lasso_reg,
                    'biasness': 'small-random',
                    'replace': False,
                }
                self.gate = MOESARTGate(config)
#             return nn.ModuleList(
#                 [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
#                  for _ in range(self.num_experts)]
#             )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
            
    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_expert:
            pass
#             return nn.ModuleList(
#                 [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
#                  for _ in range(self.num_experts)]
#             )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

#     def build_experts(self, input_dim, hidden_dim, q_noise, qn_block_size, dropout):
#     def build_gate(self, input_dim):
        
#    # TODO: Need to modify
#    def _forward_sparse_gate(self, x):
#        # TOFILL
#        bsz, seq_len, dim = x.size()  # bsz is batch size, seq_len is sequence length, dim is hidden size
#
#        x = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
#
#        def forward_expert(input_x, expert_idx):
#            input_x = self.experts[expert_idx].forward(input_x)
#            return input_x
#
#        h = [forward_expert(x, i) for i in range(self.num_experts)]
#
#        # pass the hidden states to the gate
#        y_agg, soft_averages, hard_averages, s_concat, regularization_loss = self.gate.forward((h, x))
#        # print("y_agg", y_agg.shape)
#        # print("soft_averages", soft_averages.shape)
#        # print("hard_averages", hard_averages.shape)
#        # print(y_agg)
#        # print("soft_averages", soft_averages)
#        # print("hard_averages", hard_averages)
#
#        x = y_agg.view(bsz, seq_len, dim)
#
#        return x, regularization_loss, s_concat

    # TODO: Need to modify
    def _forward_sparse_gate(self, x):
        # TOFILL
        bsz, seq_len, dim = x.size()  # bsz is batch size, seq_len is sequence length, dim is hidden size
        x_squashed = x.view(-1, dim)  # x is now a 2D tensor of size (bsz * seq_len, dim)
        # pass the hidden states to the gate
        sparse_weights, regularization_loss = self.gate.forward(x_squashed)
        weights, selected_experts = torch.topk(sparse_weights, self.k)
        y_agg = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            y_agg[batch_idx] += weights[batch_idx, nth_expert, None] * expert.forward(
                x_squashed[batch_idx]
            )
        y_agg.view_as(x)
        x = y_agg.view(bsz, seq_len, dim)
        return x, regularization_loss

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        expert_num: Optional[int] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True
        
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

#         expert = None
        gate_loss = 0.0
        if self.use_expert:
#             if self.training:
#                 if expert_num is None:
#                     expert = torch.randint(low=0, high=self.num_experts, size=(1,)).item()  # selected expert
#                 else:
#                     expert = expert_num
#                 x = self.activation_fn(self.fc1[expert](x))
#                 x = self.activation_dropout_module(x)
#                 x = self.fc2[expert](x)
#                 x, balance_loss, gate_load = self._forward_sparse_gate(x)
#             else:
#                 result = []
#                 for expert in range(self.num_experts):
#                     temp = self.activation_fn(self.fc1[expert](x))
#                     temp = self.activation_dropout_module(temp)
#                     temp = self.fc2[expert](temp)
#                     result.append(temp)
#                 result = torch.stack(result, dim=0)
#                 if self.inference_level == 0:  # token level
#                     mask = torch.randint(0, self.num_experts,
#                                          size=(result.size(1), result.size(2)), device=result.device)
#                     for i in range(self.num_experts):
#                         expert_mask = mask.eq(i)
#                         result[i] *= expert_mask.unsqueeze(-1)
#                     x = result.sum(0)
#                 elif self.inference_level == 1:  # sentence level
#                     mask = torch.randint(0, self.num_experts,
#                                          size=(result.size(1),), device=result.device)
#                     for i in range(self.num_experts):
#                         expert_mask = mask.eq(i)
#                         result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
#                     x = result.sum(0)
#            x, gate_loss, gate_load = self._forward_sparse_gate(x)
            x, gate_loss = self._forward_sparse_gate(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, gate_loss
