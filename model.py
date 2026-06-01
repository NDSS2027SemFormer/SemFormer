#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SemFormer model definitions.

This module keeps one implementation of the BERT wrapper and exposes three
semantic-link attention modes through ``config.semantic_link_mode``:

``t5``
    Scalar logit bias per relation bucket and attention head.
``qr``
    Content-to-position attention, i.e., QK + QR.
``qkr``
    Disentangled attention, i.e., QK + QR + KR.
"""

import math

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEncoder,
    BertLayer,
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
    BertSelfAttention,
)


MODEL_MODES = ("t5", "qr", "qkr")


def normalize_model_mode(mode):
    mode = str(mode or "t5").lower().replace("-", "_")
    aliases = {
        "scalar": "t5",
        "scalar_bias": "t5",
        "t5_style": "t5",
        "qk_qr": "qr",
        "qkqr": "qr",
        "qk_qr_kr": "qkr",
        "deberta": "qkr",
        "deberta_style": "qkr",
    }
    mode = aliases.get(mode, mode)
    if mode not in MODEL_MODES:
        raise ValueError(f"Unsupported model mode {mode!r}. Choose from {MODEL_MODES}.")
    return mode


def set_model_mode(config, mode):
    mode = normalize_model_mode(mode)
    config.semantic_link_mode = mode
    config.model_mode = mode
    return config


class CFGSelfAttention(BertSelfAttention):
    """BERT self-attention augmented with semantic-link relative distances."""

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.semantic_link_mode = normalize_model_mode(
            getattr(config, "semantic_link_mode", getattr(config, "model_mode", "t5"))
        )

        max_rel_dist = getattr(config, "max_rel_dist", None)
        if max_rel_dist is not None:
            max_rel_dist = int(max_rel_dist)
            self.no_edge_id = max_rel_dist + 1
            self.no_rel_id = self.no_edge_id
            self.rel_num_types = max_rel_dist + 2
        else:
            rel_num_types_cfg = int(getattr(config, "rel_num_types", 0) or 0)
            no_edge_cfg = getattr(config, "no_edge_id", None)
            if no_edge_cfg is not None:
                self.no_edge_id = int(no_edge_cfg)
                self.no_rel_id = int(getattr(config, "no_rel_id", self.no_edge_id))
            elif rel_num_types_cfg > 0:
                self.no_edge_id = rel_num_types_cfg - 1
                self.no_rel_id = self.no_edge_id
            else:
                self.no_edge_id = 0
                self.no_rel_id = 0
            self.rel_num_types = rel_num_types_cfg if rel_num_types_cfg > 0 else self.no_edge_id + 1

        if self.rel_num_types > 0:
            if self.semantic_link_mode == "t5":
                self.rel_embeddings = nn.Embedding(self.rel_num_types, self.num_attention_heads)
                nn.init.zeros_(self.rel_embeddings.weight)
            else:
                self.rel_embeddings = nn.Embedding(self.rel_num_types, self.attention_head_size)
                if self.semantic_link_mode == "qr":
                    nn.init.zeros_(self.rel_embeddings.weight)
                else:
                    nn.init.normal_(self.rel_embeddings.weight, mean=0.0, std=config.initializer_range)
        else:
            self.rel_embeddings = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_ids=None,
        path_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        rel_ids_raw = rel_ids
        content_score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = content_score / math.sqrt(self.attention_head_size)

        if rel_ids is not None and self.rel_embeddings is not None:
            if rel_ids.dim() == 2:
                rel_ids = rel_ids.unsqueeze(0).expand(attention_scores.size(0), -1, -1)

            rel_ids_clamped = rel_ids.clamp(0, self.rel_num_types - 1)

            if self.semantic_link_mode == "t5":
                rel_bias = self.rel_embeddings(rel_ids_clamped).permute(0, 3, 1, 2).contiguous()
                attention_scores = attention_scores + rel_bias
            else:
                rel_embeds = self.rel_embeddings(rel_ids_clamped)
                qr_score = torch.einsum("bhid,bijd->bhij", query_layer, rel_embeds)
                if self.semantic_link_mode == "qr":
                    attention_scores = attention_scores + qr_score / math.sqrt(self.attention_head_size)
                else:
                    kr_score = torch.einsum("bhjd,bijd->bhij", key_layer, rel_embeds)
                    attention_scores = (
                        content_score + qr_score + kr_score
                    ) / math.sqrt(self.attention_head_size)

        if rel_ids_raw is not None:
            if rel_ids_raw.dim() == 2:
                rel_mask_src = rel_ids_raw.unsqueeze(0).expand(attention_scores.size(0), -1, -1)
            else:
                rel_mask_src = rel_ids_raw
            cfg_mask = (rel_mask_src == int(self.no_edge_id)).unsqueeze(1)
            attention_scores = attention_scores.masked_fill(cfg_mask, -1e4)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CFGBertAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = CFGSelfAttention(config, position_embedding_type=position_embedding_type)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_ids=None,
        path_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_ids=rel_ids,
            path_mask=path_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class CFGBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CFGBertAttention(
            config,
            position_embedding_type=getattr(config, "position_embedding_type", "absolute"),
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_ids=None,
        path_mask=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_ids=rel_ids,
            path_mask=path_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                None,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        layer_output = self.output(self.intermediate(attention_output), attention_output)
        return (layer_output,) + outputs


class CFGBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CFGBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        rel_ids=None,
        path_mask=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.config.is_decoder) else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                rel_ids=rel_ids,
                path_mask=path_mask,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BinBertModelcfg(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        set_model_mode(config, getattr(config, "semantic_link_mode", getattr(config, "model_mode", "t5")))
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        self.embeddings.position_embedding_type = "relative_key_query"

        old_encoder_state = self.encoder.state_dict()
        self.encoder = CFGBertEncoder(config)

        self.encoder.load_state_dict(old_encoder_state, strict=False)

        self.use_cfg_rel = getattr(config, "use_cfg_rel", False)
        self.rel_num_types = getattr(config, "rel_num_types", 0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rel_ids=None,
        path_mask=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rel_ids=rel_ids,
            path_mask=path_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BinBertForMaskedLMRDP(BertPreTrainedModel):
    """BERT masked-language model with an auxiliary RDP classification head."""

    def __init__(self, config):
        set_model_mode(config, getattr(config, "semantic_link_mode", getattr(config, "model_mode", "t5")))
        super().__init__(config)

        self.bert = BinBertModelcfg(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        max_rel_dist = getattr(config, "max_rel_dist", None)
        if max_rel_dist is not None:
            max_rel_dist = int(max_rel_dist)
            self.no_edge_id = max_rel_dist + 1
            self.rel_num_types = max_rel_dist + 2
        else:
            self.no_edge_id = int(getattr(config, "no_edge_id", 0))
            self.rel_num_types = int(getattr(config, "rel_num_types", max(self.no_edge_id + 1, 1)))

        hidden_size = int(config.hidden_size)
        self.rdp_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.rel_num_types),
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        rel_ids=None,
        path_mask=None,
        pair_idx=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=kwargs.get("output_attentions", None),
            output_hidden_states=kwargs.get("output_hidden_states", None),
            return_dict=True,
            rel_ids=rel_ids,
            path_mask=path_mask,
        )
        sequence_output = outputs.last_hidden_state
        mlm_logits = self.cls(sequence_output)

        rdp_logits = None
        if pair_idx is not None:
            q_idx = pair_idx[..., 0]
            k_idx = pair_idx[..., 1]
            batch_size, _, hidden_size = sequence_output.shape
            pair_count = q_idx.shape[1]
            b_ids = torch.arange(batch_size, device=sequence_output.device).unsqueeze(1).expand(
                batch_size,
                pair_count,
            )

            q_vec = sequence_output[b_ids, q_idx]
            k_vec = sequence_output[b_ids, k_idx]
            pair_feat = torch.cat([q_vec, k_vec, q_vec - k_vec, q_vec * k_vec], dim=-1)
            rdp_logits = self.rdp_mlp(pair_feat)

        out = MaskedLMOutput(
            loss=None,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        out.rdp_logits = rdp_logits
        return out
