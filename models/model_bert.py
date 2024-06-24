import logging
import math
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling,
                                           SequenceClassifierOutput)
from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.bert.modeling_bert import (
    BertAttention, BertEmbeddings, BertEncoder, BertForQuestionAnswering,
    BertForSequenceClassification, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput, QuestionAnsweringModelOutput)
from transformers.file_utils import hf_bucket_url, cached_path
from models.indicator_vector_op import *

logger = logging.getLogger(__name__)


class GDLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden=None):
        if hidden is not None:
            remaining_index = torch.where(~hidden.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output


class GDBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = GDBertModel(config)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        if os.path.exists(pretrained_model_name_or_path):
            weights = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                                 map_location=torch.device("cpu"))
        else:
            archive_file = hf_bucket_url(pretrained_model_name_or_path, filename="pytorch_model.bin")
            resolved_archive_file = cached_path(archive_file)
            weights = torch.load(resolved_archive_file, map_location="cpu")

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in weights.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            weights[new_key] = weights.pop(old_key)

        # drop_weight_names = ["layer_transformation.weight", "layer_transformation.bias"]
        # for name in drop_weight_names:
        #     if name in weights:
        #         weights.pop(name)

        if "config" not in kwargs:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            config.do_layer_distill = False
        else:
            config = kwargs["config"]

        model = cls(config)

        #load_pruned_model(model, weights)
        return model,weights



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            head=None,
            headlayer=None,
            int=None,
            intlayer=None,
            hidden=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            head=head,
            headlayer=headlayer,
            int=int,
            intlayer=intlayer,
            hidden=hidden
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GDBertEmbeddings(BertEmbeddings):
    """ Inherit from BertEmbeddings to allow CoFiLayerNorm """

    def __init__(self, config):
        super().__init__(config)
        self.LayerNorm = GDLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, hidden=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        if hidden is not None:
            embeddings = embeddings.mul(hidden)
        embeddings = self.LayerNorm(embeddings, hidden)
        embeddings = self.dropout(embeddings)

        if hidden is not None:
            embeddings = embeddings.mul(hidden)
        return embeddings


class GDBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = GDBertEncoder(config)
        self.embeddings = GDBertEmbeddings(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            head=None,
            headlayer=None,
            int=None,
            intlayer=None,
            hidden=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            hidden=hidden
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            int=int,
            head=head,
            intlayer=intlayer,
            headlayer=headlayer,
            hidden=hidden
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GDBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([GDBertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            head=None,
            headlayer=None,
            int=None,
            intlayer=None,
            hidden=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_QK = () if output_attentions else None
        all_VV = () if output_attentions else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                int=int[i] if int is not None else None,
                head=head[i] if head is not None else None,
                intlayer=intlayer[i] if intlayer is not None else None,
                headlayer=headlayer[i] if headlayer is not None else None,
                hidden=hidden
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_QK = all_QK + (layer_outputs[1],)
                all_VV = all_VV + (layer_outputs[2],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if output_attentions:
            all_attentions = all_QK + all_VV
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_QK] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class GDBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = GDBertAttention(config)
        self.output = GDBertOutput(config)
        self.config = config

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            head=None,
            headlayer=None,
            int=None,
            intlayer=None,
            hidden=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            head=head,
            headlayer=headlayer,
            hidden=hidden
        )

        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if self.intermediate.dense is None:
            layer_output = attention_output
        else:
            self.int = int
            self.intlayer = intlayer
            self.hidden = hidden
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        outputs = (layer_output,) + outputs + (attention_output,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        if self.int is not None:
            intermediate_output = intermediate_output.mul(self.int)
        layer_output = self.output(
            intermediate_output, attention_output, self.intlayer, self.hidden)
        return layer_output


class GDBertAttention(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = GDBertSelfAttention(config)
        self.output = GDBertSelfOutput(config)
        self.config = config

    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )  # find_pruneable_heads_and_indices定位需要修剪的head，以及需要保留的维度下标

        # Prune linear layers
        if len(index) == 0:
            self.self.query = None
            self.self.key = None
            self.self.value = None
            self.output.dense = None
        else:
            self.self.query = prune_linear_layer(self.self.query, index)
            self.self.key = prune_linear_layer(self.self.key, index)
            self.self.value = prune_linear_layer(self.self.value, index)
            self.output.dense = prune_linear_layer(
                self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
                                        len(heads)
        self.self.all_head_size = self.self.attention_head_size * \
                                  self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            head=None,
            headlayer=None,
            hidden=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            head=head,
        )

        attention_output = self.output(
            self_outputs[0], hidden_states, headlayer=headlayer, hidden=hidden)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class GDBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x_shape = x.size()
        last_dim = x_shape[-1]
        size_per_head = last_dim // self.num_attention_heads
        new_x_shape = x_shape[:-1] + (self.num_attention_heads, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                head=None):
        if self.value is None:
            return (None, None) if output_attentions else (None,)

        query_hidden_states = hidden_states
        mixed_query_layer = self.query(query_hidden_states)

        key_hidden_states = hidden_states
        mixed_key_layer = self.key(key_hidden_states)

        value_hidden_states = hidden_states
        mixed_value_layer = self.value(value_hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, seq_length, seq_length).float().to(
                hidden_states.device)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)

        VV_scores = torch.matmul(
            value_layer,value_layer.transpose(-1,-2))
        VV_scores = VV_scores / \
                           math.sqrt(self.attention_head_size)
        VV_probs = nn.Softmax(dim=-1)(VV_scores)
        VV_probs = self.dropout(VV_probs)


        if head is not None:
            context_layer = context_layer.permute(0, 2, 3, 1)
            context_layer *= head
            context_layer = context_layer.permute(0, 3, 1, 2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size(
        )[:-2] + (context_layer.shape[-1] * context_layer.shape[-2],)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs,VV_probs) if output_attentions else (
            context_layer,)
        return outputs


class GDBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = GDLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, headlayer=None, hidden=None, inference=False):
        if hidden_states is None:
            return input_tensor
        hidden_states = self.dense(hidden_states)
        if headlayer is not None:
            hidden_states = hidden_states.mul(headlayer)
        if not inference and hidden_states.sum().eq(0).item():
            hidden_states = hidden_states + input_tensor
        else:
            if hidden is not None:
                hidden_states = hidden_states.mul(hidden)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(
                hidden_states + input_tensor, hidden)
            if hidden is not None:
                hidden_states = hidden_states.mul(hidden)
        return hidden_states


class GDBertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = GDLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor, intlayer=None, hidden=None, inference=False):
        if hidden_states is None:
            return input_tensor
        hidden_states = self.dense(hidden_states)
        if intlayer is not None:
            hidden_states *= intlayer
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        else:
            if hidden is not None:
                hidden_states = hidden_states.mul(hidden)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(
                hidden_states + input_tensor, hidden)
            if hidden is not None:
                hidden_states = hidden_states.mul(hidden)
        return hidden_states


