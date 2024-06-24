import torch
import numpy as np
from transformers.modeling_utils import prune_linear_layer
import os

def prune_model_with_indicator(model,indicator):
    if indicator is None:
        return None, None
    print(hasattr(model, "bert"))
    bert = model.bert if hasattr(model, "bert") else model.roberta

    if "head" in indicator:
        head_indicator = indicator.get("head", None)
        head_layer_indicator = indicator.get("headlayer", None)

        prune_heads = {}
        for layer in range(len(head_indicator)):
            head_indicator_layer = head_indicator[layer].cpu().squeeze().clone()
            if head_layer_indicator is not None:
                head_indicator_layer *= head_layer_indicator[layer]
            index = torch.where(head_indicator_layer == 0)[0].tolist()
            prune_heads[layer] = index

            print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(prune_heads)

    kept_intermediate_dims = None
    if "int" in indicator:
        kept_intermediate_dims = {}
        int_indicator = indicator["int"]
        int_layer_indicator = indicator.get("intlayer", None)
        for layer in range(len(int_indicator)):
            int_indicator_layer = int_indicator[layer].squeeze()
            int_indicator_layer = int_indicator_layer.cpu().clone()
            if int_layer_indicator is not None:
                int_indicator_layer *= int_layer_indicator[layer]
            kept_intermediate_dims[layer] = int_indicator_layer.nonzero().reshape(-1).tolist()

    def prune_layer_norm(layernorm, index):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index))
        layernorm.bias = torch.nn.parameter.Parameter(
            layernorm.bias.index_select(0, index))
        layernorm.normalized_shape = (len(index),)

    def prune_layer(layer, index, dim):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer

    if "hidden" in indicator:
        hidden = indicator["hidden"]
        index = torch.LongTensor(hidden.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach())
        bert.embeddings.word_embeddings.embedding_dim = index.shape[0]
        bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.position_embeddings.weight.index_select(1, index).clone().detach())
        bert.embeddings.position_embeddings.embedding_dim = index.shape[0]
        bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(
            bert.embeddings.token_type_embeddings.weight.index_select(1, index).clone().detach())
        bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]
        prune_layer_norm(bert.embeddings.LayerNorm, index)

        for layer in range(0, 12):
            if bert.encoder.layer[layer].attention.self.query is not None:
                bert.encoder.layer[layer].attention.self.query = \
                    prune_layer(bert.encoder.layer[layer].attention.self.query, index, dim=1)
                bert.encoder.layer[layer].attention.self.key = \
                    prune_layer(bert.encoder.layer[layer].attention.self.key, index, dim=1)
            if bert.encoder.layer[layer].attention.self.value is not None:
                bert.encoder.layer[layer].attention.self.value = \
                    prune_layer(bert.encoder.layer[layer].attention.self.value, index, dim=1)
                bert.encoder.layer[layer].attention.output.dense = \
                    prune_layer(bert.encoder.layer[layer].attention.output.dense, index, dim=0)
                prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, index)
            if bert.encoder.layer[layer].intermediate.dense is not None:
                bert.encoder.layer[layer].intermediate.dense = \
                    prune_layer(bert.encoder.layer[layer].intermediate.dense, index, dim=1)
                bert.encoder.layer[layer].output.dense = \
                    prune_layer(bert.encoder.layer[layer].output.dense, index, dim=0)
                prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)

        # accommodate for different models
        if hasattr(model, "classifier"):
            if hasattr(model.classifier, "dense"):
                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        if hasattr(model, "cls"):
            if hasattr(model.cls, "dense"):
                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        if hasattr(bert.pooler, "dense"):
            bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)
        if hasattr(model, "qa_outputs"):
            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(model.layer_transformation, index, dim=1)
            print("layer transformation", model.layer_transformation.weight.shape)
        if getattr(model, "mha_layer_transformation", None) is not None:
            model.mha_layer_transformation = prune_linear_layer(model.mha_layer_transformation, index, dim=1)
            print("layer mha_layer_transformation", model.mha_layer_transformation.weight.shape)

    if kept_intermediate_dims is not None:
        prune_intermediate_layers(model, kept_intermediate_dims)

    for layer in range(0, 12):
        print("Layer:", layer)
        if bert.encoder.layer[layer].attention.self.query is not None:
            print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)
            print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if bert.encoder.layer[layer].attention.self.value is not None:
            print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)
            print("output:", bert.encoder.layer[layer].attention.output.dense.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if bert.encoder.layer[layer].intermediate.dense is not None:
            print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)
            print("down:", bert.encoder.layer[layer].output.dense.weight.shape)
        else:
            print("up", None)
            print("down", None)

def prune_intermediate_layers(model, keep_dims):
    bert = model.bert if hasattr(model, "bert") else model.roberta
    device = model.device
    for layer in keep_dims:
        if len(keep_dims[layer]) == 0:
            bert.encoder.layer[layer].intermediate.dense = None
            bert.encoder.layer[layer].output.dense = None
        else:
            bert.encoder.layer[layer].intermediate.dense = prune_linear_layer(
                bert.encoder.layer[layer].intermediate.dense, index=torch.LongTensor(keep_dims[layer]).to(device),
                dim=0)
            bert.encoder.layer[layer].output.dense = prune_linear_layer(bert.encoder.layer[layer].output.dense,
                                                                        index=torch.LongTensor(keep_dims[layer]).to(
                                                                            device), dim=1)

def load_pruned_model(model, weights):
    config = model.config
    dim_per_head = config.hidden_size // config.num_attention_heads
    indicator = {}

    architecture = config.architectures[0].lower()
    bert_name = "roberta" if "roberta" in architecture else "bert"

    hidden_indicator = torch.zeros(config.hidden_size)
    hidden_indicator[:weights[f"{bert_name}.embeddings.word_embeddings.weight"].shape[1]] = 1
    indicator["hidden"] = hidden_indicator

    head_indicator = torch.zeros(config.num_hidden_layers, config.num_attention_heads)
    head_layer_indicator = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"{bert_name}.encoder.layer.{i}.attention.output.dense.weight"
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_indicator[i, :remaining_heads] = 1
            head_layer_indicator[i] = 1
    indicator["head"] = head_indicator
    indicator["headlayer"] = head_layer_indicator

    int_indicator = torch.zeros(config.num_hidden_layers, config.intermediate_size)
    int_layer_indicator = torch.zeros(config.num_hidden_layers)
    for i in range(config.num_hidden_layers):
        key = f"bert.encoder.layer.{i}.output.dense.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_indicator[i, :remaining_int_dims] = 1
            int_layer_indicator[i] = 1
    indicator["int"] = int_indicator
    indicator["intlayer"] = int_layer_indicator
    print(indicator)
    prune_model_with_indicator(model, indicator)
    model.load_state_dict(weights, strict=False)
    return model

def load_indicator(model_path):
    if model_path.endswith("indicator.pt"):
        indicator_path = model_path
    else:
        indicator_path = os.path.join(model_path, "indicator.pt")

    if os.path.exists(indicator_path):
        indicator = torch.load(indicator_path, map_location="cpu")
        if indicator is None:
            alpha_module = torch.load(os.path.join(model_path, "indicator_module.pt"), map_location="cpu")
            indicator,_ = alpha_module.forward()
        return indicator
    else:
        return None

def calculate_parameters(module):
    return sum(p.numel() for n, p in module.named_parameters())


