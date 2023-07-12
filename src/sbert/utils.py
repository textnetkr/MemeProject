import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_emb = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expand = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return torch.sum(token_emb * input_mask_expand, 1) / torch.clamp(
        input_mask_expand.sum(1),
        min=1e-9,
    )
