#model.py
from typing import List, Dict
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, query, key_value):
        #print(f"Query shape: {query.shape}") #Added this
        #print(f"KeyValue shape: {key_value.shape}") #Added this
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return attn_output

class Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_heads,
        encoder_layers,
        decoder_layers,
        d_categories: List[int],  # List of number of classes per field
        encoders: List[str],
        d_output  # Not needed anymore, but keep for compatibility
    ):
        super(Model, self).__init__()

        self.encoders_list = encoders
        self.encoder_streams = nn.ModuleDict()

        for stream in self.encoders_list:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                batch_first=True
            )
            self.encoder_streams[stream] = nn.TransformerEncoder(
                encoder_layer,
                num_layers=encoder_layers
            )

        self.input_proj = nn.Linear(d_input, d_model)
        self.cross_attention = CrossAttention(d_model, n_heads)

        # One decoder head per field
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, category_size)
            )
            for category_size in d_categories
        ])

    def forward(
        self,
        src_events: Dict[str, List[torch.Tensor]],
        tgt_events: torch.Tensor,
        encodings_to_carry: int = 1
    ) -> List[torch.Tensor]:

        batch_size = list(src_events.values())[0].shape[0]

        encoder_outputs = []

        for stream_name, transformer_encoder in self.encoder_streams.items():
            stream_input = src_events[stream_name]
            stream_input = stream_input.to(dtype=self.input_proj.weight.dtype, device=self.input_proj.weight.device)
            stream_input = self.input_proj(stream_input)

            if stream_input.dim() == 2:
                stream_input = stream_input.unsqueeze(0)
            elif stream_input.dim() == 1:
                stream_input = stream_input.unsqueeze(0).unsqueeze(1)

            encoded = transformer_encoder(stream_input)

            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(0)

            last_vector = encoded[:, -1:, :]
            encoder_outputs.append(last_vector)

        encoder_outputs = torch.cat(encoder_outputs, dim=1)
        query = torch.zeros((batch_size, 1, encoder_outputs.size(-1)), device=encoder_outputs.device, dtype=encoder_outputs.dtype)
        cross_attended = self.cross_attention(query, encoder_outputs)
        cross_attended = cross_attended.squeeze(1)

        # Outputs for each field
        output_logits_list = [decoder(cross_attended) for decoder in self.decoders]

        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for i, output_logits in enumerate(output_logits_list):
            target_i = tgt_events[:, i]
            target_i = target_i.to(output_logits.device)
            loss_i = loss_fn(output_logits, target_i)
            losses.append(loss_i)

        total_loss = sum(losses) / len(losses)

        return total_loss, losses

class ModelTrainer(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_heads,
        encoder_layers,
        decoder_layers,
        d_categories: List[int],
        encoders: List[str],
        d_output
    ):
        super(ModelTrainer, self).__init__()

        self.model = Model(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_categories=d_categories,
            encoders=encoders,
            d_output=d_output
        )

    def forward(
        self,
        batch_src_events: Dict[str, torch.Tensor],  # no longer List
        batch_tgt_events: torch.Tensor,
        batch_masks,
        run_backward=False,
    ):
        # Direct call
        total_loss, separated_losses = self.model(batch_src_events, batch_tgt_events)

        if run_backward:
            total_loss.backward()

        return total_loss, separated_losses
