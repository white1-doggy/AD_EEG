from typing import Optional, Tuple

import torch
from torch import nn


class EEGMaskedAutoencoder(nn.Module):
    """Masked Autoencoder for EEG windows with per-channel/time tokens."""

    def __init__(
        self,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")
        self.mask_ratio = mask_ratio
        self.num_channels = 63
        self.num_timesteps = 50
        self.num_tokens = self.num_channels * self.num_timesteps
        self.band_dim = 5

        self.input_proj = nn.Linear(self.band_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_pred = nn.Linear(decoder_dim, self.band_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstruction: [B, 5, 63, 50]
            loss: scalar tensor
            per_sample_loss: [B] tensor
        """
        band_tokens = self._to_band_tokens(x)
        tokens = self.input_proj(band_tokens)
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        x_vis, mask, ids_restore, ids_keep = self._apply_mask(tokens, mask_ratio)
        x_vis = x_vis + self._gather_pos_embed(self.pos_embed, ids_keep)
        latent = self.encoder(x_vis)

        reconstruction_tokens = self._decode(latent, ids_restore)
        reconstruction = self._detokenize(reconstruction_tokens)

        loss, per_sample_loss = self._reconstruction_loss(
            band_tokens, reconstruction_tokens, mask, valid_mask
        )
        return reconstruction, loss, per_sample_loss

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute per-window reconstruction error for anomaly detection."""
        _, _, per_sample_loss = self.forward(x, valid_mask=valid_mask, mask_ratio=mask_ratio)
        return per_sample_loss

    def _to_band_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input with shape [B, 5, 63, 50].")
        x = x.permute(0, 2, 3, 1)  # [B, 63, 50, 5]
        x = x.reshape(x.shape[0], self.num_tokens, self.band_dim)
        return x

    def _detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.view(tokens.shape[0], self.num_channels, self.num_timesteps, self.band_dim)
        x = x.permute(0, 3, 1, 2)
        return x

    def _apply_mask(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = tokens.shape
        len_keep = int(num_tokens * (1 - mask_ratio))

        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        visible_tokens = torch.gather(
            tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        )

        mask = torch.ones(batch_size, num_tokens, device=tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return visible_tokens, mask, ids_restore, ids_keep

    def _gather_pos_embed(self, pos_embed: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        batch_size = ids_keep.shape[0]
        pos_visible = torch.gather(
            pos_embed.repeat(batch_size, 1, 1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1)),
        )
        return pos_visible

    def _decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        batch_size, num_visible, _ = latent.shape
        num_tokens = ids_restore.shape[1]
        latent = self.decoder_embed(latent)

        mask_tokens = self.mask_token.repeat(batch_size, num_tokens - num_visible, 1)
        decoder_tokens = torch.cat([latent, mask_tokens], dim=1)
        decoder_tokens = torch.gather(
            decoder_tokens,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, decoder_tokens.size(-1)),
        )
        decoder_tokens = decoder_tokens + self.decoder_pos_embed
        decoded = self.decoder(decoder_tokens)
        return self.decoder_pred(decoded)

    def _reconstruction_loss(
        self,
        target_tokens: torch.Tensor,
        pred_tokens: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = target_tokens
        pred = pred_tokens
        loss_per_token = torch.mean((pred - target) ** 2, dim=-1)

        masked_loss = loss_per_token * mask
        per_sample_loss = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if valid_mask is not None:
            valid_mask = valid_mask.float()
            per_sample_loss = per_sample_loss * valid_mask
            denom = valid_mask.sum().clamp(min=1)
            loss = per_sample_loss.sum() / denom
        else:
            loss = per_sample_loss.mean()

        return loss, per_sample_loss
