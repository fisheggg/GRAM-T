from typing import List, Tuple
from einops import repeat

import pytorch_lightning as pl
import torch
import random
import torchaudio

from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers

from timm.models.vision_transformer import Block
from timm.layers.config import set_fused_attn, use_fused_attn
from torch import nn


from ..patching import PatchStrategy
from .layers import MWMHABlock
from .pos_embed import get_2d_sincos_pos_embed
from .utils import PatchEmbed, create_pretrained_model, plot_fbank, repeat_token


from ..data_modules.scene_module import generate_scenes_batch
from ..data_modules.dataset_functions import pad_or_truncate_batch

from .ambisonic_feature_extractor import FeatureExtractor

try:
    from einops import rearrange
except ImportError as e:
    print(f"Got {e} this is expected if you are training.")


set_fused_attn(True)
use_fused_attn(True)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim=0, end_dim=1)


# It won't actually have the branching, because we already pad everything to 10 seconds.
pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)
collate_fn = torch.compile(collate_fn)


def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class GRAMT_SPR(pl.LightningModule):
    """
    A Transformer-based model for spatial audio processing, supporting pretraining with masked auto encoder like
    training.
    Parameters
    ----------
    model_size : str, optional
        The size of the Transformer model, by default 'base'.
    lr : float, optional
        Learning rate, by default 1e-3.
    trainer : str, optional
        Optimizer type, by default 'adam'.
    b1 : float, optional
        Beta1 hyperparameter for the optimizer, by default 0.95.
    b2 : float, optional
        Beta2 hyperparameter for the optimizer, by default 0.999.
    weight_decay : float, optional
        Weight decay for the optimizer, by default 5e-7.
    mask_patch : int, optional
        Number of patches to mask during pretraining, by default 400.
    patch_strategy : PatchStrategy, optional
        Masking strategy to use, by default None.
    """

    def __init__(
        self,
        model_size="base",
        lr=2e-5,
        trainer="adam",
        b1=0.9,
        b2=0.95,
        weight_decay=0.01,
        mlp_ratio: float = 4.0,
        mask_patch=200,
        patch_strategy: PatchStrategy = None,
        in_channels: int = 2,
        log_every_n_steps: int = 1000,
        cluster: bool = False,
        use_mwmae_decoder: bool = True,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        decoder_embedding_dim: int = 512,
        decoder_window_sizes: List[int] = [2, 5, 10, 25, 50, 100, 0, 0],
        sr: int = 32000,
        num_mel_bins: int = 128,
        nr_samples_per_audio: int = 16,
        target_length: int = 200,
        input_length: int = 1024,
        clean_data_ratio: float = 0.0,
        pseudolabel_type: str = "foa_to_iv",
        multitask_type: str = "linear_probe",
        w_recon_loss: float = 1.0,
        w_spr_loss: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # Dataloading params
        self.clean_data_ratio = clean_data_ratio
        self.sr = sr
        self.nr_samples_per_audio = nr_samples_per_audio
        self.target_length = target_length
        self.input_length = input_length
        self.num_mel_bins = num_mel_bins

        # Initialize hyperparameters
        self.lr = lr
        self.trainer_name = trainer
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.mask_patch = mask_patch
        self.patch_strategy = patch_strategy
        self.in_channels = in_channels
        self.cluster = cluster
        self.mlp_ratio = mlp_ratio
        self.use_mwmae_decoder = use_mwmae_decoder
        self.w_recon_loss = w_recon_loss
        self.w_spr_loss = w_spr_loss
        assert self.w_recon_loss >= 0.0, "w_recon_loss must be non-negative"
        assert self.w_spr_loss >= 0.0, "w_spr_loss must be non-negative"
        self.multitask_type = multitask_type
        assert self.multitask_type in ["linear", "decoder"], (
            f"Invalid multitask_type: {self.multitask_type}. "
            "Must be one of 'linear' or 'decoder'."
        )

        # Set later in the forward pass
        self.input_shape = None
        self.log_every_n_steps = log_every_n_steps

        # Calculate intermediate shape after masking
        self.p_f_dim, self.p_t_dim = self.patch_strategy.get_patch_size()
        self.num_patches = self.p_f_dim * self.p_t_dim
        self.grid_size = (self.p_f_dim, self.p_t_dim)

        # SPR model specifics
        assert pseudolabel_type in ["w_to_xyz", "orthogonal", "pair", "foa_to_iv"], (
            f"Invalid pseudolabel_type: {pseudolabel_type}. "
            "Must be one of 'w_to_xyz', 'orthogonal', 'pair', or 'foa_to_iv'."
        )
        self.pseudolabel_type = pseudolabel_type
        print(f"=> Using pseudolabel_type: {self.pseudolabel_type}")

        ########## Network modules
        # --------------------------------------------------------------------------

        # Transformer
        (
            self.encoder,
            self.encoder_embedding_dim,
        ) = create_pretrained_model(model_size)
        self.encoder_cls_token_num = 1

        # Patch Embedder
        self.patch_embed = PatchEmbed()
        self._update_patch_embed_layers(self.patch_embed)

        # Norm/Pos
        self.cls_token = nn.Parameter(
            nn.init.normal_(torch.empty([1, 1, self.encoder_embedding_dim]))
        )

        # This is our decoder.
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_window_sizes = decoder_window_sizes
        if self.multitask_type == "linear":
            self.decoder_embed = nn.Linear(
                self.encoder_embedding_dim, self.decoder_embedding_dim, bias=True
            )

            self.register_buffer(
                "mask_token",
                nn.Parameter(
                    torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
                ),
            )
            torch.nn.init.normal_(self.mask_token, std=0.02)

            # Init the nn.Parameters here
            if self.use_mwmae_decoder:
                self.decoder_blocks = nn.ModuleList(
                    [
                        MWMHABlock(
                            dim=self.decoder_embedding_dim,
                            num_heads=self.decoder_num_heads,
                            window_sizes=self.decoder_window_sizes,
                            shift_windows=False,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            norm_layer=nn.LayerNorm,
                        )
                        for i in range(self.decoder_depth)
                    ]
                )

            else:
                self.decoder_blocks = nn.ModuleList(
                    [
                        Block(
                            self.decoder_embedding_dim,
                            num_heads=self.decoder_num_heads,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            norm_layer=nn.LayerNorm,
                        )
                        for _ in range(self.decoder_depth)
                    ]
                )

            cls_token_num = 0 if self.use_mwmae_decoder else self.encoder_cls_token_num
            self.encoder.pos_embedding = self._get_pos_embed_params()
            # Pos Embed init w/o the cls token num
            self.register_buffer(
                "decoder_pos_embed",
                nn.Parameter(
                    torch.zeros(1, self.num_patches, self.decoder_embedding_dim),
                    requires_grad=False,
                ),
            )
            pos_embed = get_2d_sincos_pos_embed(
                self.decoder_embedding_dim, self.grid_size, cls_token_num=cls_token_num
            )
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
            self.decoder_norm = nn.LayerNorm(self.decoder_embedding_dim)

        # decoder setups for multitask_type == "decoder"
        if self.multitask_type == "decoder":
            # decoder for the recon task
            self.decoder_embed_recon = nn.Linear(
                self.encoder_embedding_dim, self.decoder_embedding_dim, bias=True
            )

            self.register_buffer(
                "mask_token_recon",
                nn.Parameter(
                    torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
                ),
            )
            torch.nn.init.normal_(self.mask_token_recon, std=0.02)

            # Init the nn.Parameters here
            if self.use_mwmae_decoder:
                self.decoder_blocks_recon = nn.ModuleList(
                    [
                        MWMHABlock(
                            dim=self.decoder_embedding_dim,
                            num_heads=self.decoder_num_heads,
                            window_sizes=self.decoder_window_sizes,
                            shift_windows=False,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            norm_layer=nn.LayerNorm,
                        )
                        for i in range(self.decoder_depth)
                    ]
                )

            else:
                self.decoder_blocks_recon = nn.ModuleList(
                    [
                        Block(
                            self.decoder_embedding_dim,
                            num_heads=self.decoder_num_heads,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            norm_layer=nn.LayerNorm,
                        )
                        for _ in range(self.decoder_depth)
                    ]
                )

            cls_token_num = 0 if self.use_mwmae_decoder else self.encoder_cls_token_num
            self.encoder.pos_embedding = self._get_pos_embed_params()
            # Pos Embed init w/o the cls token num
            self.register_buffer(
                "decoder_pos_embed_recon",
                nn.Parameter(
                    torch.zeros(1, self.num_patches, self.decoder_embedding_dim),
                    requires_grad=False,
                ),
            )
            pos_embed = get_2d_sincos_pos_embed(
                self.decoder_embedding_dim, self.grid_size, cls_token_num=cls_token_num
            )
            self.decoder_pos_embed_recon.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )

            self.decoder_norm_recon = nn.LayerNorm(self.decoder_embedding_dim)
            # decoder for the spr task
            self.decoder_embed_spr = nn.Linear(
                self.encoder_embedding_dim, self.decoder_embedding_dim, bias=True
            )
            self.register_buffer(
                "mask_token_spr",
                nn.Parameter(
                    torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
                ),
            )
            torch.nn.init.normal_(self.mask_token_spr, std=0.02)
            self.decoder_blocks_spr = nn.ModuleList(
                [
                    Block(
                        self.decoder_embedding_dim,
                        num_heads=self.decoder_num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for _ in range(self.decoder_depth)
                ]
            )
            cls_token_num = 0 if self.use_mwmae_decoder else self.encoder_cls_token_num
            self.encoder.pos_embedding = self._get_pos_embed_params()
            self.register_buffer(
                "decoder_pos_embed_spr",
                nn.Parameter(
                    torch.zeros(1, self.num_patches, self.decoder_embedding_dim),
                    requires_grad=False,
                ),
            )
            pos_embed = get_2d_sincos_pos_embed(
                self.decoder_embedding_dim, self.grid_size, cls_token_num=cls_token_num
            )
            self.decoder_pos_embed_spr.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
            self.decoder_norm_spr = nn.LayerNorm(self.decoder_embedding_dim)

        # linear head setups
        self.spec_pred = nn.Sequential(
            nn.Linear(
                self.decoder_embedding_dim,
                self.patch_strategy.fshape
                * self.patch_strategy.tshape
                * self.in_channels,
                bias=True,
            ),
        )
        # define SPR pred head
        self.SPR_pred = nn.Sequential(
            nn.Linear(
                self.decoder_embedding_dim,
                3 * self.in_channels,
                bias=True,
            ),
        )

        if self.in_channels == 2 or self.in_channels == 1:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=320,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float()
        else:
            self.melspec = FeatureExtractor(
                sample_rate=self.sr,
                n_fft=1024,
                win_length=1024,
                hop_length=self.sr // 100,
                f_min=50,
                f_max=self.sr // 2,
                n_mels=self.num_mel_bins,
                power=2.0,
            ).float()

        # Normalize binaural/ambisonic spectrograms with Layer norm later.
        self.spectrogram_normalize = nn.LayerNorm(
            [self.in_channels, self.num_mel_bins, self.target_length],
            elementwise_affine=False,
        )
        self.input_shape = [self.num_mel_bins, self.target_length]
        # Save hyperparameters for checkpointing
        self.apply(self._init_weights)

        compile_modules = kwargs.get("compile_modules", None)
        if (compile_modules is not None) and (compile_modules):
            self._compile_operations()

    def _wav2fbank(self, waveform):
        with torch.amp.autocast("cuda", enabled=False):  # Force FP32 computation
            waveform = waveform.float()
            mel = self.melspec(waveform)  # Ensure input is float32
            if self.in_channels == 2 or self.in_channels == 1:
                log_mel = torch.log(mel + 1e-5).transpose(3, 2)
            else:
                # Otherwise we have already the log mel spec.
                log_mel = mel.transpose(3, 2)
        return log_mel

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.forward = torch.compile(self.forward, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_pos_embed_params(self):
        """Calculates the pos embedding embedding parameters and returns them."""
        # Update positional embedding
        pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + self.encoder_cls_token_num,
                self.encoder_embedding_dim,
            ),
            requires_grad=False,
        )
        pos_embed_data = get_2d_sincos_pos_embed(
            self.encoder_embedding_dim,
            self.grid_size,
            cls_token_num=self.encoder_cls_token_num,
        )
        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed

    def _update_patch_embed_layers(self, patch_embed):
        """Updates the patch embedding embedding layers."""
        # Update patch projection layer
        # Use 2, as the spectrogram has 2 channels
        patch_embed.proj = torch.nn.Conv2d(
            self.in_channels,
            self.encoder_embedding_dim,
            kernel_size=(self.patch_strategy.fshape, self.patch_strategy.tshape),
            stride=(self.patch_strategy.fstride, self.patch_strategy.tstride),
        )
        patch_embed.num_patch = self.num_patches

    def pass_through_encoder(self, x, non_mask_index, B):
        """
        Passes the input through the Encoder Transformer network.
        inputs:
            x:
            non_mask_index:
            B:
        output:
            x shape: (B, 1, encoder_embedding_dim)
        """
        # Add positional embeddings to the x.
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num :, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = (
            self.cls_token.expand(B, -1, -1) + self.encoder.pos_embedding[:, :1, :]
        )

        try:
            dist_token = (
                self.encoder.dist_token.expand(B, -1, -1)
                + self.encoder.pos_embedding[:, 1:2, :]
            )
            x = torch.cat((cls_token, dist_token, x), dim=1)

        except Exception as e:
            x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.dropout(x)
        for block in self.encoder.layers:
            x = block(x)
        return self.encoder.ln(x)

    def pass_through_encoder_until_block(self, x, non_mask_index, B, block_num=-1):
        """Passes the input through the Encoder Transformer network."""
        # Add positional embeddings to the x.
        x = x + self.encoder.pos_embedding[:, self.encoder_cls_token_num :, :]
        x = x[non_mask_index, :].reshape((B, -1, x.shape[-1]))
        cls_token = (
            self.cls_token.expand(B, -1, -1) + self.encoder.pos_embedding[:, :1, :]
        )
        x = torch.cat((cls_token, x), dim=1)
        # Encoder forward pass here.
        x = self.encoder.dropout(x)
        for block in self.encoder.layers[:block_num]:
            x = block(x)
        return self.encoder.ln(x)

    def pass_through_decoder(self, encoder_output, non_mask_index, B, return_latent=False):
        """
        Passes the input through the Decoder Transformer network.
        inputs:
            encoder_output:
            non_mask_index:
            B:
            return_latent: additionally return the decoder output the two
                prediction heads are read from.
        output:
            recon_pred shape: (B, n_frames, patch_strategy.fshape * patch_strategy.tshape * in_channels)
            spr_pred shape: (B, n_frames, 3 * in_channels)
            latent shape: (B, n_frames, decoder_embedding_dim), only if return_latent
        """

        ## linear probe, goes through the only decoder and two prediction heads
        if self.multitask_type == "linear":
            encoder_output = self.decoder_embed(encoder_output)
            x_ = repeat_token(self.mask_token, (B, self.num_patches)).type_as(
                encoder_output
            )
            # print(f"x_ shape: {x_.shape}, encoder_output shape: {encoder_output.shape}")
            x_[non_mask_index, :] = encoder_output[
                :, self.encoder_cls_token_num :, :
            ].reshape((-1, encoder_output.shape[-1]))
            x_ = x_.reshape((B, -1, encoder_output.shape[-1]))

            # Concatenate the CLS and Possibly Distill tokens from the encoder
            # We can not do it with multi windowed attention though!
            # So remove the CLS token from the decoder!
            if self.use_mwmae_decoder:
                x = x_
                return_cut = 0
            else:
                x = torch.cat(
                    [encoder_output[:, : self.encoder_cls_token_num, :], x_], dim=1
                )
                return_cut = self.encoder_cls_token_num

            x = x + self.decoder_pos_embed  # add the pos embeds
            # Pass through transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            recon_pred = self.spec_pred(x)
            recon_pred = recon_pred[:, return_cut:, :]
            spr_pred = self.SPR_pred(x)
            spr_pred = spr_pred[:, return_cut:, :]

        ## decoder, goes through the two decoders and two prediction heads
        if self.multitask_type == "decoder":
            ## recon branch
            encoder_output_recon = self.decoder_embed_recon(encoder_output)
            x__recon = repeat_token(self.mask_token_recon, (B, self.num_patches)).type_as(
                encoder_output_recon
            )
            # print(f"x_ shape: {x_.shape}, encoder_output shape: {encoder_output.shape}")
            x__recon[non_mask_index, :] = encoder_output_recon[
                :, self.encoder_cls_token_num :, :
            ].reshape((-1, encoder_output_recon.shape[-1]))
            x__recon = x__recon.reshape((B, -1, encoder_output_recon.shape[-1]))

            # Concatenate the CLS and Possibly Distill tokens from the encoder
            # We can not do it with multi windowed attention though!
            # So remove the CLS token from the decoder!
            if self.use_mwmae_decoder:
                x_recon = x__recon
                return_cut = 0
            else:
                x_recon = torch.cat(
                    [encoder_output[:, : self.encoder_cls_token_num, :], x_], dim=1
                )
                return_cut = self.encoder_cls_token_num

            x_recon = x__recon + self.decoder_pos_embed_recon  # add the pos embeds
            for blk in self.decoder_blocks_recon:
                x_recon = blk(x_recon)
            x_recon = self.decoder_norm_recon(x_recon)
            recon_pred = self.spec_pred(x_recon)
            recon_pred = recon_pred[:, return_cut:, :]

            ## spr branch
            encoder_output_spr = self.decoder_embed_spr(encoder_output)
            x__spr = repeat_token(self.mask_token_spr, (B, self.num_patches)).type_as(
                encoder_output_spr
            )
            x__spr[non_mask_index, :] = encoder_output_spr[
                :, self.encoder_cls_token_num :, :
            ].reshape((-1, encoder_output_spr.shape[-1]))
            x__spr = x__spr.reshape((B, -1, encoder_output_spr.shape[-1]))


            # Concatenate the CLS and Possibly Distill tokens from the encoder
            # We can not do it with multi windowed attention though!
            # So remove the CLS token from the decoder!
            if self.use_mwmae_decoder:
                x_spr = x__spr
                return_cut = 0
            else:
                x_spr = torch.cat(
                    [encoder_output[:, : self.encoder_cls_token_num, :], x_], dim=1
                )
                return_cut = self.encoder_cls_token_num

            x_spr = x__spr + self.decoder_pos_embed_spr  # add the pos embeds
            for blk in self.decoder_blocks_spr:
                x_spr = blk(x_spr)
            x_spr = self.decoder_norm_spr(x_spr)
            spr_pred = self.SPR_pred(x_spr)
            spr_pred = spr_pred[:, return_cut:, :]

        if return_latent:
            latent = x if self.multitask_type == "linear" else x_spr
            return recon_pred, spr_pred, latent[:, return_cut:, :]
        return recon_pred, spr_pred

    def log_first_spectrogram(self, patches, title, loss, **kwargs):
        sample_img = patches[:1].clone()
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = (
            torch.min(patches_unflattened),
            torch.max(patches_unflattened),
        )
        combined = (
            self.patch_strategy.combine_patches(patches_unflattened, self.input_shape)
            .detach()
            .cpu()
            .float()
            .numpy()
        )
        combined = combined.transpose(0, 1, 3, 2)

        title_plot = (
            "Azimuth: {} Elevation: {}".format(
                kwargs["direction"][0], kwargs["direction"][1]
            )
            if "direction" in kwargs
            else f"Loss: {loss}"
        )
        fig = plot_fbank(combined[0], vmin=color_min, vmax=color_max, title=title_plot)
        self.logger.experiment.add_figure(f"{title}", fig, global_step=self.global_step)

    def log_first_spectrogram_with_mask(self, patches, mask, title):
        sample_img = patches[:1].clone().detach().cpu()
        # Selecting using boolean indexing
        sample_img[:, mask[0], :] = 0
        patches_unflattened = sample_img.unflatten(2, self.patches_shape[2:])
        color_min, color_max = (
            torch.min(patches_unflattened),
            torch.max(patches_unflattened),
        )
        combined = (
            self.patch_strategy.combine_patches(patches_unflattened, self.input_shape)
            .cpu()
            .float()
            .numpy()
        )
        combined = combined.transpose(0, 1, 3, 2)
        fig = plot_fbank(combined[0], vmin=color_min, vmax=color_max)
        self.logger.experiment.add_figure(f"{title}", fig, global_step=self.global_step)

    def mse_loss(self, pred, target, mask):
        """
        Mask is the boolean tensor containing 1 for masked indices, and 0 for non-mask-indices
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (
            loss * mask.int().float()
        ).sum() / mask.int().float().sum()  # mean loss on masked patches
        loss = loss * self.in_channels  # Weight the loss w.r.t in channels!
        return loss

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        # print("=> starting training step")
        # Just making sure that they are in training mode
        audio_input, mask = self._prepare_batch(batch)
        # Use Flash Attention if possible.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x, patches, recon_loss, spr_loss = self.pretrain_forward(audio_input, mask)
        summed_loss = self.w_recon_loss * recon_loss + self.w_spr_loss * spr_loss

        if self.global_step % self.log_every_n_steps == 0:
            self.log_first_spectrogram(
                patches, title="input_spectrogram", loss=summed_loss
            )
            self.log_first_spectrogram_with_mask(
                patches, ~mask, title="target_spectrogram"
            )
            self.log_first_spectrogram_with_mask(
                patches, mask, title="masked_spectrogram"
            )
            self.log_first_spectrogram(
                x, title="generated_spectrogram", loss=summed_loss
            )

        self.log_dict(
            {
                "recon_loss": recon_loss,
                "spr_loss": spr_loss,
                "summed_loss": summed_loss,
            }, on_step=True, on_epoch=False,
        )

        return summed_loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step."""
        audio_input, mask = batch
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x, patches, recon_loss, spr_loss = self.pretrain_forward(audio_input, mask)
        summed_loss = self.w_recon_loss * recon_loss + self.w_spr_loss * spr_loss
        self.log_dict(
            {
                "valid_recon_loss": recon_loss,
                "valid_spr_loss": spr_loss,
                "valid_summed_loss": summed_loss,
            }
        )
        return summed_loss

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        audio_trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = None
        if self.trainer_name == "adamW":
            optimizer = torch.optim.AdamW(
                audio_trainables,
                self.lr,
                weight_decay=self.weight_decay,
                betas=(self.b1, self.b2),
            )

        cosine_annealing = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=10000, num_training_steps=self.trainer.max_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": cosine_annealing, "interval": "step"},
        }

    def calculate_rec_loss(self, rec, target):
        """FROM SPR"""
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()
        return rec_loss

    @torch.no_grad()
    def _prepare_batch(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the batch before the training step.
        This version is vectorized for performance and memory efficiency.
        """
        # print("=> start preparing batch")
        audio, noise, noise_length, noise_idx, source_rir, noise_rirs, snr, mask = batch

        clean_batch = False

        # Accessing this if statement produces clean data.
        if random.random() < self.clean_data_ratio:
            # Do it in a list to make it look like a "batch"
            noise = [None]
            source_rir = [None]
            noise_rirs = [None]
            snr = [None]
            clean_batch = True

        # 1. Generate scene
        generated_scene = generate_scenes_batch.generate_scene(
            source_rir=source_rir,
            noise_rirs=noise_rirs,
            source=audio,
            noise=noise,
            snr=snr,
            real_noise_length=noise_length,
            noise_start_idx=noise_idx,
        )

        # Stack the generated scene on the channel axis, this means that we did not augment our audio, and we are using the one-channel audio.
        if clean_batch:
            # If we are utilizing ambisonics, then stack the one-channel audio on the 4 channels, the feature extraction will give the Intesity Vectors
            if self.in_channels == 7:
                generated_scene = repeat(
                    generated_scene, "B C L_full -> B (repeat C) L_full", repeat=4
                )
            # If we are utilizing binaural training, then stack them on left and right channels.
            elif self.in_channels == 2:
                generated_scene = repeat(
                    generated_scene, "B C L_full -> B (repeat C) L_full", repeat=2
                )
            elif self.in_channels == 1:
                generated_scene = generated_scene
            else:
                raise Exception(f"Unknown channel count {self.in_channels}")

        fbank = self._wav2fbank(generated_scene)
        fbank = pad_or_truncate_batch(fbank, self.input_length)  # B, C, T, F
        B, C, T, F_mel = fbank.shape

        # Generate all random start indices at once
        rand_starts = torch.randint(
            0,
            T - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device,
        )

        # Create indices for gathering
        indices = rand_starts.unsqueeze(-1) + torch.arange(
            self.target_length, device=self.device
        )

        # Expand fbank and indices for gathering
        # THIS IS THE FIX: Expand fbank to match the number of samples dimension
        fbank_expanded = fbank.unsqueeze(1).expand(
            -1, self.nr_samples_per_audio, -1, -1, -1
        )
        indices_expanded = indices.view(
            B, self.nr_samples_per_audio, 1, self.target_length, 1
        ).expand(-1, -1, C, -1, F_mel)

        # Use gather to select all windows at once. The shapes now match correctly.
        return_fbank = torch.gather(fbank_expanded, 3, indices_expanded)

        # 4. Flatten, shuffle, and cast (no change)
        flattened = collate_fn(return_fbank.to(torch.bfloat16))  # B*N, C, L, F
        idx = torch.randperm(flattened.size(0))
        # print("=> Batch preparing done")
        return flattened[idx, ...], collate_fn(mask)

    # def forward(self, x, mask):
    #     """X is expected to be in B,C,T,F if normal TAR module
    #     # Here we are actually doing frequency first
    #     # When we do the forward, it is indeed frequency first!
    #        Otherwise 1,B,C,T,F ..
    #     """
    #     assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
    #     B = x.shape[0]
    #     x = x.transpose(2, 3)
    #     x = self.spectrogram_normalize(x)
    #     # 1. Patch the input X, we use these later to mask them.
    #     patches = self.patch_strategy.patch(x)
    #     self.patches_shape = patches.shape  # patched shape: (B, T, C, F_out1, F_out2)
    #     patches = patches.flatten(2)  # flattened shape: (B, T, C*F_out1*F_out2)
    #     # 2. Encode downsampled input
    #     encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
    #     print(f"encoded_patches shape: {encoded_patches.shape}")
    #     x = self.pass_through_encoder(
    #         encoded_patches, ~mask, B
    #     )  # encoder out shape: (B, 1, encoder_embedding_dim)
    #     x = self.pass_through_decoder(
    #         x, ~mask, B
    #     )  # decoder out shape: (B, T, decoder_embedding_dim)
    #     print(f"decoder output shape: {x.shape}")
    #     # Calculate loss on the masked patches
    #     mse_loss = self.mse_loss(x, patches, mask)

    #     return x, patches, mse_loss

    def pretrain_forward(self, x, global_mask, return_latent=False):
        """
        Forward pass for pretraining with SPR pretext tasks.
        inputs:
            x: (B, C, T, F) input mel spectrogram
            global_mask: (B, T) boolean mask indicating which patches are masked
            return_latent: additionally return the decoder output, so a caller
                can attach further losses without a second forward pass.
        outputs:
        """
        assert x.ndim == 4, f"Have to be B,C,T,F got {x.shape}"
        B, C, T, F = x.shape
        # get SPR pseudolabels and valid mask
        SPR_target, SPR_mask = self.get_channel_pair_pseudolabels(
            x, global_mask, pretext_type=self.pseudolabel_type
        )  # SPR_target: (B, C, T, 3), SPR_mask: (B, C, T)

        x = x.transpose(2, 3)  # (B, C, F, T)
        x = self.spectrogram_normalize(x)
        # 1. Patch the input X, we use these later to mask them.
        patches = self.patch_strategy.patch(x)  # (B, T, C, F_out1, F_out2)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)  # (B, T, C*F_out1*F_out2)
        # 2. Encode downsampled input
        encoded_patches = self.patch_strategy.embed(
            x, self.patch_embed
        )  # (B, T, encoder_embedding_dim)
        x = self.pass_through_encoder(
            encoded_patches, ~global_mask, B
        )  # (B, 1, encoder_embedding_dim)
        decoded = self.pass_through_decoder(
            x, ~global_mask, B, return_latent=return_latent
        )  # (B, T, decoder_output_dim)
        if return_latent:
            pred_recon, pred_spr, latent = decoded
        else:
            pred_recon, pred_spr = decoded
        # 3. Calculate reconstruction loss on the masked patches
        recon_loss = self.mse_loss(pred_recon, patches, global_mask)
        # 4. Calculate SPR loss on the masked patches
        pred_spr = pred_spr.reshape(B, T, C, 3).permute(0, 2, 1, 3)  # (B, C, T, 3)
        SPR_target_flat = SPR_target.reshape(-1, 3)
        pred_spr_flat = pred_spr.reshape(-1, 3)
        SPR_mask_flat = SPR_mask.reshape(-1)
        SPR_target_flat = SPR_target_flat[SPR_mask_flat]
        pred_spr_flat = pred_spr_flat[SPR_mask_flat]
        SPR_loss = nn.functional.mse_loss(pred_spr_flat, SPR_target_flat)

        if return_latent:
            return pred_recon, patches, recon_loss, SPR_loss, latent
        return pred_recon, patches, recon_loss, SPR_loss

    def get_audio_representation(self, x, strategy="mean"):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)
        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder(encoded_patches, ~mask, B)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            # We have 25 time patches in 2 second audio. We need to have 20 for STARSS22.
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")

    def get_audio_representation_from_layer(self, x, strategy="mean", block_num=-1):
        """Extract audio representation using different strategies."""
        # Put the model in eval mode when getting representations.
        B = x.shape[0]
        x = x.transpose(2, 3)

        x = self.spectrogram_normalize(x)
        patches = self.patch_strategy.patch(x)
        self.patches_shape = patches.shape
        patches = patches.flatten(2)
        encoded_patches = self.patch_strategy.embed(x, self.patch_embed)
        # Do not mask anything.
        mask = torch.zeros((B, self.num_patches), dtype=torch.bool, device=self.device)
        x = self.pass_through_encoder_until_block(encoded_patches, ~mask, B, block_num)
        if strategy == "mean":
            return x[:, self.encoder_cls_token_num :, :].mean(axis=1)
        elif strategy == "sum":
            return x[:, self.encoder_cls_token_num :, :].sum(axis=1)
        elif strategy == "cls":
            return x[:, 0, :]
        elif strategy == "raw":
            x = x[:, self.encoder_cls_token_num :, :]
            grid_size = self.grid_size
            f, t = grid_size
            outcome = rearrange(
                x, "b (f t) d -> b t (f d)", f=f, d=self.encoder_embedding_dim
            )
            return outcome
        else:
            raise ValueError(f"Strategy '{strategy}' is unrecognized.")

    def get_channel_pair_pseudolabels(self, x, mask, pretext_type="w_to_xyz"):
        """
        From SPR.

        Compute the pretext target and a per-position valid mask for one of
        the three physics-based channel-pair pretext tasks.

        All three tasks share the same output format (3-dim regression per
        masked (channel, time) patch) and the same SPR model output head, but
        differ in (i) which channels are valid for the loss, and (ii) what the
        3-dim target represents at each valid position.

        Args:
            x:           (B, 7, T_frames, 64) intensity spectrogram.
            mask:        (B, 7, T_frames) boolean mask, True = masked.
            pretext_type: one of:
                'w_to_xyz'    - Type 1 (W-X, W-Y, W-Z): mask the W channel and
                                predict the 3-dim FOA direction (X, Y, Z) at
                                that time. Learns "pressure -> direction".
                'orthogonal'  - Type 2 (X-Y, X-Z, Y-Z): mask one of the 3 FOA
                                directional channels and predict the OTHER
                                TWO FOA directional channels at that time.
                                Learns "orthogonal axes consistency".
                'pair'        - Type 3 (X-I_x, Y-I_y, Z-I_z): mask one of the
                                3 FOA directional channels and predict the
                                CORRESPONDING intensity vector component
                                (the strongest single correspondence).
                'foa_to_iv'   - The current default: mask any FOA channel and
                                predict the full 3-dim IV (I_x, I_y, I_z).

        Returns:
            target:     (B, C, T, 3) 3-dim regression target per position.
            valid_mask: (B, C, T) bool — True for positions that contribute
                        to the loss. The trainer combines this with the random
                        chunk mask to decide which positions are active.
        """
        assert x.ndim == 4 and x.shape[1] == 7
        B, C, T, F = x.shape
        # Average the 7 channels' spectrograms across mel bins.
        ch_avg = x.mean(dim=-1)  # (B, 7, T) — [W, Y, Z, X, I_x, I_y, I_z]

        target = torch.zeros(B, C, T, 3, device=x.device, dtype=x.dtype)
        valid_mask = torch.zeros(B, C, T, dtype=torch.bool, device=x.device)

        if pretext_type == "w_to_xyz":
            # Type 1: mask W (channel 0); predict (X, Y, Z) at that time.
            # X, Y, Z are channels 3, 1, 2 in the [W, Y, Z, X, I_x, I_y, I_z]
            # ordering, so the target vector is (X, Y, Z).
            target[:, 0, :, 0] = ch_avg[:, 3, :]  # X
            target[:, 0, :, 1] = ch_avg[:, 1, :]  # Y
            target[:, 0, :, 2] = ch_avg[:, 2, :]  # Z
            valid_mask[:, 0, :] = True  # only W positions

        elif pretext_type == "orthogonal":
            # Type 2: mask X/Y/Z (channels 1, 2, 3); predict the OTHER two.
            # For X (channel 1) mask: target = (0, Y, Z)
            target[:, 1, :, 1] = ch_avg[:, 1, :]  # Y
            target[:, 1, :, 2] = ch_avg[:, 2, :]  # Z
            # For Y (channel 2) mask: target = (X, 0, Z)
            target[:, 2, :, 0] = ch_avg[:, 3, :]  # X
            target[:, 2, :, 2] = ch_avg[:, 2, :]  # Z
            # For Z (channel 3) mask: target = (X, Y, 0)
            target[:, 3, :, 0] = ch_avg[:, 3, :]  # X
            target[:, 3, :, 1] = ch_avg[:, 1, :]  # Y
            valid_mask[:, 1:4, :] = True  # only X, Y, Z positions

        elif pretext_type == "pair":
            # Type 3 (strongest single correspondence): mask X/Y/Z; predict
            # the SINGLE corresponding IV component, placed in the matching
            # axis slot. The other two slots are zeroed.
            # For X (channel 1) mask: target = (I_x, 0, 0)
            target[:, 1, :, 0] = ch_avg[:, 4, :]  # I_x
            # For Y (channel 2) mask: target = (0, I_y, 0)
            target[:, 2, :, 1] = ch_avg[:, 5, :]  # I_y
            # For Z (channel 3) mask: target = (0, 0, I_z)
            target[:, 3, :, 2] = ch_avg[:, 6, :]  # I_z
            valid_mask[:, 1:4, :] = True

        elif pretext_type == "foa_to_iv":
            # The current default: any FOA channel mask -> predict full IV.
            # Channel 0 (W) and 1 (Y), 2 (Z), 3 (X) are all valid mask sources.
            for ch in (0, 1, 2, 3):
                target[:, ch, :, 0] = ch_avg[:, 4, :]  # I_x
                target[:, ch, :, 1] = ch_avg[:, 5, :]  # I_y
                target[:, ch, :, 2] = ch_avg[:, 6, :]  # I_z
            valid_mask[:, 0:4, :] = True

        else:
            raise ValueError(f"Unknown pretext_type: {pretext_type}")

        return target, valid_mask
