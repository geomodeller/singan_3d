## Update Note =================================================================
# 2025.5.14: Updated to v_0.0.3
#  - add a visualization every 5000 epoch
#  - fix initialization
#  - make all print into logging so that it can be read external file 
# =============================================================================

# =============================================================================
# 3D SinGAN PyTorch Implementation
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import logging

# Optional imports for data loading/saving
try:
    import nibabel as nib
except ImportError:
    logging.info("nibabel not found. Install if needed for NIfTI files.")
# Optional import for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.info("matplotlib not found. Install if needed for visualization.")

# =============================================================================
# Utility Functions
# =============================================================================
def initialize_model(model, scale=1.):
    """
    Initializes weights for Conv and BatchNorm layers with specific distributions.
    Applied recursively to all modules in a model.
    """
    for m in model.modules():
        # Initialize Conv weights with a normal distribution
        if isinstance(m, nn.Conv3d):
            m.weight.data.normal_(0.0, 0.02)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        # Initialize BatchNorm weights and biases
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        # Other layer types (like Linear, etc.) are not initialized by this function
        else:
            continue


def calculate_gradient_penalty(netD, real_data, fake_data, device, lambda_gp):
    #... (implementation as in Section 4.2)...
    batch_size = real_data.size(0) # Should be 1 for SinGAN
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device) # Expand for 5D tensor

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True)

    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# def generate_noise(size, device):
#     #... (implementation as in Section 4.2)...
#     # size = (channels, depth, height, width)
#     noise = torch.randn(1, 1, size, size, size, device=device)
#     return noise
def generate_noise(size_tuple_cdhw, device, batch_size=1): # Renamed 'size' to be more explicit
    """
    Generates a 5D noise tensor.
    The original comment was "size = (channels, depth, height, width)".
    This function interprets the 'size_tuple_cdhw' argument as that tuple.

    Args:
        size_tuple_cdhw (tuple): A tuple representing (channels, depth, height, width).
        device: The torch device to create the tensor on.
        batch_size (int): The batch size for the noise. Defaults to 1.
    Returns:
        torch.Tensor: A noise tensor of shape (batch_size, C, D, H, W).
    """
    if not (isinstance(size_tuple_cdhw, tuple) and len(size_tuple_cdhw) == 4):
        raise ValueError("size_tuple_cdhw must be a tuple of (channels, depth, height, width)")

    # Unpack the shape tuple for torch.randn
    # Shape will be (batch_size, channels, depth, height, width)
    noise = torch.randn(batch_size, *size_tuple_cdhw, device=device)
    return noise

def upsample_3d(x, scale_factor=2.0):
    #... (implementation as in Section 4.2)...
    return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False)

def downsample_3d(x, scale_factor=0.5):
    #... (implementation as in Section 4.2)...
    return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False, recompute_scale_factor=False)

def create_scale_pyramid(real_vol, scale_factor_base, min_size, device):
    #... (implementation as in Section 4.2, using size interpolation)...
    pyramid = []
    current_vol = real_vol.clone() # Clone to avoid modifying original
    target_scale_factor = scale_factor_base # This is the factor for *downsampling*

    # Add original volume (finest scale) first
    pyramid.append(current_vol)

    # Calculate dimensions iteratively
    current_dims = np.array(current_vol.shape[-3:])
    while True:
        # Calculate next scale dimensions using ceiling
        next_dims_float = current_dims * target_scale_factor
        # Use ceiling for dimensions, ensure minimum size of 1
        next_dims = np.maximum(1, np.ceil(next_dims_float)).astype(int)

        # Check if minimum dimension is reached
        if min(next_dims) <= min_size:
            break

        # Downsample using interpolate with target size
        current_vol = F.interpolate(current_vol, size=tuple(next_dims), mode='trilinear', align_corners=False)
        pyramid.append(current_vol)
        current_dims = next_dims

    return pyramid[::-1] # Return coarsest to finest

def load_real_volume(path, device):
    # Example loading function (adapt based on your file format)
    if path.endswith('.npy'):
        vol = np.load(path)
    elif path.endswith('.nii') or path.endswith('.nii.gz'):
        if 'nib' not in globals():
            raise ImportError("nibabel required for NIfTI files. Please install it.")
        vol = nib.load(path).get_fdata()
    else:
        raise ValueError("Unsupported file format. Please use.npy or.nii/.nii.gz or adapt the loading function.")

    # Ensure single channel and correct dimension order (C, D, H, W)
    if vol.ndim == 3: # Add channel dimension if missing
        vol = vol[np.newaxis,...]
    elif vol.ndim == 4: # Assume (D, H, W, C) or (C, D, H, W) - adjust if needed
        if vol.shape[-1] == 1: # Assume (D, H, W, C), permute
             vol = np.transpose(vol, (3, 0, 1, 2))
        elif vol.shape[0] != 1: # If first dim is not 1, and it's 4D, assume it's (D, H, W, C) where C>1
             raise ValueError("Input volume has 4 dimensions and the first dimension is not 1 (channel). Please provide single channel data (C,D,H,W or D,H,W,C=1).")
        # If shape[0] == 1, assume it's already (C, D, H, W)
    else:
        raise ValueError(f"Unsupported number of dimensions: {vol.ndim}. Expected 3 or 4.")

    # Convert to PyTorch tensor
    vol_tensor = torch.from_numpy(vol.astype(np.float32))

    # Normalize to [-1, 1] (adjust if your data has a different range)
    min_val = torch.min(vol_tensor)
    max_val = torch.max(vol_tensor)
    if max_val > min_val:
        vol_tensor = 2 * ((vol_tensor - min_val) / (max_val - min_val)) - 1
    else: # Handle constant volume case
        vol_tensor = torch.zeros_like(vol_tensor)

    # Add batch dimension (B, C, D, H, W)
    vol_tensor = vol_tensor.unsqueeze(0).to(device)
    return vol_tensor

def save_snapshot(vol_tensor, path):
    # Example saving function (adapt as needed)
    vol_np = vol_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove Batch and Channel
    plt.figure(figsize=(10,5))
    plt.imshow(vol_np.squeeze()[0])
    plt.savefig(path)
    plt.close()

def save_volume(vol_tensor, path):
    # Example saving function (adapt as needed)
    vol_np = vol_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove Batch and Channel
    # Denormalize if needed before saving (assuming original range was [0, X])
    # vol_np = (vol_np + 1) / 2 * MAX_ORIGINAL_VALUE
    if path.endswith('.npy'):
        np.save(path, vol_np)
    elif path.endswith('.nii') or path.endswith('.nii.gz'):
        if 'nib' not in globals():
            raise ImportError("nibabel required for NIfTI files. Please install it.")
        # Create NIfTI image object (affine matrix might need adjustment)
        ni_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
        nib.save(ni_img, path)
    else:
        logging.warning(f"Unsupported save format for {path}. Saving as .npy")
        np.save(path.replace(os.path.splitext(path)[1], '.npy'), vol_np)

# =============================================================================
# Network Definitions
# =============================================================================

class ConvBlock3D(nn.Module):
    #... (implementation as in Section 4.3)...
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, batch_norm=True, activation=True):
        super(ConvBlock3D, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias=bias))
        if batch_norm:
            # Use track_running_stats=True, affine=True by default
            layers.append(nn.BatchNorm3d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class Generator3D(nn.Module):
    #... (implementation as in Section 4.4, using opt attributes)...
    def __init__(self, opt):
        super(Generator3D, self).__init__()
        self.opt = opt
        N = opt.nfc # Base number of filters

        # Head: ConvBlock3D
        self.head = ConvBlock3D(int(opt.nc_im), int(N), opt.ker_size, 1, opt.padd_size)

        # Body: Sequence of ConvBlock3Ds
        body_blocks = []
        for _ in range(opt.num_layer):
            body_blocks.append(ConvBlock3D(N, N, opt.ker_size, 1, opt.padd_size))
        self.body = nn.Sequential(*body_blocks)

        # Tail: Final Conv3d layer
        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, opt.ker_size, 1, opt.padd_size),
            nn.Tanh() # Output normalized to [-1, 1]
        )

    def forward(self, noise, prev_out_upsampled):
        # Combine noise and previous output (simple addition here)
        # Ensure noise has same spatial dims as prev_out_upsampled
        if noise.shape[-3:]!= prev_out_upsampled.shape[-3:]:
             noise_resized = F.interpolate(noise, size=prev_out_upsampled.shape[-3:], mode='trilinear', align_corners=False)
        else:
             noise_resized = noise

        x = noise_resized + prev_out_upsampled

        x = self.head(x)
        x_body = self.body(x)
        residual = self.tail(x_body)

        # Apply residual connection - ensure sizes match for addition
        # We need to crop prev_out_upsampled to match residual's size if padding caused differences
        target_size = residual.shape[-3:]
        input_size = prev_out_upsampled.shape[-3:]

        # Calculate cropping indices (center crop)
        diff_d = input_size[0] - target_size[0]
        diff_h = input_size[1] - target_size[1]
        diff_w = input_size[2] - target_size[2]

        start_d = diff_d // 2
        start_h = diff_h // 2
        start_w = diff_w // 2

        # Handle cases where target is larger (shouldn't happen with padding=1, kernel=3, stride=1)
        if diff_d < 0 or diff_h < 0 or diff_w < 0:
            # Pad residual instead if it's smaller (less likely)
            pad_d = max(0, -diff_d)
            pad_h = max(0, -diff_h)
            pad_w = max(0, -diff_w)
            residual = F.pad(residual, (pad_w//2, pad_w - pad_w//2,
                                        pad_h//2, pad_h - pad_h//2,
                                        pad_d//2, pad_d - pad_d//2))
            prev_cropped = prev_out_upsampled
        else:
            # Crop prev_out_upsampled
            prev_cropped = prev_out_upsampled[:, :, start_d:start_d + target_size[0],
                                                    start_h:start_h + target_size[1],
                                                    start_w:start_w + target_size[2]]

        output = prev_cropped + residual
        return output


class Discriminator3D(nn.Module):
    #... (implementation as in Section 4.5, using opt attributes)...
    def __init__(self, opt):
        super(Discriminator3D, self).__init__()
        self.opt = opt
        N = opt.nfc # Base number of filters

        layers = []
        # Initial layer
        # No BN on first layer, bias might be okay
        layers.append(ConvBlock3D(opt.nc_im, N, opt.ker_size, 1, opt.padd_size, batch_norm=False, activation=True))

        # Downsampling layers
        current_filters = N
        # num_layer determines depth, adjust strides for downsampling
        # Example: 3 layers with stride 2 reduces size by 8x
        num_downsample = opt.num_layer # Or define separately
        for i in range(num_downsample):
            # Use stride=2 for downsampling
            # Increase filters, cap at 512 or opt.max_nfc
            next_filters = min(current_filters * 2, getattr(opt, 'max_nfc', 512))
            # Padding=1, kernel=3, stride=2 usually works well
            layers.append(ConvBlock3D(current_filters, next_filters, opt.ker_size, stride=1, padding=opt.padd_size, batch_norm=True, activation=True))
            current_filters = next_filters

        # Final layer for WGAN-GP (1 output channel, no activation/BN)
        # Kernel size might need adjustment based on final feature map size
        # Using kernel=3, padding=1 keeps size if stride=1
        layers.append(nn.Conv3d(current_filters, 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =============================================================================
# Training Function
# =============================================================================

def train_3d_singan(real_volume, opt, device):
    #... (implementation as in Section 4.7)...
    # 1. Create image pyramid
    scale_factor_base = 1 / opt.scale_factor # For downsampling
    pyramid = create_scale_pyramid(real_volume, scale_factor_base, opt.min_size, device)
    num_scales = len(pyramid)
    logging.info(f"Created pyramid with {num_scales} scales.")
    for i, vol in enumerate(pyramid):
        logging.info(f"  Scale {num_scales - 1 - i} (Coarsest={i==0}): {vol.shape}")


    # 2. Initialize storage
    stored_reconstructions = {} # Key: scale index (N, N-1,... 0)
    trained_generators = [] # List of state dicts, finest to coarsest
    fixed_noise_maps = {} # Key: scale index (N, N-1,... 0)

    # Generate fixed noise z* for reconstruction at coarsest scale (index N = num_scales - 1)
    coarsest_scale_idx = num_scales - 1
    coarsest_shape = pyramid[0].shape # pyramid is coarsest
    fixed_noise_maps[coarsest_scale_idx] = generate_noise(coarsest_shape[1:], device)

    # 3. Loop through scales from Coarsest (index 0 in pyramid) to Finest (index num_scales-1 in pyramid)
    scale_factor_r = opt.scale_factor # For upsampling between scales

    # Ensure output directory exists
    os.makedirs(opt.outdir, exist_ok=True)

    for n in range(num_scales): # n iterates through pyramid list [0, num_scales-1]
        scale_idx_actual = num_scales - 1 - n # Actual scale index (N, N-1,..., 0)
        logging.info(f"\n--- Training Scale {scale_idx_actual} ---")
        real_vol_at_scale = pyramid[n].to(device) # Ensure it's on the correct device
        opt.nc_im = real_vol_at_scale.shape[1] # Set number of input channels for the model
        current_size = real_vol_at_scale.shape[-3:]
        logging.info(f"Volume size: {current_size}")

        # Adjust nfc potentially based on scale (optional, start simple)
        # opt.nfc = max(opt.min_nfc, opt.nfc // (2**(scale_idx_actual//4))) # Example decrease

        # Initialize Generator and Discriminator for this scale
        netG = Generator3D(opt).to(device)
        netD = Discriminator3D(opt).to(device)
        netG.apply(initialize_model)
        netD.apply(initialize_model)

        # Optimizers
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # Schedulers (adjust milestones/gamma as needed)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[opt.niter // 2, opt.niter * 3 // 4], gamma=0.1)
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[opt.niter // 2, opt.niter * 3 // 4], gamma=0.1)

        # Store zero reconstruction noise for finer scales
        if scale_idx_actual < coarsest_scale_idx:
             fixed_noise_maps[scale_idx_actual] = torch.zeros((1,) + real_vol_at_scale.shape[1:], device=device)

        # Inner training loop for the current scale
        for iter_num in range(opt.niter):
            # --- Discriminator Training ---
            netD.zero_grad()
            # Real data loss
            output_real = netD(real_vol_at_scale)
            errD_real = -output_real.mean() # WGAN loss for real

            # Fake data loss
            noise = generate_noise(real_vol_at_scale.shape[1:], device) # Use current scale size for noise
            # Get previous scale reconstruction (handle coarsest scale where prev is None)
            if scale_idx_actual == coarsest_scale_idx: # Coarsest scale
                # Input to G is just noise, prev_rec is effectively zero
                prev_rec_detached = torch.zeros_like(real_vol_at_scale, device=device)
            else:
                # Get reconstruction from the *previous actual scale* (coarser scale)
                prev_rec_detached = stored_reconstructions[scale_idx_actual + 1].detach()

            # Upsample previous reconstruction
            prev_rec_upsampled = upsample_3d(prev_rec_detached, scale_factor=scale_factor_r)
            # Ensure size matches current scale exactly
            prev_rec_upsampled = F.interpolate(prev_rec_upsampled, size=current_size, mode='trilinear', align_corners=False)

            fake_vol = netG(noise, prev_rec_upsampled).detach() # Detach fake_vol for D training
            output_fake = netD(fake_vol)
            errD_fake = output_fake.mean() # WGAN loss for fake

            # Gradient Penalty
            gradient_penalty = calculate_gradient_penalty(netD, real_vol_at_scale.data, fake_vol.data, device, opt.lambda_gp)

            # Total D loss
            errD = errD_real + errD_fake + gradient_penalty
            errD.backward()
            optimizerD.step()

            # Clip discriminator weights if not using GP (legacy WGAN) - Not recommended
            # for p in netD.parameters():
            #     p.data.clamp_(-opt.clip_value, opt.clip_value)

            # --- Generator Training (typically less frequent than D, e.g., every opt.Dsteps) ---
            if iter_num % opt.Gsteps == 0:
                netG.zero_grad()
                # We need to regenerate fake_vol for G update as graph was potentially freed
                # Reuse noise and prev_rec_upsampled from D step
                fake_vol_for_G = netG(noise, prev_rec_upsampled) # Don't detach here
                output_fake_G = netD(fake_vol_for_G)
                errG_adv = -output_fake_G.mean() # Generator wants to maximize D's output for fake

                # Reconstruction loss
                if opt.alpha > 0:
                    rec_noise_this_scale = fixed_noise_maps[scale_idx_actual] # Get appropriate noise
                    if scale_idx_actual == coarsest_scale_idx: # Coarsest scale
                        prev_rec_for_G = torch.zeros_like(real_vol_at_scale, device=device)
                    else:
                        prev_rec_for_G = stored_reconstructions[scale_idx_actual + 1].detach() # Use stored reconstruction

                    prev_rec_upsampled_for_G = upsample_3d(prev_rec_for_G, scale_factor=scale_factor_r)
                    prev_rec_upsampled_for_G = F.interpolate(prev_rec_upsampled_for_G, 
                                                             size=current_size, 
                                                             mode='trilinear', 
                                                             align_corners=False)

                    reconstruction = netG(rec_noise_this_scale, prev_rec_upsampled_for_G)
                    errG_rec = F.mse_loss(reconstruction, real_vol_at_scale) * opt.alpha
                else:
                    errG_rec = torch.tensor(0.0, device=device)

                # Total G loss
                errG = errG_adv + errG_rec
                errG.backward()
                optimizerG.step()

            # Log progress
            if iter_num % 1000 == 0:
                logging.info(f"Scale [{scale_idx_actual}] Iter [{iter_num}/{opt.niter}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} (Adv: {errG_adv.item():.4f} Rec: {errG_rec.item():.4f})")

            # Step schedulers
            schedulerG.step()
            schedulerD.step()

            # Optional: Save intermediate generated sample visualization
            if ((iter_num+1) % opt.save_every_epoch == 0) or (iter_num == 0):
                 with torch.no_grad():
                     vis_noise = generate_noise(real_vol_at_scale.shape[1:], device)
                     vis_fake = netG(vis_noise, prev_rec_upsampled).detach()
                     save_snapshot(vis_fake, os.path.join(opt.outdir, f"sample_scale{scale_idx_actual}_iter{iter_num}.png"))
                     # save_volume(vis_fake, os.path.join(opt.outdir, f"sample_scale{scale_idx_actual}_iter{iter_num}.npy"))


        # --- End of scale training ---
        # Save trained generator state dict (append to list)
        trained_generators.append(netG.state_dict())
        # Save final reconstruction for this scale (needed for next scale's input)
        if opt.alpha > 0:
             with torch.no_grad():
                 rec_noise_this_scale = fixed_noise_maps[scale_idx_actual]
                 if scale_idx_actual == coarsest_scale_idx:
                     prev_rec_final = torch.zeros_like(real_vol_at_scale, device=device)
                 else:
                     prev_rec_final = stored_reconstructions[scale_idx_actual + 1].detach()
                 prev_rec_upsampled_final = upsample_3d(prev_rec_final, scale_factor=scale_factor_r)
                 prev_rec_upsampled_final = F.interpolate(prev_rec_upsampled_final, size=current_size, mode='trilinear', align_corners=False)
                 final_reconstruction = netG(rec_noise_this_scale, prev_rec_upsampled_final).detach()
                 stored_reconstructions[scale_idx_actual] = final_reconstruction
                 # Save final reconstruction image for inspection
                 save_volume(final_reconstruction, os.path.join(opt.outdir, f"reconstruction_scale{scale_idx_actual}.npy"))
                 save_snapshot(final_reconstruction, os.path.join(opt.outdir, f"reconstruction_scale{scale_idx_actual}.png"))

        else:
             # If no reconstruction loss, store a zero tensor or handle differently
             stored_reconstructions[scale_idx_actual] = torch.zeros_like(real_vol_at_scale, device=device)

        # Save model checkpoint for this scale
        torch.save({
            'scale': scale_idx_actual,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, os.path.join(opt.outdir, f"model_scale{scale_idx_actual}.pth"))


    logging.info("\n--- Training Complete ---")
    # Return trained generators (state dicts, ordered finest to coarsest)
    # and fixed noise maps (dict by scale index)
    return trained_generators[::-1], fixed_noise_maps, pyramid # Return Gs finest to coarsest

# =============================================================================
# Generation Function
# =============================================================================
def generate_3d_sample(trained_generators_state_dicts, 
                       # fixed_noise_maps, 
                       pyramid, opt, device, gen_start_scale=0, custom_noise_shape=None):
    num_scales = len(trained_generators_state_dicts)
    generators = []
    # Load generators from state dicts
    for i in range(num_scales):
        netG = Generator3D(opt).to(device)
        # Load state dict corresponding to scale i (0=finest, N=coarsest)
        # Note: trained_generators was returned finest-to-coarsest
        netG.load_state_dict(trained_generators_state_dicts[i])
        netG.eval() # Set to evaluation mode
        generators.append(netG)

    # Determine starting scale index (N = num_scales - 1)
    start_scale_idx_actual = num_scales - 1 - gen_start_scale

    # Generate initial noise at the starting scale
    if custom_noise_shape:
        # Use custom shape (C, D, H, W)
        noise_shape = (opt.nc_im,) + tuple(custom_noise_shape)
    else:
        # Use shape from the corresponding pyramid level
        noise_shape = pyramid[num_scales - 1 - start_scale_idx_actual].shape[1:] # Get C, D, H, W

    current_noise = generate_noise(noise_shape, device)
    current_vol = torch.zeros((1,) + noise_shape, device=device) # Initial previous output is zero

    scale_factor_r = opt.scale_factor

    # Generate through the pyramid from start_scale down to 0 (finest)
    with torch.no_grad():
        for scale_idx in range(start_scale_idx_actual, -1, -1): # Iterate N, N-1,..., start_scale,..., 0
            # Get the generator for this scale (index maps directly: 0=finest, N=coarsest)
            # Need to map scale_idx (N..0) to list index (0..N)
            generator_list_idx = num_scales - 1 - scale_idx
            netG = generators[generator_list_idx]

            # Upsample previous volume
            prev_vol_upsampled = upsample_3d(current_vol, scale_factor=scale_factor_r)

            # Determine target size for this scale
            if custom_noise_shape and scale_idx == start_scale_idx_actual:
                 target_size = noise_shape[-3:]
            elif scale_idx < num_scales -1 : # Not the coarsest scale being generated
                 # Infer target size by scaling up from the next coarser scale's pyramid shape
                 coarser_pyramid_idx = num_scales - 1 - (scale_idx + 1)
                 coarser_dims = np.array(pyramid[coarser_pyramid_idx].shape[-3:])
                 target_dims_float = coarser_dims * scale_factor_r
                 target_size = tuple(np.round(target_dims_float).astype(int))
                 # Ensure minimum size 1
                 target_size = tuple(max(1, d) for d in target_size)
            else: # Coarsest scale being generated (scale_idx == num_scales - 1)
                 target_size = noise_shape[-3:] # Use noise shape directly


            # Resize upsampled volume and noise to target size
            prev_vol_upsampled = F.interpolate(prev_vol_upsampled, size=target_size, mode='trilinear', align_corners=False)
            noise_this_scale = F.interpolate(current_noise, size=target_size, mode='trilinear', align_corners=False)

            # Generate volume for this scale
            current_vol = netG(noise_this_scale, prev_vol_upsampled)

            # Prepare noise for the next finer scale (if any)
            if scale_idx > 0:
                current_noise = generate_noise(target_size, device) # Generate new noise based on current size

    return current_vol


# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    import logging
    parser = argparse.ArgumentParser(description='Train 3D SinGAN')
    # Input/Output
    parser.add_argument('--input_name', default = '3d_data_new.npy', help='Path to the single input 3D volume (.npy or.nii/.nii.gz)')
    parser.add_argument('--outdir', default='Output3D', help='Directory to save results')
    # Model parameters
    parser.add_argument('--min_size', type=int, default=20, help='Minimum spatial dimension at coarsest scale')
    parser.add_argument('--scale_factor', type=float, default=4/3, help='Upscaling factor between scales (r)')
    parser.add_argument('--nfc', type=int, default=32, help='Base number of filters in conv layers')
    parser.add_argument('--ker_size', type=int, default=3, help='Kernel size in conv layers')
    parser.add_argument('--num_layer', type=int, default=5, help='Number of layers in G/D body')
    # Training parameters
    parser.add_argument('--niter', type=int, default=2000, help='Number of training iterations per scale')
    parser.add_argument('--lr_g', type=float, default=0.0005, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='Learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--lambda_gp', type=float, default=0.1, help='Gradient penalty lambda') # WGAN-GP lambda
    parser.add_argument('--alpha', type=float, default=10.0, help='Reconstruction loss weight')
    parser.add_argument('--Gsteps', type=int, default=1, help='Train G every Gsteps D iterations')
    parser.add_argument('--save_every_epoch', type=int, default=5000, help='save visual snapshot every this many epochs')
    # Execution
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--mode', default='train', choices=['train', 'generate'], help='Operation mode')
    parser.add_argument('--gen_start_scale', type=int, default=0, help='Scale to start generation from (0=finest, 1=second finest,...)')
    parser.add_argument('--gen_output_name', default='generated_sample.npy', help='Filename for generated sample')
    parser.add_argument('--gen_size', type=str, default=None, help='Custom generation size D,H,W (e.g., "32,128,128")')
    parser.add_argument('--log_file', default='singan_3d.log', help='File to save log output')
    parser.add_argument('--log_level', default='INFO', help='File to save log output')


    opt = parser.parse_args()

    if os.path.isdir(opt.outdir) == False:
        os.mkdir(opt.outdir)


    print("Logging completed. Check demo.log file.")
    # --- Configure Logging ---
    # Moved logging configuration to be the very first thing
    # so all messages, including potential errors during arg parsing, are logged.
        
    if opt.log_level.upper() == 'DEBUG':
        level = logging.DEBUG
    elif opt.log_level.upper() == 'INFO':
        level = logging.INFO
    elif opt.log_level.upper() == 'WARNING':
        level = logging.WARNING
    elif opt.log_level.upper() == 'ERROR':
        level = logging.ERROR

    logging.basicConfig(
                        filename=os.path.join(opt.outdir, opt.log_file),            # Log file name
                        level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s',        
                        filemode='w', force=True  # Overwrite existing log file
                        )

    # Suppress verbose matplotlib logging if it's imported
    if 'matplotlib' in globals() and matplotlib is not None:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


    
    # --- Derived/Calculated Parameters ---
    opt.padd_size = opt.ker_size // 2 # Calculate padding for 'same' convolution

    # --- Device Setup ---
    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{opt.gpu}")
        logging.info(f"Using GPU: {opt.gpu}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # --- Mode Selection ---
    if opt.mode == 'train':
        logging.info("Starting Training Mode...")
        # Load real volume
        try:
            real_volume = load_real_volume(opt.input_name, device)
            logging.info(f"Loaded real volume: {real_volume.shape}")
        except Exception as e:
            logging.error(f"Error loading real volume: {e}")
            exit()

        opt.nc_im = real_volume.shape[1] # Set number of input channels from loaded volume

        # Train the model
        trained_generators, fixed_noise_maps, pyramid = train_3d_singan(real_volume, opt, device)

        # Save final trained models and noise maps
        save_data = {
            'generators_state_dicts': trained_generators, # Finest to Coarsest
            'fixed_noise_maps': fixed_noise_maps,       # Dict by scale index
            'pyramid_shapes': [list(p.shape) for p in pyramid], # Coarsest to Finest shapes
            'pyramid': [p for p in pyramid], # Coarsest to Finest shapes
            'opt': vars(opt) # Save options used for training
        }
        torch.save(save_data, os.path.join(opt.outdir, 'trained_model_3d.pth'))
        logging.info(f"Trained model saved to {os.path.join(opt.outdir, 'trained_model_3d.pth')}")

    elif opt.mode == 'generate':
        logging.info("Starting Generation Mode...")
        # Load trained model data
        model_path = os.path.join(opt.outdir, 'trained_model_3d.pth')
        if not os.path.exists(model_path):
             logging.error(f"Trained model not found at {model_path}. Please train first.")
             exit()

        saved_data = torch.load(model_path, map_location=device)
        trained_generators_state_dicts = saved_data['generators_state_dicts']
        fixed_noise_maps = saved_data['fixed_noise_maps']
        pyramid_shapes = saved_data['pyramid_shapes']
        train_opt_dict = saved_data['opt']

        # Create dummy opt object and populate with training options
        class DummyOpt: pass
        train_opt = DummyOpt()
        for k, v in train_opt_dict.items(): setattr(train_opt, k, v)
        # Ensure nc_im is correctly set for generation if it wasn't explicitly saved
        if not hasattr(train_opt, 'nc_im'):
            # Try to infer from pyramid_shapes, assuming (B, C, D, H, W)
            if pyramid_shapes and len(pyramid_shapes[0]) >= 2:
                train_opt.nc_im = pyramid_shapes[0][1]
            else:
                train_opt.nc_im = 1 # Default to 1 if info is missing
                logging.warning("Number of input channels (nc_im) not found in saved options. Defaulting to 1.")


        # Create dummy pyramid list just for shape reference during generation
        # Ensure dimensions match (B, C, D, H, W) if saved as (C, D, H, W)
        dummy_pyramid = []
        for s_shape in pyramid_shapes:
            if len(s_shape) == 4: # Assuming saved shape is (C, D, H, W)
                dummy_pyramid.append(torch.empty((1,) + tuple(s_shape)))
            else: # Assuming saved shape is already (B, C, D, H, W)
                dummy_pyramid.append(torch.empty(tuple(s_shape)))
        logging.info(f"Reconstructed dummy pyramid shapes for generation: {[p.shape for p in dummy_pyramid]}")


        custom_shape = None
        if opt.gen_size:
            try:
                custom_shape = [int(d) for d in opt.gen_size.split(',')]
                if len(custom_shape)!= 3: raise ValueError
                logging.info(f"Generating with custom size: {custom_shape}")
            except:
                logging.error(f"Error: Invalid gen_size format '{opt.gen_size}'. Use D,H,W (e.g., '32,128,128'). Exiting.")
                exit()


        # Generate sample
        generated_volume = generate_3d_sample(trained_generators_state_dicts,
                                              fixed_noise_maps,
                                              dummy_pyramid, # Pass dummy pyramid for shape reference
                                              train_opt, # Use options from training
                                              device,
                                              gen_start_scale=opt.gen_start_scale,
                                              custom_noise_shape=custom_shape)

        # Save generated sample
        output_path = os.path.join(opt.outdir, opt.gen_output_name)
        save_volume(generated_volume, output_path)
        logging.info(f"Generated sample saved to {output_path}")

    else:
        logging.error(f"Error: Unknown mode '{opt.mode}'")