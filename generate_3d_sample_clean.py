import torch
import torch.nn.functional as F
from main import Generator3D, generate_noise
import numpy as np

def generate_3d_sample(trained_generators_state_dicts, 
                       pyramid, 
                       options, device, 
                       starting_scale=0, 
                       noise_shape_override=None,
                       evaluation_mode=False):
    """
    Generates a 3D volume sample using a trained SinGAN model.

    This function utilizes a list of trained generator state dictionaries to
    create a 3D volume by progressively generating through a pyramid of scales.
    The generation can start from a custom scale and optionally use a custom
    noise shape.

    Args:
        trained_generators_state_dicts (list): List of state dictionaries for
            the trained generator models, ordered from finest to coarsest scale.
        pyramid (list): A list of tensors representing the pyramid levels,
            ordered from coarsest to finest scale.
        options: Options object containing generation parameters such as 'nc_im'
            and 'scale_factor'.
        device: The torch device to perform the generation on.
        starting_scale (int, optional): The starting scale index for
            generation, with 0 being the finest scale. Defaults to 0.
        noise_shape_override (tuple, optional): A tuple specifying a custom
            noise shape (D, H, W) to use at the starting scale. If None, uses
            the shape from the pyramid.
        evaluation_mode (bool, optional): If True, sets the generators to evaluation mode.

    Returns:
        torch.Tensor: A 5D tensor representing the generated 3D volume
        sample with shape (batch_size, C, D, H, W).
    """

    num_scales = len(trained_generators_state_dicts)
    generators = []

    for i in range(num_scales):
        generator = Generator3D(options).to(device)
        generator.load_state_dict(trained_generators_state_dicts[i])
        if evaluation_mode:
            generator.eval()
        else:
            generator.train()
        generators.append(generator)

    generators.reverse()
    actual_starting_scale = num_scales - 1 - starting_scale

    if noise_shape_override:
        noise_shape = (options.nc_im,) + tuple(noise_shape_override)
    else:
        noise_shape = pyramid[starting_scale].shape[1:]

    current_noise = generate_noise(noise_shape, device)
    if starting_scale == 0:
        current_volume = torch.zeros((1,) + noise_shape, device=device)
    else:
        current_volume = pyramid[::-1][starting_scale]

    with torch.no_grad():
        for scale_index in range(actual_starting_scale, -1, -1):
            generator_index = num_scales - 1 - scale_index
            generator = generators[generator_index]
            current_volume = generator(current_noise, current_volume)

            if scale_index > 0:
                target_dimensions = pyramid[::-1][scale_index - 1].shape[2:]
                current_volume = F.interpolate(
                    current_volume, size=target_dimensions, 
                    mode='trilinear', align_corners=False
                )
                current_noise = generate_noise((1, *target_dimensions), device)

    return current_volume


def cdf_mapping(source, target):
    """
    Apply CDF mapping to make the distribution of 'source' match that of 'target'.
    
    Parameters:
        source (np.ndarray): The input array to be transformed.
        target (np.ndarray): The reference array whose distribution will be matched.

    Returns:
        np.ndarray: Transformed array with the same shape as 'source', but with
                    distribution matched to 'target'.
    """
    # Flatten both arrays
    source_flat = source.ravel()
    target_flat = target.ravel()

    # Sort and compute CDF values for source
    source_sorted = np.sort(source_flat)
    source_cdf = np.linspace(0, 1, len(source_sorted), endpoint=False)

    # Sort and compute quantiles for target
    target_sorted = np.sort(target_flat)

    # Create interpolator for mapping source values to target quantiles
    source_to_cdf = np.interp(source_flat, source_sorted, source_cdf)
    mapped_values = np.interp(source_to_cdf, np.linspace(0, 1, len(target_sorted), endpoint=False), target_sorted)

    # Reshape back to original shape
    return mapped_values.reshape(source.shape)
