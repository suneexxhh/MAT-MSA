"""Scale configurations used by the MAT+MSA SMAC experiments.

The entries correspond to Table 1 of the MSCM manuscript.  A configuration
is represented as ``(number_of_scales, first_intermediate_chunk_size)``.
For three scales, the resulting hierarchy is ``[1, c1, N]``.  For four
scales, it is ``[1, c1, c1**2, N]``.
"""

TABLE1_SCALE_CONFIG = {
    "1c3s5z": (3, 3),
    "3s5z": (3, 4),
    "5m_vs_6m": (3, 2),
    "8m_vs_9m": (3, 4),
    "10m_vs_11m": (3, 4),
    "6h_vs_8z": (3, 4),
    "3s5z_vs_3s6z": (3, 3),
    "MMM2": (3, 3),
    "27m_vs_30m": (4, 3),
}


def get_scale_config(map_name, default_levels=3, default_chunk_size=4):
    """Return ``(n_levels, c1)`` for a map, or the supplied defaults."""
    return TABLE1_SCALE_CONFIG.get(
        map_name, (default_levels, default_chunk_size)
    )


def get_chunk_sizes(map_name, n_agents, n_levels=None, chunk_size=None,
                    use_map_adaptive=True):
    """Build the concrete attention hierarchy for a SMAC map."""
    if use_map_adaptive:
        n_levels, chunk_size = get_scale_config(
            map_name,
            default_levels=n_levels if n_levels is not None else 3,
            default_chunk_size=chunk_size if chunk_size is not None else 4,
        )
    else:
        n_levels = n_levels if n_levels is not None else 3
        chunk_size = chunk_size if chunk_size is not None else 4

    if n_levels < 2:
        raise ValueError("n_levels must be at least 2")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    sizes = [1]
    for level in range(1, n_levels - 1):
        sizes.append(chunk_size ** level)
    sizes.append(n_agents)
    return n_levels, chunk_size, sizes
