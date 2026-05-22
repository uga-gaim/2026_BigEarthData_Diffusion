import numpy as np
import matplotlib as mpl


# Color map referenced from NWP
# https://unidata.github.io/python-gallery/examples/Precipitation_Map.html
#
clevs = np.array(
    [
        0,
        1,
        2.5,
        5,
        7.5,
        10,
        15,
        20,
        30,
        40,
        50,
        70,
        100,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        750,
    ]
)

cmap_data = np.array(
    [
        (1.0, 1.0, 1.0),
        (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
        (0.0, 1.0, 1.0),
        (0.0, 0.8784313797950745, 0.501960813999176),
        (0.0, 0.7529411911964417, 0.0),
        (0.501960813999176, 0.8784313797950745, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.6274510025978088, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.125490203499794, 0.501960813999176),
        (0.9411764740943909, 0.250980406999588, 1.0),
        (0.501960813999176, 0.125490203499794, 1.0),
        (0.250980406999588, 0.250980406999588, 1.0),
        (0.125490203499794, 0.125490203499794, 0.501960813999176),
        (0.125490203499794, 0.125490203499794, 0.125490203499794),
        (0.501960813999176, 0.501960813999176, 0.501960813999176),
        (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
        (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
        (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
        (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
        (0.4000000059604645, 0.20000000298023224, 0.0),
    ]
)


assert len(cmap_data) == len(
    clevs
), f"Mismatch of lengths: {len(cmap_data)} != {len(clevs)}"


class Precipitation:
    @staticmethod
    def get_cmap(start_idx=0, end_idx=len(cmap_data)):
        assert start_idx >= 0, "Start index must to non negative"
        assert end_idx <= len(cmap_data), "End index too large"

        return mpl.colors.ListedColormap(cmap_data[start_idx:end_idx], "precipitation")

    @staticmethod
    def get_norm(start_idx=0, end_idx=len(clevs)):
        assert start_idx >= 0, "Start index must to non negative"
        assert end_idx <= len(cmap_data), "End index too large"

        return mpl.colors.BoundaryNorm(clevs[start_idx:end_idx], end_idx - start_idx)

    @staticmethod
    def cmap_norm(start_idx=0, end_idx=len(clevs)):
        return (
            Precipitation.get_cmap(start_idx=start_idx, end_idx=end_idx),
            Precipitation.get_norm(start_idx=start_idx, end_idx=end_idx),
        )