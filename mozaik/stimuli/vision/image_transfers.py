"""Built-in image transfers for the Experanto pattern sampler."""

from numbers import Real

import numpy as np
import param
from imagen.transferfn import MaximumDynamicRange, TransferFn


DEFAULT_IMAGE_TRANSFER = "MaximumDynamicRange"
DEFAULT_CONTRAST_WINDOW = 0.2


class CenteredSymmetricContrast(TransferFn):
    """Map the image mean to 0.5 and the largest deviation to window / 2.

    Mozaik scales the result by twice the background luminance B, giving
    a mean of B and a maximum absolute luminance deviation of window * B.
    A constant image maps to 0.5.
    """

    window = param.Number(default=DEFAULT_CONTRAST_WINDOW, bounds=(0, 1))

    def __call__(self, pixels):
        pixels -= pixels.mean()
        amplitude = np.max(np.abs(pixels))
        if not np.isfinite(amplitude):
            raise ValueError("CenteredSymmetricContrast requires finite pixels")
        if pixels.min() == pixels.max():
            pixels.fill(0.5)
        else:
            pixels /= amplitude
            pixels *= self.window
            pixels += 1.0
            pixels *= 0.5


IMAGE_TRANSFERS = {
    DEFAULT_IMAGE_TRANSFER: MaximumDynamicRange,
    "CenteredSymmetricContrast": CenteredSymmetricContrast,
}


def resolve_image_transfer(name, window):
    """Validate a registered name and return its effective, serializable window."""
    if not isinstance(name, str) or name not in IMAGE_TRANSFERS:
        choices = ", ".join(IMAGE_TRANSFERS)
        raise ValueError(f"Unknown image transfer {name!r}. Choose from {choices}")
    if IMAGE_TRANSFERS[name] is CenteredSymmetricContrast:
        if window is None:
            window = DEFAULT_CONTRAST_WINDOW
        if (
            isinstance(window, bool)
            or not isinstance(window, Real)
            or not 0 <= window <= 1
        ):
            raise ValueError("CenteredSymmetricContrast window must be finite and in [0, 1]")
        return float(window)
    if window is not None:
        raise ValueError(f"{name} does not accept an image transfer window")
    return None
