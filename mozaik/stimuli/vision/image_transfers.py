"""Built-in image transfers for Experanto stimuli."""

from numbers import Real

import numpy as np
import param
from imagen.transferfn import MaximumDynamicRange


DEFAULT_IMAGE_TRANSFER = "MaximumDynamicRange"
DEFAULT_CONTRAST_WINDOW = 0.2


class CenteredSymmetricContrast(param.Parameterized):
    """Center a rendered luminance image at the background luminance B.

    The largest absolute deviation is window * B. Apply this transfer after
    spatial sampling and luminance scaling, as in the original centered_c020.
    """

    window = param.Number(default=DEFAULT_CONTRAST_WINDOW, bounds=(0, 1))

    def __call__(self, image, background_luminance):
        if not np.isfinite(image).all():
            raise ValueError("CenteredSymmetricContrast requires finite luminance")
        centered = image - image.mean()
        amplitude = np.max(np.abs(centered))
        z = centered / amplitude if amplitude > 0 else np.zeros_like(image)
        return background_luminance * (1 + self.window * z)


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
