import numpy as np
from NSFopen.read import read
from pymoo.core.problem import ElementwiseProblem
from scipy.optimize import curve_fit
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from afm_runtime import runtime


def scan_image(PGain, IGain, DGain, file):
    latest_file = runtime().call(
        "update_and_acquire",
        {
            "pgain": PGain,
            "igain": IGain,
            "dgain": DGain,
        },
        file,
    )
    afm = read(latest_file)
    data = afm.data
    im_file_fw = data["Image"]["Forward"]["Z-Axis"]
    im_file_bw = data["Image"]["Backward"]["Z-Axis"]
    similarity_index, _diff = ssim(
        im_file_bw,
        im_file_fw,
        full=True,
        data_range=im_file_bw.max() - im_file_bw.min(),
    )
    mse = mean_squared_error(im_file_bw, im_file_fw)
    return similarity_index, mse


def corrected_image(image):
    def poly5d(xy, *params):
        x, y = xy
        return (
            params[0]
            + params[1] * x
            + params[2] * y
            + params[3] * x**2
            + params[4] * y**2
            + params[5] * x * y
            + params[6] * x**3
            + params[7] * y**3
            + params[8] * x**2 * y
            + params[9] * x * y**2
            + params[10] * x**4
            + params[11] * y**4
            + params[12] * x**3 * y
            + params[13] * x * y**3
            + params[14] * x**2 * y**2
        )

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    image_flat = image.flatten()
    params, _ = curve_fit(poly5d, (x, y), image_flat, p0=np.zeros(15))
    baseline = poly5d((x, y), *params).reshape(image.shape)
    return image - baseline


def scan_image_poly(PGain, IGain, DGain, file):
    latest_file = runtime().call(
        "update_and_acquire",
        {
            "pgain": PGain,
            "igain": IGain,
            "dgain": DGain,
        },
        file,
    )
    afm = read(latest_file)
    data = afm.data
    im_file_fw = corrected_image(data["Image"]["Forward"]["Z-Axis"])
    im_file_bw = corrected_image(data["Image"]["Backward"]["Z-Axis"])
    similarity_index, _diff = ssim(
        im_file_bw,
        im_file_fw,
        full=True,
        data_range=im_file_bw.max() - im_file_bw.min(),
    )
    mse = mean_squared_error(im_file_bw, im_file_fw)
    return similarity_index, mse


class MyProblem(ElementwiseProblem):
    def __init__(self, baseline=True):
        super().__init__(
            n_var=3, n_obj=1, xl=np.array([0, 500, 0]), xu=np.array([500, 9000, 100])
        )
        self.baseline = baseline

    def _evaluate(self, x, out, *args, **kwargs):  # noqa: ARG002
        if self.baseline:
            scan_outputs = scan_image_poly(
                x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_"
            )
        else:
            scan_outputs = scan_image(x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_")

        ssim = scan_outputs[0]
        f1 = (1 - ssim) * 10000
        out["F"] = [f1]
