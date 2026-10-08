"""Assisted namespace bootstrap; imported only for the assisted arm.

Preloading this factory before confinement keeps public numerical modules
available without mounting source. The blind arm never imports this factory.
"""

from stargazer.public_rv import Observations, PublicFitContext, PublicRV
from stargazer.tools import _create_worker_namespace


def create_assisted_namespace(public_data):
    namespace = _create_worker_namespace(public_data)
    context = PublicFitContext(
        observations=Observations(
            **{
                key: tuple(public_data[key])
                for key in ("times_days", "rvs_ms", "sigmas_ms", "instruments")
            }
        ),
        star_mass_sun=float(public_data["star_mass_sun"]),
        maximum_rms_factor=public_data.get("maximum_rms_factor", 1.5),
        los_axis=public_data.get("los_axis", "x"),
        integrator_preference=public_data.get("integrator_preference", "whfast"),
    )
    helper = PublicRV(context)
    namespace.update(
        stargazer_predict=helper.predict,
        stargazer_diagnostics=helper.diagnostics,
        STARGAZER_PUBLIC_RESOURCES=dict(public_data.get("public_resources", {})),
    )
    return namespace
