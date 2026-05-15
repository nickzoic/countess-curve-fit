"""CountESS VAMPseq2 Plugin"""
import logging
from typing import Iterable, Optional, Any, Dict

from countess.core.plugins import DuckdbTransformPlugin
from countess.core.parameters import (
    BooleanParam,
    FloatParam,
    PerColumnArrayParam,
    PerNumericColumnArrayParam,
    TabularMultiParam,
    ColumnGroupChoiceParam,
    ColumnGroupOrNoneChoiceParam,
)
from scipy.stats import norm
from scipy.optimize import curve_fit

VERSION = "0.0.1"

logger = logging.getLogger(__name__)

def func(x, mu, sigma):
    return norm.cdf(x+1, loc=mu, scale=sigma) - norm.cdf(x, loc=mu, scale=sigma)

def fit(counts: list[float], sigmas: Optional[list[float]] = None) -> tuple[float, float, float]:
    total = sum(counts)
    popt, pcov, *_ = curve_fit(
        func,
        xdata = list(range(0, len(counts))),
        ydata = [ c / total for c in counts ],
        sigma = [ s / total for s in sigmas ] if sigmas else None,
    )
    return popt[0], popt[1], pcov[0][0]

class VampSeq2Plugin(DuckdbTransformPlugin):
    """Implement enhanced VAMP-seq scoring"""

    name = "VAMP-seq Scoring (Enhanced)"
    description = "VAMP-seq Scoring (Enhanced)"
    version = VERSION

    columns = ColumnGroupChoiceParam("Frequency Columns")
    sigmas = ColumnGroupOrNoneChoiceParam("Sigma Columns")

    def add_fields(self):
        return {
            "mu": float,
            "sigma": float,
            "var_mu": float,
        }

    def transform(self, data: dict[str, Any]) -> Optional[Dict[str, Any]]:
        logger.debug(data)
        prefix = self.columns.get_column_prefix()
        counts = [ data[x] for x in data if x.startswith(prefix) ]
        sigmas = [ data[x] for x in data if x.startswith(self.sigmas.get_column_prefix()) ] if self.sigmas.is_not_none() else None
        data["mu"], data["sigma"], data["var_mu"] = fit(counts)
        return data
