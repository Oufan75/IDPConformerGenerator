"""Bond geomtry strategies."""
import argparse

from idpconfgen.components.bgeo_strategies.int2cart.bgeo_int2cart import (
    has_int2cart,
    )
from idpconfgen.components.bgeo_strategies.int2cart.bgeo_int2cart import \
    name as bgeo_int2cart_name
from idpconfgen.components.bgeo_strategies.fixed import \
    name as bgeo_fixed_name
from idpconfgen.components.bgeo_strategies.sampling import \
    name as bgeo_sampling_name


bgeo_strategies_default = bgeo_sampling_name
"""The default bond geometry sampling strategy."""

bgeo_strategies = (
    bgeo_fixed_name,
    bgeo_sampling_name,
    bgeo_int2cart_name,
    )
"""Available bond geometry sampling strategies."""

_bgeo_arg_name = '--bgeo-strategy'


def add_bgeo_strategy_arg(parse):
    """Add the bond geometry choice argument to client."""
    parse.add_argument(
        _bgeo_arg_name,
        dest='bgeo_strategy',
        help="Which strategy to use for bond geometries. Defaults to `sampling`.",
        choices=bgeo_strategies,
        default=bgeo_strategies_default,
        action=CheckBgeoInstallation,
        )

    parse.add_argument(
        '-bgeo_path',
        '--bgeo_path',
        help=(
            'Path to the bond geometry database as generated by `bgeo` CLI. '
            'Defaults to using the internal library. '
            f'This option works with \'{_bgeo_arg_name} {bgeo_sampling_name}\' '
            f'and \'{_bgeo_arg_name} {bgeo_int2cart_name}\'.'
            ),
        default=None,
        )


class CheckBgeoInstallation(argparse.Action):
    """Controls if input is folder, files or tar."""

    def __call__(self, parser, namespace, value, option_string=None):
        """Hello."""
        if value == bgeo_int2cart_name:
            if not has_int2cart:
                emsg = (
                    f"Please install Int2Cart software to access '{_bgeo_arg_name} {bgeo_int2cart_name}'. "  # noqa: E501
                    "See the INSTALL page of IDPConfGen documentation."
                    )
                raise parser.error(emsg)

        setattr(namespace, self.dest, value)


bgeo_error_msg = (
    f'{_bgeo_arg_name} valid options are "{", ".join(bgeo_strategies)}": '
    '{!r} received.'
    )
