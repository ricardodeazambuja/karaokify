"""Allow `python -m karaokify`."""

import sys

from .cli import main

sys.exit(main())
