"""pysift."""

import pkg_resources

from pysift.pysift import computeKeypointsAndDescriptors

version = pkg_resources.get_distribution(__package__).version
