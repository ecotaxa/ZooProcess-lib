
import unittest
import pytest

from zooprocess2 import zooprocessv10

class test_zooprocess2(unittest.TestCase):

    def test_process(self):

        z = zooprocessv10()

        z.process()


