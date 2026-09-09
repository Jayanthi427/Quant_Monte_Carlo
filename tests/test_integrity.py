import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import quant_engine

class TestIntegrity(unittest.TestCase):
    def test_engine_load(self):
        self.assertIsNotNone(quant_engine)

if __name__ == '__main__':
    unittest.main()
