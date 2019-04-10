import unittest

from lisa import rl


class TestTranslator(unittest.TestCase):

    def setUp(self):
        self.translator = rl.Translator(values=("a", "c", "b"))

    def test_encode_valid(self):
        self.assertEqual(self.translator.encode("a"), 0)
        self.assertEqual(self.translator.encode("c"), 1)
        self.assertEqual(self.translator.encode("b"), 2)

    def test_encode_invalid(self):
        with self.assertRaises(ValueError):
            self.translator.encode("d")

    def test_decode_valid(self):
        self.assertEqual(self.translator.decode(0), "a")
        self.assertEqual(self.translator.decode(1), "c")
        self.assertEqual(self.translator.decode(2), "b")

    def test_decode_invalid(self):
        with self.assertRaises(ValueError):
            self.translator.decode(3)
