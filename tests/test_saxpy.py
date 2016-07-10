from saxpy import SAX
from numpy import allclose

class TestSAX(object):
    def setUp(self):
        # All tests will be run with 6 letter words
        # and 5 letter alphabet
        self.sax = SAX(6, 5, 1e-6)

    def test_to_letter_rep(self):
        arr = [7,1,4,4,4,4]
        (letters, indices, letter_boundries) = self.sax.to_letter_rep(arr)
        assert letters == 'eacccc'

    def test_long_to_letter_rep(self):
        long_arr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,10,100]
        (letters, indices, letter_boundries) = self.sax.to_letter_rep(long_arr)
        assert letters == 'bbbbce'

    def test_compare_strings(self):
        base_string = 'aaabbc'
        similar_string = 'aabbbc'
        dissimilar_string = 'ccddbc'
        similar_score = self.sax.compare_strings(base_string, similar_string)
        dissimilar_score = self.sax.compare_strings(base_string, dissimilar_string)
        assert similar_score < dissimilar_score

    def test_from_letter_rep(self):
        arr = [7,1,4,4,4,4]
        (letters, indices, letter_boundries) = self.sax.to_letter_rep(arr)
        reconstructed = self.sax.from_letter_rep(letters, indices, letter_boundries)
        assert allclose(reconstructed, [5.45, 2.54, 4.0, 4.0, 4.0, 4.0], atol=0.01)

    def test_breakpoints(self):
        assert allclose(self.sax.breakpoints(3), [-0.43, 0.43], atol=0.01)
        assert allclose(self.sax.breakpoints(2), [0], atol=0.01)
        assert allclose(self.sax.breakpoints(20), [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64], atol=0.01)