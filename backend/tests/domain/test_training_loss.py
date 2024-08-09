import unittest
from domain.training_loss import TrainingLoss


class TestTrainingLoss(unittest.TestCase):

    def test_calculate_loss_rank_zero(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(0), 0.0)

    def test_calculate_loss_rank_below_first_rank(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.1), 0.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.4), 0.0)

    def test_calculate_loss_rank_first_rank(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.5), 0.5)

    def test_calculate_loss_rank_second_rank(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.7), 0.5)
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.9), 0.5)

    def test_calculate_loss_rank_third_rank(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(1.0), 1.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(1.4), 1.0)

    def test_calculate_loss_rank_mid_range(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(2.3), 2.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(3.7), 3.5)

    def test_calculate_loss_rank_upper_boundary(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(5.0), 5.0)

    def test_calculate_loss_rank_above_upper_boundary(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(5.1), 5.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(10.0), 10.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(100.0), 10.0)

    def test_calculate_loss_rank_negative(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(-1.0), 0.0)
        self.assertEqual(TrainingLoss.calculate_loss_rank(-0.5), 0.0)

    def test_calculate_loss_rank_float_precision(self):
        self.assertEqual(TrainingLoss.calculate_loss_rank(0.99999), 0.5)
        self.assertEqual(TrainingLoss.calculate_loss_rank(1.00001), 1.0)

    def test_calculate_loss_rank_exact_boundaries(self):
        for i in range(21):
            expected = min(i * 0.5, 10.0)
            self.assertEqual(TrainingLoss.calculate_loss_rank(i * 0.5), expected)


if __name__ == "__main__":
    unittest.main()
