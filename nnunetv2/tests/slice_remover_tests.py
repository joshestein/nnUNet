import numpy as np
import unittest

from nnunetv2.training.dataloading.slice_remover import SliceRemover


class TestSliceRemover(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.random.rand(1, 9, 1, 1)
        self.seg = np.random.rand(1, 9, 1, 1)
        self.non_even_data = np.random.rand(1, 10, 1, 1)
        self.non_even_seg = np.random.rand(1, 10, 1, 1)

    def test_invalid_areas(self):
        self.assertRaises(AssertionError, SliceRemover, sample_regions=["invalid"])

    def test_maintain_shape(self):
        slicer = SliceRemover(percentage_slices=0.2, maintain_shape=True)
        sliced_data, sliced_seg = slicer.remove_slices(self.data, self.seg)
        self.assertEqual(sliced_data.shape, self.data.shape)
        self.assertEqual(sliced_seg.shape, self.seg.shape)

    def test_no_maintain_shape(self):
        slicer = SliceRemover(percentage_slices=0.2, maintain_shape=False)
        sliced_data, sliced_seg = slicer.remove_slices(self.data, self.seg)
        self.assertNotEqual(sliced_data.shape, self.data.shape)
        self.assertNotEqual(sliced_seg.shape, self.seg.shape)
        self.assertEqual(sliced_data.shape[1], int(self.data.shape[1] * slicer.percentage_slices))
        self.assertEqual(sliced_seg.shape[1], int(self.seg.shape[1] * slicer.percentage_slices))
        self.assertEqual(np.count_nonzero(sliced_data == 0), 0)
        self.assertEqual(np.count_nonzero(sliced_seg == 0), 0)

    def test_percentage_one_is_full_dataset(self):
        slicer = SliceRemover()
        self.assertEqual(slicer.percentage_slices, 1.0)

        sliced_data, sliced_seg = slicer.remove_slices(self.data, self.seg)
        assert np.array_equal(self.data, sliced_data)
        assert np.array_equal(self.seg, sliced_seg)
        self.assertEqual(np.count_nonzero(sliced_data == 0), 0)
        self.assertEqual(np.count_nonzero(sliced_seg == 0), 0)

    def test_base_samples_first_third(self):
        slicer = SliceRemover(sample_regions=["base"])

        slice_data, slice_seg = slicer.remove_slices(self.data, self.seg)
        self.assertEqual(np.count_nonzero(slice_data[:, :3] == 0), 0)  # There are no zeros in first 3 elements
        self.assertEqual(np.count_nonzero(slice_seg[:, :3] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_data[:, 3:]), 0)  # Everything after 3rd element is zero
        self.assertEqual(np.count_nonzero(slice_seg[:, 3:]), 0)

        slice_non_even_data, slice_non_even_seg = slicer.remove_slices(self.non_even_data, self.non_even_seg)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, :4] == 0), 0)  # There are no zeros in first 4 elements
        self.assertEqual(np.count_nonzero(slice_non_even_seg[:, :4] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 4:]), 0)  # Everything after 4th element is zero
        self.assertEqual(np.count_nonzero(slice_non_even_seg[:, 4:]), 0)

    def test_mid_samples_second_third(self):
        slicer = SliceRemover(sample_regions=["mid"])

        slice_data, _ = slicer.remove_slices(self.data, self.seg)
        self.assertEqual(np.count_nonzero(slice_data[:, 0:3]), 0)
        self.assertEqual(np.count_nonzero(slice_data[:, 3:6] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_data[:, 6:]), 0)

        slice_non_even_data, _ = slicer.remove_slices(self.non_even_data, self.non_even_seg)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 0:4]), 0)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 4:7] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 7:]), 0)

    def test_apex_samples_last_third(self):
        slicer = SliceRemover(sample_regions=["apex"])

        slice_data, _ = slicer.remove_slices(self.data, self.seg)
        self.assertEqual(np.count_nonzero(slice_data[:, 0:6]), 0)
        self.assertEqual(np.count_nonzero(slice_data[:, 6:] == 0), 0)

        slice_non_even_data, _ = slicer.remove_slices(self.non_even_data, self.non_even_seg)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 0:7]), 0)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 7:] == 0), 0)

    def test_combined_base_and_mid(self):
        slicer = SliceRemover(sample_regions=["base", "mid"])

        slice_data, _ = slicer.remove_slices(self.data, self.seg)
        self.assertEqual(np.count_nonzero(slice_data[:, 0:6] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_data[:, 6:]), 0)

        slice_non_even_data, _ = slicer.remove_slices(self.non_even_data, self.non_even_seg)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 0:7] == 0), 0)
        self.assertEqual(np.count_nonzero(slice_non_even_data[:, 7:]), 0)

    def test_reduced_slices(self):
        slicer = SliceRemover(percentage_slices=0.5)
        data = np.random.rand(1, 100, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data), 50)

        slicer = SliceRemover(percentage_slices=0.2)
        data = np.random.rand(1, 100, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data), 20)

        slicer = SliceRemover(percentage_slices=0.8)
        data = np.random.rand(1, 100, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data), 80)

    def test_reduced_slices_with_constrained_areas(self):
        slicer = SliceRemover(percentage_slices=0.1, sample_regions=["base"])
        data = np.random.rand(1, 99, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data[:, 0:33]), 9)
        self.assertEqual(np.count_nonzero(sliced_data[:, 33:]), 0)

    def test_non_random_slices_full_set(self):
        slicer = SliceRemover(randomise_slices=False, sample_regions=["base", "mid", "apex"])
        data = np.random.rand(1, 99, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data == 0), 0)

    def test_non_random_slices_reduced_set(self):
        slicer = SliceRemover(randomise_slices=False, sample_regions=["base"])
        data = np.random.rand(1, 99, 1, 1)
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data[:, 0:33] == 0), 0)
        self.assertEqual(np.count_nonzero(sliced_data[:, 33:]), 0)

    def test_non_random_slices_reduced_percentage(self):
        slicer = SliceRemover(percentage_slices=0.5, randomise_slices=False)
        data = np.random.rand(1, 99, 1, 1)

        # Total slices to sample = 99 * 0.5 = 49.5 ~ 49
        # 49 / 3 = 16.33 ~ 16 slices per region
        # So we actually sample 16 * 3 = 48 slices
        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data), 48)

        # Base
        self.assertEqual(np.count_nonzero(sliced_data[:, 0:33]), 16)
        self.assertEqual(np.count_nonzero(sliced_data[:, 8:24] == 0), 0)

        # Mid
        self.assertEqual(np.count_nonzero(sliced_data[:, 33:66]), 16)
        self.assertEqual(np.count_nonzero(sliced_data[:, 41:57] == 0), 0)

        # Apex
        self.assertEqual(np.count_nonzero(sliced_data[:, 66:]), 16)
        self.assertEqual(np.count_nonzero(sliced_data[:, 74:90] == 0), 0)

    def test_non_random_slices_reduced_percentage_with_constrained_area(self):
        slicer = SliceRemover(percentage_slices=0.1, randomise_slices=False, sample_regions=["base"])
        data = np.random.rand(1, 99, 1, 1)

        sliced_data, _ = slicer.remove_slices(data, data)
        self.assertEqual(np.count_nonzero(sliced_data[:, 12:20] == 0), 0)
        self.assertEqual(np.count_nonzero(sliced_data), 8)


if __name__ == "__main__":
    unittest.main()
