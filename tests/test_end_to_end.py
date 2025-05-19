import os
import numpy as np
from unittest.mock import patch

from zener.generate import generate_all_testing_data
from zener.optimize import GAAll


def test_generate_and_optimize_variable_tp(tmpdir):
    os.chdir(tmpdir)

    Tp = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
    Rsig = [1, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1]
    test_label = 'test_label'

    with (patch('zener.generate.parse_signal_file') as mock_parse_signal_file_gen,
         patch('zener.optimize.parse_signal_file') as mock_parse_signal_file_opt):
        mock_parse_signal_file_gen.return_value = \
            mock_parse_signal_file_opt.return_value = {test_label: Rsig}

        generate_all_testing_data(Tp=Tp)

        ga_all = GAAll()
        subj = ga_all.subjects[test_label]
        ga_all.ga.num_generations = 1
        ga_all.run()
        ga_all.save_result()

        solution, fitness, _ = ga_all.ga.best_solution()
        assert solution is not None
        assert isinstance(fitness, (float, np.floating, np.ndarray))

        merged = subj.compute_and_merge(solution)

        # Check that the last times and total sum of Tp match
        last_time_merged = merged['time'].iloc[-1]
        last_time_target = subj.df_target_sorted['time'].iloc[-1]
        assert np.isclose(sum(subj.Tp), last_time_target, rtol=0.05)
        assert np.isclose(last_time_merged, last_time_target, rtol=0.05)

        # Check that the number of rows is similar (allowing for a small difference)
        n_merged = len(merged)
        n_target = len(subj.df_target_sorted)
        assert abs(n_merged - n_target) < max(2, 0.05 * n_target)
