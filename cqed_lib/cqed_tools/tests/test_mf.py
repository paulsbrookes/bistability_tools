from numpy.testing import assert_, assert_equal, run_module_suite
from cqed_tools.mf.hamiltonian import *
import pandas as pd


def test_collapse_operators_mf():

    params = pd.Series([4, 1.0, 2.0, 3.0], index=['t_levels', 'gamma', 'gamma_phi', 'n_t'], dtype=object)
    c_ops = collapse_operators_mf(params)
    assert_equal(c_ops[0], destroy(params.t_levels) * np.sqrt(params.gamma * (params.n_t + 1)))
    assert_equal(c_ops[1], create(params.t_levels) * np.sqrt(params.gamma * params.n_t))
    assert_equal(len(c_ops), 3)

    params = pd.Series([14, 0.0, 2.5, 3.7], index=['t_levels', 'gamma', 'gamma_phi', 'n_t'], dtype=object)
    c_ops = collapse_operators_mf(params)
    assert_equal(len(c_ops), 1)

    params = pd.Series([8, 0.1, 0.0, 5.6], index=['t_levels', 'gamma', 'gamma_phi', 'n_t'], dtype=object)
    c_ops = collapse_operators_mf(params)
    assert_equal(c_ops[0], destroy(params.t_levels) * np.sqrt(params.gamma * (params.n_t + 1)))
    assert_equal(c_ops[1], create(params.t_levels) * np.sqrt(params.gamma * params.n_t))
    assert_equal(len(c_ops), 2)


if __name__ == "__main__":
    run_module_suite()