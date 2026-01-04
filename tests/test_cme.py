import unittest

import jax.numpy as jnp
import jumpax as jx
from jax.scipy.special import gammaln


def poisson_pmf(k, lam):
    """Poisson probability mass function."""
    return jnp.exp(k * jnp.log(lam) - lam - gammaln(k + 1))


def binomial_pmf(k, n, p):
    """Binomial probability mass function."""
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return jnp.exp(log_comb + k * jnp.log(p) + (n - k) * jnp.log(1 - p))


def make_birth_death_problem(lam=5.0, mu=0.5):
    """Create a standard birth-death MassActionJump problem."""
    reactants = jnp.array([[0], [1]])
    net_stoich = jnp.array([[1], [-1]])
    rates = jnp.array([lam, mu])
    return jx.MassActionJump(reactants, net_stoich, rates=rates)


class TestGeneratorMatrix(unittest.TestCase):
    def setUp(self):
        self.problem = make_birth_death_problem()
        self.Q, self.cme_state = jx.build_generator(self.problem, jnp.array([20]))

    def test_columns_sum_to_zero(self):
        col_sums = jnp.sum(self.Q, axis=0)
        self.assertTrue(jnp.allclose(col_sums, 0.0, atol=1e-6))

    def test_off_diagonal_nonnegative(self):
        Q_offdiag = self.Q.at[jnp.diag_indices(self.Q.shape[0])].set(0.0)
        self.assertTrue(jnp.all(Q_offdiag >= 0))

    def test_diagonal_nonpositive(self):
        diagonal = jnp.diag(self.Q)
        self.assertTrue(jnp.all(diagonal <= 0))


class TestProbabilityConservation(unittest.TestCase):
    def setUp(self):
        self.problem = make_birth_death_problem()
        self.Q, self.cme_state = jx.build_generator(self.problem, jnp.array([30]))
        self.p0 = jnp.zeros(self.Q.shape[0]).at[0].set(1.0)

    def test_probability_sums_to_one(self):
        for t in [0.1, 0.5, 1.0, 5.0, 10.0]:
            p_t = jx.solve_cme(self.Q, self.p0, t)
            self.assertAlmostEqual(jnp.sum(p_t), 1.0, places=4)

    def test_probabilities_nonnegative(self):
        p0_mid = jnp.zeros(self.Q.shape[0]).at[5].set(1.0)
        for t in [0.1, 1.0, 5.0]:
            p_t = jx.solve_cme(self.Q, p0_mid, t)
            self.assertTrue(jnp.all(p_t >= -1e-10))


class TestBirthDeathSteadyState(unittest.TestCase):
    """Test birth-death process converges to Poisson steady state.

    For birth rate lam and death rate mu, the steady state is Poisson(lam/mu).
    """

    def test_steady_state_is_poisson(self):
        lam, mu = 5.0, 0.5
        expected_mean = lam / mu  # Poisson parameter = lam/mu

        problem = make_birth_death_problem(lam, mu)
        max_n = 40
        Q, _ = jx.build_generator(problem, jnp.array([max_n]))

        p0 = jnp.zeros(Q.shape[0]).at[0].set(1.0)
        p_ss = jx.solve_cme(Q, p0, t=50.0)

        k = jnp.arange(max_n + 1)
        poisson_expected = poisson_pmf(k, expected_mean)

        self.assertTrue(jnp.allclose(p_ss[:30], poisson_expected[:30], atol=1e-4))


class TestPureDeathProcess(unittest.TestCase):
    """Test pure death process matches binomial distribution.

    Starting with n0 particles, each survives independently with probability
    exp(-mu * t). The count at time t is Binomial(n0, exp(-mu * t)).
    """

    def test_death_process_binomial(self):
        mu, n0, t = 0.5, 20, 2.0
        survival_prob = jnp.exp(-mu * t)  # Binomial(n0, survival_prob)

        reactants = jnp.array([[1]])
        net_stoich = jnp.array([[-1]])
        problem = jx.MassActionJump(reactants, net_stoich, rates=jnp.array([mu]))

        Q, _ = jx.build_generator(problem, jnp.array([n0]))
        p0 = jnp.zeros(Q.shape[0]).at[n0].set(1.0)
        p_t = jx.solve_cme(Q, p0, t)

        k = jnp.arange(n0 + 1)
        binomial_expected = binomial_pmf(k, n0, survival_prob)

        self.assertTrue(jnp.allclose(p_t, binomial_expected, atol=1e-6))


class TestMarginalDistribution(unittest.TestCase):
    def setUp(self):
        reactants = jnp.array([[0, 0], [1, 0], [0, 1]])
        net_stoich = jnp.array([[1, 0], [-1, 1], [0, -1]])
        rates = jnp.array([2.0, 1.0, 0.5])
        self.problem = jx.MassActionJump(reactants, net_stoich, rates=rates)

    def test_marginal_sums_to_one(self):
        Q, cme_state = jx.build_generator(self.problem, jnp.array([10, 15]))
        p0 = jnp.zeros(Q.shape[0]).at[0].set(1.0)
        p_t = jx.solve_cme(Q, p0, t=5.0)

        marginal_0 = jx.marginal_distribution(p_t, cme_state, species=0)
        marginal_1 = jx.marginal_distribution(p_t, cme_state, species=1)

        self.assertAlmostEqual(jnp.sum(marginal_0), 1.0, places=5)
        self.assertAlmostEqual(jnp.sum(marginal_1), 1.0, places=5)

    def test_marginal_correct_length(self):
        max_counts = jnp.array([8, 12])
        Q, cme_state = jx.build_generator(self.problem, max_counts)
        p = jnp.ones(Q.shape[0]) / Q.shape[0]

        marginal_0 = jx.marginal_distribution(p, cme_state, species=0)
        marginal_1 = jx.marginal_distribution(p, cme_state, species=1)

        self.assertEqual(len(marginal_0), 9)
        self.assertEqual(len(marginal_1), 13)


class TestCMEState(unittest.TestCase):
    def test_state_space_size(self):
        reactants = jnp.array([[0, 0], [1, 0]])
        net_stoich = jnp.array([[1, 0], [-1, 1]])
        rates = jnp.array([1.0, 0.5])
        problem = jx.MassActionJump(reactants, net_stoich, rates=rates)

        max_counts = jnp.array([5, 7])
        Q, cme_state = jx.build_generator(problem, max_counts)

        expected_size = 6 * 8
        self.assertEqual(Q.shape[0], expected_size)
        self.assertEqual(cme_state.states.shape[0], expected_size)

    def test_states_in_bounds(self):
        problem = make_birth_death_problem()
        max_counts = jnp.array([15])
        Q, cme_state = jx.build_generator(problem, max_counts)

        self.assertTrue(jnp.all(cme_state.states >= 0))
        self.assertTrue(jnp.all(cme_state.states <= max_counts))


if __name__ == "__main__":
    unittest.main()
