"""Test that the README example runs correctly."""

import unittest

import jax.numpy as jnp
import jax.random as jr
import jumpax as jpx


class TestReadmeExample(unittest.TestCase):
    def test_readme_example(self):
        """Test the birth-death process example runs and produces valid output."""
        reactants = jnp.array([[0], [1]])
        net_stoich = jnp.array([[1], [-1]])
        rates = jnp.array([10.0, 0.1])

        jumps = jpx.MassActionJump(reactants, net_stoich, rates=rates)
        solver = jpx.SSA()
        save = jpx.Save(states=True)

        u0 = jnp.array([50.0])
        key = jr.key(0)

        sol = jpx.solve(jumps, solver, save, u0, t0=0.0, t1=1.0, key=key)

        self.assertIsNotNone(sol.ts)
        self.assertIsNotNone(sol.us)
        self.assertEqual(sol.us.shape[1], 1)

        mask = jnp.isfinite(sol.ts)
        ts = sol.ts[mask]
        us = sol.us[mask]

        self.assertGreater(len(ts), 0)
        self.assertTrue(jnp.isfinite(us).all())
        self.assertGreaterEqual(us[-1, 0], 0)


if __name__ == "__main__":
    unittest.main()
