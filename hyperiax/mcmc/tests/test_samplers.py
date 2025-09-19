import pytest
import jax.numpy as jnp

from hyperiax.mcmc.samplers import (
    PCNNoiseSampler,
    MetropolisHastingsParameterSampler,
    GibbsParameterSampler,
    AlternatingProposalSampler,
    MetropolisHastingsSampler,
    create_mh_sampler,
    create_gibbs_sampler,
)


class TestPCNNoiseSampler:
    """Test the Preconditioned Crank-Nicolson noise sampler."""

    def test_init(self):
        """Test PCN sampler initialization."""
        eta = 0.9
        sampler = PCNNoiseSampler(eta)
        assert sampler.eta == eta

    def test_propose_noise_shape(self, rng_key, noise):
        """Test that proposed noise has correct shape."""
        sampler = PCNNoiseSampler(eta=0.8)
        new_noise = sampler.propose_noise(rng_key, noise)

        assert new_noise.shape == noise.shape

    def test_propose_noise_eta_limits(self, rng_key, noise):
        """Test PCN behavior at eta limits."""
        # When eta = 0, should be pure random noise
        sampler_eta0 = PCNNoiseSampler(eta=0.0)
        new_noise_0 = sampler_eta0.propose_noise(rng_key, noise)

        # When eta = 1, should be unchanged
        sampler_eta1 = PCNNoiseSampler(eta=1.0)
        new_noise_1 = sampler_eta1.propose_noise(rng_key, noise)

        # Check that eta=1 keeps noise unchanged
        assert jnp.allclose(new_noise_1, noise)

        # Check that eta=0 gives different result
        assert not jnp.allclose(new_noise_0, noise)


class TestMetropolisHastingsParameterSampler:
    """Test the Metropolis-Hastings parameter sampler."""

    def test_init(self):
        """Test MH parameter sampler initialization."""
        sampler = MetropolisHastingsParameterSampler()
        assert sampler is not None

    def test_propose_params(self, rng_key, params, data):
        """Test parameter proposal."""
        sampler = MetropolisHastingsParameterSampler()

        new_params, log_correction = sampler.propose_params(rng_key, params, data)

        assert type(new_params) is type(params)
        assert not jnp.allclose(
            new_params["dummy"].value, params["dummy"].value
        )  # Params should change
        assert not jnp.allclose(
            new_params["obs_var"].value, params["obs_var"].value
        )  # Params should change


class TestGibbsParameterSampler:
    """Test the Gibbs parameter sampler."""

    def test_init(self, tree):
        """Test Gibbs sampler initialization."""
        sampler = GibbsParameterSampler(tree)
        assert sampler.tree == tree
        assert not sampler.update_obs_var

    def test_propose_params_params_update(self, rng_key, tree, params, data):
        """Test parameter proposal when not updating obs_var."""
        sampler = GibbsParameterSampler(tree)
        sampler.update_obs_var = False  # Ensure we're not updating obs_var

        new_params, log_correction = sampler.propose_params(rng_key, params, data)
        assert type(new_params) is type(params)
        assert not jnp.allclose(new_params["dummy"].value, params["dummy"].value)
        assert jnp.allclose(new_params["obs_var"].value, params["obs_var"].value)
        assert not sampler.update_obs_var  # Flag remains until outer sampler toggles

    def test_propose_params_obs_var_update(self, rng_key, tree, params, data):
        """Test parameter proposal when updating obs_var."""
        sampler = GibbsParameterSampler(tree)
        sampler.update_obs_var = True  # Set to update obs_var

        new_params, log_correction = sampler.propose_params(rng_key, params, data)

        assert type(new_params) is type(params)
        assert jnp.allclose(new_params["dummy"].value, params["dummy"].value)
        assert not jnp.allclose(new_params["obs_var"].value, params["obs_var"].value)
        assert log_correction == 0.0  # Gibbs should have no correction
        assert sampler.update_obs_var  # Flag remains until outer sampler toggles

    def test_compute_residuals(self, tree, data):
        """Test residual computation."""
        sampler = GibbsParameterSampler(tree)
        residuals = sampler._compute_residuals(data)

        expected_residuals = tree.data["value"][tree.is_leaf] - data
        assert jnp.allclose(residuals, expected_residuals)

    def test_sample_obs_var_posterior(self, rng_key, tree, params):
        """Test obs_var posterior sampling."""
        sampler = GibbsParameterSampler(tree)
        residuals = jnp.zeros_like(tree.data["value"][tree.is_leaf])

        obs_var_param = params["obs_var"]

        new_obs_var = sampler._sample_obs_var_posterior(
            rng_key, obs_var_param, residuals
        )

        assert type(new_obs_var) is type(obs_var_param)
        assert not jnp.allclose(new_obs_var.value, obs_var_param.value)


class TestAlternatingProposalSampler:
    """Test the alternating proposal sampler."""

    @pytest.fixture
    def param_sampler(self):
        """Fixture for parameter sampler."""
        return MetropolisHastingsParameterSampler()

    @pytest.fixture
    def noise_sampler(self):
        """Fixture for noise sampler."""
        return PCNNoiseSampler(eta=0.9)

    def test_init(self, tree, param_sampler, noise_sampler):
        """Test alternating sampler initialization."""
        sampler = AlternatingProposalSampler(tree, param_sampler, noise_sampler)

        assert sampler.param_sampler == param_sampler
        assert sampler.noise_sampler == noise_sampler
        assert sampler.tree == tree
        assert sampler.update_params

    def test_propose_state_params_update(
        self,
        rng_key,
        tree,
        param_sampler,
        noise_sampler,
        params,
        noise,
        data,
    ):
        """Test state proposal when updating parameters."""
        sampler = AlternatingProposalSampler(tree, param_sampler, noise_sampler)
        sampler.update_params = True

        current_state = (params, noise)

        new_state, log_correction = sampler.propose_state(rng_key, current_state, data)

        assert not sampler.update_params  # Should flip
        assert not jnp.allclose(
            new_state[0]["dummy"].value, params["dummy"].value
        )  # Params changed
        assert jnp.allclose(new_state[1], noise)  # Noise unchanged

    def test_propose_state_noise_update(
        self,
        rng_key,
        tree,
        param_sampler,
        noise_sampler,
        params,
        noise,
        data,
    ):
        """Test state proposal when updating noise."""
        sampler = AlternatingProposalSampler(tree, param_sampler, noise_sampler)
        sampler.update_params = False

        current_state = (params, noise)

        new_state, log_correction = sampler.propose_state(rng_key, current_state, data)

        assert sampler.update_params  # Should flip
        assert jnp.allclose(
            new_state[0]["dummy"].value, params["dummy"].value
        )  # Params unchanged
        assert not jnp.allclose(new_state[1], noise)  # Noise changed


class TestMetropolisHastingsSampler:
    """Test the main Metropolis-Hastings sampler."""

    @pytest.fixture
    def proposal_sampler(self, tree):
        """Fixture for a proposal sampler."""
        param_sampler = MetropolisHastingsParameterSampler()
        noise_sampler = PCNNoiseSampler(eta=0.9)
        return AlternatingProposalSampler(tree, param_sampler, noise_sampler)

    def test_init(self, proposal_sampler):
        """Test MH sampler initialization."""
        sampler = MetropolisHastingsSampler(proposal_sampler)

        assert sampler.proposal_sampler == proposal_sampler
        assert sampler.is_alternating_sampling
        assert sampler.accepted_count == 0

    def test_reset_counts(self, proposal_sampler):
        """Test count reset functionality."""
        sampler = MetropolisHastingsSampler(proposal_sampler)

        # Set some counts
        sampler.accepted_count = 10
        sampler.accepted_params_count = 5
        sampler.accepted_noise_count = 3

        sampler.reset_counts()

        assert sampler.accepted_count == 0
        assert sampler.accepted_params_count == 0
        assert sampler.accepted_noise_count == 0

    def test_is_updating_params(self, proposal_sampler):
        """Test parameter update detection."""
        sampler = MetropolisHastingsSampler(proposal_sampler)

        # Test with alternating sampler
        proposal_sampler.update_params = True
        assert sampler._is_updating_params()

        proposal_sampler.update_params = False
        assert not sampler._is_updating_params()

    def test_is_gibbs_update(self, tree):
        """Test Gibbs update detection."""
        # Create Gibbs proposal sampler
        gibbs_param_sampler = GibbsParameterSampler(tree)
        noise_sampler = PCNNoiseSampler(eta=0.9)
        gibbs_proposal_sampler = AlternatingProposalSampler(
            tree, gibbs_param_sampler, noise_sampler
        )

        sampler = MetropolisHastingsSampler(gibbs_proposal_sampler)

        # Test when updating params and obs_var
        gibbs_proposal_sampler.update_params = True
        gibbs_param_sampler.update_obs_var = True
        assert sampler._is_gibbs_update()

        # Test when not updating obs_var
        gibbs_param_sampler.update_obs_var = False
        assert not sampler._is_gibbs_update()

        # Test when not updating params
        gibbs_proposal_sampler.update_params = False
        assert not sampler._is_gibbs_update()

    def test_sample_basic(self, rng_key, proposal_sampler, params, noise, data):
        """Test basic sampling functionality."""
        sampler = MetropolisHastingsSampler(proposal_sampler)

        # Mock log posterior and likelihood functions
        def mock_log_posterior(state, data):
            return 0.0

        def mock_log_likelihood(data, state):
            return 0.0

        init_state = (params, noise)

        log_likes, samples = sampler.sample(
            rng_key=rng_key,
            log_posterior=mock_log_posterior,
            log_likelihood=mock_log_likelihood,
            data=data,
            init_state=init_state,
            num_samples=5,
            num_burn_in=2,
            thinning=1,
        )

        assert len(log_likes) == 5  # num_samples
        assert len(samples) == 5

    def test_gibbs_flag_toggles_after_iteration(
        self, rng_key, tree, params, noise, data
    ):
        """Ensure the Gibbs sampler defers flag toggling to the outer loop."""
        gibbs_param_sampler = GibbsParameterSampler(tree)
        noise_sampler = PCNNoiseSampler(eta=0.9)
        gibbs_proposal_sampler = AlternatingProposalSampler(
            tree, gibbs_param_sampler, noise_sampler
        )

        sampler = MetropolisHastingsSampler(gibbs_proposal_sampler)

        def mock_log_posterior(state, data):
            return 0.0

        def mock_log_likelihood(data, state):
            return 0.0

        init_state = (params, noise)

        assert not gibbs_param_sampler.update_obs_var

        sampler.sample(
            rng_key=rng_key,
            log_posterior=mock_log_posterior,
            log_likelihood=mock_log_likelihood,
            data=data,
            init_state=init_state,
            num_samples=0,
            num_burn_in=2,
            thinning=1,
        )

        assert gibbs_param_sampler.update_obs_var


class TestFactoryFunctions:
    """Test the factory functions for creating samplers."""

    def test_create_mh_sampler(self, tree):
        """Test MH sampler factory function."""
        eta = 0.8
        sampler = create_mh_sampler(tree, eta=eta)

        assert isinstance(sampler, MetropolisHastingsSampler)
        assert isinstance(sampler.proposal_sampler, AlternatingProposalSampler)
        assert isinstance(
            sampler.proposal_sampler.param_sampler, MetropolisHastingsParameterSampler
        )
        assert isinstance(sampler.proposal_sampler.noise_sampler, PCNNoiseSampler)
        assert sampler.proposal_sampler.noise_sampler.eta == eta

    def test_create_gibbs_sampler(self, tree):
        """Test Gibbs sampler factory function."""
        eta = 0.7
        sampler = create_gibbs_sampler(tree, eta=eta)

        assert isinstance(sampler, MetropolisHastingsSampler)
        assert isinstance(sampler.proposal_sampler, AlternatingProposalSampler)
        assert isinstance(sampler.proposal_sampler.param_sampler, GibbsParameterSampler)
        assert isinstance(sampler.proposal_sampler.noise_sampler, PCNNoiseSampler)
        assert sampler.proposal_sampler.noise_sampler.eta == eta


class TestIntegration:
    """Integration tests that test the full workflow."""

    def test_mh_sampler_workflow(self, rng_key, tree, params, noise, data):
        """Test complete MH sampler workflow."""
        init_state = (params, noise)

        # Create sampler
        sampler = create_mh_sampler(tree, eta=0.9)

        # Simple log posterior (just return 0 for testing)
        def log_posterior(state, data):
            return 0.0

        def log_likelihood(data, state):
            return 0.0

        # Run a short sampling
        log_likes, samples = sampler.sample(
            rng_key=rng_key,
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            data=data,
            init_state=init_state,
            num_samples=3,
            num_burn_in=1,
            thinning=1,
        )

        assert len(log_likes) == 3
        assert len(samples) == 3
        assert all(isinstance(sample, tuple) for sample in samples)

    def test_gibbs_sampler_workflow(self, rng_key, tree, params, noise, data):
        """Test complete Gibbs sampler workflow."""
        init_state = (params, noise)

        # Create sampler
        sampler = create_gibbs_sampler(tree, eta=0.9)

        # Simple log posterior
        def log_posterior(state, data):
            return 0.0

        def log_likelihood(data, state):
            return 0.0

        # Run a short sampling
        log_likes, samples = sampler.sample(
            rng_key=rng_key,
            log_posterior=log_posterior,
            log_likelihood=log_likelihood,
            data=data,
            init_state=init_state,
            num_samples=3,
            num_burn_in=1,
            thinning=1,
        )

        assert len(log_likes) == 3
        assert len(samples) == 3
        assert all(isinstance(sample, tuple) for sample in samples)


if __name__ == "__main__":
    pytest.main([__file__])
