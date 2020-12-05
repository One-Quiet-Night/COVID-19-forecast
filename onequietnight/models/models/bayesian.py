import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.base import BaseEstimator, RegressorMixin


class LinearModel(BaseEstimator, RegressorMixin):
    def model(self, X, y=None):
        num_features = X.shape[1]
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(num_features), 100 * jnp.ones(num_features))
        )
        alpha = numpyro.sample("alpha", dist.Normal(0, 100))
        theta = jnp.dot(X, beta) + alpha
        sigma = numpyro.sample("sigma", dist.HalfNormal(100))
        return numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)

    def fit(self, X, y):
        kernel = NUTS(self.model, target_accept_prob=0.9)

        mcmc = MCMC(kernel, num_samples=1000, num_warmup=1000, num_chains=1)
        mcmc.run(random.PRNGKey(0), X, y)

        self.samples = mcmc.get_samples()
        self.predictive = Predictive(self.model, self.samples)

    def predict(self, X):
        assert hasattr(self, "predictive")
        predictions = self.predictive(random.PRNGKey(0), X)["obs"]
        predictions = np.array(predictions)
        predictions[~np.isfinite(predictions)] = np.nan
        y = np.nanmean(predictions, axis=0)
        return y

    def predict_proba(self, X):
        assert hasattr(self, "predictive")
        predictions = self.predictive(random.PRNGKey(0), X)["obs"]
        predictions = np.array(predictions)
        predictions[~np.isfinite(predictions)] = np.nan
        probs = [0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = {}
        for prob in probs:
            a, b, = (
                1 - prob
            ) / 2, (1 + prob) / 2
            results[a], results[b] = hpdi(predictions, prob)
        results[0.5] = np.nanpercentile(predictions, 50, axis=0)
        results = pd.DataFrame(results)
        return results


class ClippedModel(LinearModel):
    def model(self, X, y=None):
        num_features = X.shape[1]
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(num_features), 100 * jnp.ones(num_features))
        )
        alpha = numpyro.sample("alpha", dist.Normal(0, 100.))
        theta = jnp.dot(X, beta) + alpha
        sigma = numpyro.sample("sigma", dist.HalfNormal(100.))
        return numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)


class TruncatedModel(LinearModel):
    def model(self, X, y=None):
        num_features = X.shape[1]
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(num_features), 100 * jnp.ones(num_features))
        )
        alpha = numpyro.sample("alpha", dist.Normal(0, 100))
        theta = jnp.dot(X, beta) + alpha
        sigma = numpyro.sample("sigma", dist.HalfNormal(100))
        return numpyro.sample(
            "obs", dist.TruncatedNormal(low=0, loc=theta.clip(0), scale=sigma), obs=y
        )


class SoftPlusModel(LinearModel):
    def model(self, X, y=None):
        num_features = X.shape[1]
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(num_features), 100 * jnp.ones(num_features))
        )
        alpha = numpyro.sample("alpha", dist.Normal(0, 100))
        theta = jnp.dot(X, beta) + alpha
        sigma = numpyro.sample("sigma", dist.HalfNormal(100))
        return numpyro.sample(
            "obs", dist.Normal(jnp.log(1 + jnp.exp(theta)), sigma), obs=y
        )


class RobustModel(LinearModel):
    def model(self, X, y=None):
        num_features = X.shape[1]
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(num_features), 100 * jnp.ones(num_features))
        )
        alpha = numpyro.sample("alpha", dist.Normal(0, 100))
        theta = jnp.dot(X, beta) + alpha
        sigma = numpyro.sample("sigma", dist.HalfNormal(100))
        return numpyro.sample("obs", dist.StudentT(2, theta, sigma), obs=y)
