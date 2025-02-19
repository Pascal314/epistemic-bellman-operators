import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import distrax
import blackjax
from functools import partial


gamma = 0.9

def generate_TR(key, S, A, sparsity=0.):
    alpha = jnp.ones((S, A, S))
    sparse_key, t_key, r_key = jax.random.split(key, 3)
    mask = jax.random.bernoulli(sparse_key, 1 - sparsity, shape=alpha.shape)
    alpha = alpha * mask

    T = jax.random.dirichlet(t_key, alpha)
    R = jax.random.normal(r_key, (S, A))
    return T, R

def mcmc_sampler(rng_key, T, R, noise_level):
    sigma = jnp.ones_like(R.flatten()) * noise_level
    S, A = R.shape

    def td_error(q, q_t):
        target = R.flatten() + gamma * T.reshape(S * A, S) @ jnp.max(q_t, axis=1)
        return (q.flatten() - target).reshape(S, A)
             
    def likelihood(q):
        return distrax.Normal(jnp.zeros_like(q), sigma).log_prob(td_error(q.reshape(S, A), q.reshape(S, A)).flatten()).sum()

    def log_density(q):
        return 0 * distrax.Normal(0, 1).log_prob(q).sum() + likelihood(q)
    
    step_size = 1e-1
    q_flat = jnp.zeros_like(R.flatten())
    inverse_mass_matrix = jnp.eye(q_flat.shape[0]) * noise_level**2
    hmc = blackjax.hmc(log_density, step_size, inverse_mass_matrix, 100)

    # Initialize the state
    initial_position = q_flat
    state = hmc.init(initial_position)

    # Iterate
    step = jax.jit(hmc.step)
    def _to_scan(state, rng_key):
        state, info = step(rng_key, state)
        return state, {'state': state, 'info': info}

    state, info = jax.lax.scan(_to_scan, state, jax.random.split(rng_key, 1000))
    return info['state'].position.reshape(-1, *R.shape)[100:], info

def double_mcmc_sampler(rng_key, T, R, noise_level):
    S, A = R.shape
    sigma = jnp.ones_like(S * A * 2) * noise_level


    def double_td_error(q, q_t, q_s):
        q = q.reshape(S, A)
        q_t = q_t.reshape(S, A)
        q_s = q_s.reshape(S, A)
        a = jax.nn.one_hot(jnp.argmax(q_s, axis=1), A)

        target = R.flatten() + gamma * T.reshape(S * A, S) @ (a * q_t).sum(axis=-1)
        return (q.flatten() - target).reshape(S, A)
             
    def likelihood(qqs):
        q, q_s = qqs[:S * A], qqs[S * A:]
        q_td = double_td_error(q, q, q_s)
        q_s_td = double_td_error(q_s, q_s, q)

        return distrax.Normal(jnp.zeros_like(q), sigma).log_prob((q_td + q_s_td).flatten()).sum()
    
    def log_density(q):
        return 0 * distrax.Normal(0, 1).log_prob(q).sum() + likelihood(q)
    
    step_size = 1e-4
    q_flat = jnp.zeros_like(R.flatten())
    inverse_mass_matrix = jnp.eye(q_flat.shape[0] * 2) * noise_level**2
    hmc = blackjax.hmc(log_density, step_size, inverse_mass_matrix, 100)

    # Initialize the state
    initial_position = jnp.concatenate([q_flat, q_flat], axis=0)
    print(initial_position.shape)
    state = hmc.init(initial_position)

    # Iterate
    step = jax.jit(hmc.step)
    def _to_scan(state, rng_key):
        state, info = step(rng_key, state)
        return state, {'state': state, 'info': info}

    state, info = jax.lax.scan(_to_scan, state, jax.random.split(rng_key, 1000))
    return info['state'].position.reshape(-1, *R.shape)[100:], info

def ebo_sampler(rng_key, T, R, noise_level):
    S, A = R.shape
    def _pushforward_method(carry, rng_key):
        q = carry
        epsilon = jax.random.normal(rng_key, q.shape) * noise_level
        q = 0.0 * q + 1.0 * (R + gamma * (T * q.max(1)[None, None, :]).sum(-1) + epsilon)
        return q, q

    def pushforward_method(rng_key):
        init = jnp.zeros_like(R)
        q, _ = jax.lax.scan(_pushforward_method, init, jax.random.split(rng_key, 200))
        return q
    
    qs = jax.vmap(pushforward_method)(jax.random.split(rng_key, 500))
    return qs, None


def double_ebo_sampler(rng_key, T, R, noise_level):
    S, A = R.shape
    def _pushforward_method(carry, rng_key):
        rng_key, rng_key_s = jax.random.split(rng_key)
        q, q_s = carry

        epsilon = (jax.random.normal(rng_key, R.shape) * noise_level)
        epsilon_s = (jax.random.normal(rng_key_s, R.shape) * noise_level)


        q_a = q[jnp.arange(S), q_s.argmax(1)][None, :]
        q_s_a = q_s[jnp.arange(S), q.argmax(1)][None, :]

        q = (R + gamma * (T * q_a[None, :]).sum(axis=-1) + epsilon)
        q_s = (R + gamma * (T * q_s_a[None, :]).sum(axis=-1) + epsilon_s)
        return (q, q_s), None

    def pushforward_method(rng_key):
        init = (jnp.ones_like(R), jnp.ones_like(R))
        (q, q_s), _ = jax.lax.scan(_pushforward_method, init, jax.random.split(rng_key, 200))
        return q
    
    qs = jax.vmap(pushforward_method)(jax.random.split(rng_key, 500))
    return qs, None


def compute_true_value(pi, T, R):
    S, A = R.shape
    q_true = jnp.linalg.solve( jnp.eye(S * A) - gamma * (T[:, :, :, None] * pi[None, None, :, :]).reshape(S * A, S * A), R.flatten())
    return (pi * q_true.reshape(S, A)).sum(-1)

def average_greedy_policy(qs):
    pi = qs == qs.max(-1, keepdims=True)
    pi = pi.mean(0)
    pi = pi / pi.sum(-1, keepdims=True)
    return pi

def experiment(rng_key, noise_level, S, A, sampler):
    tr_key, sampler_key = jax.random.split(rng_key, 2)
    T, R = generate_TR(tr_key, S, A, 0.)
    R = 0.1 * R

    qs, sampler_info = sampler(sampler_key, T, R, noise_level)
    policy = average_greedy_policy(qs)

    true_values = compute_true_value(policy, T, R)
    imaginary_values = (qs.mean(0) * policy).sum(-1)

    average_difference = imaginary_values.mean() - true_values.mean()
    info = dict(
        sampler_info=sampler_info,
        true_values=true_values,
        imaginary_values=imaginary_values,
        policy=policy,
    )
    
    return average_difference, info
    
partialed_experiment = partial(experiment, S=30, A=5, sampler=double_mcmc_sampler)


samplers = {
    "MCMC": mcmc_sampler,
    "EBO": ebo_sampler,
    "Double-Q EBO": double_ebo_sampler,
    "Double-Q MCMC": double_mcmc_sampler
}

noise_levels = jnp.linspace(0., 2., 10)

results = {}
rng_key = jax.random.PRNGKey(42)
for name, sampler in samplers.items():

    partialed_experiment = partial(experiment, S=30, A=5, sampler=sampler)
    err, _ = jax.vmap(jax.vmap(partialed_experiment, in_axes=(0, None)), in_axes=(None, 0)) (jax.random.split(rng_key, 10), noise_levels)
    results[name] = err


plt.rcParams["text.usetex"] = True

colors = {
    "MCMC": "blue",
    "Double-Q MCMC": "red",
    "EBO": "green",
    "Double-Q EBO": "purple",
}

linestyles = {
    "MCMC": "-",
    "Double-Q MCMC": "-",
    "EBO": "",
    "Double-Q EBO": "",
}

markers = {"MCMC": "",
    "Double-Q MCMC": "",
    "EBO": "x",
    "Double-Q EBO": "x",}


sorted_results = {name: results[name] for name in ["MCMC", "Double-Q MCMC", "EBO", "Double-Q EBO"]}

for name, result in sorted_results.items():
    plt.plot(noise_levels, result.mean(1), label=name, color=colors[name], linestyle=linestyles[name], linewidth = 2., marker=markers[name])


plt.ylabel('Value gap: ($V_{\\texttt{ts}} - V_{\\texttt{true}}$)')
plt.xlabel(r'$\epsilon$-scale')
plt.gcf().set_size_inches(5, 2.5)
plt.legend()
plt.tight_layout()
plt.savefig("ts_experiment.pdf")


plt.figure()
no_doubles = {name: results[name] for name in ["MCMC", "EBO",]}
for name, result in no_doubles.items():
    plt.plot(noise_levels, result.mean(1), label=name, color=colors[name], linestyle=linestyles[name], linewidth = 2., marker=markers[name])


plt.ylabel('Value gap: ($V_{\\texttt{ts}} - V_{\\texttt{true}}$)')
plt.xlabel(r'$\epsilon$-scale')
plt.gcf().set_size_inches(5, 2.5)
plt.legend()
plt.tight_layout()
plt.savefig("ts_experiment-no-doubles.pdf")
