# UASM_VI

## Usage

```python
from uasm_vi import uncertainty_aware_saliency
res = uncertainty_aware_saliency(lambda x: (score(x), grad(x)), x, seed=0)
res.saliency, res.uncertainty  # E[mask], Std[mask] under the variational posterior
```

A mean-field Gaussian posterior over mask logits is fitted by maximising a
Monte-Carlo ELBO (reparameterisation trick). Requires `numpy`; tests use `pytest`.
