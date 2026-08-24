# MoEClinic

MoEClinic diagnoses mixture-of-experts router collapse from traces you already have. It does not load model weights, require a GPU, import a training framework, or make benchmark claims. The CLI is dependency-free and runs post-hoc on CPU in local development or CI.

```bash
python -m moeclinic examples/healthy.jsonl
python -m moeclinic examples/collapsed.jsonl --json
python -m moeclinic run.jsonl --fail-on warning --output report.json
```

Exit status is `0` when the configured gate passes, `1` when `--fail-on` is reached, and `2` for invalid input or output errors.

## What it measures

- auxiliary router load-balancing loss;
- router z-loss when logits or pre-aggregated log-sum-exp squares are available;
- per-expert assignment shares and mean routing probabilities;
- normalized assignment entropy, load coefficient of variation, and Gini coefficient;
- dead experts and dominant-expert concentration;
- per-batch capacity overflow and reported dropped assignments;
- a bounded `healthy`, `warning`, or `collapse` diagnosis suitable for CI.

The implemented load-balancing term is:

```text
L_aux = alpha * E * sum(f_i * P_i)
```

where `alpha` defaults to `0.01`, `E` is the expert count, `f_i` is expert `i`'s share of recorded assignments, and `P_i` is its mean routing probability. For top-1 routing this is the standard dispatched-token fraction. For top-k traces, MoEClinic normalizes by all recorded assignments so `sum(f_i) = 1` and reports the observed mean top-k separately.

The optional router stabilization term is:

```text
L_z = z_coeff * mean(logsumexp(router_logits)^2)
```

with `z_coeff=0.0001` by default. MoEClinic reports these terms; it does not add them to a training loss or claim that its default diagnosis thresholds fit every architecture.

## Trace formats

One JSON object is accepted per line. Raw records are convenient during debugging:

```json
{
  "step": 12,
  "router_probs": [[0.70, 0.10, 0.10, 0.10], [0.15, 0.60, 0.15, 0.10]],
  "selected_experts": [0, 1],
  "router_logits": [[2.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, -0.5]],
  "capacity": 32,
  "dropped_assignments": 0
}
```

Compact records avoid retaining token-level router output:

```json
{
  "step": 12,
  "token_count": 100,
  "assignment_counts": [25, 25, 25, 25],
  "probability_sums": [25.0, 25.0, 25.0, 25.0],
  "logsumexp_sq_sum": 192.1812055673,
  "logit_token_count": 100,
  "capacity": 32,
  "dropped_assignments": 0
}
```

Every record must use the same expert count. Probability rows must sum to one; compact probability sums must total `token_count`. Invalid traces fail closed with a line-numbered error.

## Concrete distinction

Current public MoE projects commonly focus on training frameworks, learned composition, or research probes over released weights. MoEClinic occupies a narrower operational gap: it accepts framework-neutral router traces, can retain only aggregates, calculates the anti-collapse terms exactly, and turns regression signals into a stable machine-readable CI decision without a model, accelerator, or third-party package.

This is a practical distinction, not a claim that no related diagnostic script exists.

## Test

```bash
python -m unittest discover -s tests -v
python -m compileall -q moeclinic tests
```

## Fund more development

Donations directly fund more RequiredTruth development. The Bitcoin, Ethereum/EVM, and Dogecoin addresses and the confirmed-transaction request process are in [`SUPPORT.md`](SUPPORT.md). A donor may open an issue with a public transaction hash and ask for more work in a specific direction; never post a private key or seed phrase.

Apache-2.0 licensed.
