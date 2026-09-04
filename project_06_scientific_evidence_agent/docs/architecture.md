# Architecture boundary

```text
claim text
  -> retrieval
  -> reranking
  -> sentence evidence selection
  -> stance verification
  -> citation audit
  -> support / contradict / no-evidence decision
```

Runtime code receives claim text and the public corpus only. Evaluation code
receives frozen predictions and gold annotations separately. This separation is
the primary control against accidental data leakage.
