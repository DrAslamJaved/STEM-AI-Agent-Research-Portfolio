# SciFact validation contract

`validate-data` must complete before retrieval or verification code may consume
the dataset. It verifies:

1. required corpus and train/dev/test claim files exist and are valid JSONL;
2. corpus document IDs are unique, titles and abstracts are non-empty, and
   sentence lists contain text;
3. claim IDs are unique across the primary splits and claim text is non-empty;
4. every gold evidence document exists in the corpus;
5. rationale labels are `SUPPORT` or `CONTRADICT` and all sentence IDs point to
   valid abstract sentences;
6. the official five-fold train/dev files exist and contain only primary
   train/dev claim IDs; and
7. a JSON summary records counts, labels, and any cited documents unavailable
   in the corpus.

The validator may inspect gold annotations because it belongs to the evaluator
boundary. Runtime retrieval code must still receive only the safe claim object
defined in `evidence_agent.contracts`.
