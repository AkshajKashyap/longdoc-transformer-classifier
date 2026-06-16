# Limitations

## Smoke Runs Are Not Final Rankings

Small sample sizes are useful for engineering validation, but they are not enough to rank modeling
families. Reported smoke scores should be treated as pipeline checks.

## bert-tiny Is Not A Performance Model

`prajjwal1/bert-tiny` is used because it makes CPU smoke runs practical. Its scores should not be read
as evidence against transformer methods in general.

## Whitespace Token Counts Are Approximations

Length analysis uses whitespace token counts as a cheap approximation. Actual tokenizer-specific token
counts can differ, especially for scientific notation, symbols, and rare vocabulary.

## arXiv Labels May Have Lexical Cues

The arXiv labels may be predictable from domain vocabulary. That can make TF-IDF unusually strong and
does not necessarily prove the model understands long-range structure.

## Summary-First Reads The Front Portion

The current summary-first setup may summarize only the front portion of very long documents because the
summarizer has its own input limit.

## Chunking Does Not Know Evidence Location

Fixed word chunking does not know which chunks contain class evidence. It gives structure to the
pipeline, but it does not solve evidence selection by itself.

## IDF Top-K Is Lexical

IDF top-k ranks chunks by lexical rarity from training chunks. It is not semantic and can overvalue rare
terms that are not useful evidence for the class.

## Uniform Coverage Can Miss Evidence

Uniform selection improves coverage across a document, but important evidence can still fall between
selected chunks.

## Chunk Selection Is Unsupervised

The current chunk selection strategies do not learn evidence relevance from labels or model feedback.
They are heuristics for making capped chunked runs less prefix-biased.

## Long-Context Models Still Have Limits

Longformer-style models still have a maximum context window. A `max_length=1024` smoke run does not use
the full 4096-token capacity of `allenai/longformer-base-4096`.

## Long-Context Smoke Runs Are Expensive

Long-context models use more memory and compute than TF-IDF or chunk heuristics. Batch size 1, tiny
sample sizes, and encoder freezing keep smoke runs practical but limit what the results prove.

## Frozen Encoders Limit Adaptation

Freezing the encoder trains only the classification head where possible. That is useful for memory and
speed, but it can understate what a fully fine-tuned long-context model could do.
