# Dataset provenance

The JSON files created in this directory are generated educational fixtures.
They are not exports or copies of Wikipedia, Project Gutenberg, SQuAD, or
OpenAssistant. Legacy filenames are generated locally to avoid breaking older
workflows, while the dataset builder normalizes their source labels to
`synthetic_articles`, `synthetic_books`, `synthetic_qa`, and
`synthetic_dialogue`.

Generated JSON files are intentionally excluded from Git. Use these fixtures
for smoke tests and pipeline experiments only. Meaningful
language-model quality requires a larger, diverse, licensed corpus supplied by
the user. Generated and processed outputs are excluded from Git.
