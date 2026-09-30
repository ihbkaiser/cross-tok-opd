# Gemma terminal controls in MP-OPD

Gemma added vocabulary includes ordinary newline and whitespace content. Reject
special IDs, not every added ID, when tokenizer special metadata is available.
The legacy unsupported_added_token reason remains for compatibility.

MP-OPD strips trailing EOS and a registered special <end_of_turn> from the
teacher response text and atom credit coverage. Student sampled IDs and behavior
log probabilities remain unchanged, including the sampled turn terminator.
An internal special token still fails closed. Existing masked_eos metrics count
both kinds of excluded terminal control. Non-MP-OPD response construction is
unchanged. Synthetic EOS and the existing normalization denominator are retained.

Run CPU regressions with:
python -m pytest -q tests/mp_opd/test_atoms.py tests/mp_opd/test_gemma_controls.py

The borrowed B200 rollout replay accepted all 64 responses with content-only
atomization. That replay is not evidence of optimizer updates or GPU parity;
atomic and fixed canaries must pass after applying this fix.
