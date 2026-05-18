# Draft examples

`.ferx` files in this directory show **proposed syntax** for features whose
parser support has not yet landed. They are intentionally not picked up by
the parser smoke test in `src/parser/model_parser.rs`, and running
`cargo run -- examples/drafts/<file>.ferx` today will fail.

The point is to give reviewers something concrete to comment on while the
larger feature is being staged across multiple PRs. When the corresponding
parser hookup lands, the file is moved up to `examples/` and becomes a
real, runnable example.

Current contents:

| File                  | Feature                | Tracking         |
|-----------------------|------------------------|------------------|
| `warfarin_dcm.ferx`   | Deep Compartment Model | Phase A M1 / M2 in `plans/dcm-and-low-dim-node.md` |
