# Full project reference (not part of the paper)

These four LaTeX files are the long, fully source-commented drafts the condensed appendix
(`paper/sec/appendix/Z_condensed.tex`) was cut from. Each number carries a `% src:` comment with the file it came from.

- `A_method.tex`: extended method (Tesseract partition, features, network, targets, roll transform, orbit rules, distillation, translation, flat baselines)
- `B_protocol.tex`: datasets, splits, crops, training hyper-parameters, metric definitions, pre-registration, compute
- `C_results.tex`: complete results (68 tables generated from the run CSVs: all variants, modes, per-object, per-seed)
- `D_log.tex`: complete experiment log, all decisions, negative results, claim boundaries

They are not `\input` in `main.tex`; use them for lookup.
