# Changelog

## [🚧Unreleased](https://github.com/dhruvdcoder/structured_prediction_baselines/tree/HEAD)

[Full Changelog](https://github.com/dhruvdcoder/structured_prediction_baselines/compare/v0.1.0...HEAD)

### 💥 Breaking Changes:

- Reverse the order of param updates and various other breaking changes. [\#22](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/22) ([dhruvdcoder](https://github.com/dhruvdcoder))

### ✨ Features and Enhancements:

- Add original infnet setup \(Tu, Gimpel\) in the reverse order  [\#32](https://github.com/dhruvdcoder/structured_prediction_baselines/issues/32)
- Add unnormalized score loss for training tasknn [\#30](https://github.com/dhruvdcoder/structured_prediction_baselines/issues/30)
- Changes to make sequence-tagging work again [\#61](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/61) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Adds SPEN Loss and configs [\#58](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/58) ([purujitgoyal](https://github.com/purujitgoyal))
- Adapters and configs with blurb genre dataset [\#55](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/55) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Adds Best Run Configs [\#50](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/50) ([purujitgoyal](https://github.com/purujitgoyal))
- Made changes for applying ranking loss directly to the TaskNN [\#48](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/48) ([leejayyoon](https://github.com/leejayyoon))
- Score-nn Evaluation [\#42](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/42) ([purujitgoyal](https://github.com/purujitgoyal))
- Text MLC with any text encoder including BERT [\#37](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/37) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Original inference\_net setup as in Tu & Gimpel [\#33](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/33) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Added 'multi-label-score-loss' and changes in relevant configs \(bibtex\_strat\_nce\) [\#31](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/31) ([dhruvdcoder](https://github.com/dhruvdcoder))
- sweep and model configs for dvn, nce reverse order [\#29](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/29) ([dhruvdcoder](https://github.com/dhruvdcoder))

### 🐛 Bug Fixes:

- Default "nce discrete" jsonnet has "-" signs now. [\#51](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/51) ([leejayyoon](https://github.com/leejayyoon))
- Making GBI work again. [\#47](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/47) ([dhruvdcoder](https://github.com/dhruvdcoder))

### 📖 Documentation updates

- Add new dependency version info in README [\#27](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/27) ([leejayyoon](https://github.com/leejayyoon))

### 📦 Dependencies

- Tracking notebooks using jupytext [\#54](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/54) ([dhruvdcoder](https://github.com/dhruvdcoder))

### 👷 Build and CI

- Automatic generation of changelog [\#23](https://github.com/dhruvdcoder/structured_prediction_baselines/issues/23)
- Add new labels and update changelog generator [\#34](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/34) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Tests workflow, end2end pytest for training. \(closes 25\) [\#26](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/26) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Labeler and automatic changelog generation. [\#24](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/24) ([dhruvdcoder](https://github.com/dhruvdcoder))

### ⚙️  Model and sweep configs

- NYT and RCV text datasets, readers and configs [\#59](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/59) ([dhruvdcoder](https://github.com/dhruvdcoder))
- No cross entropy configs for DVN+tasknn [\#53](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/53) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Feat/from pretrained dvn [\#52](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/52) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Added sweeps & jsonnet for eurlex \(and other small sweep config changes\) [\#45](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/45) ([leejayyoon](https://github.com/leejayyoon))
- Organized sweep configs for general data & specific datasets. [\#40](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/40) ([leejayyoon](https://github.com/leejayyoon))
- Feat/general data configs [\#39](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/39) ([leejayyoon](https://github.com/leejayyoon))
- Pushing general sweep configs & corresponding jsonnet. + Sweep configs for 'delicious' dataset. [\#38](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/38) ([leejayyoon](https://github.com/leejayyoon))
- Original infnet remaining configs [\#36](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/36) ([dhruvdcoder](https://github.com/dhruvdcoder))
- Dev/v1.2/jy  Created sweeps/configs for testing effect of pretrained model + general data sweeps.  [\#35](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/35) ([leejayyoon](https://github.com/leejayyoon))
- Added 6 more feature based datasets and blurb and nyt [\#19](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/19) ([dhruvdcoder](https://github.com/dhruvdcoder))

### 🧪 Peripheral utilities

- Scripts to count instances and print env vars in jsonnet [\#60](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/60) ([dhruvdcoder](https://github.com/dhruvdcoder))

**Merged pull requests:**

- Added expr and spo, go and fun datasets. [\#44](https://github.com/dhruvdcoder/structured_prediction_baselines/pull/44) ([dhruvdcoder](https://github.com/dhruvdcoder))

## [v0.1.0](https://github.com/dhruvdcoder/structured_prediction_baselines/tree/v0.1.0) (2021-07-08)

[Full Changelog](https://github.com/dhruvdcoder/structured_prediction_baselines/compare/ef23891a32a0dcc7b9ca02a8c11e008cbe412dbb...v0.1.0)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
