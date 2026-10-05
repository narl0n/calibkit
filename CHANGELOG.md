# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-05
### Added
- Calibration metrics: ECE, MCE, Brier score, per-bin reliability statistics.
- `TemperatureScaler` for post-hoc recalibration of binary classifiers.
- Reliability diagram plotting (optional `matplotlib` dependency).
- `calibkit report` command-line tool.
- Test suite (pytest, 100% coverage) and GitHub Actions CI/CD pipeline.
