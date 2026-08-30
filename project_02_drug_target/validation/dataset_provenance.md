# Davis Dataset Provenance

## Dataset

Davis drug-target affinity benchmark in DeepDTA-format representation.

## Original Scientific Source

Davis, M. I., Hunt, J. P., Herrgard, S., Ciceri, P., Wodicka, L. M.,
Pallares, G., Hocker, M., Treiber, D. K., and Zarrinkar, P. P. (2011).
Comprehensive analysis of kinase inhibitor selectivity.
Nature Biotechnology, 29(11), 1046-1051.
DOI: https://doi.org/10.1038/nbt.1990
PMID: 22037378

The original study reports measurements for 72 kinase inhibitors and
442 kinases. The local benchmark contains 68 canonical drug records and
442 target records; it is therefore a processed machine-learning
representation, not a claim that the original study contained only
68 compounds.

## Benchmark Source

Repository: hkmztrk/DeepDTA  
Repository URL: https://github.com/hkmztrk/DeepDTA  
Pinned commit: a546a8433a6822e958f36171c4356ad6f414d623  
Commit date: 2023-08-16  
Commit message: updated installation instructions  
Local acquisition date: 2026-08-24  
Local raw-data directory: data/raw/davis/

## Dataset Representation

- Drug representation: canonical SMILES in `ligands_can.txt`
- Target representation: amino-acid sequences in `proteins.txt`
- Affinity matrix: pickled `Y` matrix containing Kd values in nM
- Missing values: represented by `NaN`
- Benchmark modelling transformation: pKd = -log10(Kd / 1e9)
- Predefined folds: indices of observed affinity positions

The structural-validation result is 68 drugs, 442 targets, 30,056
observed affinity values, zero missing values, and no overlap between
the provided train and test index sets.

## Relationship to the Original Dataset

This project uses a DeepDTA-format secondary benchmark representation
of the Davis measurements. Raw data are not committed to this
repository. Their SHA-256 hashes are stored in
`validation/davis_sha256.csv`.

## Licensing and Usage

No explicit licence file was detected in the local DeepDTA clone during
this review. The raw data must not be redistributed through this
repository. Use remains subject to the applicable upstream repository
and original-data terms.

## Human Verification Decision

Status: Approved with documented limitations

Decision rationale:

- upstream repository URL and pinned commit were verified;
- raw-file SHA-256 values were recorded;
- executable structural validation passed;
- predictive results must not be interpreted as biological causality.