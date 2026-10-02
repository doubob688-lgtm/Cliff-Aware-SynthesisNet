# Immutable Buchwald-Hartwig HTE source workbook

Source: [rxn4chemistry/rxn_yields at d9e6b87](https://github.com/rxn4chemistry/rxn_yields/blob/d9e6b87ce1b881978490d68bfc00021e3b48127a/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx).
Downloaded directly from that commit's raw GitHub URL. All 16 worksheets and
original XLSX bytes are retained. Read `FullCV_01` only; do not concatenate the
alternative benchmark split sheets.

- File: `Dreher_and_Doyle_input_data.xlsx`
- Size: 2,134,762 bytes
- SHA-256: `d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e`
- Data range: `FullCV_01!A1:E3956`, including the header.
- 3,955 benchmark rows; no further filtering or imputation.

`cliffnet.buchwald.download_raw()` verifies the checksum and refuses to overwrite
an existing different file. Processed experimental route records and preference
pairs are written separately to `results/pathway/`, without numerical yields.
Yield values remain in the raw workbook and are used for auditing and deriving
preference labels/eligibility, never as ranking features or regression targets.
