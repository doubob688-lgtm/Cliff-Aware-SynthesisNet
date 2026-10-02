"""Pinned HTE workbook audit and outcome-derived route preferences (no TDC)."""

import hashlib
import itertools
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdChemReactions
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE_COMMIT = "d9e6b87ce1b881978490d68bfc00021e3b48127a"
SOURCE_PATH = "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"
SOURCE_URL = f"https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/{SOURCE_COMMIT}/{SOURCE_PATH}"
RAW_SHA256 = "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e"
RAW_FILE = Path("data/raw/buchwald_hartwig/Dreher_and_Doyle_input_data.xlsx")
COMPONENTS = ("Aryl halide", "Ligand", "Base", "Additive")
AMINE = "Cc1ccc(N)cc1"
# Exact product-generation SMARTS from pinned rxn_yields/data.py.
PRODUCT_SMARTS = "[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]"


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_raw(path=RAW_FILE):
    """Fetch pinned bytes once; refuse to overwrite or accept a different file."""
    path = Path(path)
    if not path.exists():
        payload = urllib.request.urlopen(SOURCE_URL, timeout=60).read()
        if hashlib.sha256(payload).hexdigest() != RAW_SHA256:
            raise ValueError("Downloaded XLSX checksum differs from pinned source")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if checksum(path) != RAW_SHA256:
        raise ValueError("Existing raw XLSX differs from pinned source; left unchanged")
    return path


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid component SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def product_for(aryl):
    reaction = rdChemReactions.ReactionFromSmarts(PRODUCT_SMARTS)
    products = set()
    for result in reaction.RunReactants((Chem.MolFromSmiles(aryl), Chem.MolFromSmiles(AMINE))):
        Chem.SanitizeMol(result[0])
        products.add(Chem.MolToSmiles(result[0]))
    if len(products) != 1:
        raise ValueError(f"Expected one benchmark product for {aryl}: {products}")
    return products.pop()


def load_and_audit(path=RAW_FILE):
    path = download_raw(path)
    with pd.ExcelFile(path) as book:
        sheets = book.sheet_names
        raw = pd.read_excel(book, sheet_name="FullCV_01")
    if list(raw.columns) != ["Ligand", "Additive", "Base", "Aryl halide", "Output"]:
        raise ValueError(f"Unexpected benchmark columns: {list(raw.columns)}")
    frame = raw.copy()
    for column in COMPONENTS:
        frame[column] = frame[column].map(canonical)
    values = pd.to_numeric(frame["Output"], errors="raise")
    if frame.isna().any().any() or not np.isfinite(values).all() or not values.between(0, 100).all():
        raise ValueError("Missing/invalid observations: explicit review required")
    frame["Output"] = values
    if frame.duplicated(list(COMPONENTS)).any():
        raise ValueError("Duplicate experimental conditions require explicit replicate handling")
    products = {aryl: product_for(aryl) for aryl in frame["Aryl halide"].unique()}
    frame["target"] = frame["Aryl halide"].map(products)
    frame["reaction_id"] = [f"FullCV_01:{i + 2}" for i in range(len(frame))]
    observed_conditions = set(frame[list(COMPONENTS)].itertuples(index=False, name=None))
    audit = {
        "source_url": SOURCE_URL, "source_commit": SOURCE_COMMIT, "raw_sha256": checksum(path),
        "raw_bytes": path.stat().st_size, "primary_sheet": "FullCV_01", "sheets": sheets,
        "raw_reactions": len(raw), "usable_reactions": len(frame), "columns": list(raw.columns),
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "unique_components_raw": {c: int(raw[c].nunique()) for c in COMPONENTS},
        "unique_components_canonical": {c: int(frame[c].nunique()) for c in COMPONENTS},
        "duplicates_exact_rows": int(raw.duplicated().sum()),
        "duplicates_canonical_conditions": int(frame.duplicated(list(COMPONENTS)).sum()),
        "yield_percent_summary": {c: float(v) for c, v in values.describe().items()},
        "yield_histogram_edges": [0, 20, 40, 60, 80, 100],
        "yield_histogram_counts": np.histogram(values, [0, 20, 40, 60, 80, 100])[0].tolist(),
        "yield_zero_count": int((values == 0).sum()), "unique_targets": len(set(products.values())),
        "filtering": "FullCV_01 already contains the 3955-row benchmark; no extra row filtering or imputation",
        "factorial_expected": int(np.prod([frame[c].nunique() for c in COMPONENTS])),
        "products_by_aryl": products,
        "missing_factorial_combinations": [dict(zip(COMPONENTS, combination))
                                            for combination in itertools.product(*(sorted(frame[c].unique()) for c in COMPONENTS))
                                            if combination not in observed_conditions],
    }
    return frame, audit


def split_targets(frame, seed=42, fold=0):
    """Group routes and target analogues by product Bemis-Murcko scaffold.

    One scaffold each trains, validates and tests. The largest scaffold (three
    phenyl targets) trains the primary fold. Order the remaining scaffolds with
    an outcome-independent seed and rotate three outer folds.
    """
    targets = sorted(frame["target"].unique())
    scaffolds = {t: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(t)) for t in targets}
    unique = sorted(set(scaffolds.values()))
    if len(unique) != 3:
        raise ValueError("This initial scaffold-disjoint protocol expects three product scaffolds")
    largest = sorted(unique, key=lambda s: (-sum(v == s for v in scaffolds.values()), s))[0]
    others = [s for s in unique if s != largest]
    np.random.RandomState(seed).shuffle(others)
    order = [largest] + others
    training, validation, test = order[fold % 3], order[(fold + 1) % 3], order[(fold + 2) % 3]
    return {t: "train" if scaffolds[t] == training else "validation" if scaffolds[t] == validation else "test" for t in targets}


def construct_pairs(frame, gap=20.0, high_similarity=0.90, cliff_gap=40.0, seed=42, fold=0):
    """Exactly one changed component, same product/family, all other roles fixed.

    Labels depend on outcomes; orientation, grouping and similarities do not.
    No reverse-pair augmentation. The exported training records omit yields.
    """
    if gap <= 0 or cliff_gap < gap:
        raise ValueError("Require positive preference gap and cliff_gap >= gap")
    split = split_targets(frame, seed, fold)
    fps = {s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
           for c in COMPONENTS for s in frame[c].unique()}
    similarity_cache = {}
    pairs, gap_counts, eligible, seen = [], {str(g): 0 for g in (10, 20, 30, 40)}, 0, set()
    for changed in COMPONENTS:
        fixed = ["target"] + [c for c in COMPONENTS if c != changed]
        for _, matched in frame.groupby(fixed, sort=True):
            for a, b in itertools.combinations(matched.index.tolist(), 2):
                if frame.at[a, changed] == frame.at[b, changed]:
                    continue
                key = tuple(sorted((a, b)))
                if key in seen:
                    raise AssertionError("Duplicate unordered pair")
                seen.add(key)
                eligible += 1
                difference = abs(float(frame.at[a, "Output"] - frame.at[b, "Output"]))
                for threshold in gap_counts:
                    gap_counts[threshold] += int(difference >= float(threshold))
                if difference < gap:
                    continue
                smiles_key = tuple(sorted((frame.at[a, changed], frame.at[b, changed])))
                if smiles_key not in similarity_cache:
                    similarity_cache[smiles_key] = DataStructs.TanimotoSimilarity(fps[smiles_key[0]], fps[smiles_key[1]])
                changed_similarity = similarity_cache[smiles_key]
                # Role-aligned six-role reaction similarity: aryl, fixed amine,
                # product, ligand, base, additive. Constant catalyst excluded.
                reaction_similarity = (5 + changed_similarity) / 6
                substrate_similarity = (1 + changed_similarity) / 2 if changed == "Aryl halide" else 1.0
                tree_similarity = (2 + changed_similarity) / 3 if changed == "Aryl halide" else 1.0
                high = reaction_similarity >= high_similarity and tree_similarity >= 0.80
                # Hash decides orientation independently of outcome; row IDs stable.
                if hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest()[0] & 1:
                    a, b = b, a
                ra, rb = frame.loc[a], frame.loc[b]
                context = {"target": ra["target"], "reaction_family": "Buchwald-Hartwig",
                           "fixed_components": {c: ra[c] for c in COMPONENTS if c != changed}}
                pairs.append({"target_context": context, "tree_A": ra["reaction_id"], "tree_B": rb["reaction_id"],
                              "preference_label": int(ra["Output"] > rb["Output"]), "split": split[ra["target"]],
                              "changed_component": changed, "changed_component_similarity": changed_similarity,
                              "reaction_similarity": reaction_similarity, "substrate_similarity": substrate_similarity,
                              "tree_similarity": tree_similarity, "topology_similarity": 1.0,
                              "high_similarity": bool(high), "route_cliff": bool(high and difference >= cliff_gap)})
    summary = {"eligible_one_change_pairs": eligible, "threshold_pair_counts": gap_counts,
               "preference_gap_pp": gap, "high_similarity_cutoff": high_similarity,
               "tree_similarity_cutoff": 0.80, "cliff_gap_pp": cliff_gap, "pairs": len(pairs),
               "high_similarity_pairs": sum(p["high_similarity"] for p in pairs),
               "route_cliff_candidates": sum(p["route_cliff"] for p in pairs), "target_split": split,
               "by_split": {s: sum(p["split"] == s for p in pairs) for s in ("train", "validation", "test")},
               "by_changed_component": {c: sum(p["changed_component"] == c for p in pairs) for c in COMPONENTS}}
    return pairs, summary


def route_records(frame):
    """Experimental single-step route records; conditions are annotations only."""
    return [{"reaction_id": r["reaction_id"], "target": r["target"], "reaction_family": "Buchwald-Hartwig",
             "tree": {"reactants": [r["Aryl halide"], AMINE], "product": r["target"], "rxn_id": 41},
             "conditions": {c: r[c] for c in COMPONENTS if c != "Aryl halide"}}
            for _, r in frame.iterrows()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
