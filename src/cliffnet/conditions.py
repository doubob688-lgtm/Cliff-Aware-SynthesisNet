"""Train-vocabulary categorical conditions, separate from native synthetic trees."""

from collections import Counter

import numpy as np


CONDITION_ROLES = ("Ligand", "Base", "Additive")


class ConditionEncoder:
    """Sorted canonical-SMILES one-hot blocks with one unknown slot per role.

    Fit only on training observations. No molecular descriptors, outcome values,
    learned parameters or held-out categories enter the vocabulary.
    """

    def __init__(self, vocabularies):
        if set(vocabularies) != set(CONDITION_ROLES):
            raise ValueError("Expected ligand, base and additive vocabularies")
        self.vocabularies = {r: tuple(vocabularies[r]) for r in CONDITION_ROLES}
        self.offsets, self.lookups, offset = {}, {}, 0
        for role, values in self.vocabularies.items():
            if list(values) != sorted(set(values)):
                raise ValueError("Vocabulary must be sorted and unique")
            self.offsets[role] = offset
            self.lookups[role] = {v: i for i, v in enumerate(values)}
            offset += len(values) + 1
        self.width = offset

    @classmethod
    def fit(cls, training_conditions):
        rows = list(training_conditions)
        if not rows:
            raise ValueError("Condition vocabulary requires training observations")
        return cls({r: sorted({row[r] for row in rows}) for r in CONDITION_ROLES})

    def transform(self, conditions):
        rows = list(conditions)
        result = np.zeros((len(rows), self.width), dtype=np.float32)
        for i, row in enumerate(rows):
            for role in CONDITION_ROLES:
                value = row[role]
                if not isinstance(value, str):
                    raise ValueError("Condition categories must be strings")
                column = self.lookups[role].get(value, len(self.vocabularies[role]))
                result[i, self.offsets[role] + column] = 1
        return result

    def unknown_counts(self, conditions):
        rows = list(conditions)
        return {r: sum(row[r] not in self.lookups[r] for row in rows) for r in CONDITION_ROLES}

    def metadata(self):
        return {"type": "deterministic_categorical_one_hot", "roles": list(CONDITION_ROLES),
                "vocabularies": {r: list(v) for r, v in self.vocabularies.items()},
                "offsets": self.offsets, "width": self.width,
                "unknown_rule": "one final unknown slot in each role block",
                "fit_scope": "training observations participating in training preference pairs only",
                "trainable_parameters": 0, "outcomes_as_features": False}


def representations(z_tree, z_condition):
    tree = np.asarray(z_tree, dtype=np.float32)
    condition = np.asarray(z_condition, dtype=np.float32)
    if tree.ndim != 2 or condition.ndim != 2 or len(tree) != len(condition):
        raise ValueError("Expected two matrices with the same observation count")
    if not np.isfinite(tree).all() or not np.isfinite(condition).all():
        raise ValueError("Nonfinite representations")
    return {"tree_only": tree, "condition_only": condition,
            "tree_condition": np.concatenate((tree, condition), axis=1)}


def training_statistics(features, indices):
    """Original Torch train-only mean/population-SD and 1e-3 floor."""
    import torch

    values = torch.as_tensor(features, dtype=torch.float32)[sorted(set(indices))]
    if len(values) == 0:
        raise ValueError("No training representations")
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-3)


def collision_counts(features, route_index, pairs):
    """Exact vector equality, including all unordered observation collisions."""
    features = np.asarray(features)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected finite feature matrix")
    rows = features[[route_index[r] for r in sorted(route_index)]]
    # tuple equality treats +/- zero as equal, matching np.array_equal.
    counts = Counter(map(tuple, rows.tolist()))
    by_component = {}
    for component in ("Aryl halide", *CONDITION_ROLES):
        selected = [p for p in pairs if p["changed_component"] == component]
        equal = sum(np.array_equal(features[route_index[p["tree_A"]]],
                                   features[route_index[p["tree_B"]]]) for p in selected)
        by_component[component] = {"pairs": len(selected), "collisions": equal}
    return {"observations": len(rows), "unique_representations": len(counts),
            "duplicate_observations_beyond_first": len(rows) - len(counts),
            "unordered_observation_collisions": sum(n * (n - 1) // 2 for n in counts.values()),
            "preference_pair_collisions": sum(v["collisions"] for v in by_component.values()),
            "condition_only_pair_collisions": sum(by_component[r]["collisions"] for r in CONDITION_ROLES),
            "by_changed_component": by_component,
            "equality_rule": "exact vector equality before normalization"}
