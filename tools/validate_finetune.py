"""Check a bounded fine-tuning run's frozen inputs, updates and final networks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    settings = json.loads(args.config.read_text())
    training = ROOT / settings["log_root"] / settings["run_name"]
    plan = json.loads((training / "training_plan.json").read_text())
    run = json.loads((training / "run_report.json").read_text())
    assert settings == {key: plan[key] for key in settings}
    assert run["ok"] and run["optimizer_updates"] > 0
    assert 0 < run["additional_steps"] <= settings["additional_steps"]
    assert run["final_iteration"]-run["initial_iteration"] == run["additional_steps"]
    assert digest(ROOT / settings["checkpoint"]) == plan["checkpoint_sha256"]
    assert digest(ROOT / settings["motion"]) == plan["motion_sha256"]
    assert digest(training / "launcher_frozen.py") == plan["launcher_sha256"]
    assert digest(Path(run["saved_checkpoint"])) == run["saved_checkpoint_sha256"]
    updates = [json.loads(line) for line in (training / "updates.jsonl").read_text().splitlines()]
    counts = {}
    for component in ("wm", "policy"):
        rows = [row for row in updates if row["component"] == component]
        expected = run["optimizer_updates"]
        if component == "policy":
            expected *= plan["effective_config"]["schedule"]["policy_updates_per_wm_update"]
        assert len(rows) == expected
        assert all(math.isfinite(float(value)) for row in rows for key, value in row.items() if key != "component")
        counts[component] = len(rows)
    original = torch.load(ROOT / settings["checkpoint"], map_location="cpu", weights_only=False)
    final = torch.load(run["saved_checkpoint"], map_location="cpu", weights_only=False)
    from heading_checkpoint import migrate_checkpoint
    original, migration = migrate_checkpoint(original, final["config"]["policy"]["goal_dim"])
    assert final["iteration"] == run["final_iteration"]
    assert final["num_updates"]-original["num_updates"] == run["optimizer_updates"]
    networks = {}
    for name in ("wm", "policy", "critic"):
        assert original[name].keys() == final[name].keys()
        changed, squared_change, squared_initial = 0, 0., 0.
        for key, value in final[name].items():
            previous = original[name][key]
            assert value.shape == previous.shape
            if value.is_floating_point():
                assert torch.isfinite(value).all()
                changed += int(not torch.equal(value, previous))
                squared_change += float((value.double()-previous.double()).square().sum())
                squared_initial += float(previous.double().square().sum())
        assert changed > 0
        networks[name] = {"changed_tensors": changed, "all_finite": True,
                          "relative_l2_parameter_change": math.sqrt(squared_change/squared_initial)}
    result = {"config_matches_frozen_plan": True, "motion_unchanged": True,
              "original_checkpoint_unchanged": True, "final_checkpoint_hash_matches": True,
              "stop_reason": run["stop_reason"], "additional_steps": run["additional_steps"],
              "update_counts": counts, "update_metrics_finite": True, "networks": networks}
    result["heading_input_migration"] = migration
    if migration["migrated"]:
        from heading_checkpoint import POLICY_WEIGHT
        columns = final["policy"][POLICY_WEIGHT][:, 680:]
        mode = final["config"]["env"]["heading_obs_mode"]
        result["heading_input_weights"] = {"mode": mode, "nonzero_elements": int(columns.count_nonzero()),
                                           "l2_norm": float(columns.norm())}
        if mode == "zero":
            assert not columns.count_nonzero(), "Zero-input control learned nonzero extra columns"
        else:
            assert columns.count_nonzero(), "True heading input never affected its new weights"
    with (training / "training_validation.json").open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
