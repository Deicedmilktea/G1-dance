"""Zero-extend a 680-goal checkpoint and its Adam moments for two heading inputs."""

import copy

POLICY_WEIGHT = "encoders.motion.mlp.net.0.weight"
CRITIC_WEIGHT = "mlp.net.0.weight"


def insert_zero_columns(value, offset, extra):
    import torch
    assert value.ndim == 2 and 0 <= offset <= value.shape[1]
    zeros = value.new_zeros((value.shape[0], extra))
    return torch.cat((value[:, :offset], zeros, value[:, offset:]), dim=1)


def migrate_optimizer(optimizer, state, weight_name, offset, extra):
    """Keep the parameter ordering/step count; add zero moments only to new columns."""
    import torch
    names = [name for name in state if name.endswith((".weight", ".bias"))]
    ids = [value for group in optimizer["param_groups"] for value in group["params"]]
    assert len(names) == len(ids), "Unsupported optimizer parameter ordering"
    parameter_id = ids[names.index(weight_name)]
    result = {**optimizer, "state": dict(optimizer["state"]),
              "param_groups": copy.deepcopy(optimizer["param_groups"])}
    if parameter_id not in optimizer["state"]:
        return result, []
    previous = optimizer["state"][parameter_id]
    updated, expanded = dict(previous), []
    for name, value in previous.items():
        if isinstance(value, torch.Tensor) and value.ndim:
            assert value.shape == state[weight_name].shape, (weight_name, name, value.shape)
            updated[name] = insert_zero_columns(value, offset, extra)
            assert torch.equal(updated[name][:, :offset], value[:, :offset])
            assert torch.equal(updated[name][:, offset+extra:], value[:, offset:])
            expanded.append(name)
    result["state"][parameter_id] = updated
    return result, expanded


def migrate_checkpoint(checkpoint, target_goal_dim):
    """Return a new dictionary; leave the caller's checkpoint and its tensors intact."""
    import torch
    previous_dimension = checkpoint["policy"]["encoders.motion.scale"].numel()
    if previous_dimension == target_goal_dim:
        return checkpoint, {"migrated": False, "goal_dim": target_goal_dim}
    assert previous_dimension == 680 and target_goal_dim == 682, "Only the audited 680 -> 682 interface is supported"
    extra = target_goal_dim - previous_dimension
    result = dict(checkpoint)
    report = {"migrated": True, "old_goal_dim": previous_dimension, "new_goal_dim": target_goal_dim,
              "new_weight_columns_zero": True, "original_weight_slices_exact": True,
              "optimizer_step_counts_preserved": True, "optimizer_moments": {}}
    for network, weight, scale, offset in (
        ("policy", POLICY_WEIGHT, "encoders.motion.scale", previous_dimension),
        ("critic", CRITIC_WEIGHT, "goal_scale", 71+previous_dimension),
    ):
        state = checkpoint[network]
        assert state[scale].numel() == previous_dimension
        expected_input = previous_dimension if network == "policy" else 71+previous_dimension+33+4
        assert state[weight].shape[1] == expected_input
        migrated = dict(state)
        migrated[weight] = insert_zero_columns(state[weight], offset, extra)
        migrated[scale] = torch.cat((state[scale], state[scale].new_ones(extra)))
        assert torch.equal(migrated[weight][:, :offset], state[weight][:, :offset])
        assert torch.equal(migrated[weight][:, offset+extra:], state[weight][:, offset:])
        assert not migrated[weight][:, offset:offset+extra].count_nonzero()
        result[network] = migrated
        optimizer_key = "policy_opt" if network == "policy" else "critic_opt"
        result[optimizer_key], expanded = migrate_optimizer(checkpoint[optimizer_key], state, weight, offset, extra)
        report["optimizer_moments"][network] = expanded
    return result, report
