"""CPU checks for the future-only representation entry. They do not train."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.world_model.future_only import (
    ACTION_CONTROL_RELATIVE_BAND,
    D_MODEL,
    FEATURE_DIM,
    GRID,
    HEADS,
    LAYERS,
    MAX_PARAMETERS,
    NAVIGATION_CONNECTED,
    POSE_LOSS,
    TOKENS,
    FutureGridDecoder,
    action_control_verdict,
    assert_split,
    assign_scene_folds,
    batch_plan,
    build_memory,
    candidate_lists,
    conv_flatten_tokens,
    decoder_flops,
    executed_action_features,
    fit_channel_scale,
    future_loss,
    parameter_count,
    parse_shard_spec,
    residual_injection_verdict,
    retrieval_evaluation,
    shard_slots,
    siglip_grid_report,
    siglip_grid_side,
    spatial_descriptor,
    stage_a_verdict,
    synthetic_routing_check,
    token_coordinate,
    token_index,
    uses_memory_coordinates,
)


def test_siglip_contract_is_a_27_grid_without_a_class_token():
    assert siglip_grid_side(384, 14) == 27
    assert GRID * GRID == TOKENS == 729
    assert FEATURE_DIM == 1152
    assert NAVIGATION_CONNECTED is False
    assert POSE_LOSS is False


def test_cached_order_matches_siglip_conv_flatten():
    grid = torch.arange(GRID * GRID * 4, dtype=torch.float32).reshape(GRID, GRID, 4)
    tokens = conv_flatten_tokens(grid)
    assert tokens.shape == (TOKENS, 4)
    for row in range(GRID):
        for col in range(GRID):
            index = token_index(row, col)
            assert token_coordinate(index) == (row, col)
            assert torch.equal(tokens[index], grid[row, col])
    assert token_index(1, 0) == 27
    assert token_index(0, 1) == 1


def test_decoder_queries_use_the_same_row_major_order():
    model = FutureGridDecoder("grid", use_action=False)
    assert int(model.rows[0]) == 0 and int(model.cols[0]) == 0
    assert int(model.rows[1]) == 0 and int(model.cols[1]) == 1
    assert int(model.rows[27]) == 1 and int(model.cols[27]) == 0


def test_arms_reject_each_others_inputs():
    patches = torch.zeros(2, TOKENS, FEATURE_DIM)
    pooler = torch.zeros(2, FEATURE_DIM)
    assert build_memory("pooler", patches=None, pooler=pooler).shape == (2, 1, FEATURE_DIM)
    assert build_memory("mean", patches=patches, pooler=None).shape == (2, 1, FEATURE_DIM)
    assert build_memory("grid", patches=patches, pooler=None).shape == (2, TOKENS, FEATURE_DIM)
    with pytest.raises(ValueError, match="cannot read patch"):
        build_memory("pooler", patches=patches, pooler=pooler)
    with pytest.raises(ValueError, match="cannot read the official pooler"):
        build_memory("mean", patches=patches, pooler=pooler)
    with pytest.raises(ValueError, match="cannot read the official pooler"):
        build_memory("grid", patches=patches, pooler=pooler)


def test_grid_order_changes_the_prediction_and_the_mean_discards_it():
    torch.manual_seed(0)
    patches = torch.randn(2, TOKENS, FEATURE_DIM)
    control = torch.zeros(2, 3)
    grid = FutureGridDecoder("grid", use_action=True)
    mean = FutureGridDecoder("mean", use_action=True)
    permutation = torch.randperm(TOKENS)
    grid_memory = build_memory("grid", patches=patches, pooler=None)
    grid_permuted = build_memory("grid", patches=patches[:, permutation], pooler=None)
    assert not torch.allclose(grid(grid_memory, control), grid(grid_permuted, control))
    mean_memory = build_memory("mean", patches=patches, pooler=None)
    mean_permuted = build_memory("mean", patches=patches[:, permutation], pooler=None)
    assert torch.allclose(mean(mean_memory, control), mean(mean_permuted, control), rtol=1e-5, atol=1e-5)


def test_no_action_control_matches_parameter_count_and_ignores_the_command():
    torch.manual_seed(1)
    with_action = FutureGridDecoder("pooler", use_action=True)
    torch.manual_seed(1)
    no_action = FutureGridDecoder("pooler", use_action=False)
    action_count = sum(parameter.numel() for parameter in with_action.parameters())
    no_action_count = sum(parameter.numel() for parameter in no_action.parameters())
    assert action_count == no_action_count
    memory = torch.randn(2, 1, FEATURE_DIM)
    first = no_action(memory, torch.randn(2, 3))
    second = no_action(memory, torch.zeros(2, 3))
    assert torch.equal(first, second)
    assert not torch.allclose(with_action(memory, torch.zeros(2, 3)), with_action(memory, torch.ones(2, 3)))


def test_each_arm_accepts_only_its_memory_length():
    control = torch.zeros(2, 3)
    pooled = FutureGridDecoder("pooler", use_action=False)
    spatial = FutureGridDecoder("grid", use_action=False)
    assert pooled(torch.randn(2, 1, FEATURE_DIM), control).shape == (2, TOKENS, FEATURE_DIM)
    assert spatial(torch.randn(2, TOKENS, FEATURE_DIM), control).shape == (2, TOKENS, FEATURE_DIM)
    with pytest.raises(ValueError, match="pooler memory"):
        pooled(torch.randn(2, TOKENS, FEATURE_DIM), control)
    with pytest.raises(ValueError, match="grid memory"):
        spatial(torch.randn(2, 1, FEATURE_DIM), control)


def test_future_loss_is_position_mse_and_has_no_pose_term():
    predicted = torch.zeros(2, TOKENS, FEATURE_DIM)
    target = torch.ones(2, TOKENS, FEATURE_DIM)
    assert future_loss(predicted, target).item() == 1.0
    assert future_loss.__code__.co_varnames[: future_loss.__code__.co_argcount] == ("predicted", "target")


def test_descriptor_keeps_the_block_that_contains_the_token():
    tokens = np.zeros((TOKENS, FEATURE_DIM), dtype=np.float32)
    tokens[token_index(0, 0), 0] = 1.0
    tokens[token_index(26, 26), 1] = 1.0
    descriptor = spatial_descriptor(tokens)
    assert descriptor.shape == (9 * FEATURE_DIM,)
    blocks = descriptor.reshape(9, FEATURE_DIM)
    assert abs(blocks[0, 0]) > abs(blocks[8, 0])
    assert abs(blocks[8, 1]) > abs(blocks[0, 1])


def test_split_rejects_shared_scenes_routes_and_transitions():
    fit = [{"transition_id": "a", "scene": "s1", "route_group_id": "r1"}]
    development = [{"transition_id": "b", "scene": "s2", "route_group_id": "r2"}]
    counts = assert_split(fit, development)
    assert counts["fit_transitions"] == 1
    assert counts["development_transitions"] == 1
    with pytest.raises(ValueError, match="scene"):
        assert_split(fit, [{"transition_id": "c", "scene": "s1", "route_group_id": "r9"}])
    with pytest.raises(ValueError, match="route"):
        assert_split(fit, [{"transition_id": "c", "scene": "s9", "route_group_id": "r1"}])
    with pytest.raises(ValueError, match="both"):
        assert_split(fit, [{"transition_id": "a", "scene": "s9", "route_group_id": "r9"}])


def _meta(step: int, digest: str, transition: str) -> dict:
    return {
        "episode_uid": "e",
        "t": step,
        "transition_id": transition,
        "rgb_pixel_sha256": digest,
        "scene": "s",
        "route_group_id": "r",
    }


def test_retrieval_shares_candidates_and_dedupes_raw_rgb_hashes():
    meta = [
        _meta(0, "h1", "a"),
        _meta(1, "h1", "b"),
        _meta(2, "h2", "c"),
        _meta(3, "h3", "d"),
        _meta(40, "h4", "e"),
    ]
    prepared = candidate_lists(meta)
    assert prepared["indices"][4] == [0, 2, 3]
    assert prepared["duplicates_removed"] == 1
    assert prepared["eligible"] == 1
    assert prepared["candidate_lag"] == 5
    assert prepared["dedup"] == "admitted_rgb_pixel_sha256"
    assert prepared["development_is_blind_test"] is False
    assert prepared["eligible"] + prepared["excluded_before_dedup"] + prepared["excluded_after_dedup"] == 5
    bank = np.eye(5, dtype=np.float64)
    scored = retrieval_evaluation({"current": bank, "future_oracle": bank}, bank, meta)
    assert scored["methods"]["current"]["recall_at_2"] == 1.0
    assert scored["methods"]["future_oracle"]["mrr_oracle_top1"] == 1.0
    assert scored["methods"]["current"]["n"] == scored["methods"]["random"]["n"] == 1
    short = [_meta(step, f"h{step}", f"t{step}") for step in range(4)]
    assert candidate_lists(short)["eligible"] == 0


def test_registered_decoder_is_shared_and_within_the_parameter_cap():
    assert D_MODEL == 128 and HEADS == 4 and LAYERS == 2
    counts = {arm: parameter_count(arm, use_action=True) for arm in ("pooler", "mean", "grid")}
    assert len(set(counts.values())) == 1
    assert counts["grid"] <= MAX_PARAMETERS
    assert counts["grid"] == parameter_count("grid", use_action=False)
    model = FutureGridDecoder("grid", use_action=True)
    assert model.action_injection == "memory_token"
    assert len(model.blocks) == 2
    assert uses_memory_coordinates("grid") is True
    assert uses_memory_coordinates("pooler") is False
    assert uses_memory_coordinates("mean") is False
    assert decoder_flops("grid") > decoder_flops("pooler")
    assert decoder_flops("pooler") == decoder_flops("mean")
    film_counts = {
        arm: parameter_count(arm, use_action=True, action_injection="query_film")
        for arm in ("pooler", "mean", "grid")
    }
    assert len(set(film_counts.values())) == 1
    assert counts["grid"] < film_counts["grid"] <= MAX_PARAMETERS
    assert film_counts["grid"] == parameter_count("grid", use_action=False, action_injection="query_film")


def test_query_film_starts_unmodulated_and_ignores_a_hidden_command():
    torch.manual_seed(0)
    memory = torch.randn(2, TOKENS, FEATURE_DIM)
    stopped = torch.zeros(2, 3)
    moving = torch.tensor([[1.5, 0.0, 0.1], [0.2, 1.0, 0.1]])
    model = FutureGridDecoder("grid", use_action=True, action_injection="query_film")
    assert not hasattr(model, "action_proj")
    assert torch.count_nonzero(model.film.weight) == 0
    assert torch.count_nonzero(model.film.bias) == 0
    assert torch.equal(model(memory, stopped), model(memory, moving))
    with torch.no_grad():
        model.film.weight.copy_(torch.ones_like(model.film.weight) * 0.01)
    assert not torch.allclose(model(memory, stopped), model(memory, moving))
    blinded = FutureGridDecoder("grid", use_action=False, action_injection="query_film")
    with torch.no_grad():
        blinded.film.weight.copy_(torch.ones_like(blinded.film.weight))
    assert torch.equal(blinded(memory, stopped), blinded(memory, moving))
    with pytest.raises(ValueError, match="action injection"):
        FutureGridDecoder("grid", use_action=True, action_injection="concat")


def test_query_film_weight_learns_only_from_a_visible_command():
    torch.manual_seed(0)
    memory = torch.randn(1, TOKENS, FEATURE_DIM)
    target = torch.randn(1, TOKENS, FEATURE_DIM)
    command = torch.tensor([[1.0, 0.5, 0.1]])
    visible = FutureGridDecoder("grid", use_action=True, action_injection="query_film")
    future_loss(visible(memory, command), target).backward()
    assert visible.film.weight.grad is not None
    assert int(torch.count_nonzero(visible.film.weight.grad)) > 0
    blinded = FutureGridDecoder("grid", use_action=False, action_injection="query_film")
    future_loss(blinded(memory, command), target).backward()
    assert blinded.film.weight.grad is not None
    assert int(torch.count_nonzero(blinded.film.weight.grad)) == 0


def test_action_features_match_the_executed_command_and_the_zero_control():
    control = torch.tensor([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]])
    features = executed_action_features(control, use_action=True)
    assert features.shape == (2, 6)
    assert features[0, 5].item() == 1.0
    assert features[1, 5].item() == 0.0
    assert features[1, 0].item() == pytest.approx(1.0 / 2.5)
    assert features[1, 3].item() == pytest.approx(0.1 / 0.25)
    assert torch.equal(executed_action_features(control, use_action=False), torch.zeros(2, 6))


def test_siglip_grid_records_the_uncovered_margin():
    report = siglip_grid_report(384, 14)
    assert report["grid"] == 27
    assert report["tokens"] == 729
    assert report["uncovered_pixels_after_last_patch"] == 6
    assert report["class_token"] is False
    assert report["coordinate_space"] == "resized image plane, not world meters"


def test_scene_folds_keep_routes_inside_one_scene():
    scenes = [
        "5LpN3gDmAk7",
        "7y3sRwLe3Va",
        "82sE5b5pLXE",
        "EDJbREhghzL",
        "VzqfbhrpDEA",
        "cV4RVeZvu5T",
        "jh4fc5c5qoQ",
        "kEZ7cmS4wCh",
    ]
    rows = [
        {"scene": scene, "route_group_id": f"{scene}:{index}", "transition_id": f"{scene}-{index}"}
        for scene in scenes
        for index in range(2)
    ]
    sealed = assign_scene_folds(rows)
    assert sealed["blocked"] is False
    assert sealed["random_frame_split"] is False
    assert sealed["folds"][0]["validation_scenes"] == ["5LpN3gDmAk7", "EDJbREhghzL", "jh4fc5c5qoQ"]
    assert sealed["folds"][2]["validation_scenes"] == ["82sE5b5pLXE", "cV4RVeZvu5T"]
    for fold in sealed["folds"]:
        assert fold["scene_overlap"] == []
        assert fold["route_overlap"] == []
    rows.append({"scene": "other", "route_group_id": "5LpN3gDmAk7:0", "transition_id": "x"})
    with pytest.raises(ValueError, match="blocked"):
        assign_scene_folds(rows)


def test_batch_plan_is_truncated_and_repeats_for_the_same_seed():
    first = batch_plan(300, 128, 3, 20261029)
    second = batch_plan(300, 128, 3, 20261029)
    assert len(first) == 3
    assert all(len(batch) == 128 for batch in first)
    assert set(first[0]).isdisjoint(first[1])
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[2], second[2])


def test_channel_scale_uses_a_floor_and_refuses_nonfinite_tokens():
    future = np.zeros((2, TOKENS, FEATURE_DIM), dtype=np.float32)
    future[0, :, 0] = 5.0
    future[1, :, 0] = -5.0
    scale = fit_channel_scale(future)
    assert scale.shape == (FEATURE_DIM,)
    assert scale[0] == pytest.approx(5.0)
    assert scale[1] == pytest.approx(0.01)
    future[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_channel_scale(future)


def test_pooler_backward_does_not_read_patches():
    patches = torch.randn(1, TOKENS, FEATURE_DIM, requires_grad=True)
    pooler = torch.randn(1, FEATURE_DIM, requires_grad=True)
    memory = build_memory("pooler", patches=None, pooler=pooler)
    predicted = FutureGridDecoder("pooler", use_action=False)(memory, torch.zeros(1, 3))
    predicted.sum().backward()
    assert pooler.grad is not None
    assert patches.grad is None


def test_synthetic_grid_distinguishes_a_move_the_mean_cannot_represent():
    result = synthetic_routing_check()
    assert result["mean_outputs_tied"] is True
    assert result["grid_outputs_differ"] is True
    assert result["tied_output_loss_floor"] == 1.0


def _verdict_row(arm: str, seed: int, ratio: float, recall_delta: float, mrr_delta: float) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "ratio_to_copy": ratio,
        "recall_delta": recall_delta,
        "mrr_delta": mrr_delta,
    }


def _control_pair(no_action_mse: float, no_action_change: float) -> tuple[list[dict], list[dict]]:
    action = []
    no_action = []
    for arm in ("pooler", "mean", "grid"):
        for seed in (20261029, 20261030, 20261031):
            action.append(
                {"arm": arm, "seed": seed, "scene_macro_mse": 2.0, "change_vector_error": 1000.0}
            )
            no_action.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "scene_macro_mse": no_action_mse,
                    "change_vector_error": no_action_change,
                }
            )
    return action, no_action


def test_action_control_requires_every_seed_outside_the_one_percent_band():
    assert ACTION_CONTROL_RELATIVE_BAND == 0.01
    action, tied = _control_pair(2.0, 1000.0)
    within = action_control_verdict(action, tied)
    assert within["arms"]["grid"]["mse_call"] == "within_band"
    assert within["grid_action_adds_future_grid_information"] is False
    _action, inside = _control_pair(2.01, 1005.0)
    assert action_control_verdict(action, inside)["arms"]["grid"]["mse_call"] == "within_band"
    _action, higher = _control_pair(2.03, 1020.0)
    added = action_control_verdict(action, higher)
    assert added["arms"]["grid"]["mse_call"] == "action_lower"
    assert added["grid_action_adds_future_grid_information"] is True
    assert added["retrieval_recomputed"] is False
    assert added["evidence_grade"] == "不足"
    assert added["stage_b_started"] is False
    _action, lower = _control_pair(1.9, 900.0)
    assert action_control_verdict(action, lower)["arms"]["grid"]["mse_call"] == "no_action_lower"
    mixed = [dict(row) for row in higher]
    mixed[0]["scene_macro_mse"] = 1.9
    with pytest.raises(ValueError, match="three arms"):
        action_control_verdict(action, higher[:-1])
    assert action_control_verdict(action, mixed)["arms"]["pooler"]["mse_call"] == "mixed"
    broken = [dict(row) for row in higher]
    broken[0]["scene_macro_mse"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        action_control_verdict(action, broken)


def test_shard_slots_cover_every_job_once():
    assert parse_shard_spec("0/6") == (0, 6)
    with pytest.raises(ValueError, match="shard spec"):
        parse_shard_spec("0-6")
    with pytest.raises(ValueError, match="outside"):
        parse_shard_spec("6/6")
    for job_count in (27, 9):
        assigned = [shard_slots(job_count, shard, 6) for shard in range(6)]
        assert sorted(index for slots in assigned for index in slots) == list(range(job_count))
        assert all(set(left).isdisjoint(right) for left, right in zip(assigned, assigned[1:]))


def test_verdict_keeps_eight_scenes_from_counting_as_a_blind_result():
    seeds = (20261029, 20261030, 20261031)
    met = []
    for seed in seeds:
        met.extend(
            [
                _verdict_row("grid", seed, 0.8, 0.03, 0.01),
                _verdict_row("pooler", seed, 0.99, 0.0, 0.0),
                _verdict_row("mean", seed, 0.97, 0.0, 0.0),
            ]
        )
    verdict = stage_a_verdict(met)
    assert verdict["representation_target_met"] is True
    assert verdict["retrieval_target_met"] is True
    assert verdict["evidence_grade"] == "不足"
    assert verdict["development_is_blind_test"] is False
    assert verdict["stage_b_started"] is False
    missed = [
        _verdict_row("grid", seeds[0], 1.1, 0.03, 0.01) if row["arm"] == "grid" and row["seed"] == seeds[0] else row
        for row in met
    ]
    assert stage_a_verdict(missed)["representation_target_met"] is False


def _residual_rows(action_mse: float, no_action_mse: float) -> list[dict]:
    seeds = (20261029, 20261030, 20261031)
    rows = []
    for seed in seeds:
        rows.append({"seed": seed, "use_action": True, "scene_macro_residual_mse": action_mse})
        rows.append({"seed": seed, "use_action": False, "scene_macro_residual_mse": no_action_mse})
    return rows


def test_residual_injection_uses_the_same_one_percent_band():
    tied = residual_injection_verdict(_residual_rows(0.87, 0.87))
    assert tied["call"] == "within_band"
    assert tied["action_lowers_residual_on_every_seed"] is False
    assert tied["action_injection"] == "query_film"
    assert tied["navigation_connected"] is False
    assert tied["habitat_started"] is False
    added = residual_injection_verdict(_residual_rows(0.80, 0.82))
    assert added["call"] == "action_lower"
    assert added["action_lowers_residual_on_every_seed"] is True
    mixed = _residual_rows(0.80, 0.82)
    mixed[1]["scene_macro_residual_mse"] = 0.79
    assert residual_injection_verdict(mixed)["call"] == "mixed"
    with pytest.raises(ValueError, match="three registered seeds"):
        residual_injection_verdict(_residual_rows(0.80, 0.82)[:-1])
