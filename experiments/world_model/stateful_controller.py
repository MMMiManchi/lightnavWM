"""Real event feedback and guarded STOP around the stateful LightNav Harness."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace

import numpy as np

from experiments.world_model.entrance_alignment import EntranceAligner, ImageDirectionTransport
from experiments.world_model.event_verification import QwenEventVerifier
from experiments.world_model.execution_feedback import ExecutionFeedback
from experiments.world_model.exit_crossing import ObservedExitCrossing
from experiments.world_model.online_input import online_info, online_observation
from experiments.world_model.recovery import RecoveryDecision, RecoveryManager
from experiments.world_model.route_memory import ObservedRouteJournal
from experiments.world_model.skill_contracts import GoalContract, SkillCommand
from experiments.world_model.stop_guard import is_implicit_stop
from experiments.world_model.task_completion import QwenTaskAwareProgressVerifier
from experiments.world_model.task_progress import (
    QwenTaskDecomposer,
    TaskProgress,
    TaskStage,
)
from experiments.world_model.tssm_queries import StatefulTSSMHarness
from experiments.world_model.visual_harness import QwenBranchProposer, QwenTaskPlanner


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v), allow_nan=False) + "\n")


class StatefulEpisodeController:
    def __init__(self, *, session, executor, vision, images, context, episode_id, output,
                 initial_rgb, planning_interval=12, verification_interval=6, observation_budget=8,
                 proposer=None, planner=None, progress_verifier=None, event_verifier=None,
                 unknown_stop_policy="defer", maximum_recovery_cycles=3, depth_scale=10.0,
                 exploration_overrides=False, option_max_steps=100, camera_hfov=120.0,
                 exit_guard_factory=ObservedExitCrossing, require_measured_progress=True,
                 stop_thresholds=None, prediction_control=None, route_memory=False, policy_stop_roundtrip=False):
        self.session, self.executor, self.images = session, executor, images
        self.route_journal = ObservedRouteJournal(session, output / "route_observations.json") if route_memory else None
        self.context, self.episode_id, self.output = context, episode_id, output
        self.output.mkdir(parents=True, exist_ok=False)
        self.initial = f"{episode_id}:view:0"
        images.put(self.initial, initial_rgb)
        decomposer = QwenTaskDecomposer(vision, require_verbatim=True)
        try:
            stages = decomposer(executor.original_instruction)
            error = None
        except (ValueError, TypeError, KeyError) as exc:
            stages = (TaskStage("stage-0", executor.original_instruction,
                "Actual target and requested stopping relation visibly established", "goal"),)
            error = str(exc)
        policy = executor.policy
        contract = {"dt_seconds": policy.dt, "lin_vel_range": policy.lin_vel_range, "ang_vel_range": policy.ang_vel_range}
        self.progress = TaskProgress(episode_id=episode_id, instruction=executor.original_instruction,
            stages=stages, initial_observation=self.initial, motion_contract=contract,
            require_current_target=GoalContract.from_instruction(stages[-1].instruction).require_current_target,
            require_measured_motion=require_measured_progress)
        self.decomposition = {"stages": [asdict(s) for s in stages], "raw": decomposer.last_raw, "error": error}
        self.progress_verifier = progress_verifier or QwenTaskAwareProgressVerifier(vision, images)
        self.event_verifier = event_verifier or QwenEventVerifier(vision, images)
        self.transport = ImageDirectionTransport(images, EntranceAligner(hfov_degrees=camera_hfov))
        self.harness = StatefulTSSMHarness(session=session, context=context, episode_id=episode_id,
            executor=executor, images=images, proposer=proposer or QwenBranchProposer(vision, images, hfov_deg=camera_hfov),
            planner=planner or QwenTaskPlanner(vision, images), progress_tracker=self.progress,
            alignment_provider=self.transport, exploration_overrides=exploration_overrides,
            prediction_control=prediction_control)
        self.planning_interval, self.verification_interval = planning_interval, verification_interval
        self.observation_budget, self.observations_used = observation_budget, 0
        self.current_reference, self.current_step = None, -1
        self.state_step = 0
        self.pending = None
        self.commands, self.planning_trace, self.progress_trace, self.event_trace, self.abandoned = [], [], [], [], []
        self.last_progress_step, self.last_event_step = -1, -1
        self.recovery = RecoveryManager(observation_budget=observation_budget, unknown_stop_policy=unknown_stop_policy,
                                        maximum_cycles=maximum_recovery_cycles)
        self.feedback = ExecutionFeedback(depth_scale=depth_scale, hfov_degrees=camera_hfov)
        self.execution_trace, self.skill_trace, self.recovery_trace = [], [], []
        self.command_version, self.option_max_steps = 0, option_max_steps
        self.backtracking, self.goal_verdict = None, "unknown"
        self.observed_poses = {}
        self.last_action_source = None
        self.looking = None
        self.policy_stop_roundtrip = policy_stop_roundtrip
        self.pending_policy_stop = None
        self.exit_guard_factory = exit_guard_factory
        self.require_measured_progress = bool(require_measured_progress)
        self.stop_thresholds = stop_thresholds or {"linear_m_per_s": .01, "angular_deg_per_s": .5}
        if not all(math.isfinite(float(self.stop_thresholds[key])) and self.stop_thresholds[key] > 0
                   for key in ("linear_m_per_s", "angular_deg_per_s")):
            raise ValueError("invalid physical velocity STOP thresholds")

    def observe(self, step, rgb, feature, columns, depth=None):
        reference = f"{self.episode_id}:view:{step}"
        self.harness.observe(reference, rgb, feature, columns)
        self.progress.observe(reference, step)
        if self.pending is not None:
            self.pending["tracker"].observe(reference, step)
        self.current_reference, self.current_step = reference, step
        self.transport.current_depth = depth
        if hasattr(self.progress_verifier, "set_depth"):
            self.progress_verifier.set_depth(depth, reference, self.feedback.depth_scale)
        receipt = self.commands[-1] if self.commands and self.commands[-1]["step"] == step-1 else None
        measured = self.feedback.observe(step, reference, rgb, depth, receipt)
        self.progress.record_observed_motion(reference, measured)
        if self.pending is not None:
            self.pending["tracker"].record_observed_motion(reference, measured)
        if self.pending is not None:
            self.pending["exit_guard"].observe(measured)
        self.observed_poses[reference] = measured
        if receipt is not None:
            self.execution_trace.append({**measured, "command_id": receipt.get("command_id"),
                                         "action_source": receipt.get("action_source")})
            if (self.pending is not None and receipt.get("command_id") == self.pending["command"].command_id and
                    receipt.get("action_source") == "Harness_align_selected_exit" and measured["measurement"]["valid"]):
                self.pending["alignment_remaining"] += receipt.get("issued_yaw_left_rad", 0)-measured["measurement"]["yaw_left_rad"]
            if (self.looking is not None and receipt.get("action_source") == "Harness_active_observation" and
                    measured["measurement"]["valid"]):
                self.looking["remaining"] += receipt.get("issued_yaw_left_rad", 0)-measured["measurement"]["yaw_left_rad"]
        self.recovery.observe_progress(self.motion_context(), self.progress.active_index)
        self.observations_used = self.recovery.total_observations
        if self.session.episode_id != self.episode_id:
            history, valid = self.harness.visual_history()
            self.session.begin_episode(self.context, self.episode_id, history, reference,
                                       instruction=self.executor.original_instruction, valid=valid)
        if self.route_journal is not None:
            self.route_journal.observe(reference, feature, measured)

    def _sync_state(self):
        if self.session.current_ref == self.current_reference:
            return
        commands = [c for c in self.commands if self.state_step <= c["step"] < self.current_step]
        if not commands:
            raise ValueError("new current view lacks actual execution acknowledgements")
        self.harness.report_observed_milestone(self.harness.features[-1], self.current_reference,
            execution_ref=str(self.output / "commands.jsonl"), executed=True)
        self.state_step = self.current_step

    def abandon(self, reason):
        if self.pending is not None:
            self.abandoned.append({"source": self.pending["source"], "at": self.current_reference, "reason": reason})
            self.harness.abandon_event(reason)
            self.skill_trace.append({"command": asdict(self.pending["command"]), "status": "aborted",
                                    "at": self.current_reference, "reason": reason})
            self.pending = None
        else:
            self.executor.restore_global_goal()

    def check_event(self, step):
        if self.pending is None or self.last_event_step == step:
            return False
        self.last_event_step = step
        tracker = self.pending["tracker"]
        commands = list(tracker.executions.values())
        # Commanded forward motion is only a prerequisite. Visual evidence is
        # still required; neither step counts nor issued metres prove arrival.
        forward = sum(max(0, r.get("commanded_motion", {}).get("forward_metres", 0)) for r in commands)
        tracker.execution_feedback = self.motion_context()
        crossing = self.pending["exit_guard"]
        if not crossing.eligible(step):
            self.event_trace.append({"step": step, "event_supported": False,
                "exit_crossing_guard": crossing.snapshot(),
                "reason": "measured_source_exit_crossing_required_before_visual_event"})
            return False
        if len(commands) < 3 or forward < 0.2 or tracker.execution_feedback["stalled"]:
            return False
        try:
            evidence = self.event_verifier(tracker, self.pending["branch"], self.current_reference)
            kind = getattr(self.event_verifier, "last_trace", {}).get("event_kind", "unknown")
            if evidence.completed and tracker.confirmation is not None and kind != self.pending["confirmation_kind"]:
                tracker.confirmation = None
            advanced = tracker.consider(evidence)
            self.pending["confirmation_kind"] = kind if evidence.completed else None
            trace = {"step": step, **getattr(self.event_verifier, "last_trace", {}),
                "exit_crossing_guard": crossing.snapshot(),
                "status": tracker.last_status, "event_supported": advanced,
                "support_kind": "temporal_visual_event_support_not_ground_truth"}
            if advanced:
                self._sync_state()
                branch = self.pending["branch"]
                result = self.harness.report_verified_event(self.harness.features[-1],
                    target_observation=self.current_reference, execution_ref=str(self.output / "commands.jsonl"),
                    executed=True, arrived=True, executed_direction=[math.sin(branch.bearing_rad), math.cos(branch.bearing_rad)])
                trace["writeback"] = result
                trace["supporting_views"] = tracker.completed[-1]["supporting_views"]
                trace["measured_motion_support"] = tracker.completed[-1]["measured_motion_support"]
                self.skill_trace.append({"command": asdict(self.pending["command"]), "status": "completed",
                                        "at": self.current_reference, "event_kind": kind})
                self.pending = None
                self.executor.restore_global_goal()
            self.event_trace.append(trace)
            return advanced
        except (ValueError, TypeError, KeyError) as exc:
            self.event_trace.append({"step": step, "error": str(exc), "event_supported": False})
            return False

    def check_progress(self, step):
        if self.progress.complete or self.last_progress_step == step:
            return
        self.last_progress_step = step
        try:
            if hasattr(self.progress_verifier, "set_execution_context"):
                self.progress_verifier.set_execution_context(self.motion_context())
            evidence = self.progress_verifier(self.progress, self.current_reference)
            self.goal_verdict = evidence.verdict if self.progress.active.kind == "goal" else "unknown"
            if self.decomposition["error"] is not None and self.goal_verdict == "reached":
                self.goal_verdict = "unknown"
            self.recovery.evidence(self.goal_verdict)
            if self.decomposition["error"] is not None:
                # The original task still drives LightNav, but a destination
                # image alone cannot certify an unverified multi-clause route.
                self.progress.last_status = "task_decomposition_unverified"
                self.progress_trace.append({"step": step, **getattr(self.progress_verifier, "last_trace", {}),
                    "status": self.progress.last_status, "advanced": False,
                    "reason": "original_instruction_retained_but_route_clauses_not_independently_verified"})
                return
            advanced = self.progress.consider(evidence)
            if advanced:
                # A verified stage starts a fresh recovery phase immediately,
                # including when it completes during an observation option.
                self.recovery.observe_progress(self.motion_context(), self.progress.active_index)
            trace = {"step": step, **getattr(self.progress_verifier, "last_trace", {}),
                "status": self.progress.last_status, "advanced": advanced}
            if advanced:
                trace["accepted_stage"] = self.progress.completed[-1]
            self.progress_trace.append(trace)
            if advanced:
                # Task progress can release local control without cancelling an
                # independent observation of the actually executed entrance.
                if self.pending is not None:
                    self.pending["monitor_only"] = True
                    self.skill_trace.append({"command": asdict(self.pending["command"]),
                        "status": "control_released_without_event_claim", "at": self.current_reference})
                self.executor.restore_global_goal()
        except (ValueError, TypeError, KeyError) as exc:
            self.progress_trace.append({"step": step, "error": str(exc), "advanced": False})

    def plan(self, step, *, force_exploration=False):
        if self.pending is not None:
            return None
        try:
            self._sync_state()
            previous = self.harness.exploration_overrides
            self.harness.exploration_overrides = previous or force_exploration
            try:
                result = self.harness.plan()
            finally:
                self.harness.exploration_overrides = previous
            self.planning_trace.append({"step": step, **result.trace, "alignment": self.transport.last_trace.copy()})
            self.transport.last_trace.clear()
            if self.harness.committed_candidate is not None:
                candidate = self.harness.committed_candidate
                ticket = self.harness.query.tickets[candidate]
                # The proposer already bound each candidate to this actual view.
                branch = next(b for b in self.harness.query.current_branches if b.candidate_id == candidate)
                local_instruction = f"Pass through the currently visible opening: {branch.description}. End this entrance option once on its far side."
                if result.goal is not None:
                    self.executor.commit_goal(replace(result.goal, instruction=local_instruction),
                        current_observation=self.current_reference, visible_sequence=ticket.visible_sequence)
                tracker = TaskProgress(episode_id=self.episode_id, instruction=local_instruction,
                    stages=[TaskStage("event", local_instruction, "Selected exit crossed; next room or decision visible", "goal")],
                    initial_observation=self.current_reference, motion_contract=self.progress.motion_contract, initial_step=step,
                    require_measured_motion=self.require_measured_progress)
                tracker.record_observed_motion(self.current_reference, self.observed_poses[self.current_reference])
                self.command_version += 1
                command = SkillCommand.bind(branch, version=self.command_version, step=step,
                    task_key=self.harness.router.last_decision.task_key, mode="active" if result.goal is not None else "monitor")
                self.pending = {"source": self.current_reference, "branch": branch, "tracker": tracker,
                    "confirmation_kind": None, "command": command, "alignment_remaining": branch.bearing_rad, "alignment_steps": 0,
                    "exit_guard": self.exit_guard_factory(branch, self.transport.current_depth,
                        self.observed_poses[self.current_reference], step=step, scale=self.feedback.depth_scale,
                        hfov=self.feedback.hfov)}
                self.skill_trace.append({"command": asdict(command), "status": "running",
                                        "exit_crossing_guard": self.pending["exit_guard"].snapshot()})
                source_depth = None
                if self.transport.current_depth is not None:
                    depth_file = self.output / f"source-depth-{command.command_id}.npy"
                    if depth_file.exists():
                        raise ValueError("source depth binding must be immutable")
                    np.save(depth_file, self.transport.current_depth)
                    source_depth = {"file": depth_file.name, "sha256": hashlib.sha256(depth_file.read_bytes()).hexdigest(),
                                    "metres_per_depth_unit": self.feedback.depth_scale, "hfov_degrees": self.feedback.hfov}
                save_json(self.output / f"query-{step:04d}.json", {"source": ticket.source_observation,
                    "cutoff": ticket.visible_sequence, "query_kind": ticket.evidence_kind,
                    "world_model_called": ticket.world_model_called, "past_sequences": ticket.memory_sequences,
                    "retrieved_sequences": [m.sequence for m in self.session.query_actual_memories(ticket)],
                    "used_evidence_sequences": result.trace.get("used_evidence_sequences", []),
                    "control_source": result.trace.get("control_source"),
                    "prediction_control": result.trace.get("prediction_control"),
                    "prediction_saved_before_execution": True, "source_actual_depth": source_depth})
                np.save(self.output / f"query-{step:04d}.npy", ticket.future_feature)
            return result
        except (ValueError, TypeError, KeyError) as exc:
            self.abandon("planning_failed")
            self.planning_trace.append({"step": step, "error": str(exc)})
            return None

    def motion_context(self):
        issued = list(self.progress.executions.values())[-12:]
        forward = sum(max(0, r.get("commanded_motion", {}).get("forward_metres", 0)) for r in issued)
        context = self.feedback.context(commanded_forward=forward)
        context["active_skill"] = asdict(self.pending["command"]) if self.pending else None
        context["option_is_passive_monitor"] = bool(self.pending and
            (self.pending["command"].mode == "monitor" or self.pending.get("monitor_only")))
        return context

    def _target_bearing(self):
        identities = getattr(self.progress_verifier, "identities", {})
        if not identities:
            return None
        ref = next(reversed(identities))
        box = identities[ref].get("box")
        if box is None:
            return None
        bearing = math.atan((.5-(box[0]+box[2])/2000)*2*math.tan(math.radians(self.feedback.hfov)/2))
        old = self.observed_poses.get(ref, {})
        if (self.feedback.pose_valid and old.get("local_pose") is not None and
                old["pose_epoch"] == self.feedback.epoch):
            bearing += old["local_pose"][2]-self.feedback.pose[2]
        elif ref != self.current_reference:
            return None
        return math.atan2(math.sin(bearing), math.cos(bearing))

    def _result(self, action, trace):
        policy = self.executor.policy
        def implicit(candidate):
            return is_implicit_stop(candidate, linear_range=policy.lin_vel_range,
                                    angular_range=policy.ang_vel_range, thresholds=self.stop_thresholds)

        if action is not None and not trace.get("stop_dispatched", False) and implicit(action):
            # A near-zero velocity command is a STOP in Habitat, even when it
            # came from an observation or a non-STOP model waypoint. Route it
            # through recovery instead of letting it bypass the global gate.
            trace["implicit_stop_intercepted"] = True
            trace["implicit_stop_original_action"] = action
            self.abandon("implicit_zero_velocity_stop_intercepted")
            decision = self.recovery.observation(reason="implicit_stop_requires_distinct_view")
            if decision is None:
                self.looking = None
                trace["end_reason"] = "implicit_stop_blocked_without_observation_budget"
                action = None
            else:
                self.observations_used = self.recovery.total_observations
                self.looking = {"remaining": decision.angle, "started": self.current_step, "reason": decision.reason}
                replacement = self._motion(turn=max(-.3, min(.3, decision.angle)))
                if implicit(replacement):
                    self.looking = None
                    trace["end_reason"] = "no_non_stopping_observation_possible"
                    action = None
                else:
                    trace.update(action_source="Harness_active_observation", recovery_action="implicit_stop_observe",
                                 observation_number=self.observations_used, recovery_reason=decision.reason)
                    action = replacement
        self.last_action_source = trace.get("action_source", trace.get("stop_source"))
        trace["command_id"] = self.pending["command"].command_id if self.pending else None
        trace["command_version"] = self.pending["command"].version if self.pending else None
        trace["goal_verdict"] = self.goal_verdict
        trace["semantic_arrival_confirmed"] = self.progress.complete
        trace["recovery"] = self.recovery.snapshot()
        return action, trace

    def _motion(self, forward=0.0, turn=0.0):
        from lightnav.velocity import first_waypoint_to_velocity_cmd

        policy = self.executor.policy
        return first_waypoint_to_velocity_cmd(np.array([forward*policy.dt, 0., turn]),
            policy.dt, policy.lin_vel_range, policy.ang_vel_range)

    def _look(self, observation, info, trace, *, already_observed=False, angle=None, preserve_local=False,
              decision=None):
        decision = decision or self.recovery.observation(reason="inspect_current_ambiguity", target_bearing=self._target_bearing())
        if decision is None:
            return None
        self.observations_used = self.recovery.total_observations
        if not preserve_local:
            self.abandon("insert_actual_observation")
        if not already_observed:
            self.executor.policy.observe(online_observation(observation), online_info(info))
        trace.update(action_source="Harness_active_observation", observation_number=self.observations_used,
                     recovery_reason=decision.reason)
        requested = decision.angle if angle is None else max(-math.pi/6, min(math.pi/6, angle))
        if angle is not None:
            self.recovery.scan_offset += requested-decision.angle
        self.looking = {"remaining": requested, "started": self.current_step, "reason": decision.reason}
        return self._motion(turn=max(-.3, min(.3, requested)))

    def _continue_look(self, observation, info, trace):
        if self.looking is None:
            return None
        if abs(self.looking["remaining"]) <= .07 or self.current_step-self.looking["started"] >= 24:
            self.recovery_trace.append({"step": self.current_step, "action": "observation_option_finished",
                "remaining_yaw_rad": self.looking["remaining"], "reason": self.looking["reason"]})
            self.looking = None
            self.check_event(self.current_step)
            self.check_progress(self.current_step)
            return None
        self.executor.policy.observe(online_observation(observation), online_info(info))
        trace.update(action_source="Harness_active_observation", observation_number=self.observations_used,
                     recovery_reason=self.looking["reason"], remaining_yaw_rad=self.looking["remaining"])
        return self._motion(turn=max(-.3, min(.3, self.looking["remaining"])))

    def _finish_policy_stop(self, observation, info, trace):
        action = self._continue_look(observation, info, trace)
        if action is not None:
            return action
        pending = self.pending_policy_stop
        if pending["phase"] == "inspect":
            pending["phase"] = "return"
            before, now = pending["pose"], self.feedback.context()
            if (before.get("local_pose") is not None and now.get("local_pose") is not None and
                    before["pose_epoch"] == now["pose_epoch"]):
                turn = before["local_pose"][2] - now["local_pose"][2]
                trace["policy_stop_return_yaw_source"] = "actual_rgbd_pnp"
            else:
                turn = -sum(c["issued_yaw_left_rad"] for c in self.commands[pending["commands_start"]:])
                trace["policy_stop_return_yaw_source"] = "issued_estimate_without_arrival_claim"
            turn = math.atan2(math.sin(turn), math.cos(turn))
            if abs(turn) > .07:
                self.looking = {"remaining": turn, "started": self.current_step, "reason": "return_to_original_policy_STOP_heading"}
                self.executor.policy.observe(online_observation(observation), online_info(info))
                trace.update(action_source="Harness_active_observation", recovery_reason=self.looking["reason"])
                return self._motion(turn=max(-.3, min(.3, turn)))
        self.pending_policy_stop = None
        if self.goal_verdict == "not_reached":
            trace.update(end_reason="original_STOP_contradicted_after_roundtrip",
                         policy_stop_rejected_by_new_evidence=True)
            return None
        self.executor.policy.observe(online_observation(observation), online_info(info))
        trace.update(stop_dispatched=True, stop_source="LightNav_original_STOP_after_roundtrip",
                     policy_STOP_deferred=not self.progress.complete, policy_stop_latched=True)
        return self.executor.policy.stop_action()

    def _orient(self, observation, info, trace, *, already_observed=False):
        if self.pending is None or self.pending["command"].mode != "active" or self.pending.get("monitor_only"):
            return None
        remaining = self.pending["alignment_remaining"]
        if abs(remaining) <= .12 or self.pending["alignment_steps"] >= 8:
            return None
        if not already_observed:
            self.executor.policy.observe(online_observation(observation), online_info(info))
        self.pending["alignment_steps"] += 1
        trace.update(action_source="Harness_align_selected_exit",
                     command_id=self.pending["command"].command_id, remaining_bearing_rad=remaining,
                     alignment_evidence="RGB-D yaw when available; otherwise issued-control estimate")
        return self._motion(turn=max(-.3, min(.3, remaining)))

    def _start_backtrack(self):
        anchor = self.feedback.backtrack_anchor()
        if anchor is None:
            return False
        self.backtracking = {**anchor, "started": self.current_step}
        return True

    def _backtrack(self, observation, info, trace, *, already_observed=False):
        if self.backtracking is None:
            return None
        target = self.backtracking
        if (not self.feedback.pose_valid or target["epoch"] != self.feedback.epoch or
                self.current_step-target["started"] > 60):
            self.recovery_trace.append({"step": self.current_step, "action": "backtrack_unavailable",
                                        "reason": "lost_local_odometry_or_recovery_timeout"})
            self.backtracking = None
            return None
        delta = np.asarray(target["xy"])-self.feedback.pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance < .25:
            self.recovery_trace.append({"step": self.current_step, "action": "backtrack_completed",
                                        "actual_past_reference": target["reference"]})
            self.backtracking = None
            self.executor.restore_global_goal()
            return None
        yaw = math.atan2(-delta[0], delta[1])-self.feedback.pose[2]
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        clearance = self.motion_context()["clearance_m"]
        if abs(yaw) <= .25 and (clearance is None or clearance < .45):
            self.recovery_trace.append({"step": self.current_step, "action": "backtrack_unavailable",
                                        "reason": "current_depth_does_not_support_forward_clearance"})
            self.backtracking = None
            return None
        if not already_observed:
            self.executor.policy.observe(online_observation(observation), online_info(info))
        trace.update(action_source="Harness_backtrack_to_observed_anchor",
                     actual_past_reference=target["reference"], estimated_local_distance=distance)
        return self._motion(forward=.15 if abs(yaw) <= .25 else 0., turn=max(-.2, min(.2, yaw)))

    def _recover(self, decision, step, observation, info, trace, *, already_observed=True):
        self.recovery_trace.append({"step": step, **asdict(decision), "goal_verdict": self.goal_verdict})
        trace["recovery_action"] = decision.action
        if decision.action == "defer_stop":
            self.abandon("unverified_original_policy_STOP")
            trace.update(stop_dispatched=True, stop_source="LightNav_global_STOP_unverified_after_recovery",
                         semantic_arrival_confirmed=False, policy_STOP_deferred=True)
            return self.executor.policy.stop_action()
        if decision.action == "abstain":
            trace.update(end_reason="unverified_STOP_no_supported_continuation", recovery_reason=decision.reason)
            return None
        if decision.action in {"observe", "await_confirmation"}:
            action = self._look(observation, info, trace, already_observed=already_observed,
                                decision=decision if decision.action == "observe" else None,
                                preserve_local=bool(self.pending and self.pending["tracker"].confirmation is not None))
            if action is None:
                trace.update(end_reason="insufficient_distinct_evidence_after_recovery", recovery_reason=decision.reason)
            return action
        self.abandon("insert_"+decision.action)
        if decision.action == "backtrack" and self._start_backtrack():
            action = self._backtrack(observation, info, trace, already_observed=already_observed)
            if action is not None:
                return action
        self.harness.report_execution("Recovery: "+decision.reason+"; goal verdict="+self.goal_verdict+
                                      ". Retry only a currently grounded exit; old intentions are not arrival evidence.")
        planned = self.plan(step, force_exploration=self.goal_verdict == "not_reached" or self.motion_context()["stalled"])
        if not already_observed:
            self.executor.policy.observe(online_observation(observation), online_info(info))
        if planned is not None and planned.action == "observe":
            return self._look(observation, info, trace, already_observed=True, angle=planned.observation_turn_rad)
        action = self._orient(observation, info, trace, already_observed=True)
        if action is None:
            action, _ = self.executor.replan_observed_frame()
            if action is not None:
                trace["action_source"] = "LightNav_after_relocalization"
        if action is None:
            action = self._look(observation, info, trace, already_observed=True)
        if action is None:
            trace.update(end_reason="no_supported_recovery_action", recovery_reason=decision.reason)
        return action

    def next_action(self, step, observation, info):
        trace = {"step": step, "source_observation": self.current_reference}
        if step and step % self.verification_interval == 0:
            self._sync_state()
            self.check_event(step)
            self.check_progress(step)
        if self.pending_policy_stop is not None:
            return self._result(self._finish_policy_stop(observation, info, trace), trace)
        if self.progress.accept_global_stop():
            self.looking = self.backtracking = None
            self.executor.policy.observe(online_observation(observation), online_info(info))
            trace.update(stop_dispatched=True, stop_source="Harness_verified_task_progress")
            return self._result(self.executor.policy.stop_action(), trace)
        action = self._continue_look(observation, info, trace)
        if action is not None:
            return self._result(action, trace)
        if self.progress.accept_global_stop():
            self.executor.policy.observe(online_observation(observation), online_info(info))
            trace.update(stop_dispatched=True, stop_source="Harness_verified_task_progress")
            return self._result(self.executor.policy.stop_action(), trace)
        if (self.pending is not None and self.pending["command"].mode == "monitor" and
                not self.pending["exit_guard"].snapshot().get("motion_chain_valid", False) and
                step-self.pending["command"].issued_step >= self.planning_interval and
                step % self.planning_interval == 0):
            self.abandon("unmeasurable_exit_hypothesis_refresh_at_new_view")
        if self.pending is not None and step-self.pending["command"].issued_step >= self.option_max_steps:
            self.abandon("option_execution_budget_exhausted_without_verified_event")
        if (self.motion_context()["stalled"] and self.pending is not None and
                self.pending["command"].mode == "active" and not self.pending.get("monitor_only")):
            action = self._recover(RecoveryDecision("backtrack", "measured_motion_stalled"), step, observation, info,
                                   trace, already_observed=False)
            return self._result(action, trace)
        action = self._backtrack(observation, info, trace)
        if action is not None:
            return self._result(action, trace)
        if step % self.planning_interval == 0:
            planned = self.plan(step)
            if planned is not None and planned.action == "observe":
                action = self._look(observation, info, trace, angle=planned.observation_turn_rad)
                if action is not None:
                    return self._result(action, trace)
                # Budget exhaustion changes the strategy; it does not mark this task failed.
                self.abandon("uncertain_memory_returned_to_original_policy")
                trace["recovery_action"] = "resume_original_policy"
        action = self._orient(observation, info, trace)
        if action is not None:
            return self._result(action, trace)
        action, details = self.executor.propose_action(online_observation(observation), online_info(info))
        trace["rvq_codes"] = details.get("rvq_codes")
        if action is not None:
            trace["action_source"] = "LightNav_local" if self.executor.active_goal is not None else "LightNav_current_task"
            return self._result(action, trace)
        if self.executor.last_tool_status != "stop_proposed":
            trace["end_reason"] = self.executor.last_tool_status
            return self._result(None, trace)
        trace["stop_proposal_source"] = "LightNav_local" if self.executor.active_goal is not None else "LightNav_original_task"
        if self.executor.active_goal is not None:
            trace["local_stop_intercepted"] = True
            self.check_event(step)
            self.check_progress(step)
            if self.progress.accept_global_stop():
                trace.update(stop_dispatched=True, stop_source="LightNav_STOP_verified_by_Harness")
                return self._result(self.executor.policy.stop_action(), trace)
            if self.pending is not None and self.pending["tracker"].confirmation is not None:
                action = self._look(observation, info, trace, already_observed=True, preserve_local=True)
                if action is not None:
                    return self._result(action, trace)
            self.abandon("unverified_local_STOP")
            action, _ = self.executor.replan_observed_frame()
            if action is not None:
                trace["action_source"] = "LightNav_original_task_after_local_STOP"
                return self._result(action, trace)
            if self.executor.last_tool_status != "stop_proposed":
                trace["end_reason"] = self.executor.last_tool_status
                return self._result(None, trace)
        self.check_event(step)
        self.check_progress(step)
        if self.progress.accept_global_stop():
            trace.update(stop_dispatched=True, stop_source="LightNav_STOP_verified_by_Harness")
            return self._result(self.executor.policy.stop_action(), trace)
        trace["global_stop_intercepted"] = True
        if self.policy_stop_roundtrip and self.recovery.unknown_stop_policy == "defer" and self.goal_verdict == "unknown":
            self.pending_policy_stop = {"phase": "inspect", "pose": self.feedback.context(),
                                        "commands_start": len(self.commands)}
            action = self._look(observation, info, trace, already_observed=True, angle=math.pi/6)
            if action is None:
                self.pending_policy_stop["phase"] = "return"
                self.pending_policy_stop = None
                trace.update(stop_dispatched=True, stop_source="LightNav_original_STOP_without_observation_budget",
                             policy_STOP_deferred=True, policy_stop_latched=True)
                return self._result(self.executor.policy.stop_action(), trace)
            trace["policy_stop_latched"] = True
            return self._result(action, trace)
        if (self.progress.confirmation is not None and
                self.recovery.phase_observations < self.recovery.observation_budget):
            decision = RecoveryDecision("await_confirmation", "current task stage has one supported view")
        else:
            decision = self.recovery.on_global_stop(verdict=self.goal_verdict, target_bearing=self._target_bearing(),
                                                    can_backtrack=self.feedback.backtrack_anchor() is not None)
        action = self._recover(decision, step, observation, info, trace)
        return self._result(action, trace)

    def executed(self, step, action, *, rgb_changed, command_id=None, action_source=None):
        if step != self.current_step:
            raise ValueError("execution receipt does not match the actual current view")
        current_id = self.pending["command"].command_id if self.pending else None
        matched = command_id is None or command_id == current_id
        if not matched:
            self.abandon("execution_receipt_from_superseded_command")
        record = {"step": step, "action": action, "rgb_changed": bool(rgb_changed), "environment_step_returned": True,
            "action_source": action_source or self.last_action_source, "matched_current_command": matched,
            "command_id": command_id or (self.pending["command"].command_id if self.pending else None),
            "command_version": self.pending["command"].version if self.pending else None}
        self.progress.executed(step, environment_step_returned=True, rgb_changed=rgb_changed, action=action)
        record["issued_yaw_left_rad"] = math.radians(self.progress.executions[step].get("commanded_motion", {}).get("yaw_degrees_left_positive", 0))
        if self.pending is not None and self.last_action_source == "Harness_align_selected_exit":
            self.pending["alignment_remaining"] -= record["issued_yaw_left_rad"]
        if self.looking is not None and self.last_action_source == "Harness_active_observation":
            self.looking["remaining"] -= record["issued_yaw_left_rad"]
        if self.pending is not None:
            self.pending["tracker"].executed(step, environment_step_returned=True, rgb_changed=rgb_changed, action=action)
        self.commands.append(record)
        with (self.output / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def persist(self):
        if self.route_journal is not None:
            save_json(self.output / "route_observations.json", self.route_journal.records)
        for name, value in [("decomposition", self.decomposition), ("planning", self.planning_trace),
                            ("progress", self.progress_trace), ("events", self.event_trace), ("abandoned", self.abandoned),
                            ("execution_feedback", self.execution_trace), ("skills", self.skill_trace), ("recovery", self.recovery_trace)]:
            save_json(self.output / f"{name}.json", value)
        (self.output / "progress_state.json").write_text(self.progress.context() + "\n")

    def finish(self, reason):
        self.abandon(reason)
        self.session.end_episode()
        self.persist()
