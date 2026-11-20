"""Compose: audio and video features, gesture vocabulary, score IR, planner, renderer."""

from syncsummoner.compose.features import (
    AudioFeatures,
    Features,
    Section,
    VideoFeatures,
    analyze,
    analyze_audio,
    analyze_frames,
)
from syncsummoner.compose.planner import Curve, plan_automation, reachable, search, thin
from syncsummoner.compose.master import common_duration, master_command, probe_duration
from syncsummoner.compose.render import RenderConfig, Rig, render_cuts, render_played
from syncsummoner.compose.score import GestureInstance, Layer, Score, control_rate
from syncsummoner.compose.vocabulary import GESTURES, Anchor, Automation, Gesture, GestureContext
from syncsummoner.compose import features, master, planner, render, score, vocabulary

__all__ = [
    "GESTURES",
    "Anchor",
    "AudioFeatures",
    "Automation",
    "Curve",
    "Features",
    "Gesture",
    "GestureContext",
    "GestureInstance",
    "Layer",
    "RenderConfig",
    "Rig",
    "Score",
    "Section",
    "VideoFeatures",
    "analyze",
    "analyze_audio",
    "analyze_frames",
    "common_duration",
    "render_cuts",
    "render_played",
    "control_rate",
    "features",
    "master",
    "master_command",
    "plan_automation",
    "planner",
    "probe_duration",
    "reachable",
    "render",
    "score",
    "search",
    "thin",
    "vocabulary",
]
