"""Astra as a RoboCasa365-style VLA: one 16-step chunk of native 12-D actions per model call (open loop).

A variant of :mod:`robocasa_astra.astra_robodawn` (discrete skills) that changes only the model's output.
Scenes, step budgets, annotated images, memory, success conditions, the few-shot layout, the model caller
and all recording are imported from ``astra_robodawn`` unchanged, so the two can be compared run for run.
"""
