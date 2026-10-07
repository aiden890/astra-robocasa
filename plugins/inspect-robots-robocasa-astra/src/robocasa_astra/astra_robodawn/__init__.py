"""RoboDawn-style discrete-command control of RoboCasa PandaOmron by a Codex subscription model.

Host-side modules (commands, prompts, memory, codex, loop, cost, run) need only NumPy and PIL.
Simulator-side modules (executor, views, recorder, sim_server) import robosuite and run inside the
simulator subprocess.
"""
