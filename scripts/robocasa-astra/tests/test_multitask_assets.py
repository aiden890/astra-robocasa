"""Check new containers can resolve retained asset links without changing live mounts."""

from astra_ops.assets import activate as assets


def test_new_container_keeps_source_asset_links_read_only(monkeypatch):
    """Expose the old source path so absolute symlink targets resolve inside the new mount."""
    commands = []
    staged = []
    monkeypatch.setattr(
        assets, "stage_worker", lambda name, seed: staged.append((name, seed)) or True
    )
    monkeypatch.setattr(assets, "remote", lambda args: commands.append(args))
    assets.create("test-isolated-container", 2, "12g")
    command = commands[0]
    assert assets.ASSETS + "/assets:/opt/robocasa/robocasa/models/assets:ro" in command
    assert assets.OLD_ASSETS + ":" + assets.OLD_ASSETS + ":ro" in command
    assert command[command.index("--name") + 1] == "test-isolated-container"
    assert command[command.index("--network") + 1] == "none"
    assert staged == [("test-isolated-container", 0)]
