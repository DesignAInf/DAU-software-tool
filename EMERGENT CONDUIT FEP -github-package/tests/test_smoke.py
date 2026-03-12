from emergent_conduit_fep import Config, simulate


def test_smoke_runs_short_simulation():
    cfg = Config(steps=8, frame_stride=2, n_electrons=24)
    designer_df, passive_df, frames = simulate(cfg)
    assert len(designer_df) == 8
    assert len(passive_df) == 8
    assert len(frames) >= 1
    assert "efe" in designer_df.columns
