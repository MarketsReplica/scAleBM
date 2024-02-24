import polars as pl

from .grid import Grid


def test_grid():
    width = 10
    grid = Grid.init_random(width=width)
    df = pl.DataFrame(data=dict(agent_id=range(10))).with_columns(grid.positions)

    flatten_positions = df.select(grid.all_positions)
    assert len(flatten_positions) == width * width

    reshaped_positions = df.select(grid.reshape_positions)
    assert len(reshaped_positions) == width

    empties = df.select(grid.empty_positions)
    assert len(empties) == width * width - width

    img = grid.as_image(df)
    assert img.shape == (width, width)
