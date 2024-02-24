import uuid

import polars as pl


class Grid:
    """
    Lazy expression of a grid data structure

    # TODO for now only a squared grid, but can be easily adpated to a more generic "panel form"
    """

    def __init__(self, width: int, positions: pl.Expr):
        self.positions = positions
        self.width = width

    @classmethod
    def init_random(cls, width: int):
        position_col = f"grid_{str(uuid.uuid4())[:8]}"
        positions = (
            pl.int_range(width * width).sample(pl.len(), with_replacement=False).alias(position_col)
        )
        return cls(width, positions)

    @property
    def all_positions(self):
        return pl.int_range(0, self.width * self.width).alias(
            f"{self.width}x{self.width}_all_positions"
        )

    @property
    def reshape_positions(self):
        """
        Reshape positions for them to fit in the original context.
        For example, position 13 on a 5x5 grid will be asigned a struct of the form {grid_x=2, grid_y=3}
        """
        x_name = self.positions.meta.output_name() + "_x"
        y_name = self.positions.meta.output_name() + "_y"
        return (self.positions // self.width).alias(x_name), (self.positions % self.width).alias(
            y_name
        )

    @property
    def empty_positions(self):
        empty_name = f"{self.positions.meta.output_name()}_empty_positions"
        return (
            self.all_positions.implode()
            .list.set_difference(self.positions.implode())
            .explode()
            .alias(empty_name)
        )

    def as_image(self, df: pl.DataFrame):
        assert (
            self.positions.meta.output_name() in df.columns
        ), "position column must be present in `df`"
        df = df.select(self.all_positions.is_in(self.positions).cast(pl.Int8))
        return df.to_numpy().reshape((self.width, self.width))
