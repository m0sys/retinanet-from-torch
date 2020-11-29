from collections import namedtuple


class ShapeSpec(namedtuple("__ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    Simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    @classmethod
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)
