from .updownmodel import UpModel, UpDownModel, DownModel
from .updatemodel import UpdateModel

class UpLambda(UpModel):
    """Lambda model that only contains an up interface

    :param UpModel: Requires an up and fuse function.
    """
    def __init__(self, up_fn, fuse_fn) -> None:
        """Lambda model that only contains an up interface

        :param up_fn: param intputs from node to fuse_fn
        :param fuse_fn: function to fuse or do calculations form up parameters
        """
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn

    def up(self, *args, **kwargs):
        """ Up function to define values to fuse function
        :return: input arguments to fuse function
        """
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        """ Fuse function to define calculations from up parameters
        :return: calculated values from up parameters to parent node
        """
        return self.fuse_fn(*args, **kwargs)
    
class UpDownLambda(UpDownModel):
    """Lambda model that only contains both an up and down interface.    Requires an up, fuse and down function.

    :param UpDownModel: Requires an up, fuse and down function.
    """
    def __init__(self, up_fn, fuse_fn, down_fn) -> None:
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn
        self.down_fn = down_fn

    def up(self, *args, **kwargs):
        """ Up function to define values to fuse function
        :return: input arguments to fuse function
        """
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        """ Fuse function to define calculations from up parameters
        :return: calculated values from up parameters to parent node
        """
        return self.fuse_fn(*args, **kwargs)
    
    def down(self, *args, **kwargs):
        """ Down function to define values to fuse function
        :return: input arguments to fuse function
        """
        return self.down_fn(*args, **kwargs)
    
class DownLambda(DownModel):
    """ Lambda model that only contains a down interface

    :param DownModel  Requires a down function.
    """

    def __init__(self, down_fn) -> None:
        """Initialize DownLambda model

        :param down_fn: function to define values to fuse function
        """
        super().__init__()

        self.down_fn = down_fn
    
    def down(self, *args, **kwargs):
        """ down function to define values to fuse function
        :return: input arguments to fuse function
        """
        return self.down_fn(*args, **kwargs)
    
class UpdateLambda(UpdateModel):
    """L ambda model that only contains a local update interface

    :param UpdateModel: Requires an update function.
    """
    def __init__(self, update_fn) -> None:
        """Initialize DownLambda model

        :param down_fn: function to define values to fuse function
        """
        super().__init__()

        self.update_fn = update_fn

    def update(self, *args, **kwargs):
        """ Update function to define values to fuse function

        :return: input arguments to fuse function
        """
        return self.update_fn(*args, **kwargs)