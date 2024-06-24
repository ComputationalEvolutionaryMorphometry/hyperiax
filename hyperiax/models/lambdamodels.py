from .updownmodel import UpModel, DownModel, FuseModel
from .updatemodel import UpdateModel

class UpLambda(UpModel):
    """ Lambda model that only contains an up interface

    :param UpModel: Requires an up and fuse function.
    """
    def __init__(self, up_fn, transform_fn, reductions) -> None:
        """Lambda model that only contains an up interface

        :param up_fn: param intputs from node to transform_fn
        :param transform_fn: function to transform or do calculations form up parameters
        """

        self.up = up_fn
        self.transform = transform_fn

        super().__init__(reductions=reductions)

    def up(self, *args, **kwargs):
        """ Up function to define values to fuse function
        :return: input arguments to fuse function
        """
        raise ValueError('Model does not have a valid up function')
    
    def transform(self, *args, **kwargs):
        """ Fuse function to define calculations from up parameters
        :return: calculated values from up parameters to parent node
        """
        raise ValueError('Model does not have a valid fuse function')
    
class DownLambda(DownModel):
    """ Lambda model that only contains a down interface

    :param DownModel  Requires a down function.
    """

    def __init__(self, down_fn) -> None:
        """ Initialize DownLambda model

        :param down_fn: function to define values to fuse function
        """
        self.down = down_fn # need to do this to carry the argspec
    
        super().__init__()

    def down(self, *args, **kwargs):
        """ Down function to define values to fuse function
        :return: input arguments to fuse function
        """
        raise ValueError('Model does not have a valid down function')
    
class UpdateLambda(UpdateModel):
    """Lambda model that only contains a local update interface

    :param UpdateModel: Requires an update function.
    """
    def __init__(self, up_fn, update_fn, reductions) -> None:
        """Lambda model that only contains an up interface

        :param up_fn: param intputs from node to update_fn
        :param update_fn: function to update or do calculations form up parameters
        """
        self.up = up_fn
        self.update = update_fn

        super().__init__(reductions=reductions)

    def up(self, *args, **kwargs):
        """ Up function to define values to fuse function
        :return: input arguments to fuse function
        """
        raise ValueError('Model does not have a valid up function')
    
    def update(self, *args, **kwargs):
        """ Fuse function to define calculations from up parameters
        :return: calculated values from up parameters to parent node
        """
        raise ValueError('Model does not have a valid fuse function')
    
class FuseLambda(FuseModel):
    """ Lambda model that only contains a fuse interface

    :param FuseModel: Requires a fuse function.
    """
    def __init__(self, fuse_fn) -> None:
        """ Initialize FuseLambda model

        :param fuse_fn: function to define values to fuse function
        """

        self.fuse = fuse_fn
        super().__init__()

    def fuse(self, *args, **kwargs):
        """ Fuse function to define values to fuse function
        :return: input arguments to fuse function
        """
        raise ValueError('Model does not have a valid fuse function')