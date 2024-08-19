from .updownmodel import UpReducer, DownModel, UpModel
from .updatemodel import UpdateReducer, UpdateModel

class UpLambdaReducer(UpReducer):
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
    
class UpdateLambdaReducer(UpdateReducer):
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
    

class UpdateLambda(UpdateModel):
    """Lambda model that only contains a local update interface

    :param UpdateModel: Requires an update function.
    """
    def __init__(self, update_fn) -> None:
        """Lambda model that only contains an up interface

        :param update_fn: function to update or do calculations from both parent and child nodes
        """
        self.update = update_fn

        super().__init__()
    

    def update(self, *args, **kwargs):
        """ update function to define calculations from up parameters
        :return: calculated values from child and parent parameters
        """
        raise ValueError('Model does not have a valid update function')
    
class UpLambda(UpModel):
    """Lambda model that only contains an upward interface
    """
    def __init__(self, up_fn) -> None:
        """Lambda model that only contains an up interface

        :param up_fn: function to take current node and children node values and proive values to se in current
        """
        self.up = up_fn

        super().__init__()
    

    def up(self, *args, **kwargs):
        """ Up function to calculate the upward action on the tree
        :return: calculated values from child and current values
        """
        raise ValueError('Model does not have a valid up function')