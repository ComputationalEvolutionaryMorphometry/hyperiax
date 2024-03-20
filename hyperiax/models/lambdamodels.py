from .updownmodel import UpModel, UpDownModel, DownModel
from .updatemodel import UpdateModel

class UpLambda(UpModel):
    """Lambda model that only contains an up interface

    Requires an up and fuse function.
    """
    def __init__(self, up_fn, fuse_fn) -> None:
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn

    def up(self, *args, **kwargs):
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        return self.fuse_fn(*args, **kwargs)
    
class UpDownLambda(UpDownModel):
    """Lambda model that only contains both an up and down interface

    Requires an up, fuse and down function.
    """
    def __init__(self, up_fn, fuse_fn, down_fn) -> None:
        super().__init__()

        self.up_fn = up_fn
        self.fuse_fn = fuse_fn
        self.down_fn = down_fn

    def up(self, *args, **kwargs):
        return self.up_fn(*args, **kwargs)
    
    def fuse(self, *args, **kwargs):
        return self.fuse_fn(*args, **kwargs)
    
    def down(self, *args, **kwargs):
        return self.down_fn(*args, **kwargs)
    
class DownLambda(DownModel):
    """Lambda model that only contains a down interface

    Requires a down function.
    """
    def __init__(self, down_fn) -> None:
        super().__init__()

        self.down_fn = down_fn
    
    def down(self, *args, **kwargs):
        return self.down_fn(*args, **kwargs)
    
class UpdateLambda(UpdateModel):
    """Lambda model that only contains a local update interface

    Requires an update function.
    """
    def __init__(self, update_fn) -> None:
        super().__init__()

        self.update_fn = update_fn

    def update(self, *args, **kwargs):
        return self.update_fn(*args, **kwargs)