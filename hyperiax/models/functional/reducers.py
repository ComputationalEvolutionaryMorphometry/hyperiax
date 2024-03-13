

def sum_fuse_children(axis=-1):
    def _fuse(**kwargs):
        return {k[6:]:v.sum(axis=axis) for k,v in kwargs.items() if k.startswith('child_')}
    return _fuse

