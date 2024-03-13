
def pass_up(*args):
    def _f(**kwargs):
        return {k: kwargs[k] for k in args}
    return _f