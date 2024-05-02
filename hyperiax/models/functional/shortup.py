
def pass_up(*args):
    """ Returns a function that passes a set of keys on.

    """
    def _f(**kwargs):
        """ _F to pass_up

        :return: `pass_up('value', 'noise')` will give a function that given a dictionary, returns a new one containing only `value` and `noise`
        """
        return {k: kwargs[k] for k in args}
    return _f