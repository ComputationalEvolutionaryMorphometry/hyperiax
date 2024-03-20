from .executor import OrderedExecutor


class LevelwiseTreeExecutor(OrderedExecutor):
    """Executor for running one level on a tree at a time
    """
    def _determine_execution_order(self, tree):
        return list(tree.iter_levels())
    