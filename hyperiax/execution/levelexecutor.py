from .executor import OrderedExecutor


class LevelwiseTreeExecutor(OrderedExecutor):
    def _determine_execution_order(self, tree):
        return list(tree.iter_levels())
    