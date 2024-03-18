from hyperiax.tree.tree import HypTree
from .unorderedexecutor import UnorderedExecutor
import itertools

class RedBlackExecutor(UnorderedExecutor):
    def _determine_execution_pools(self, tree: HypTree):
        levels = tree.iter_levels()
        l1, l2 = [], []
        for level in levels:
            if len(l1) <= len(l2):
                l1 += level
            else:
                l2 += level
        return [l1, l2]
    
    def _iter_pools(self, pools):
        return itertools.chain(*pools)