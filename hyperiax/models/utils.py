from jax import ops

def map_reduction(reduction_string):
    map = {
        'sum': ops.segment_sum,
        'prod': ops.segment_prod,
        'max': ops.segment_max,
        'min': ops.segment_min
    }
    if reduction_string not in map.keys():
        raise ValueError(f"Reduction '{reduction_string}' not supported")
    return map[reduction_string]


def filter_keywords(keys):
    reserved_keywords = ['self', 'params', 'key', 'root_mask', 'leaf_mask']

    return [k for k in keys if k not in reserved_keywords]