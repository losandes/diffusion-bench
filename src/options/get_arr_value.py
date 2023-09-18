def get_value_or_first (arr, idx=0):
  """
  Gets the value from an array for this idx or the first idx.

  NOTE: Why is this necessary? Multiple prompts may be passed.
  Negative prompts might be None. There may be a negative prompt,
  per-pipeline. There may be a single prompot or negative prompt
  to use for all indexes.

  Args:
    arr {array} - the array to evaluate
    idx {[int]=0} - the zero-based index of the parent loop iteration

  Returns:
    string|None - the value in the array at this index, otherwise
                  the value in the array at index 0, otherwise None
  """
  val = None

  if (len(arr) >= (idx + 1)):
    val = arr[idx]
  elif (len(arr) > 0):
    val = arr[0]

  return val

def get_value_or_none (arr, idx=0):
  """
  Gets the value from an array for this idx, otherwise None.

  NOTE: Why is this necessary? There may or may not be a
  seed for this idx and we prefer to generate a random
  seed when one isn't specified.

  Args:
    arr {array} - the array to evaluate
    idx {[int]=0} - the zero-based index of the parent loop iteration

  Returns:
    string|None - the value in the array at this idx, otherwise None
  """
  if (len(arr) >= (idx + 1)):
    return arr[idx]
  else:
    return None
