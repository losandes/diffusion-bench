def split_to_list (delimiter):
  """
  Returns a function that splits a string on the given
  delimiter and strips each value

  Usage:
    split_to_list(",")("apple, pear, orange")

  @curried
  """
  def split (maybe_string):
    """
    splits a string on a delimiter and strips each value

    Usage:
      split_to_list(",")("apple, pear, orange")
    """
    raw_output = []
    output = []

    if maybe_string is not None:
      raw_output = maybe_string.split(delimiter)

    for idx, item in enumerate(raw_output):
      output.append(item.strip())

    return output
  return split
