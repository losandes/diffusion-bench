def maybe_str_to_list (input):
  """
  Converts a string to a list if the value is a string,
  otherwise returns the input unmodified

  Usage:
    str_to_list("apple")
    # returns ["apple"]

    str_to_list(["apple", "pear", "orange"])
    # returns ["apple", "pear", "orange"]

    str_to_list(None)
    # returns None
  """

  if input is not None and type(input).__name__ == 'str':
    return [input]
  else:
    return input
