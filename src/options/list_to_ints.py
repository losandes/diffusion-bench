def list_to_ints (list):
  """
  converts a list of strings to a list of arrays
  """
  output = []

  for _idx, item in enumerate(list):
    output.append(int(item))

  return output
