from str_to_list import maybe_str_to_list


def test_maybe_str_to_list():
  # Test that a string is converted to a list
  assert maybe_str_to_list("apple") == ["apple"]

  # Test that a list is returned unmodified
  assert maybe_str_to_list(["apple", "pear", "orange"]) == ["apple", "pear", "orange"]

  # Test that None is returned unmodified
  assert maybe_str_to_list(None) == None

  # Test that a non-string input is returned unmodified
  assert maybe_str_to_list(123) == 123
