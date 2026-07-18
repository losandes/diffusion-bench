from split_to_list import split_to_list


def test_split_with_commas():
  split = split_to_list(",")

  # Test that a string is split correctly
  assert split("apple, pear, orange") == ["apple", "pear", "orange"]

  # Test that a string with no delimiter returns a list with one element
  assert split("apple") == ["apple"]

  # Test that a string with multiple delimiters returns a list with empty elements removed
  assert split("apple, , pear, orange, ") == ["apple", "pear", "orange"]

  # Test that None is returned as an empty list
  assert split(None) == []

  # Test that an empty string is returned as an empty list
  assert split("") == []

def test_split_with_pipes():
  split = split_to_list("|")

  # Test that a string is split correctly
  assert split("apple | pear| orange") == ["apple", "pear", "orange"]

  # Test that a string with no delimiter returns a list with one element
  assert split("apple") == ["apple"]

  # Test that a string with multiple delimiters returns a list with empty elements removed
  assert split("apple| | pear| orange| ") == ["apple", "pear", "orange"]

  # Test that None is returned as an empty list
  assert split(None) == []

  # Test that an empty string is returned as an empty list
  assert split("") == []
