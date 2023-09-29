def maybe_update_path (funcs):
  def try_update (path):
    if path is None:
      return None
    else:
      updated = path

      for idx, func in enumerate(funcs):
        updated = func(updated)

      return updated
  return try_update

def add_name_to_path (model_name):
  def update (path):
    return path.replace(":name", model_name)
  return update

def add_seed_to_path (seed):
  def update (path):
    if seed is None:
      return path
    else:
      return path.replace(":seed", str(seed))
  return update

def add_dir_to_path (dir):
  def update (path):
    return path.replace(":dir_path", dir)
  return update

def add_count_to_path (idx):
  def update (path):
    return path.replace(":count_idx", (str(idx + 1)).zfill(2))
  return update

def add_model_idx_to_path (idx):
  def update (path):
    return path.replace(":model_idx", (str(idx + 1)).zfill(2))
  return update

def make_all_paths (template):
  def with_context (output_path, model_idx, count=1):
    """
    generates output paths for the files that will be saved

    Properties:
      output_path (str) - the path to the directory where files will be saved
      model_idx (int) - the index of the model in the models array
      count (int) - the number of images, per model, that will be produced

    Returns:
      [[string]] - an array of arrays of file paths where the outer array
        is count based, and each inner array is based on the model index
    """
    paths = []

    for i in range(count):
      paths.append(maybe_update_path([
        add_dir_to_path(output_path),
        add_count_to_path(i),
        add_model_idx_to_path(model_idx),
      ])(template))

    return paths
  return with_context

def make_empty_paths (count):
  """
  Makes an array of Nones so the array has appropriate indexes
  even through an item won't be saved
  """
  paths = []

  for _i in range(count):
    paths.append(None)

  return paths
