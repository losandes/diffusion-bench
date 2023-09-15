def make_paths (template):
  def with_context (output_path, type, model_idx, count=1):
    """
    generates output paths for the files that will be saved

    Properties:
      output_path (str) - the path to the directory where files will be saved
      type (str) - the typeof model (GENERATOR|REFINER|UPSCALER)
      model_idx (int) - the index of the model in the models array
      count (int) - the number of images, per model, that will be produced

    Returns:
      [[string]] - an array of arrays of file paths where the outer array
        is count based, and each inner array is based on the model index
    """
    paths = []

    for i in range(count):
      path = template.replace(":path", output_path)
      path = path.replace(":type", type.lower())
      path = path.replace(":count_idx", (str(i + 1)).zfill(2))
      path = path.replace(":model_idx", (str(model_idx + 1)).zfill(2))
      paths.append(path)

    return paths
  return with_context
