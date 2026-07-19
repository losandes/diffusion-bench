from .app.txt2img import txt2img
from .app.img2img import img2img
from .app.vid2vid import vid2vid
from .options.get_args import parse_args, map_terminal_input
from .constants import GENERATOR, VID2VID

passes = map_terminal_input(parse_args())

for idx, one_pass in enumerate(passes):
  # VID2VID-typed models (e.g. AnimateDiff) always route to vid2vid; --naive
  # routes any img2img-capable model through the per-frame video path.
  if one_pass['model']['type'] == VID2VID or one_pass.get('naive'):
    vid2vid(one_pass, idx)
  elif one_pass['model']['type'] == GENERATOR:
    txt2img(one_pass, idx)
  else:
    img2img(one_pass, idx)
