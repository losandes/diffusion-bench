from app.txt2img import txt2img
from app.img2img import img2img
from options.get_args import parse_args, map_terminal_input
from constants import GENERATOR

passes = map_terminal_input(parse_args())

for idx, one_pass in enumerate(passes):
  if one_pass['model']['type'] == GENERATOR:
    txt2img(one_pass)
  else:
    img2img(one_pass)
