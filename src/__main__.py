from app.txt2img import txt2img
from app.img2img import img2img
from options.get_args import get_args
from constants import GENERATOR

ARGS = get_args()

for idx, model in enumerate(ARGS['MODELS']):
  if model[0] == GENERATOR:
    txt2img(ARGS, idx)
  else:
    img2img(ARGS, idx)
