import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def initialize_figure(palette='bwr'):
  '''Initialize a subplots figure with some default values.'''
  sns.set_style('whitegrid', {'axis.grid': False})
  sns.set_palette(palette)
  fig, ax = plt.subplots()
  ax.axis('off')
  return fig, ax
  
def render_line(data, y_pos_pct, fig, ax, fontsize=16, name='monospace', alpha=0.6, 
  offset_factor=1.29):
  '''Draw a single line of text.'''
  
  cmap = ListedColormap(sns.color_palette())
  r = fig.canvas.get_renderer()
  
  fig_size_pix = fig.get_size_inches()[0] * fig.dpi
  num_tokens = len(data)
  x_pos_pct = 0.
  x_pos_pix = 0.
  idx = 0
  
  while (x_pos_pix < fig_size_pix) and (idx < num_tokens):
    props = dict(alpha=alpha, facecolor=cmap(data[0][1]), pad=0., lw=0.)
    t = ax.text(x_pos_pct, y_pos_pct, data[0][0] + ' ', bbox=props, horizontalalignment='left',
                fontsize=fontsize, verticalalignment='bottom', name=name)
    x_pos_pix += (offset_factor * t.get_window_extent(renderer=r).width)
    x_pos_pct = x_pos_pix / fig_size_pix
    data.pop(0)
    idx += 1
    
  return fig, ax, t, data

def render(tokens, scores, sigma=10, fontsize=16, name='monospace', alpha=0.6, offset_factor=1.29, 
  typical_wpl=10, palette='bwr'):
  '''Render highlighted text.
  Args:
    tokens: list of tokens to render
    scores: list or 1d-array of scores between 0. and 1., one per token
    sigma: sigmoid parameter for squishing scores towards 0/1. Higher values make the
      function more step-like.
    fontsize:  text font size
    name:  font family name
    alpha: text box transparency between 0. and 1.
    offset_factor: fudge factor for aligning text boxes.
    typical_wpl: desired (approximate) number of words per line
    palette: Seaborn or matplotlib color palette
  Returns:
    fig, ax: figure and axis handles
  '''

  fig, ax = initialize_figure(palette)

  # Transform the scores using sigmoid function.
  scores = 1. / (1. + np.exp( -2 * sigma * (scores - 0.5)))

  # Get the dimensions needed to align text.
  props = dict(facecolor='white', pad=0.0, lw=0.)
  t_tmp = ax.text(0., 0., 'typical?', bbox=props, fontsize=fontsize, name=name, alpha=0.)
  extent = t_tmp.get_window_extent(renderer=fig.canvas.get_renderer())
  
  # Set figure dimensions.
  fig_height_pix = 1. * extent.height * (1. * len(tokens) / typical_wpl)
  fig_width_pix = 1. * extent.width * typical_wpl
  fig.set_size_inches(fig_width_pix / fig.dpi, fig_height_pix / fig.dpi)

  # Plot text lines.
  data = zip(tokens, scores)
  num_tokens = len(tokens)
  line_num = 0
  while data:
    y_pos_pct = 1. - (offset_factor * line_num * extent.height / fig_height_pix)
    fig, ax, t, data = render_line(data, y_pos_pct, fig, ax, fontsize=fontsize,
      name=name, alpha=alpha, offset_factor=offset_factor)
    line_num += 1
    
  return fig, ax
