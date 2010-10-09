import numpy as np
import matplotlib.pyplot as mplt

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis'
__email__ = 'lucas@tuebingen.mpg.de'

def sample(img, patch_size, sample_size):
	"""
	Generates a random sample of image patches from an image.
	"""

	# uniformly sample patch locations
	xpos = np.floor(np.random.uniform(0, img.shape[0] - patch_size[0] + 1, sample_size))
	ypos = np.floor(np.random.uniform(0, img.shape[1] - patch_size[1] + 1, sample_size))

	# collect sample patches
	samples = []
	for i in range(sample_size):
		samples.append(img[xpos[i]:xpos[i] + patch_size[0], ypos[i]:ypos[i] + patch_size[1]])

	return np.array(samples)



def show(samples, num_rows=None, num_cols=None, num_patches=None, line_width=1, margin=20):
	"""
	Displays a sample of image patches.
	"""

	# process and check parameters
	samples = np.array(samples)

	if not num_patches:
		num_patches = samples.shape[0]
	if not num_rows and not num_cols:
		num_cols = np.ceil(np.sqrt(num_patches))
		num_rows = np.ceil(num_patches / num_cols)
	elif not num_rows:
		num_rows = np.ceil(num_patches / num_cols)
	elif not num_cols:
		num_cols = np.ceil(num_patches / num_rows)

	num_patches = min(min(num_patches, samples.shape[0]), num_rows * num_cols)
	num_rows = int(num_rows)
	num_cols = int(num_cols)

	patch_size = samples.shape[1:3]

	# normalize patches
	smin = float(samples.min())
	smax = float(samples.max())
	samples = (samples - smin) / (smax - smin)

	# allocate memory
	if len(samples.shape) > 3:
		patchwork = np.zeros((
				num_rows * patch_size[0] + (num_rows + 1) * line_width,
				num_cols * patch_size[1] + (num_cols + 1) * line_width, 3))
	else:
		patchwork = np.zeros((
				num_rows * patch_size[0] + (num_rows + 1) * line_width,
				num_cols * patch_size[1] + (num_cols + 1) * line_width))

	# stitch patches together
	for i in range(num_patches):
		r = i / num_cols
		c = i % num_cols

		r_off = r * patch_size[0] + (r + 1) * line_width
		c_off = c * patch_size[1] + (c + 1) * line_width

		patchwork[r_off:r_off + patch_size[0], c_off:c_off + patch_size[1], ...] = samples[i]

	# display patches
	h = mplt.imshow(patchwork, cmap='gray', interpolation='nearest', aspect='equal')

	xmargin = float(margin) / (patchwork.shape[1] + 2 * margin + 1)
	ymargin = float(margin) / (patchwork.shape[0] + 2 * margin + 1)
	xwidth = 1 - 2 * xmargin
	ywidth = 1 - 2 * ymargin

	# make sure that 1 pixel is actually represented by 1 pixel
	dpi = h.figure.get_dpi()
	h.figure.set_figwidth(patchwork.shape[0] / (ywidth * dpi))
	h.figure.set_figheight(patchwork.shape[1] / (xwidth * dpi))
	h.figure.canvas.resize(patchwork.shape[1] + 2 * margin + 1, patchwork.shape[0] + 2 * margin + 1)

	h.axes.set_position([xmargin, ymargin, xwidth, ywidth])
	h.axes.set_xlim(-1, patchwork.shape[1])
	h.axes.set_ylim(patchwork.shape[0], -1)

	mplt.axis('off')
	mplt.draw()

	return h
