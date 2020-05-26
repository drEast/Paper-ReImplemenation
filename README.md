# Paper_Re-Implementation

## Deep Learning

### A Neural Algorithm of Artistic Style (Gatys, L.) - [Paper](https://arxiv.org/pdf/1508.06576.pdf)
Repaint a content image with the style of another image:

<img src="StyleTransfer/pyramids.jpg" height="150"> + <img src="StyleTransfer/night.jpg" height="150"> = <img src="StyleTransfer/starry_pyramids.jpg" height="150">

<img src="StyleTransfer/sphinx.jpg" height="150"> + <img src="StyleTransfer/ship.jpg" height="150"> = <img src="StyleTransfer/wrecked_sphinx.jpg" height="150">

Note: Clipping image pixels to 0-1 causes bright red and yellow clusters. Maybe have a look there.
