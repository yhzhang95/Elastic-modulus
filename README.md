# Elastic-modulus


### Initiation

	POINTS = 2000
	DPI = 600
	CVoigt[1, 1:7] = [153, -4.9, 1.9, 0, 0, 0]
	CVoigt[2, 2:7] = [267, 1.42, 0, 0, 0]
	CVoigt[3, 3:7] = [6.4, 0, 0, 0]
	CVoigt[4, 4:7] = [2.34, 0, 0]
	CVoigt[5, 5:7] = [2.35, 0]
	CVoigt[6, 6] = 28

`POINTS: How many points will be drawn.`

`DPI: The DPI of output figure.`

`Cvoigt: 7X7 matrix, but actually only (1-6, 1-6) will work.`

### Color Setting
	Young_color = (55 / 255., 126 / 255., 184 / 255.)
	LC_color = (50 / 255., 177 / 255., 101 / 255.)
	shear_color_positive = (76 / 255., 114 / 255., 176 / 255.)
	shear_color_negative = (228 / 255., 26 / 255., 28 / 255.)
	Poisson_color_positive = (76 / 255., 114 / 255., 176 / 255.)
	Poisson_color_negative = (228 / 255., 26 / 255., 28 / 255.)

`color in RGB mode or RGBA mode, you can rewrite into (R,G,B,A) like (55 / 255., 126 / 255., 184 / 255., 0.5)`

### Example

###### *Youngs modulus*

<img src="https://github.com/RJAtouT/Elastic-modulus/blob/master/Elastic%20modulus/Youngs_modulus.png" width="50%" height="50%">

###### *Shear modulus*

<img src="https://github.com/RJAtouT/Elastic-modulus/blob/master/Elastic%20modulus/Shear_modulus.png" width="50%" height="50%">

###### *Poissons ratio*

<img src="https://github.com/RJAtouT/Elastic-modulus/blob/master/Elastic%20modulus/Poissons_ratio.png" width="50%" height="50%">

###### *Poissons ratio nega*

<img src="https://github.com/RJAtouT/Elastic-modulus/blob/master/Elastic%20modulus/Poissons_ratio_nega.png" width="50%" height="50%">

###### *Linear compressibility*

<img src="https://github.com/RJAtouT/Elastic-modulus/blob/master/Elastic%20modulus/Linear_compressibility.png" width="50%" height="50%">
