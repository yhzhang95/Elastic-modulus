import numpy as np
from numpy import cos, sin, array, pi
from matplotlib import pyplot as plt
from matplotlib import ticker as tik

# Initialize
CVoigt = np.zeros((7, 7))

# INPUT!
POINTS = 2000
DPI = 600
CVoigt[1, 1:7] = [153, -4.9, 1.9, 0, 0, 0]
CVoigt[2, 2:7] = [267, 1.42, 0, 0, 0]
CVoigt[3, 3:7] = [6.4, 0, 0, 0]
CVoigt[4, 4:7] = [2.34, 0, 0]
CVoigt[5, 5:7] = [2.35, 0]
CVoigt[6, 6] = 28

Young_color = (55 / 255., 126 / 255., 184 / 255.)
LC_color = (50 / 255., 177 / 255., 101 / 255.)
shear_color_positive = (76 / 255., 114 / 255., 176 / 255.)
shear_color_negative = (228 / 255., 26 / 255., 28 / 255.)
Poisson_color_positive = (76 / 255., 114 / 255., 176 / 255.)
Poisson_color_negative = (228 / 255., 26 / 255., 28 / 255.)

# https://jerkwin.github.io/pic/2016/%E5%BC%B9%E6%80%A7%E6%A8%A1%E9%87%8F_CSmat.png
# S = C.Inverse
# Triclinic system
#           | C11 C12 C13 C14 C15 C16 |
#           |     C22 C23 C24 C25 C26 |
#     C  =  |         C33 C34 C35 C36 |
#           |             C44 C45 C46 |
#           |                 C55 C56 |
#           |                     C66 |
# Name='Triclinic system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = 90; CVoigt[1,4] = 0; CVoigt[1,5] = -9; CVoigt[1,6] = 0;
# CVoigt[2,2] = 169; CVoigt[2,3] = 105; CVoigt[2,4] = 0; CVoigt[2,5] = -7; CVoigt[2,6] = 0;
# CVoigt[3,3] = 128; CVoigt[3,4] = 0; CVoigt[3,5] = 11; CVoigt[3,6] = 0;
# CVoigt[4,4] = 53; CVoigt[4,5] = 0; CVoigt[4,6] = -0.6;
# CVoigt[5,5] = 36; CVoigt[5,6] = 0;
# CVoigt[6,6] = 48;

# Hexagonal system
#           | C11 C12 C13  0   0   0  |
#           |     C11 C13  0   0   0  |
#     C  =  |         C33  0   0   0  |
#           |             C44  0   0  |
#           |                 C44  0  |
#           |                      X  |    X = (C11-C12)/2
# Name='Hexagonal system'
# CVoigt[1,1] = 581; CVoigt[1,2] = 55; CVoigt[1,3] = 121;
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,3];
# CVoigt[3,3] = 445;
# CVoigt[4,4] = 240;
# CVoigt[5,5] = CVoigt[4,4];
# CVoigt[6,6] = [CVoigt[1,1]-CVoigt[1,2]]/2;

# Orthorhombic system
#           | C11 C12 C13  0   0   0  |
#           |     C22 C23  0   0   0  |
#     C  =  |         C33  0   0   0  |
#           |             C44  0   0  |
#           |                 C55  0  |
#           |                     C66 |
# Name='Orthorhombic system'
# CVoigt[1,1] = 115.9; CVoigt[1,2] = 35.3; CVoigt[1,3] = 46.8;
# CVoigt[2,2] = 174.1; CVoigt[2,3] = 38.7;
# CVoigt[3,3] = 153.1;
# CVoigt[4,4] = 50.9;
# CVoigt[5,5] = 70.2;
# CVoigt[6,6] = 26.6;

# Cubic system
#           | C11 C12 C12  0   0   0  |
#           |     C11 C12  0   0   0  |
#     C  =  |         C11  0   0   0  |
#           |             C44  0   0  |
#           |                 C44  0  |
#           |                     C44 |
# Name='Cubic system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = CVoigt[1,2];
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,2];
# CVoigt[3,3] = CVoigt[1,1];
# CVoigt[4,4] = 53;
# CVoigt[5,5] = CVoigt[4,4];
# CVoigt[6,6] = CVoigt[4,4];

# Trigonal system type 1(3, 3-)
#           | C11  C12  C13  C14  C15   0  |
#           |      C11  C13 -C14 -C15   0  |
#     C  =  |           C33   0    0    0  |
#           |                C44   0  -C15 |
#           |                     C44  C14 |
#           |                           X  |    X = (C11-C12)/2
# Name='Trigonal system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = CVoigt[1,2]; CVoigt[1,4] = 0; CVoigt[1,5] = -9;
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,3]; CVoigt[2,4] = -CVoigt[1,4]; CVoigt[2,5] = -CVoigt[1,5];
# CVoigt[3,3] = 55;
# CVoigt[4,4] = 53; CVoigt[4,6] = -CVoigt[1,5];
# CVoigt[5,5] = CVoigt[4,4]; CVoigt[5,6] = CVoigt[1,4]
# CVoigt[6,6] = [CVoigt[1,1]-CVoigt[1,2]]/2;

# Trigonal system type 2(32, 3m, 3-m)
#           | C11  C12  C13  C14   0    0  |
#           |      C11  C13 -C14   0    0  |
#     C  =  |           C33   0    0    0  |
#           |                C44   0    0  |
#           |                     C44  C14 |
#           |                           X  |    X = (C11-C12)/2
# Name='Trigonal system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = CVoigt[1,2]; CVoigt[1,4] = 0;
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,3]; CVoigt[2,4] = -CVoigt[1,4];
# CVoigt[3,3] = 55;
# CVoigt[4,4] = 53;
# CVoigt[5,5] = CVoigt[4,4]; CVoigt[5,6] = CVoigt[1,4]
# CVoigt[6,6] = [CVoigt[1,1]-CVoigt[1,2]]/2;

# Tetragonal system type 1(4, 4-, 4/m)
#           | C11  C12  C13   0    0   C16 |
#           |      C11  C13   0    0  -C16 |
#     C  =  |           C33   0    0    0  |
#           |                C44   0    0  |
#           |                     C44   0  |
#           |                          C66 |
# Name='Tetragonal system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = CVoigt[1,2]; CVoigt[1,6] = 7;
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,3]; CVoigt[2,6] = -CVoigt[1,6];
# CVoigt[3,3] = 55;
# CVoigt[4,4] = 53;
# CVoigt[5,5] = CVoigt[4,4];
# CVoigt[6,6] = 77;

# Tetragonal system type 2(422, 4mm, 4-2m, 4/mmm)
#           | C11 C12 C13  0   0   0  |
#           |     C11 C13  0   0   0  |
#     C  =  |         C33  0   0   0  |
#           |             C44  0   0  |
#           |                 C44  0  |
#           |                     C66 |
# Name='Tetragonal system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = CVoigt[1,2];
# CVoigt[2,2] = CVoigt[1,1]; CVoigt[2,3] = CVoigt[1,3];
# CVoigt[3,3] = 55;
# CVoigt[4,4] = 53;
# CVoigt[5,5] = CVoigt[4,4];
# CVoigt[6,6] = 77;

# Monoclinic system
#           | C11 C12 C13  0  C15  0  |
#           |     C22 C23  0  C25  0  |
#     C  =  |         C33  0  C35  0  |
#           |             C44  0  C46 |
#           |                 C55  0  |
#           |                     C66 |
# Name='Monoclinic system'
# CVoigt[1,1] = 125; CVoigt[1,2] = 87; CVoigt[1,3] = 17; CVoigt[1,5] = 53;
# CVoigt[2,2] = 13; CVoigt[2,3] = 77; CVoigt[2,5] = 55;
# CVoigt[3,3] = 55; CVoigt[3,5] = 42;
# CVoigt[4,4] = 53; CVoigt[4,6] = 2;
# CVoigt[5,5] = 77;
# CVoigt[6,6] = 77;

# # By C and vector L1, L2, L3 get Elastic modulus(E)
# def GetE(C, Lx, Ly, Lz):
#     S = np.zeros((7, 7))
#     S[1:7, 1:7] = np.matrix(C).I
#     E_Inv = S[1, 1] * Lx**4
#     E_Inv += S[2, 2] * Ly**4
#     E_Inv += S[3, 3] * Lz**4
#     E_Inv += (S[4, 4] + 2 * S[2, 3]) * Ly**2 * Lz**2
#     E_Inv += (S[5, 5] + 2 * S[1, 3]) * Lx**2 * Lz**2
#     E_Inv += (S[6, 6] + 2 * S[1, 2]) * Lx**2 * Ly**2
#     E_Inv += 2 * (S[1, 4] + S[5, 6]) * Lx**2 * Ly * Lz
#     E_Inv += 2 * S[2, 4] * Ly**3 * Lz
#     E_Inv += 2 * S[3, 4] * Ly * Lz**3
#     E_Inv += 2 * S[1, 5] * Lx**2 * Lz
#     E_Inv += 2 * (S[2, 5] + S[4, 6]) * Lx * Ly**2 * Lz
#     E_Inv += 2 * S[3, 5] * Lx * Lz**3
#     E_Inv += 2 * S[1, 6] * Lx**3 * Ly
#     E_Inv += 2 * S[2, 6] * Lx * Ly**3
#     E_Inv += 2 * (S[3, 6] + S[4, 5]) * Lx * Ly * Lz**2
#     return 1. / E_Inv


def SVoigtCoeff(P, Q):
    return 1. / ((1 + P / 3) * (1 + Q / 3))


def Vector(theta, phi):
    return array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])


def Vector2(theta, phi, chi):
    return array([
        cos(theta) * cos(phi) * cos(chi) - sin(phi) * sin(chi),
        cos(theta) * sin(phi) * cos(chi) + cos(phi) * sin(chi),
        -sin(theta) * cos(chi)
    ])


def Young(Smat, theta, phi):
    L = Vector(theta, phi)
    L = L.reshape(-1, 1, 1, 1) * L.reshape(1, -1, 1, 1) * L.reshape(1, 1, -1, 1) * L.reshape(1, 1, 1, -1)
    return 1. / np.sum(L * Smat)


def LinearCompressibility(Smat, theta, phi):
    L = Vector(theta, phi)
    L = L.reshape(-1, 1, 1) * L.reshape(1, -1, 1)
    LC = L * Smat[:, :].diagonal()
    return 1000 * np.sum(LC)


def shear(Smat, theta, phi, chi):
    L1 = Vector(theta, phi)
    L2 = Vector2(theta, phi, chi)
    L = L1.reshape(-1, 1, 1, 1) * L2.reshape(1, -1, 1, 1) * L1.reshape(1, 1, -1, 1) * L2.reshape(1, 1, 1, -1)
    return 1. / (4 * np.sum(L * Smat))


def Poisson(Smat, theta, phi, chi):
    L1 = Vector(theta, phi)
    L2 = Vector2(theta, phi, chi)
    L = L1.reshape(-1, 1, 1, 1) * L1.reshape(1, -1, 1, 1) * L2.reshape(1, 1, -1, 1) * L2.reshape(1, 1, 1, -1)
    L_ = L1.reshape(-1, 1, 1, 1) * L1.reshape(1, -1, 1, 1) * L1.reshape(1, 1, -1, 1) * L1.reshape(1, 1, 1, -1)
    return -1. * np.sum(L * Smat) / np.sum(L_ * Smat)


## Pre data
CVoigt = np.matrix(CVoigt[1:7, 1:7])
CVoigt = np.triu(CVoigt)
CVoigt += CVoigt.T - np.diag(CVoigt.diagonal())
SVoigt = np.matrix(CVoigt).I
VoigtMat = array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
Coeff = SVoigtCoeff(VoigtMat.reshape(3, 3, 1, 1), VoigtMat.reshape(1, 1, 3, 3))
SVoigtSort = SVoigt[VoigtMat.reshape(3, 3, 1, 1), VoigtMat.reshape(1, 1, 3, 3)]
Smat = array(Coeff) * array(SVoigtSort)

## Plot_data
Theta = np.linspace(0, 2 * pi, 5 + POINTS * 4)
R_Young = array(Young(Smat, pi / 2, Theta))
R_LC = array(LinearCompressibility(Smat, pi / 2, Theta))
R_shear_positive = array(shear(Smat, pi / 2, Theta, pi / 2))
R_shear_negative = array(shear(Smat, pi / 2, Theta, pi / 2)) * -1
R_Poisson_positive = array(Poisson(Smat, pi / 2, Theta, pi / 2))
R_Poisson_negative = array(Poisson(Smat, pi / 2, Theta, pi / 2)) * -1


## Matplotlib plot
def figure_init():
    plt.close()
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(polar=True)
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    ax.set_rlabel_position(45)
    plt.thetagrids(np.arange(0, 360, 30), size=10)
    ax.spines['polar'].set_visible(False)
    ax.grid(linestyle=':', alpha=0.6, linewidth=2, color='black')
    return fig, ax


# Young's modulus
fig, ax = figure_init()
plt.title("Young's modulus in (xy) plane")
R = R_Young
ax.plot(Theta, R, '-', color=Young_color, linewidth=5)
ax.set_rlim(0, max(R) * 1.1)
frmtr = tik.FormatStrFormatter("%d GPa")
ax.yaxis.set_major_formatter(frmtr)
plt.savefig("Youngs_modulus.png", format='png', bbox_inches='tight', dpi=DPI)

# Linear compressibility
fig, ax = figure_init()
plt.title("Linear compressibility in (xy) plane")
R = R_LC
ax.plot(Theta, R, '-', color=LC_color, linewidth=5)
ax.set_rlim(0, max(R) * 1.1)
frmtr = tik.FormatStrFormatter("%d")
ax.yaxis.set_major_formatter(frmtr)
plt.savefig("Linear_compressibility.png", format='png', bbox_inches='tight', dpi=DPI)

# Shear modulus
fig, ax = figure_init()
plt.title("Shear modulus in (xy) plane")
R = R_shear_positive
ax.plot(Theta, R, '-', color=shear_color_positive, linewidth=5)
R = R_shear_negative
ax.plot(Theta, R, '-', color=shear_color_negative, linewidth=5)
ax.set_rlim(0, max(max(R_shear_positive), max(R_shear_negative)) * 1.1)
frmtr = tik.FormatStrFormatter("%d GPa")
ax.yaxis.set_major_formatter(frmtr)
legend = plt.legend(labels=['Positive', 'Negative'], loc=1, bbox_to_anchor=(0.19, 0.19), fontsize=12)
legend.get_frame().set_alpha(0)
plt.savefig("Shear_modulus.png", format='png', bbox_inches='tight', dpi=DPI)

# Poisson's ratio
fig, ax = figure_init()
plt.title("Poisson's ratio in (xy) plane")
ax.set_rlim(0, max(max(R_Poisson_positive), max(R_Poisson_negative)) * 1.1)
R = R_Poisson_positive
ax.plot(Theta, R, '-', color=Poisson_color_positive, linewidth=2)
R = R_Poisson_negative
ax.plot(Theta, R, '-', color=Poisson_color_negative, linewidth=2)
legend = plt.legend(labels=['Positive', 'Negative'], loc=1, bbox_to_anchor=(0.19, 0.19), fontsize=12)
legend.get_frame().set_alpha(0)
plt.savefig("Poissons_ratio.png", format='png', bbox_inches='tight', dpi=DPI)
ax.set_rlim(0, max(R_Poisson_negative) * 1.1)
ax.fill(Theta, R_Poisson_positive, color=Poisson_color_positive, alpha=0.4)
ax.fill(Theta, R_Poisson_negative, color=Poisson_color_negative, alpha=0.4)
plt.savefig("Poissons_ratio_nega.png", format='png', bbox_inches='tight', dpi=DPI)
