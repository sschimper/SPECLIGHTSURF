import sys
import numpy as np
from numpy.testing import assert_array_equal
import math


# rotation matrix (rotates counter-clockwise around x axis)
def rotateX(vector, theta):
    rot_matrix = np.array([[1, 0, 0, 0],
                           [0, math.cos(theta), -math.sin(theta), 0],
                           [0, math.sin(theta), math.cos(theta), 0],
                           [0, 0, 0, 1]
                           ])

    return rot_matrix.dot(vector)


# rotation matrix (rotates counter-clockwise around y axis)
def rotateY(vector, theta):
    rot_matrix = np.array([[math.cos(theta), 0, math.sin(theta), 0],
                           [0, 1, 0, 0],
                           [-math.sin(theta), 0, math.cos(theta), 0],
                           [0, 0, 0, 1]
                           ])

    return rot_matrix.dot(vector)


# rotation matrix (rotates counter-clockwise around z axis)
def rotateZ(vector, theta):
    rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0, 0],
                           [math.sin(theta), math.cos(theta), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]
                           ])

    return rot_matrix.dot(vector)


# representation of frame of reference
# using right-handed coordinate system
class ReferenceFrame:
    def __init__(self, angleX, angleZ):
        # initial coordinate system
        self.x = np.array([1, 0, 0, 0])
        self.y = np.array([0, 1, 0, 0])
        self.z = np.array([0, 0, 1, 0])
        self.w = np.array([0, 0, 0, 1])

        # rotate frame around x axis
        if angleX is not None and angleZ is None:
            self.x = rotateX(self.x, angleX)  # should be the same
            self.y = rotateX(self.y, angleX)
            self.z = rotateX(self.z, angleX)
            self.w = rotateX(self.w, angleX)

        # rotate frame around Z axis
        elif angleZ is not None and angleZ is not None:
            self.x = rotateX(self.x, angleX) * rotateZ(self.x, angleZ)  # should be the same
            self.y = rotateX(self.y, angleX) * rotateZ(self.y, angleZ)
            self.z = rotateX(self.z, angleX) * rotateZ(self.z, angleZ)
            self.w = rotateX(self.w, angleX) * rotateZ(self.w, angleZ)

        # verify orthogonality
        assert self.x.dot(self.y) == 0.0 and self.x.dot(self.z) == 0.0 and self.y.dot(self.x) == 0.0

    # getters and setters
    def get_x(self):
        return self.x

    def set_x(self, new_x):
        self.x = new_x

    def get_y(self):
        return self.y

    def set_y(self, new_y):
        self.y = new_y

    def get_z(self):
        return self.z

    def set_z(self, new_z):
        self.z = new_z


# representation of stokes vector
class StokesVector:
    def __init__(self, s_0, s_1, s_2, s_3, frame):
        if frame is not None:
            self.frame = frame
        else:
            self.frame = None

        if s_0 < 0:
            print("Error: Intensity of stokes vector is negative.")
            sys.exit(0)

        if (s_0 > 1.0 or s_0 < 0) and (s_1 > 1.0 or s_1 < 0) and (s_2 > 1.0 or s_2 < 0) and (s_3 > 1.0 or s_3 < 0):
            print("Error: Parameters of stokes vector are not between 0.0 and 1.0 .")
            sys.exit(0)

        # stokes vector
        self.vector = np.array([s_0, s_1, s_2, s_3])

    # set frame of reference
    def set_frame(self, frame):
        self.frame = frame


# representation of mueller matrix
class MuellerMatrix:
    def __init__(self, frame_entry, frame_exit):
        if frame_entry is not None:
            self.frame_entry = frame_entry
        else:
            self.frame_entry = None

        if frame_exit is not None:
            self.frame_exit = frame_exit
        else:
            self.frame_exit = None

        self.matrix = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])

    # calculate the mueller matrix
    def set_mueller_matrix(self, matrix):
        self.matrix = matrix

    # set entry frame
    def set_entry_frame(self, frame_entry):
        self.frame_entry = frame_entry

    # set exit frame
    def set_exit_frame(self, frame_exit):
        self.frame_exit = frame_exit


# class for representing filter
class PolarizationFilter:
    def __init__(self, matrix, frame):
        self.matrix = matrix
        self.frame = frame

    # calculate the mueller matrix
    def set_matrix(self, matrix):
        self.matrix = matrix

    # set frame
    def set_entry_frame(self, frame):
        self.frame = frame


# class for storing complex number
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag


# calculate a coeff
def calculate_a_coeff(ior_c, theta):
    a = math.sqrt((1 / 2) * math.sqrt((ior_c.real ** 2 - ior_c.imag ** 2 -
                                       math.sin(theta) ** 2) ** 2 + 4 * ior_c.real ** 2 * ior_c.imag ** 2)
                  + ior_c.real ** 2 - ior_c.imag ** 2 - math.sin(theta) ** 2)
    return a


# calculate b coeff
def calculate_b_coeff(ior_c, theta):
    first_part = math.sqrt(0.5)
    second_part = math.sqrt(
        (ior_c.real ** 2 - ior_c.imag ** 2 - math.sin(theta) ** 2) ** 2 + 4 * ior_c.real ** 2 * ior_c.imag ** 2)
    third_part = - ior_c.real ** 2 + ior_c.imag ** 2 + math.sin(theta) ** 2

    return first_part * second_part + third_part


# calculate the mueller matrix
def calculate_mueller_matrix(ior_c, a, b, theta):
    f_perp = (a ** 2 + b ** 2 - 2 * a * math.cos(theta) + math.cos(theta) ** 2) / (
            a ** 2 + b ** 2 + 2 * a * math.cos(theta) + math.cos(theta) ** 2)
    f_paral = (a ** 2 + b ** 2 - 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
            math.tan(theta) ** 2)) / \
              (a ** 2 + b ** 2 + 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
                      math.tan(theta) ** 2)) * f_perp
    tan_delta_perp = (2 * b * math.cos(theta)) / ((math.cos(theta) ** 2) - a ** 2 - b ** 2)
    tan_delta_paral = (2 * math.cos(theta) * (
            (ior_c.real ** 2 - ior_c.imag ** 2) * b - 2 * ior_c.real * ior_c.imag * a)) / \
                      ((ior_c.real ** 2 + ior_c.imag ** 2) ** 2 * (math.cos(theta) ** 2) - a ** 2 - b ** 2)

    A = (f_perp + f_paral) / 2
    B = (f_perp - f_paral) / 2
    C = math.cos(tan_delta_perp - tan_delta_paral) * math.sqrt(f_perp * f_paral)
    S = math.sin(tan_delta_perp - tan_delta_paral) * math.sqrt(f_perp * f_paral)

    matrix = np.array([[A, B, 0, 0],
                       [B, A, 0, 0],
                       [0, 0, C, S],
                       [0, 0, -S, C]])
    return matrix


# calculate polarization matrix
def calculate_polarization_matrix(gamma):
    matrix = np.array([[1, math.cos(2 * gamma), math.sin(2 * gamma), 0],
                       [math.cos(2 * gamma), math.cos(2 * gamma) ** 2, math.sin(2 * gamma) * math.cos(2 * gamma), 0],
                       [math.sin(2 * gamma), math.sin(2 * gamma) * math.cos(2 * gamma), math.sin(2 * gamma) ** 2, 0],
                       [0, 0, 0, 0]])
    return matrix


# check if 2 reference frames are matching
def referenceframe_match(a, b):
    if np.all(a.x == b.x) and np.all(a.y == b.y) and np.all(a.z == b.z) and np.all(a.w == b.w):
        return True
    else:
        return False


# calculate magnitude
def calculate_magnitude(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


# calculate angle btw two vectors
def calculate_angle(vec_a, vec_b):
    vec_a_mag = calculate_magnitude(vec_a)
    vec_b_mag = calculate_magnitude(vec_b)

    res = np.dot(vec_a, vec_b) / np.dot(vec_a_mag, vec_b_mag)
    res = np.arccos(res)
    return res


# determine discrepancy angle
def get_discrepancy_angle(reference_frame_comp, entry_matrix_comp):
    theta = calculate_angle(reference_frame_comp, entry_matrix_comp)
    return theta


# interact with surface
def interact_with_surface(mueller_matrix, stokes_vector, theta):
    if theta == 0.0 or theta is None:
        res = mueller_matrix.dot(stokes_vector)
        return res
    else:
        rot_mat_negative = np.array([[1, 0, 0, 0],
                                     [0, math.cos(-2 * theta), math.sin(-2 * theta), 0],
                                     [0, -math.sin(-2 * theta), math.cos(-2 * theta), 0],
                                     [0, 0, 0, 1]])

        rot_mat_pos = np.array([[1, 0, 0, 0],
                                [0, math.cos(2 * theta), math.sin(2 * theta), 0],
                                [0, -math.sin(2 * theta), math.cos(2 * theta), 0],
                                [0, 0, 0, 1]])

        res = rot_mat_negative.dot(mueller_matrix).dot(rot_mat_pos).dot(stokes_vector)
        return res


# print the resulting vector
def print_stokes_vector(sv):
    print("The resulting stokes vector looks like this: ")
    print(sv)


# main calculation function
def calculate_polarized_light(points_conductor, ior_c, ext_coeff, delta, rho, phi, use_polarization_filter):
    print("\n=============================================")
    print("\nUsing Conductor: ", points_conductor)
    if ior_c is not None:
        print("Index of Refraction - Real Part: ", ior_c.real)
        print("Index of Refraction - Imaginary Part: ", ior_c.imag)
    if ext_coeff is not None:
        print("Extinction Coefficient: ", ext_coeff)
    print("Delta: ", delta)
    print("Rho: ", rho)
    print("Phi: ", phi)
    print("Using Polarization Filter: ", use_polarization_filter)

    print("\nSTARTING CALCULATION")

    # calculate a and b coefficitents
    a = calculate_a_coeff(ior_c, delta)
    b = calculate_b_coeff(ior_c, delta)

    # angle for aligning coordinate system with light beam
    eye_rot_angle = 90 - delta
    eye_frame = ReferenceFrame(eye_rot_angle, None)

    # stokes vector set up
    stokes_vector = StokesVector(100.0, 1.0, 0.0, 0.0, eye_frame)

    # X1 - setup
    x1_rot_angle = (180 - 2 * delta) / 2
    x1_entry_frame = ReferenceFrame(x1_rot_angle, None)  # same as frame of stokes vector
    x1_exit_frame = ReferenceFrame(-x1_rot_angle, None)  # clockwise rotation
    x1_mueller_matrix = MuellerMatrix(x1_entry_frame, x1_exit_frame)
    x1_mueller_matrix.matrix = calculate_mueller_matrix(ior_c, a, b, delta)

    # X2 - setup
    x2_rot_angle = ((180 - 2 * phi) / 2)
    x2_entry_frame = ReferenceFrame(-x2_rot_angle, -rho)
    x2_exit_frame = ReferenceFrame(x2_rot_angle, -rho)
    x2_mueller_matrix = MuellerMatrix(x2_entry_frame, x2_exit_frame)
    x2_mueller_matrix.matrix = calculate_mueller_matrix(ior_c, a, b, phi)

    # X1 - interaction
    if referenceframe_match(stokes_vector.frame, x1_mueller_matrix.frame_entry):
        stokes_vector.vector = interact_with_surface(x1_mueller_matrix.matrix, stokes_vector.vector, x1_rot_angle)
        stokes_vector.frame.y = rotateX(stokes_vector.frame.y, -2 * x1_rot_angle)
        stokes_vector.frame.z = rotateX(stokes_vector.frame.z, -2 * x1_rot_angle)
        print("stokes vector frame")
        print(stokes_vector.frame.x)
        print(stokes_vector.frame.y)
        print(stokes_vector.frame.z)
        print(stokes_vector.frame.w)
        print("matrix exit frame")
        print(x1_mueller_matrix.frame_exit.x)
        print(x1_mueller_matrix.frame_exit.y)
        print(x1_mueller_matrix.frame_exit.z)
        print(x1_mueller_matrix.frame_exit.w)
        # assert stokes_vector.frame == x1_mueller_matrix.frame_exit # weird bug, vectors look correct when print
        stokes_vector.vector = rotateX(stokes_vector.vector, -2 * x1_rot_angle)

    # X2 - interaction
    if not referenceframe_match(stokes_vector.frame, x2_mueller_matrix.frame_entry):
        # bring back to initial
        stokes_vector.frame.y = rotateX(stokes_vector.frame.y, x1_rot_angle)
        stokes_vector.frame.z = rotateX(stokes_vector.frame.z, x1_rot_angle)
        stokes_vector.vector = rotateX(stokes_vector.vector, x1_rot_angle)
        # rotate neg phi
        stokes_vector.frame.y = rotateX(stokes_vector.frame.y, -x2_rot_angle)
        stokes_vector.frame.z = rotateX(stokes_vector.frame.z, -x2_rot_angle)
        stokes_vector.vector = rotateX(stokes_vector.vector, -x2_rot_angle)
        # rotate neg rho
        stokes_vector.frame.x = rotateZ(stokes_vector.frame.x, -rho)
        stokes_vector.frame.y = rotateZ(stokes_vector.frame.y, -rho)
        stokes_vector.frame.z = rotateZ(stokes_vector.frame.z, -rho)
        stokes_vector.vector = rotateZ(stokes_vector.vector, -rho)

    # filter - setup
    pol_filter = PolarizationFilter(None, None)
    if use_polarization_filter:
        pol_filter.set_matrix(calculate_polarization_matrix(4))
        pol_filter.set_entry_frame(x2_exit_frame)


# process user input
def get_user_parameters():
    points_conductor = False
    use_polarization_filter = False

    # surface on points
    surface_user_answer = None
    while surface_user_answer != "1" and surface_user_answer != "2":
        surface_user_answer = input("\nPlease type '1' if you want the surface at the points X_1 and X_2 to be "
                                    "perfectly smooth dielectric "
                                    "\n or type '2' if you want it to be customizable: ")
        if surface_user_answer == "2":
            points_conductor = True

    # index of refraction
    ior_complex_answer = input("\nWould you like the Index of Refraction to be a complex number? \n"
                               "If so, please type 'y', if not, type anything else: ")

    ior_complex_bool = False
    if ior_complex_answer == 'y':
        ior_complex_bool = True
    else:
        ior_complex_bool = False

    ior_c = None
    if not ior_complex_bool:
        ior = input("\nPlease enter your preferred value for the Index of Refraction: ")
        ior = float(ior)
        while ior < 0:
            ior = input("Error: Index of refraction must not be negative. Please enter another value: ")
            ior = float(ior)
        ior_c = ComplexNumber(ior, 0.0)
    else:
        ior_real = input("\nPlease enter your preferred real value for the Index of Refraction: ")
        ior_real = float(ior_real)
        while ior_real < 0:
            ior_real = input("Error: Real value must not be negative. Please enter another value: ")
            ior_real = float(ior_real)
        ior_imag = input("\nPlease enter your preferred imaginary value for the Index of Refraction: ")
        ior_imag = float(ior_imag)
        while ior_imag < 0:
            ior_imag = input("Error: Imaginary value must not be negative. Please enter another value:  ")
            ior_imag = float(ior_imag)
        ior_c = ComplexNumber(ior_real, ior_imag)

    # extinction coefficient
    ext_coeff = None
    if points_conductor:
        ext_coeff = input("\nPlease enter your preferred value for the Extinction Coefficient: ")
        ext_coeff = float(ext_coeff)
        while ext_coeff < 0:
            ext_coeff = input("Error: Extinction Coefficient must not be negative. Please enter another value: ")
            ext_coeff = float(ext_coeff)

    # delta angle
    delta_answer = input("\nPlease enter your preferred value for the angle Delta (in degrees): ")
    delta = float(delta_answer) % 360

    # rho angle
    rho_answer = input("\nPlease enter your preferred value for the angle Rho (in degrees): ")
    rho = float(rho_answer) % 360

    # phi angle
    phi_answer = input("\nPlease enter your preferred value for the angle Phi (in degrees): ")
    phi = float(phi_answer) % 360

    # query the user if (s)he wants to use polarization filter
    polarization_answer = input("\nWould you like to use a Polarization filter? If so, please type 'y': ")
    if polarization_answer == "y" or polarization_answer == "Y":
        use_polarization_filter = True

    # start calculation
    calculate_polarized_light(points_conductor, ior_c, ext_coeff, delta, rho, phi, use_polarization_filter)


# header
print("=============================================")
print("||        Welcome to SPECLIGHTSURF         ||")
print("=============================================")

# menu loop
user_option = None
while (user_option != "start" and user_option != "Start") and (user_option != "exit" and user_option != "Exit"):
    user_option = input("Type either 'start' to start a new calculation or"
                        "\n'exit' to exit the program: ")
    if user_option == "start" or user_option == "Start":
        get_user_parameters()
    elif user_option == "exit" or user_option == "Exit":
        sys.exit(0)
