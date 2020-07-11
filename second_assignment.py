import sys
import numpy as np
import math


# convert degrees to radiant
def deg_to_rad(angle):
    return angle * (math.pi / 180.0)


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
# using left-handed coordinate system
class ReferenceFrame:
    def __init__(self, angleZ):
        # initial coordinate system
        self.x = np.array([1, 0, 0, 0])
        self.y = np.array([0, 1, 0, 0])
        self.z = np.array([0, 0, 1, 0])
        self.w = np.array([0, 0, 0, 1])

        # rotate frame around x axis
        if angleZ is not None:
            self.x = rotateZ(self.x, angleZ)  # should be the same
            self.y = rotateZ(self.y, angleZ)
            self.z = rotateZ(self.z, angleZ)
            self.w = rotateZ(self.w, angleZ)

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
    theta = deg_to_rad(theta)  # convert to radiant
    term = math.sqrt(pow((pow(ior_c.real, 2) - (pow(ior_c.imag, 2)) - pow(math.sin(theta), 2)), 2)
                     + 4 * pow(ior_c.real, 2) * pow(ior_c.imag, 2)) + pow(ior_c.real, 2) - pow(ior_c.imag, 2) - pow(
        math.sin(theta), 2)
    res = term * 0.5
    res = math.sqrt(res)
    return res


# calculate b coeff
def calculate_b_coeff(ior_c, theta):
    theta = deg_to_rad(theta)  # convert to radiant
    term = math.sqrt(pow((pow(ior_c.real, 2) - (pow(ior_c.imag, 2)) - pow(math.sin(theta), 2)), 2)
                     + 4 * pow(ior_c.real, 2) * pow(ior_c.imag, 2)) - pow(ior_c.real, 2) + pow(ior_c.imag, 2) + pow(
        math.sin(theta), 2)
    res = term * 0.5
    res = math.sqrt(res)
    return res


# calculate the mueller matrix
def calculate_mueller_matrix(ior_c, a, b, theta):
    theta = deg_to_rad(theta)  # convert theta to radiant

    f_perp = (a ** 2 + b ** 2 - 2 * a * math.cos(theta) + math.cos(theta) ** 2) / (
            a ** 2 + b ** 2 + 2 * a * math.cos(theta) + math.cos(theta) ** 2)
    f_paral = f_perp * (a ** 2 + b ** 2 - 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
            math.tan(theta) ** 2)) / (a ** 2 + b ** 2 + 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
                      math.tan(theta) ** 2))

    tan_delta_perp = (2 * b * math.cos(theta)) / (pow(math.cos(theta), 2) - pow(a, 2) - pow(b, 2))
    tan_delta_paral = (2 * math.cos(theta) * ((pow(ior_c.real, 2) - pow(ior_c.imag, 2)) * b - 2 * ior_c.real * ior_c.imag * a)) / (pow(pow(ior_c.real, 2) + pow(ior_c.imag, 2), 2) * pow(math.cos(theta), 2) - pow(a, 2) - pow(b, 2))

    delta_perp = math.atan(tan_delta_perp)
    delta_paral = math.atan(tan_delta_paral)

    A = (f_perp + f_paral) / 2
    B = (f_perp - f_paral) / 2
    C = math.cos(delta_perp - delta_paral) * math.sqrt(f_perp * f_paral)
    S = math.sin(delta_perp - delta_paral) * math.sqrt(f_perp * f_paral)

    matrix = np.array([[A, B, 0, 0],
                       [B, A, 0, 0],
                       [0, 0, C, S],
                       [0, 0, -S, C]])
    return matrix


# calculate polarization matrix
def apply_polarization(stokes_vector, gamma):
    gamma = deg_to_rad(gamma)  # convert to radians
    matrix = np.array([[1, math.cos(2 * gamma), math.sin(2 * gamma), 0],
                       [math.cos(2 * gamma), pow(math.cos(2 * gamma), 2), math.sin(2 * gamma) * math.cos(2 * gamma), 0],
                       [math.sin(2 * gamma), math.sin(2 * gamma) * math.cos(2 * gamma), pow(math.sin(2 * gamma), 2), 0],
                       [0, 0, 0, 0]])

    intermediate_res = 0.5 * matrix
    res = intermediate_res.dot(stokes_vector.vector)

    return res


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
def calculate_angle_btw_vectors(vec_a, vec_b):
    vec_a_mag = calculate_magnitude(vec_a)
    vec_b_mag = calculate_magnitude(vec_b)

    res = np.dot(vec_a, vec_b) / np.dot(vec_a_mag, vec_b_mag)
    res = np.arccos(res)
    return res


# determine discrepancy angle
def get_discrepancy_angle(reference_frame_comp, entry_matrix_comp):
    theta = calculate_angle_btw_vectors(reference_frame_comp, entry_matrix_comp)
    return theta


# interact with surface
def interact_with_surface(mueller_matrix, stokes_vector):
    if referenceframe_match(stokes_vector.frame, mueller_matrix.frame_entry):
        res = mueller_matrix.matrix.dot(stokes_vector.vector)
        return res

    else:
        discr_angle_y = get_discrepancy_angle(mueller_matrix.frame_entry.y, stokes_vector.frame.y)
        discr_angle_x = get_discrepancy_angle(mueller_matrix.frame_entry.x, stokes_vector.frame.x)

        assert discr_angle_y == discr_angle_x

        theta = discr_angle_y

        rot_mat_negative = np.array([[1, 0, 0, 0],
                                     [0, math.cos(-2 * theta), math.sin(-2 * theta), 0],
                                     [0, -math.sin(-2 * theta), math.cos(-2 * theta), 0],
                                     [0, 0, 0, 1]])

        rot_mat_pos = np.array([[1, 0, 0, 0],
                                [0, math.cos(2 * theta), math.sin(2 * theta), 0],
                                [0, -math.sin(2 * theta), math.cos(2 * theta), 0],
                                [0, 0, 0, 1]])

        res = rot_mat_negative.dot(mueller_matrix.matrix)
        res = res.dot(rot_mat_pos)
        res = res.dot(stokes_vector.vector)
        return res


# print the resulting vector
def print_stokes_vector(sv):
    print("The resulting stokes vector looks like this: ")
    print(sv)


# main calculation function
def calculate_polarized_light(ior_c_X1, ior_c_X2, delta, rho, phi, polarization_angle):
    print("\n=============================================")
    if ior_c_X1 is not None:
        print("Index of Refraction at X1 - Real Part: ", ior_c_X1.real)
        print("Index of Refraction at X1 - Imaginary Part: ", ior_c_X1.imag)
    if ior_c_X2 is not None:
        print("Index of Refraction at X2 - Real Part: ", ior_c_X2.real)
        print("Index of Refraction at X2 - Imaginary Part: ", ior_c_X2.imag)
    print("Delta: ", delta)
    print("Rho: ", rho)
    print("Phi: ", phi)
    if polarization_angle is not None:
        print("Using Polarization Filter with angle: ", polarization_angle)
    print("\nSTARTING CALCULATION")

    # calculate a and b coefficitents
    a_X1 = calculate_a_coeff(ior_c_X1, delta)
    b_X1 = calculate_b_coeff(ior_c_X1, delta)
    a_X2 = calculate_a_coeff(ior_c_X2, phi)
    b_X2 = calculate_b_coeff(ior_c_X2, phi)

    # set up of stokes vector
    stokes_vector_frame = ReferenceFrame(0)
    stokes_vector = StokesVector(100.0, 0.0, 0.0, 0.0, stokes_vector_frame)

    # X1 - setup
    x1_entry_frame = ReferenceFrame(0)  # same as frame of stokes vector
    x1_exit_frame = ReferenceFrame(-0)  # clockwise rotation
    x1_mueller_matrix = MuellerMatrix(x1_entry_frame, x1_exit_frame)
    x1_mueller_matrix.matrix = calculate_mueller_matrix(ior_c_X1, a_X1, b_X1, delta)

    # X2 - setup
    x2_entry_frame = ReferenceFrame(-rho)
    x2_exit_frame = ReferenceFrame(-rho)
    x2_mueller_matrix = MuellerMatrix(x2_entry_frame, x2_exit_frame)
    x2_mueller_matrix.matrix = calculate_mueller_matrix(ior_c_X2, a_X2, b_X2, phi)

    # X1 - interaction
    stokes_vector.vector = interact_with_surface(x1_mueller_matrix, stokes_vector)

    # X2 - interaction
    stokes_vector.vector = interact_with_surface(x2_mueller_matrix, stokes_vector)

    # filter - setup
    if polarization_angle is not None:
        stokes_vector.vector = apply_polarization(stokes_vector, polarization_angle)

    # print result
    print("--------------------------------------------------")
    print_stokes_vector(stokes_vector.vector)


# process user input
def get_user_parameters():
    use_polarization_filter = False

    # index of refraction X1
    ior_real = input("\nPlease enter your preferred real value for the Index of Refraction for X_1: ")
    ior_real = float(ior_real)
    while ior_real < 0:
        ior_real = input("Error: Real value must not be negative. Please enter another value: ")
        ior_real = float(ior_real)
    ior_imag = input("\nPlease enter your preferred imaginary value for the Index of Refraction for X_1: ")
    ior_imag = float(ior_imag)
    while ior_imag < 0:
        ior_imag = input("Error: Imaginary value must not be negative. Please enter another value:  ")
        ior_imag = float(ior_imag)
    ior_c_X1 = ComplexNumber(ior_real, ior_imag)

    # index of refraction X2
    ior_real = input("\nPlease enter your preferred real value for the Index of Refraction for X_2: ")
    ior_real = float(ior_real)
    while ior_real < 0:
        ior_real = input("Error: Real value must not be negative. Please enter another value: ")
        ior_real = float(ior_real)
    ior_imag = input("\nPlease enter your preferred imaginary value for the Index of Refraction for X_2: ")
    ior_imag = float(ior_imag)
    while ior_imag < 0:
        ior_imag = input("Error: Imaginary value must not be negative. Please enter another value:  ")
        ior_imag = float(ior_imag)
    ior_c_X2 = ComplexNumber(ior_real, ior_imag)

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
    polarization_angle = None
    polarization_answer = input("\nWould you like to use a Polarization filter? If so, please type 'y': ")
    if polarization_answer == "y" or polarization_answer == "Y":
        polarization_answer = input("\nPlease enter your preferred value for the polarization angle (in degrees):  ")
        polarization_angle = float(polarization_answer) % 360

    # start calculation
    calculate_polarized_light(ior_c_X1, ior_c_X2, delta, rho, phi, polarization_angle)


# header
print("=============================================")
print("||    2nd ASSIGNMENT Sebastian Schimper    ||")
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
