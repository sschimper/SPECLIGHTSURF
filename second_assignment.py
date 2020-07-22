import sys
import numpy as np
import math


# convert degrees to radiant
def deg_to_rad(angle):
    return angle * (math.pi / 180.0)


# rotation matrix (rotates counter-clockwise around x axis)
def rotateX(vector, theta):
    theta = math.radians(theta)

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
    def __init__(self, angleX):
        # initial coordinate system
        self.x = np.array([1, 0, 0, 0])
        self.y = np.array([0, 1, 0, 0])
        self.z = np.array([0, 0, 1, 0])
        self.w = np.array([0, 0, 0, 1])

        # rotate frame around x axis
        if angleX is not None:
            self.x = rotateX(self.x, angleX)  # should be the same
            self.y = rotateX(self.y, angleX)
            self.z = rotateX(self.z, angleX)
            self.w = rotateX(self.w, angleX)

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

        self.matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

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
    theta = math.radians(theta)  # convert degree to radiant

    f_perp = (a ** 2 + b ** 2 - 2 * a * math.cos(theta) + math.cos(theta) ** 2) \
             / (a ** 2 + b ** 2 + 2 * a * math.cos(theta) + math.cos(theta) ** 2)
    f_paral = f_perp * (a ** 2 + b ** 2 - 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
            math.tan(theta) ** 2)) \
              / (a ** 2 + b ** 2 + 2 * a * math.sin(theta) * math.tan(theta) + (math.sin(theta) ** 2) * (
            math.tan(theta) ** 2))

    delta_perp = math.atan2((pow(math.cos(theta), 2) - pow(a, 2) - pow(b, 2)), 2 * b * math.cos(theta))
    delta_paral = math.atan2(
        (pow(pow(ior_c.real, 2) + pow(ior_c.imag, 2), 2) * pow(math.cos(theta), 2) - pow(a, 2) - pow(b, 2)),
        (2 * math.cos(theta) * ((pow(ior_c.real, 2) - pow(ior_c.imag, 2)) * b - 2 * ior_c.real * ior_c.imag * a)))

    A = (f_perp + f_paral) / 2
    B = (f_perp - f_paral) / 2
    C = math.cos(delta_perp - delta_paral) * math.sqrt(f_perp * f_paral)
    S = math.sin(delta_perp - delta_paral) * math.sqrt(f_perp * f_paral) * (-1)

    matrix = np.array([[A, B, 0, 0],
                       [B, A, 0, 0],
                       [0, 0, C, S],
                       [0, 0, -S, C]])

    matrix = matrix.transpose()

    return matrix


# calculate polarization matrix
def calculate_polarizer(phi):
    phi = deg_to_rad(phi)  # convert to radians

    cos2Phi = math.cos(2 * phi)
    sin2Phi = math.sin(2 * phi)

    matrix = np.array([[1, cos2Phi, sin2Phi, 0],
                       [cos2Phi, pow(cos2Phi, 2), sin2Phi * cos2Phi, 0],
                       [sin2Phi, sin2Phi * cos2Phi, pow(sin2Phi, 2), 0],
                       [0, 0, 0, 0]])
    res = 0.5 * matrix

    return res


# check if 2 reference frames are matching
def referenceframe_match(a, b):
    epsilon = 0.00001
    if abs(a.x[0] - b.x[0]) <= epsilon and abs(a.x[1] - b.x[1]) <= epsilon and abs(a.x[2] - b.x[2]) <= epsilon \
            and abs(a.y[0] - b.y[0]) <= epsilon and abs(a.y[1] - b.y[1]) <= epsilon and abs(a.y[2] - b.y[2]) <= epsilon \
            and abs(a.z[0] - b.z[0]) <= epsilon and abs(a.z[1] - b.z[1]) <= epsilon and abs(a.z[2] - b.z[2]) <= epsilon:
        return True
    else:
        return False


# function for normalizing vector taken from here:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# maybe this function is overkill since I only need to know rho
# but now we have certainty :-)
# function for calculating the angle btw 2 vectors taken from here:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# interact with surface
def interact_with_surface(mueller_matrix_whole, matrix, stokes_vector):
    # if the frame of the stokes vector matches the entry frame of the mueller matrix
    #   then just multiply
    # else
    #   rotate to match, multiply and rotate back
    if referenceframe_match(stokes_vector.frame, mueller_matrix_whole.frame_entry):
        res = matrix.dot(stokes_vector.vector)
        return res

    else:
        # something is very wrong here since the result is incorrect
        discr_angle_y = angle_between(mueller_matrix_whole.frame_entry.y, stokes_vector.frame.y)
        discr_angle_z = angle_between(mueller_matrix_whole.frame_entry.z, stokes_vector.frame.z)

        assert discr_angle_y == discr_angle_z
        theta = discr_angle_y

        rot_mat_pos = np.array([[1, 0, 0, 0],
                                [0, math.cos(2 * theta), math.sin(2 * theta), 0],
                                [0, -math.sin(2 * theta), math.cos(2 * theta), 0],
                                [0, 0, 0, 1]])

        rot_mat_pos = rot_mat_pos.transpose()

        rot_mat_neg = np.array([[1, 0, 0, 0],
                                [0, math.cos(-2 * theta), math.sin(-2 * theta), 0],
                                [0, -math.sin(-2 * theta), math.cos(-2 * theta), 0],
                                [0, 0, 0, 1]])

        rot_mat_neg = rot_mat_neg.transpose()

        res = rot_mat_pos.dot(matrix)
        res = res.dot(rot_mat_neg)
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

    # if polarization angle is specified, calculate polarization matrix
    polarizer = None
    if polarization_angle is not None:
        polarizer = calculate_polarizer(polarization_angle)

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

    # X2 - interaction
    if polarization_angle is not None:
        stokes_vector.vector = interact_with_surface(x2_mueller_matrix, polarizer.dot(x2_mueller_matrix.matrix),
                                                     stokes_vector)
    else:
        stokes_vector.vector = interact_with_surface(x2_mueller_matrix, x2_mueller_matrix.matrix, stokes_vector)

    # X1 - interaction
    stokes_vector.vector = interact_with_surface(x1_mueller_matrix, x1_mueller_matrix.matrix, stokes_vector)

    # print result
    print("--------------------------------------------------")
    print_stokes_vector(stokes_vector.vector)
    print("--------------------------------------------------")
    print("")


# process user input
def get_user_parameters():
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
while (user_option != "exit" and user_option != "Ecit") and (user_option != "exit" and user_option != "Exit"):
    user_option = input("Type either 'start' to start a new calculation or"
                        "\n'exit' to exit the program: ")
    if user_option == "start" or user_option == "Start":
        get_user_parameters()
    elif user_option == "exit" or user_option == "Exit":
        sys.exit(0)

