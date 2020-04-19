import sys
import numpy as np


# representation of frame of reference
# using right-handed coordinate system
class ReferenceFrame:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


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
    def set_mueller_matriy(self, matrix):
        self.matrix = matrix

    # set entry frame
    def set_entry_frame(self, frame_entry):
        self.frame_entry = frame_entry

    # set exit frame
    def set_exit_frame(self, frame_exit):
        self.frame_exit = frame_exit


# get mueller matrix for reflection
def get_reflection_muller_matrix():
    t_fresnel = MuellerMatrix(None, None)



    return t_fresnel


# print the resulting vector
def print_stokes_vector(sv):
    print("The resulting stokes vector looks like this: ")
    print(sv)


def do_stuff(points_conductor, ior, ext_coeff, delta, rho, phi, use_polarization_filter):
    print("\n=============================================")
    print("\nUsing Conductor: ", points_conductor)
    print("Index of Refraction: ", ior)
    if ext_coeff is not None:
        print("Extinction Coefficient: ", ext_coeff)
    print("Delta: ", delta)
    print("Rho: ", rho)
    print("Phi: ", phi)
    print("Using Polarization Filter: ", use_polarization_filter)

    print("\nSTARTING CALCULATION")

    t_fresnel = get_reflection_muller_matrix()

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
    ior = input("\nPlease enter your preferred value for the Index of Refraction: ")
    ior = float(ior)
    while ior < 0:
        ior = input("Error: Index of refraction must not be negative. Please enter another value: ")
        ior = float(ior)

    # extinction coefficient
    ext_coeff = None
    if points_conductor:
        ext_coeff = input("\nPlease enter your preferred value for the Extinction Coefficient: ")
        ext_coeff = float(ext_coeff)
        while ext_coeff < 0:
            ext_coeff = input("Error: Extinction Coefficient must not be negative. Please enter another value: ")
            ext_coeff = float(ext_coeff)

    # delta angle
    delta_answer = input("\nPlease enter your preferred value for the angle Delta: ")
    delta = float(delta_answer) % 360

    # rho angle
    rho_answer = input("\nPlease enter your preferred value for the angle Rho: ")
    rho = float(rho_answer) % 360

    # phi angle
    phi_answer = input("\nPlease enter your preferred value for the angle Phi: ")
    phi = float(phi_answer) % 360

    # query the user if (s)he wants to use polarization filter
    polarization_answer = input("\nWould you like to use a Polarization filter? If so, please type 'y': ")
    if polarization_answer == "y" or polarization_answer == "Y":
        use_polarization_filter = True

    # start calculation
    do_stuff(points_conductor, ior, ext_coeff, delta, rho, phi, use_polarization_filter)


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
