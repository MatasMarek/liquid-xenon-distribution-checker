import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.markdown("<h1 style='text-align: center; color: white;'>Liquid xenon distribution checker</h1>", unsafe_allow_html=True)

def find_whole(array, values):
    ixs = np.round(find_fractional(array, values)).astype(int)
    return ixs


def find_fractional(array, values):
    step = (array[-1] - array[0])/(len(array) - 1)
    ixs = (values - array[0])/step
    return ixs



def create_margins(size_of_the_cube, atomic_positions):
    atomic_positions_margins = np.zeros((1, 3))
    for value_x in [-size_of_the_cube, 0, size_of_the_cube]:
        for value_y in [-size_of_the_cube, 0, size_of_the_cube]:
            for value_z in [-size_of_the_cube, 0, size_of_the_cube]:
                if not (value_x == 0 and value_y == 0 and value_z == 0):
                    atomic_positions_margins = np.concatenate((atomic_positions_margins, np.copy(atomic_positions) + np.array([value_x, value_y, value_z])))
    atomic_positions_margins = atomic_positions_margins[1:]
    return atomic_positions_margins


def average_distance_calculator(distribution, distribution_margins):
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')[:, 0]
    nbins = len(experiment)
    average_distance_random = np.zeros(nbins)
    r_min, r_max = experiment.min(), experiment.max()
    rs = np.linspace(r_min, r_max, nbins)
    progress = 0.
    for atom in distribution:
        print(100.*progress/len(distribution))
        progress += 1.
        atoms = np.concatenate((distribution, distribution_margins))
        distances = np.linalg.norm(atoms - atom, axis=1, keepdims=True)
        idxs = find_whole(rs, distances)
        for idx in idxs:
            if 0. <= idx < nbins:
                average_distance_random[idx] += 1.

    return average_distance_random, rs



def plot_results(distances, rs, no_of_atoms, error, experiment_data):
    dft_distances = [4.2657, 6.0326, 7.3884, 8.5314, 9.5384]

    fig, ax = plt.subplots()
    ax.axvline(dft_distances[0], 0., 3., label='Atomic position of a solid', linewidth=2, c='gray', linestyle='--')
    for distance in dft_distances[1:]:
        ax.axvline(distance, 0, 3, linewidth=2, c='gray', linestyle='--')

    number_density = 0.01315
    bin_width = rs[1] - rs[0]
    distances = distances / no_of_atoms / bin_width
    average = number_density * 4. * np.pi * rs ** 2
    plot_values = distances / average

    ax.plot(rs, np.array(plot_values), label='Input distribution')
    ax.plot(experiment_data[:, 0], experiment_data[:, 1], 'o', color='green', label='Experimental data')
    ax.legend(loc='upper right', facecolor='white', framealpha=1)
    ax.set_title('Error score: ' + str(round(error, 2)))
    ax.set_ylim(0., 4.)
    ax.set_xlim(3.7, 10.)
    ax.set_xlabel('Distance from atom [$\AA$]')
    ax.set_ylabel('Radial distribution function [-]')
    st.pyplot(fig)


def distribution_random(size_of_the_cube, number_of_atoms):
    return np.random.uniform(low=0., high=size_of_the_cube, size=(number_of_atoms, 3))


def random_distribution_maker(number_of_atoms):
    # Make distrbution of the points
    number_density = 1.315*10**-2  # atoms/A**3
    size_of_the_cube = np.power(number_of_atoms/number_density, 1./3.)  # in A
    # print('The length of the modeled volume for ', number_of_atoms, ' atoms is ', size_of_the_cube, ' A')
    atomic_positions = distribution_random(size_of_the_cube, number_of_atoms)
    return atomic_positions

def plot_distribution(distribution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(distribution[:, 0], distribution[:, 1], distribution[:, 2])
    ax.set_title('Visualization of the Input Distribution of Atoms')
    ax.set_xlabel('x [$\AA$]')
    ax.set_ylabel('y [$\AA$]')
    ax.set_zlabel('z [$\AA$]')
    st.pyplot(fig)

def run(student_distribution):
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')
    number_density = 0.01315

    # INPUT
    size_of_the_cube = np.power(len(student_distribution)/number_density, 1./3.)

    margins_of_a_student_distribution = create_margins(size_of_the_cube, student_distribution)
    distances, rs = average_distance_calculator(student_distribution, margins_of_a_student_distribution)

    average = number_density * 4. * np.pi * rs ** 2
    error = (np.abs((experiment[:, 1] - distances/average/len(student_distribution))/experiment[:, 1])).sum() # cumulative of fractional errors
    plot_results(distances, rs, len(student_distribution), error)
    plot_distribution(student_distribution)
    return



# Display an image, centered below the title
st.image("xenon.jpeg", use_column_width=True)

# Let the user upload multiple files via `st.file_uploader`.
uploaded_file = st.file_uploader("Upload documents [.txt], space as a delimeter", type=["txt"], accept_multiple_files=False)
if uploaded_file is not None:
    student_distribution = np.loadtxt(uploaded_file)
    run(student_distribution)
# if uploaded_files is not None and len(uploaded_files) > 0:
#     data_frames = []
#     for uploaded_file in uploaded_files:
#         student_distribution = np.loadtxt(uploaded_file)
#         run(student_distribution)




