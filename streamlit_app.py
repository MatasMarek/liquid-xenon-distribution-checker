import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Liquid Xenon Distribution Checker", layout="wide")

# Adjustable text at the top
default_text = """
Welcome to the Liquid Xenon Distribution Checker!

Upload your atomic distribution data to analyze and compare it with experimental data. You can choose to submit your results to the ranking leaderboard.
"""
custom_text = st.text_area("Introductory Text", default_text, height=150)

st.markdown(f"<div style='text-align: center; color: black;'>{custom_text}</div>", unsafe_allow_html=True)

# Initialize session state for scores
if 'scores' not in st.session_state:
    st.session_state.scores = []

def find_whole(array, values):
    ixs = np.round(find_fractional(array, values)).astype(int)
    return ixs

def find_fractional(array, values):
    step = (array[-1] - array[0]) / (len(array) - 1)
    ixs = (values - array[0]) / step
    return ixs

def create_margins(size_of_the_cube, atomic_positions):
    atomic_positions_margins = np.zeros((1, 3))
    for value_x in [-size_of_the_cube, 0, size_of_the_cube]:
        for value_y in [-size_of_the_cube, 0, size_of_the_cube]:
            for value_z in [-size_of_the_cube, 0, size_of_the_cube]:
                if not (value_x == 0 and value_y == 0 and value_z == 0):
                    shifted_positions = np.copy(atomic_positions) + np.array([value_x, value_y, value_z])
                    atomic_positions_margins = np.concatenate((atomic_positions_margins, shifted_positions))
    atomic_positions_margins = atomic_positions_margins[1:]
    return atomic_positions_margins

def average_distance_calculator(distribution, distribution_margins):
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')[:, 0]
    nbins = len(experiment)
    average_distance_random = np.zeros(nbins)
    r_min, r_max = experiment.min(), experiment.max()
    rs = np.linspace(r_min, r_max, nbins)
    atoms = np.concatenate((distribution, distribution_margins))
    for atom in distribution:
        distances = np.linalg.norm(atoms - atom, axis=1, keepdims=True)
        idxs = find_whole(rs, distances)
        for idx in idxs:
            if 0 <= idx < nbins:
                average_distance_random[int(idx)] += 1.
    return average_distance_random, rs

def plot_results(distances, rs, no_of_atoms, error):
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
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')
    ax.plot(experiment[:, 0], experiment[:, 1], 'o', color='green', label='Experimental data')
    ax.legend(loc='upper right', facecolor='white', framealpha=1)
    ax.set_title('Error score: ' + str(round(error, 2)))
    ax.set_ylim(0., 4.)
    ax.set_xlim(3.7, 10.)
    ax.set_xlabel('Distance from atom [$\\AA$]')
    ax.set_ylabel('Radial distribution function [-]')
    st.pyplot(fig)
    plt.close(fig)
    return

def distribution_random(size_of_the_cube, number_of_atoms):
    return np.random.uniform(low=0., high=size_of_the_cube, size=(number_of_atoms, 3))

def random_distribution_maker(number_of_atoms):
    number_density = 1.315e-2  # atoms/A**3
    size_of_the_cube = np.power(number_of_atoms / number_density, 1./3.)  # in A
    atomic_positions = distribution_random(size_of_the_cube, number_of_atoms)
    return atomic_positions

def plot_distribution(distribution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(distribution[:, 0], distribution[:, 1], distribution[:, 2])
    ax.set_title('Visualization of the Input Distribution of Atoms')
    ax.set_xlabel('x [$\\AA$]')
    ax.set_ylabel('y [$\\AA$]')
    ax.set_zlabel('z [$\\AA$]')
    st.pyplot(fig)
    plt.close(fig)
    return

def run(student_distribution, name, submit_to_ranking):
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')
    number_density = 0.01315
    size_of_the_cube = np.power(len(student_distribution) / number_density, 1./3.)
    margins_of_a_student_distribution = create_margins(size_of_the_cube, student_distribution)
    distances, rs = average_distance_calculator(student_distribution, margins_of_a_student_distribution)
    average = number_density * 4. * np.pi * rs ** 2
    error = (np.abs((experiment[:, 1] - distances / average / len(student_distribution)) / experiment[:, 1])).sum()

    # Plot results
    plot_results(distances, rs, len(student_distribution), error)
    plot_distribution(student_distribution)

    # Optionally store the name and error score in session state
    if submit_to_ranking:
        st.session_state.scores.append({'Name': name, 'Error Score': error})

    # Display rankings
    display_rankings()
    return

def display_rankings():
    if st.session_state.scores:
        # Create a DataFrame from session state scores
        scores_df = pd.DataFrame(st.session_state.scores)
        # Remove duplicates, keep the lowest error score for each user
        scores_df = scores_df.groupby('Name', as_index=False).min()
        # Sort by Error Score
        scores_df = scores_df.sort_values(by='Error Score')
        # Reset index
        scores_df = scores_df.reset_index(drop=True)
        # Display the rankings
        st.markdown("## Rankings")
        st.table(scores_df)
    else:
        st.markdown("## Rankings")
        st.write("No rankings to display yet.")

# Display an image, centered below the title
st.image("xenon.jpeg", use_column_width=True)

# Name input
name = st.text_input('Enter your name')

# Option to submit to ranking
submit_to_ranking = st.checkbox('Submit to ranking', value=True)

# File uploader
uploaded_file = st.file_uploader("Upload document [.txt], space as a delimiter", type=["txt"], accept_multiple_files=False)

# Submit button
if st.button('Submit'):
    if uploaded_file is not None and name:
        # Read the uploaded file as a string
        file_content = uploaded_file.read().decode('utf-8')
        # Convert the string to a StringIO object
        string_io = StringIO(file_content)
        # Load the data using numpy
        try:
            student_distribution = np.loadtxt(string_io)
            run(student_distribution, name, submit_to_ranking)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    elif not name:
        st.warning("Please enter your name before submitting.")
    elif uploaded_file is None:
        st.warning("Please upload a data file before submitting.")
