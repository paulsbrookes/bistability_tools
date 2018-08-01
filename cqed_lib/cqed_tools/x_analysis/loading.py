import h5py
import glob
import pandas as pd
import numpy as np


def iq_loader(path):
    hdf_files = glob.glob(path + '/*.hdf5')

    if len(hdf_files) == 1:

        f = h5py.File(hdf_files[0], 'r')

        idf = pd.DataFrame(f['Traces']['daqbox - CH1 In DC I'][:, 0, :].T)
        n_samples = idf.shape[1]
        dt = f['Traces']['daqbox - CH1 In DC I_t0dt'][0, 1]
        idf.columns = dt * np.arange(n_samples)

        qdf = pd.DataFrame(f['Traces']['daqbox - CH1 In DC Q'][:, 0, :].T)
        n_samples = qdf.shape[1]
        dt = f['Traces']['daqbox - CH1 In DC Q_t0dt'][0, 1]
        qdf.columns = dt * np.arange(n_samples)

        frequencies = frequencies_gen(f)

        f.close()

    else:
        with open(glob.glob(path + '/*I.txt')[0], 'r') as f:
            idf = pd.read_csv(f, skiprows=5, sep='\t')
        with open(glob.glob(path + '/*Q.txt')[0], 'r') as f:
            qdf = pd.read_csv(f, skiprows=5, sep='\t')

        with open(glob.glob(path + '/*I.txt')[0], 'r') as f:
            a = f.readline()
            a = f.readline()
            a = f.readline()
            frequencies = f.readline()
            a = f.readline()

        frequencies = frequencies.split('\n')[0]
        frequencies = frequencies.split('\t')
        frequencies = np.array([float(freq) for freq in frequencies])
        frequencies = frequencies[0::2]

    frequencies *= 1e-9

    columns = idf.columns
    new_columns = np.array([float(index) for index in columns])
    idf.columns = new_columns * 1000000

    columns = qdf.columns
    new_columns = np.array([float(index) for index in columns])
    qdf.columns = new_columns * 1000000

    iup = idf.iloc[1::2]
    iup.index = np.arange(iup.shape[0])
    idown = idf.iloc[0::2]
    idown.index = np.arange(idown.shape[0])
    idiff = iup - idown
    n_runs = idiff.shape[0]

    qup = qdf.iloc[1::2]
    qup.index = np.arange(qup.shape[0])
    qdown = qdf.iloc[0::2]
    qdown.index = np.arange(qdown.shape[0])
    # c = qdown.columns
    # nc = np.array([float(element) for element in c])
    # qdown.columns = nc
    qdiff = qup - qdown

    up = iup + qup * 1j
    down = idown + qdown * 1j

    n_frequencies = up.shape[0]
    frequencies = frequencies[0:n_frequencies]

    up.index = frequencies
    down.index = frequencies

    return down, up


def iq_loader_variation(path):
    hdf_files = glob.glob(path + '/*.hdf5')

    f = h5py.File(hdf_files[0], 'r')

    frequencies = frequencies_gen(f)
    frequencies *= 1e-9
    n_frequencies = frequencies.shape[0]
    idown = pd.DataFrame(f['Traces']['daqbox - CH1 In DC I'][:, 0, 0:n_frequencies].T)
    iup = pd.DataFrame(f['Traces']['daqbox - CH1 In DC I'][:, 0, -n_frequencies:].T)
    qdown = pd.DataFrame(f['Traces']['daqbox - CH1 In DC Q'][:, 0, 0:n_frequencies].T)
    qup = pd.DataFrame(f['Traces']['daqbox - CH1 In DC Q'][:, 0, -n_frequencies:].T)
    dt = f['Traces']['daqbox - CH1 In DC I_t0dt'][0, 1]
    n_samples = idown.shape[1]
    idown.columns = dt * np.arange(n_samples)
    qdown.columns = dt * np.arange(n_samples)
    iup.columns = dt * np.arange(n_samples)
    qup.columns = dt * np.arange(n_samples)

    columns = idown.columns
    new_columns = np.array([float(index) for index in columns])
    idown.columns = new_columns * 1000000

    columns = qdown.columns
    new_columns = np.array([float(index) for index in columns])
    qdown.columns = new_columns * 1000000

    columns = iup.columns
    new_columns = np.array([float(index) for index in columns])
    iup.columns = new_columns * 1000000

    columns = qup.columns
    new_columns = np.array([float(index) for index in columns])
    qup.columns = new_columns * 1000000

    up = iup + qup * 1j
    down = idown + qdown * 1j

    n_frequencies = up.shape[0]
    frequencies = frequencies[0:n_frequencies]

    up.index = frequencies
    down.index = frequencies

    return down, up


def frequencies_gen(f):
    frequencies = np.array([])
    info = f['Step config']['Resonator 2 RF - Frequency']['Step items']
    for block in info:
        new_frequencies = np.linspace(block[3], block[4], block[8])
        frequencies = np.hstack([frequencies, new_frequencies])
    frequencies = np.array(sorted(list(set(frequencies))))
    return frequencies