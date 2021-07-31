import pickle


def read_pickle(path):
    """
    Loads content in the pickle file from `path`.

    Parameters
    ----------
    path : str
        Path of pickle file to be loaded.

    Returns
    -------
    content
        Content of the pickle file.

    """
    opened_file = open(path, 'rb')
    content = pickle.load(opened_file)
    opened_file.close()
    return content


def write_pickle(path, variable):
    """
    Writes `variable` to `path` as a pickle file.

    Parameters
    ----------
    path : str
        Path of pickle file to be saved.
    variable
        A variable to be saved in pickle file.

    """
    opened_file = open(path, 'wb')
    pickle.dump(variable, opened_file)
    opened_file.close()
