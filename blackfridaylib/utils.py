import os

def latest_timestamp(BASE):
    """Return latests timestamped subdirectory in a savedmodel base dir"""
    subdirs = []
    filenames= os.listdir (BASE)
    for filename in filenames: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(BASE), filename)):
            subdirs.append(filename)
    return max(subdirs)

