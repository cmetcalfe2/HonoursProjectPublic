import os


def findFilesWithExtensionRecursive(path, extension):
    filePaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                filePaths.append(os.path.join(root, file))

    return filePaths


