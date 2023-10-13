import sys
import hashlib



def sha256(file_name):
    sha256 = hashlib.sha256()

    try:
        with open(file_name, 'rb') as f:
            while True:
                # Reading 1 MiB
                data = f.read(1048576)
                if not data:
                    break
                sha256.update(data)
        print(f"{file_name}: {sha256.hexdigest()}")
    except FileNotFoundError:
        print(f'File "{file_name}" not found')
    except IOError:
        print(f'Error trying to access file "{file_name}"')



if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("""
Usage: python3 sha256.py <file>

where <file> is the name of the file you want to compute the sha256.

If you pass more than one file in the command line, the sha256 of each file will
be computed and printed
""")
    else:
        for file_name in sys.argv[1:]:
            sha256(file_name)
