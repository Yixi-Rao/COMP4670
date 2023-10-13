##############################
### FILL IN THE FOLLOWING: ###
##############################

U_ID = "u6826541"  # You need to specify!

SUBMISSION_LIST = [
    # No need to submit theory question sheet `./assignment_1.pdf` or code framework `./framework/*`
    './A1_u6826541.pdf',  # Change to the PDF of your theory solutions!
    './emm_question.py',  # Make sure to include this!
    './blr_question.py',  # Make sure to include this!
    './implementation_viewer.ipynb',  # Make sure to include this!
]

# MAKE SURE TO CHECK THE CREATED ZIP FILE HAS EVERYTHING IT SHOULD HAVE BEFORE SUBMITTING!!!

##############################
### CHECKING AND PACKAGING ###
##############################

import os, zipfile

assert './emm_question.py' in SUBMISSION_LIST, "No ./emm_question.py in submission list"
assert './blr_question.py' in SUBMISSION_LIST, "No ./blr_question.py in submission list"
assert any('.pdf' in fname for fname in SUBMISSION_LIST), "No PDF solution file in submission list"

for s in SUBMISSION_LIST:
    assert os.path.exists(s), f'File {s} does not exist'

# Check length and typing of U_ID
assert(len(U_ID) == 8)
assert(type(U_ID) == str)

def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

with zipfile.ZipFile(f'{U_ID}_assignment_1.zip', 'w') as f:
    for s in SUBMISSION_LIST:
        if os.path.isdir(s):
            for s_file in get_all_file_paths(s):
                f.write(s_file)
        else:
            f.write(s)

