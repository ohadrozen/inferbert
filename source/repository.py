from shutil import copyfile
import os

def new_version_and_compare(filenames, repo_dir, exp_id=None):
    # compares each file from the input file list to its previous version ('prev_....py'), and then replaces the prev_ file with current one in case of differences.
    multi_diff = ''
    changed_files = ''
    for filename in filenames:
        prev_file = os.path.join(repo_dir, '_prev_' + filename)
        diffs, diffs_str = compare_files(filename, prev_file)
        if diffs!=[]:
            copyfile(filename, prev_file)
            if exp_id is not None:
                pathname = os.path.join(repo_dir, filename[:-3])
                repo_filename = os.path.join(pathname, str(exp_id) + '_'+ filename)
                if not os.path.exists(pathname):
                    os.mkdir(pathname)
                copyfile(filename, repo_filename)
            changed_files += ', ' + filename if changed_files != '' else filename
            multi_diff += '\n' + filename + ':\n' + diffs_str + '\n'
    if multi_diff!='':
        multi_diff = 'files changed: ' + changed_files + '\n' + multi_diff
        # saving the diff to 'diffs/[exp_id]_diff.txt'
        diff_filename = os.path.join(repo_dir, 'diffs', str(exp_id) + '_diff.txt')
        with open(diff_filename, 'w') as f:
            f.write(multi_diff)

    return multi_diff, changed_files


def compare_files(current_file, prev_file, section_size=3):
    # find all differences between two files.
    # section_size is the minimal chunk size to consider identical when looking for the end of the different chunk

    ## if a new file was added to repository
    if not os.path.exists(prev_file):
        copyfile(current_file, prev_file)
        diffs = ['** New File **']
        diffs_str = '** New File **'
        return diffs, diffs_str

    with open(current_file, 'r') as cur_f:
        cur_lines = cur_f.readlines()
    with open(prev_file, 'r') as prev_f:
        prev_lines = prev_f.readlines()

    prev_len = len(prev_lines)
    cur_len = len(cur_lines)
    diffs = []
    p_i = c_i = 0
    while (c_i < cur_len) and (p_i < prev_len):
        if prev_lines[p_i] != cur_lines[c_i]:
            ## looking for the boundries of the changed section:
            identical_part_found = False # initializing the variable
            for p_j in range(p_i, len(prev_lines) - section_size):
                if identical_part_found: break
                for c_j in range(c_i, len(cur_lines) - section_size):
                    identical_part_found = True     # initializing the variable
                    empty = j = 0
                    while j < section_size:
                        if p_j + j + empty == prev_len or c_j + j + empty == cur_len:   # end of file for either files
                            identical_part_found = False
                            break
                        if prev_lines[p_j + j + empty].strip() == cur_lines[c_j + j + empty].strip() == '':     # ignoring empty lines
                            empty += 1
                            continue
                        if prev_lines[p_j + j + empty] != cur_lines[c_j + j + empty]:
                            identical_part_found = False
                            break
                        j += 1
                    if identical_part_found:    #p_i is now the beginning of the dif, and p_j is the end (not including)
                        diff_prev = ''.join([prev_lines[k] for k in range(p_i, p_j)])
                        diff_cur = ''.join([cur_lines[k] for k in range(c_i, c_j)])
                        diffs.append((c_i, diff_prev, diff_cur))
                        p_i = p_j
                        c_i = c_j
                        break
        p_i +=1
        c_i +=1

    diffs_str = ''
    for i, dif in enumerate(diffs):
        diffs_str += 'prev:\n' + dif[1]
        diffs_str += 'current (line %d):\n'%dif[0] + dif[2]
    return diffs, diffs_str


if __name__ == '__main__':
    files_to_compare = ['modeling_edited.py', 'train_classifier_from_scratch.py']
    repo_dir = 'my_repo'
    diffs_str = new_version_and_compare(files_to_compare, repo_dir, exp_id=23)
    print(diffs_str)