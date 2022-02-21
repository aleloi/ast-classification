# Win doesn't let linux subsystem access the external drive with all
# the data. So instead of using Bash and doing

#   head -n 1 1.csv > concat.csv
#   cat *.csv | grep "Accepted" >> concat.csv

# I'm doing this:

import os


with open('output_concat.csv', 'w') as out_file:
    print('fname,id,contest_id,problem,username,language,verdict,time,memory,testcount,previd,nextid', file=out_file)
    for dirpath, dirnames, fnames in os.walk('./'):
        if dirpath != './': continue
        for fname in fnames:
            if fname == 'output_concat.csv': continue
            if fname.endswith('.csv'):
                with open(fname, 'r') as in_file:
                    for line in in_file.readlines():
                        if 'python,Accepted' in line:
                            first_item = line.split(',')[0]
                            if not first_item.endswith('.csv'):
                                line = fname+','+line
                            print(line, file=out_file, end='')
