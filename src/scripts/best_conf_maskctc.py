import os
import sys
import glob

if __name__ == "__main__":

    # -- command line arguments
    exps_dir = sys.argv[1]

    exps_metrics = glob.glob( f'{exps_dir}{os.path.sep}*.wer' )

    exps = []
    for exps_metric in exps_metrics:
        k = os.path.basename(exps_metric).split('_')[1]
        p = os.path.basename(exps_metric).split('_')[2].split('.wer')[0]

        with open(exps_metric, 'r') as f:
            lines = f.readlines()
            wer = lines[0].split()[1]
            cer = lines[1].split()[1]
            lid = lines[2].split()[1]

        exps.append( (k, p, wer, cer, lid) )

    sorted_exps = sorted(exps, key=lambda x: x[3])
    for i, sorted_exp in enumerate(sorted_exps):
        print(i, sorted_exp)
