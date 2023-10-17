import os
import subprocess
import multiprocessing

path = os.getcwd()
path = os.path.join(path, 'main.py')


def worker_0(EXP_ID):
    subprocess.run(('python', path, str(EXP_ID)))


if __name__ == '__main__':

    experiment_repeat = list(range(100))

    process_list = []

    for ii in range(len(experiment_repeat)):

        process_list.append(multiprocessing.Process(target=worker_0, args=(experiment_repeat[ii],)))

        process_list[ii].start()

        process_list[ii].join()
