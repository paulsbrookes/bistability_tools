import os

if __name__ == '__main__':
    queue_dir = 'queue'
    for filename in os.listdir(queue_dir):
        os.system('python pang_spectrum.py ../Scratch/queue -s ' + queue_dir + '/' +filename)