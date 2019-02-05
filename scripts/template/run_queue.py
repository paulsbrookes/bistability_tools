import os

if __name__ == '__main__':
    queue_dir = 'queue'
    for filename in os.listdir(queue_dir):
        os.system('python pang_simulate.py -o ../../Scratch/queue -f spectrum -r True -s ' + queue_dir + '/' +filename)
